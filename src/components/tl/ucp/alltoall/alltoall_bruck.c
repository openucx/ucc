/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"
#include "tl_ucp_sendrecv.h"
#include "components/mc/ucc_mc.h"
#include "coll_patterns/bruck_alltoall.h"

#define RADIX 2
#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->alltoall_bruck.phase = _phase;                                   \
    } while (0)

enum {
    PHASE_MERGE,
    PHASE_SENDRECV,
    PHASE_BCOPY
};

static inline int msb_pos_for_level(unsigned int nthbit, ucc_rank_t number)
{

    int msb_set = -1;
    unsigned int i;

    for (i = 0; i < nthbit - 1; i++) {
        if (1 & number >> i) {
            msb_set = i;
        }
    }

    return msb_set;
}

static inline int find_seg_index(ucc_rank_t seg_index, int level,
                                 int nsegs_per_rblock)
{
    int block, blockseg;

    if (0 == seg_index) {
        return -1;
    }

    block = msb_pos_for_level(level, seg_index);

    if (block < 0) {
        return -1;
    }

    /* remove block bit from seg_index */
    blockseg = ((seg_index >> (block + 1)) << block) |
                (seg_index & UCC_MASK(block));
    return block * nsegs_per_rblock + blockseg;
}

ucc_status_t ucc_tl_ucp_alltoall_bruck_backward_rotation(void *dst,
                                                         void *src,
                                                         ucc_rank_t trank,
                                                         ucc_rank_t tsize,
                                                         size_t seg_size)
{
    ucc_status_t st;
    ucc_rank_t index, level, nsegs_per_rblock;
    size_t snd_offset;
    int send_buffer_index;

    level = lognum(tsize);
    nsegs_per_rblock = tsize / 2;
    for (index = 1; index < tsize; index++) {
        send_buffer_index = find_seg_index(index, level + 1, nsegs_per_rblock);
        ucc_assert(send_buffer_index >= 0);
        snd_offset = send_buffer_index * seg_size;
        st = ucc_mc_memcpy(PTR_OFFSET(dst, seg_size *
                                      ((trank - index + tsize) % tsize)),
                           PTR_OFFSET(src, snd_offset), seg_size,
                           UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_HOST);
        if (ucc_unlikely(st != UCC_OK)) {
            return st;
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_alltoall_bruck_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t st, global_st;

    global_st = ucc_mc_free(task->alltoall_bruck.scratch_mc_header);
    if (ucc_unlikely(global_st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to free scratch buffer");
    }

    st = ucc_tl_ucp_coll_finalize(&task->super);
    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed finalize collective");
        global_st = st;
    }
    return global_st;
}

void ucc_tl_ucp_alltoall_bruck_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    ucc_rank_t         trank      = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         tsize      = UCC_TL_TEAM_SIZE(team);
    ucc_coll_args_t   *args       = &TASK_ARGS(task);
    void              *scratch    = task->alltoall_bruck.scratch_mc_header->addr;
    void              *mergebuf   = task->alltoall_bruck.dst;
    const ucc_rank_t   nrecv_segs = tsize / 2;
    const size_t       seg_size   = ucc_dt_size(args->src.info.datatype) *
                                    args->src.info.count / tsize;
    ucc_memory_type_t  smtype     = args->src.info.mem_type;
    ucc_memory_type_t  dmtype     = args->dst.info.mem_type;
    ucc_rank_t sendto, recvfrom, step, index;
    void *data;
    ucc_rank_t level, snd_count;
    int send_buffer_index;
    ucc_status_t status;
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t eargs;

    EXEC_TASK_TEST(task->alltoall_bruck.phase,
                   "failed to copy data from user buffer to scratch",
                   task->alltoall_bruck.etask);
    switch (task->alltoall_bruck.phase) {
    case PHASE_SENDRECV:
        goto ALLTOALL_BRUCK_PHASE_SENDRECV;
    case PHASE_BCOPY:
        task->super.status = UCC_OK;
        goto out;
    }

    step = 1 << (task->alltoall_bruck.iteration - 1);
    while (step < tsize) {
        level = task->alltoall_bruck.iteration - 1;
        sendto = get_bruck_send_peer(trank, tsize, step, 1);
        recvfrom = get_bruck_recv_peer(trank, tsize, step, 1);

        snd_count = 0;
        for (index = get_bruck_step_start(step, 1);
             index <= get_bruck_step_finish(tsize - 1, RADIX, 1, step);
             index = GET_NEXT_BRUCK_NUM(index, RADIX, step)) {
            send_buffer_index = find_seg_index(index, level + 1, nrecv_segs);
            if (send_buffer_index == -1) {
                data = PTR_OFFSET(task->alltoall_bruck.src,
                                  ((index + trank) % tsize) * seg_size);
            } else {
                data = PTR_OFFSET(scratch, send_buffer_index * seg_size);
            }
            status = ucc_mc_memcpy(PTR_OFFSET(mergebuf, seg_size * snd_count),
                                   data, seg_size, UCC_MEMORY_TYPE_HOST,
                                   UCC_MEMORY_TYPE_HOST);
            if (ucc_unlikely(UCC_OK != status)) {
                task->super.status = status;
                return;
            }
            snd_count++;
        }
        data = PTR_OFFSET(scratch, level * nrecv_segs * seg_size);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(mergebuf, snd_count * seg_size,
                                         UCC_MEMORY_TYPE_HOST, sendto, team,
                                         task),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(data, snd_count * seg_size,
                                         UCC_MEMORY_TYPE_HOST, recvfrom, team,
                                         task),
                      task, out);
ALLTOALL_BRUCK_PHASE_SENDRECV:
        if (ucc_tl_ucp_test(task) == UCC_INPROGRESS) {
            SAVE_STATE(PHASE_SENDRECV);
            return;
        }
        task->alltoall_bruck.iteration++;
        step = 1 << (task->alltoall_bruck.iteration - 1);
    }

    status = ucc_mc_memcpy(PTR_OFFSET(task->alltoall_bruck.dst, trank * seg_size),
                           PTR_OFFSET(task->alltoall_bruck.src, trank * seg_size),
                           seg_size, UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(status != UCC_OK)) {
        task->super.status = status;
        return;
    }
    status = ucc_tl_ucp_alltoall_bruck_backward_rotation(mergebuf, scratch,
                                                         trank, tsize,
                                                         seg_size);
    if (ucc_unlikely(status != UCC_OK)) {
        task->super.status = status;
        return;
    }

    if (smtype != UCC_MEMORY_TYPE_HOST || dmtype != UCC_MEMORY_TYPE_HOST) {
        task->alltoall_bruck.phase = PHASE_BCOPY;
        status = ucc_coll_task_get_executor(&task->super, &exec);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }

        eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY;
        eargs.copy.src  = mergebuf;
        eargs.copy.dst  = args->dst.info.buffer;
        eargs.copy.len  = seg_size * tsize;
        eargs.flags     = 0;
        status = ucc_ee_executor_task_post(exec, &eargs,
                                           &task->alltoall_bruck.etask);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        EXEC_TASK_TEST(PHASE_BCOPY, "failed to copy data to user buffer",
                       task->alltoall_bruck.etask);
    }

    task->super.status = UCC_OK;
out:
    return;
}

ucc_status_t ucc_tl_ucp_alltoall_bruck_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    size_t             size = ucc_dt_size(args->src.info.datatype) *
                              args->src.info.count;
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t eargs;
    ucc_status_t status;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->alltoall_bruck.iteration = 1;
    task->alltoall_bruck.phase     = PHASE_MERGE;
    task->alltoall_bruck.etask     = NULL;

    if ((args->src.info.mem_type != UCC_MEMORY_TYPE_HOST) ||
        (args->dst.info.mem_type != UCC_MEMORY_TYPE_HOST)) {
        status = ucc_coll_task_get_executor(&task->super, &exec);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }

        eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY;
        eargs.copy.src  = args->src.info.buffer;
        eargs.copy.dst  = task->alltoall_bruck.src;
        eargs.copy.len  = size;
        eargs.flags     = 0;
        status = ucc_ee_executor_task_post(exec, &eargs,
                                           &task->alltoall_bruck.etask);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoall_bruck_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *team,
                                            ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         tsize    = UCC_TL_TEAM_SIZE(tl_team);
    ucc_coll_args_t   *args     = &coll_args->args;
    size_t             ssize    = ucc_dt_size(args->src.info.datatype) *
                                  args->src.info.count;
    size_t             seg_size = ssize / tsize;
    int                is_bcopy = 0;
    size_t scratch_size;
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    ALLTOALL_TASK_CHECK(coll_args->args, tl_team);
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_alltoall_bruck_start;
    task->super.progress = ucc_tl_ucp_alltoall_bruck_progress;
    task->super.finalize = ucc_tl_ucp_alltoall_bruck_finalize;
    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;

    scratch_size = lognum(tsize) * ucc_div_round_up(tsize, 2) * seg_size;
    if ((coll_args->args.src.info.mem_type != UCC_MEMORY_TYPE_HOST) ||
        (coll_args->args.dst.info.mem_type != UCC_MEMORY_TYPE_HOST)) {
        is_bcopy = 1;
        scratch_size += 2 * ssize;
    }

    status = ucc_mc_alloc(&task->alltoall_bruck.scratch_mc_header,
                          scratch_size, UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate scratch buffer");
        ucc_tl_ucp_coll_finalize(&task->super);
        return status;
    }

    if (is_bcopy) {
        task->alltoall_bruck.src =
            PTR_OFFSET(task->alltoall_bruck.scratch_mc_header->addr,
                       lognum(tsize) * ucc_div_round_up(tsize, 2) * seg_size);
        task->alltoall_bruck.dst =
            PTR_OFFSET(task->alltoall_bruck.src, ssize);
    } else {
        task->alltoall_bruck.src = args->src.info.buffer;
        task->alltoall_bruck.dst = args->dst.info.buffer;
    }

    *task_h = &task->super;
    return UCC_OK;

out:
    return status;
}
