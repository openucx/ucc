/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

/*
 * fragmented buffered copy linear reduce scatterv algorithm
 *
 * Description:
 *      scratch buffer is split into 2 parts to guarantee data consistency e.g.
 *      when ranks in the ring are running at different steps of the algorithm.
 *      2 parts is enough since max difference between ranks steps is 1
 *
 *  Definitions:
 *      blockI  - full send buffer at Rank I
 *      fragI_J - fragment of send buffer at Rank I and step J
 *      NS      - number of steps
 *      NF      - number of fragments
 *      N       - team size
 *
 *  Setup:
 *      max_frag_size = ucc_min(ucc_max(block0, block1, ..., block N -1 ),
 *                              scratch_size / 2 / N)
 *      NF            = ucc_max(block0, block1, ..., block N-1) / max_frag_size
 *      NS            = 1 + NF
 *
 *  Algorithm
 *      for rank R
 *      step 0:    copy fragR_0 to remote scratch buffers for all ranks
 *
 *      step 1:    reduce frag1_0, frag2_0, ..., fragN_0 from local scratch buffer
 *                 to local dst buffer
 *                 copy fragR_1 from local src buffer to remote scratch buffers
 *                 for all ranks
 *      ...
 *
 *      step NS-2: reduce frag1_(NS-3), frag2_(NS-3), ..., fragN_(NS-3) from local
 *                 scratch buffer to local dst buffer
 *                 copy fragR_(NS-2) from local src buffer to remote scratch buffers
 *                 for all ranks
 *
 *      step NS-1: reduce frag1_(NS-2), frag2_(NS-2), ..., fragN_(NS-2) from local
 *                 scratch buffer to local dst buffer
 */

enum
{
    STAGE_SYNC,    /*< Wait for free SYNC segment */
    STAGE_SETUP,   /*< Wait for memhandle setup to finish */
    STAGE_COPIES,  /*< Linear algorithm is running */
    STAGE_BARRIER, /*< Linear algorithm is done, waiting for
                    *  other ranks to finish */
};

ucc_status_t
ucc_tl_cuda_reduce_scatterv_linear_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

/*
 * get maximum size of scratch memory evenly divisible by
 * (2 * team_size * datatype_size)
 */
static inline size_t get_scratch_size(ucc_tl_cuda_team_t *team,
                                      ucc_datatype_t      dt)
{
    size_t     dt_size = ucc_dt_size(dt);
    ucc_rank_t tsize   = UCC_TL_TEAM_SIZE(team);
    size_t     divisor = 2 * dt_size * tsize;

    return (UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size / divisor) * divisor;
}

/* get stride between consecutive rank slots in scratch memory */
static inline size_t get_scratch_stride(ucc_tl_cuda_team_t *team,
                                        ucc_datatype_t dt)
{
    size_t     step_ssize = get_scratch_size(team, dt) / 2;
    ucc_rank_t tsize      = UCC_TL_TEAM_SIZE(team);

    return step_ssize / tsize;
}

/* get offset of rank slot inside scratch  */
static inline size_t get_scratch_offset(ucc_tl_cuda_team_t *team,
                                        ucc_datatype_t dt, ucc_rank_t rank)
{
    size_t     step_ssize = get_scratch_size(team, dt) / 2;
    ucc_rank_t tsize      = UCC_TL_TEAM_SIZE(team);

    return rank * step_ssize / tsize;
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_linear_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_status_t        status;

    set_rank_step(task, trank, 0, 0);
    ucc_memory_cpu_store_fence();
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_linear_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);

    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_linear_copy(ucc_tl_cuda_task_t *task,
                                        ucc_ee_executor_t *exec, void *sbuf,
                                        int step, size_t remote_offset,
                                        ucc_ee_executor_task_t **etask)
{
    ucc_tl_cuda_team_t *team      = TASK_TEAM(task);
    ucc_rank_t          trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize     = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t      dt        = task->reduce_scatterv_linear.dt;
    size_t              dt_size   = ucc_dt_size(dt);
    int                 nfrags    = task->reduce_scatterv_linear.num_frags;
    size_t scratch_offset, scratch_stride, send_size, frag_size, frag_offset,
           rank_offset;
    ucc_ee_executor_task_args_t  eargs;
    ucc_rank_t i, nv, peer;

    scratch_offset  = get_scratch_offset(team, dt, trank);
    scratch_stride  = get_scratch_stride(team, dt);
    eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY_MULTI;
    eargs.flags     = 0;
    for (i = 0, nv = 0; i < tsize; i++) {
        peer = (trank + i) % UCC_TL_TEAM_SIZE(team);
        if (peer == trank) {
            continue;
        }
        send_size   = task->reduce_scatterv_linear.get_count(task, peer);
        frag_size   = ucc_buffer_block_count(send_size, nfrags, step);
        frag_offset = ucc_buffer_block_offset(send_size, nfrags, step);
        rank_offset = task->reduce_scatterv_linear.get_offset(task, peer);

        if (frag_size == 0) {
            continue;
        }
        eargs.copy_multi.src[nv]    = PTR_OFFSET(sbuf, (rank_offset + frag_offset) * dt_size);
        eargs.copy_multi.counts[nv] = frag_size * dt_size;
        if (trank < peer) {
            eargs.copy_multi.dst[nv] = PTR_OFFSET(TASK_SCRATCH(task, peer),
                remote_offset + scratch_offset);
        } else {
            eargs.copy_multi.dst[nv] = PTR_OFFSET(TASK_SCRATCH(task, peer),
                remote_offset + scratch_offset - scratch_stride);
        }
        nv++;
    }
    if (nv == 0) {
        *etask = NULL;
        return UCC_OK;
    }
    eargs.copy_multi.num_vectors = nv;
    return ucc_ee_executor_task_post(exec, &eargs, etask);
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_linear_reduce(ucc_tl_cuda_task_t *task,
                                          ucc_ee_executor_t *exec, void *sbuf,
                                          void *rbuf, int step,
                                          size_t local_offset,
                                          ucc_ee_executor_task_t **etask)
{
    ucc_tl_cuda_team_t *team      = TASK_TEAM(task);
    ucc_rank_t          trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize     = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t      dt        = task->reduce_scatterv_linear.dt;
    size_t              dt_size   = ucc_dt_size(dt);
    int                 nfrags    = task->reduce_scatterv_linear.num_frags;
    ucc_coll_args_t    *args      = &TASK_ARGS(task);
    size_t send_size, frag_size, frag_offset, rank_offset;
    ucc_ee_executor_task_args_t  eargs;

    send_size   = task->reduce_scatterv_linear.get_count(task, trank);
    frag_size   = ucc_buffer_block_count(send_size, nfrags, step);
    frag_offset = ucc_buffer_block_offset(send_size, nfrags, step);
    rank_offset = task->reduce_scatterv_linear.get_offset(task, trank);

    if (frag_size == 0) {
        *etask = NULL;
        return UCC_OK;
    }

    eargs.task_type             = UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED;
    eargs.flags                 = 0;
    eargs.reduce_strided.src1   = PTR_OFFSET(sbuf, (rank_offset + frag_offset) * dt_size);
    eargs.reduce_strided.src2   = PTR_OFFSET(TASK_SCRATCH(task, trank), local_offset);
    eargs.reduce_strided.stride = get_scratch_stride(team, dt);
    eargs.reduce_strided.count  = frag_size;
    eargs.reduce_strided.dst    = PTR_OFFSET(rbuf, frag_offset * dt_size);
    eargs.reduce_strided.op     = args->op;
    eargs.reduce_strided.n_src2 = tsize - 1;
    eargs.reduce_strided.dt     = dt;
    eargs.flags                 = 0;

    return ucc_ee_executor_task_post(exec, &eargs, etask);
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_linear_progress_frag(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *    team      = TASK_TEAM(task);
    ucc_rank_t              trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t              tsize     = UCC_TL_TEAM_SIZE(team);
    ucc_coll_args_t *       args      = &TASK_ARGS(task);
    ucc_datatype_t          dt        = task->reduce_scatterv_linear.dt;
    size_t                  ssize     = get_scratch_size(team, dt);
    int                     nfrags    = task->reduce_scatterv_linear.num_frags;
    int                     num_steps = nfrags + 1;
    ucc_ee_executor_task_t *etask;
    ucc_status_t            st;
    ucc_ee_executor_t      *exec;
    int                     step, i;
    void                   *sbuf, *rbuf;
    size_t local_offset, remote_offset, rank_offset;

    st = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(st != UCC_OK)) {
        return st;
    }

    step = get_rank_step(task, trank, 0);
    while (step < num_steps) {
        if ((task->reduce_scatterv_linear.exec_task[0] != NULL) ||
            (task->reduce_scatterv_linear.exec_task[1] != NULL)) {
            for (i = 0; i < 2; i++) {
                etask = task->reduce_scatterv_linear.exec_task[i];
                if (etask != NULL) {
                    st = ucc_ee_executor_task_test(etask);
                    if (st == UCC_OK) {
                        ucc_ee_executor_task_finalize(etask);
                        task->reduce_scatterv_linear.exec_task[i] = NULL;
                    } else {
                        if (ucc_likely(st > 0)) {
                            return UCC_INPROGRESS;
                        }
                        return st;
                    }
                }
            }
            step++;
            set_rank_step(task, trank, step, 0);
            continue;
        }

        for (i = 0; i < tsize; i++) {
            if (get_rank_step(task, i, 0) < step) {
                return UCC_INPROGRESS;
            }
        }

        if (step % 2) {
            remote_offset = ssize / 2;
            local_offset  = 0;
        } else {
            remote_offset = 0;
            local_offset  = ssize / 2;
        }

        if (UCC_IS_INPLACE(*args)) {
            rank_offset = task->reduce_scatterv_linear.get_offset(task, trank);
            sbuf = task->reduce_scatterv_linear.rbuf;
            rbuf = PTR_OFFSET(sbuf, rank_offset * ucc_dt_size(dt));
        } else {
            sbuf = task->reduce_scatterv_linear.sbuf;
            rbuf = task->reduce_scatterv_linear.rbuf;
        }

        if (step == 0) {
            st = ucc_tl_cuda_reduce_scatterv_linear_copy(task, exec, sbuf, step,
                remote_offset, &task->reduce_scatterv_linear.exec_task[0]);
        } else if (step == (num_steps - 1)) {
            st = ucc_tl_cuda_reduce_scatterv_linear_reduce(task, exec, sbuf, rbuf,
                step - 1, local_offset, &task->reduce_scatterv_linear.exec_task[1]);
        } else {
            st = ucc_tl_cuda_reduce_scatterv_linear_copy(task, exec, sbuf, step,
                remote_offset, &task->reduce_scatterv_linear.exec_task[0]);
            if (ucc_unlikely(st != UCC_OK)) {
                return st;
            }
            st = ucc_tl_cuda_reduce_scatterv_linear_reduce(task, exec, sbuf, rbuf,
                step - 1, local_offset, &task->reduce_scatterv_linear.exec_task[1]);
        }
        if (ucc_unlikely(st != UCC_OK)) {
            return st;
        }

        if ((task->reduce_scatterv_linear.exec_task[0] == NULL) &&
            (task->reduce_scatterv_linear.exec_task[1] == NULL)) {
            /* have no work to do at current step, go to next step */
            step++;
            set_rank_step(task, trank, step, 0);
            continue;
        }
    }

    return UCC_OK;
}

void ucc_tl_cuda_reduce_scatterv_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_status_t        st;

    task->super.status = UCC_INPROGRESS;
    switch (task->reduce_scatterv_linear.stage) {
    case STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        st = ucc_tl_cuda_reduce_scatterv_linear_setup_start(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->reduce_scatterv_linear.stage = STAGE_SETUP;
    case STAGE_SETUP:
        st = ucc_tl_cuda_reduce_scatterv_linear_setup_test(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->reduce_scatterv_linear.stage = STAGE_COPIES;
    case STAGE_COPIES:
        st = ucc_tl_cuda_reduce_scatterv_linear_progress_frag(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }

        st = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.status = st;
            return;
        }
        task->reduce_scatterv_linear.stage = STAGE_BARRIER;
    default:
        ucc_assert(task->reduce_scatterv_linear.stage == STAGE_BARRIER);
        break;
    }
    task->super.status =
        ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
    if (task->super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
    }
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t *   args  = &TASK_ARGS(task);
    ucc_rank_t          tsize = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t      dt    = task->reduce_scatterv_linear.dt;
    ucc_rank_t          i;
    size_t              send_size, frag_size, ssize;

    /* need to set rbuf in collective start since frag_setup of pipeline
     * schedule can update pointer
     */
    if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
        task->reduce_scatterv_linear.rbuf = args->dst.info.buffer;
    } else {
        task->reduce_scatterv_linear.rbuf = args->dst.info_v.buffer;
    }

    task->reduce_scatterv_linear.stage = STAGE_SYNC;
    task->reduce_scatterv_linear.sbuf  = args->src.info.buffer;
    send_size = task->reduce_scatterv_linear.get_count(task, 0);
    for (i = 1; i < tsize; i++) {
        send_size =
            ucc_max(send_size, task->reduce_scatterv_linear.get_count(task, i));
    }

    if (send_size == 0) {
        task->super.status = UCC_OK;
        return ucc_task_complete(&task->super);
    }

    ssize = get_scratch_size(team, dt);
    frag_size = ucc_min(ssize / 2 / ucc_dt_size(dt) / tsize, send_size);
    task->reduce_scatterv_linear.num_frags = ucc_div_round_up(send_size, frag_size);

    memset(task->reduce_scatterv_linear.exec_task, 0,
           2 * sizeof(ucc_ee_executor_task_t*));
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_linear_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *     tl_team,
                                        ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t status;

    if (coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_connected(team->topo) ||
        UCC_TL_TEAM_SIZE(team) - 1 > UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->reduce_scatterv_linear.get_count  =
        ucc_tl_cuda_reduce_scatterv_get_count;
    task->reduce_scatterv_linear.get_offset =
        ucc_tl_cuda_reduce_scatterv_get_offset;
    task->reduce_scatterv_linear.dt         =
            coll_args->args.dst.info_v.datatype;
    task->super.flags          |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_tl_cuda_reduce_scatterv_linear_start;
    task->super.progress       = ucc_tl_cuda_reduce_scatterv_linear_progress;
    task->super.finalize       = ucc_tl_cuda_reduce_scatterv_linear_finalize;
    task->bar                  = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
