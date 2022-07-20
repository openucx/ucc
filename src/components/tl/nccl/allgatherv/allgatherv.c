/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "allgatherv.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/arch/cuda_def.h"

ucc_base_coll_alg_info_t
    ucc_tl_nccl_allgatherv_algs[UCC_TL_NCCL_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_TL_NCCL_ALLGATHERV_ALG_P2P] =
            {.id   = UCC_TL_NCCL_ALLGATHERV_ALG_P2P,
             .name = "p2p",
             .desc = "allgatherv based on nccl point-to-point"},
        [UCC_TL_NCCL_ALLGATHERV_ALG_BCOPY] =
            {.id   = UCC_TL_NCCL_ALLGATHERV_ALG_BCOPY,
             .name = "bcopy",
             .desc = "allgatherv with buffered copy"},
        [UCC_TL_NCCL_ALLGATHERV_ALG_BCAST] =
            {.id   = UCC_TL_NCCL_ALLGATHERV_ALG_BCAST,
             .name = "bcast",
             .desc = "allgatherv based on nccl bcast"},
        [UCC_TL_NCCL_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

#define CHECK_INPLACE(_args, _team)                                            \
    do {                                                                       \
        if (UCC_IS_INPLACE((_args))) {                                         \
            tl_error(UCC_TL_TEAM_LIB((_team)),                                 \
                     "inplace allgatherv is not supported");                   \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while(0)

#define CHECK_USERDEFINED_DT(_args, _team)                                     \
    do {                                                                       \
        if (!UCC_DT_IS_PREDEFINED((_args).src.info.datatype) ||                \
            !UCC_DT_IS_PREDEFINED((_args).dst.info_v.datatype)) {              \
            tl_error(UCC_TL_TEAM_LIB((_team)),                                 \
                     "user defined datatype is not supported");                \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while(0)

ucc_status_t ucc_tl_nccl_allgatherv_p2p_start_gpu(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context :
                                                       team->stream;
    ucc_status_t        status = UCC_OK;
    void               *sbuf   = args->src.info.buffer;
    void               *rbuf   = args->dst.info_v.buffer;
    size_t sdt_size, rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.status = UCC_INPROGRESS;
    sdt_size           = ucc_dt_size(args->src.info.datatype);
    rdt_size           = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgatherv_start", 0);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    count = args->src.info.count;
    if (count != 0) {
        for (peer = 0; peer < size; peer++) {
            NCCLCHECK_GOTO(ncclSend(sbuf, count * sdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    for (peer = 0; peer < size; peer++) {
        count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(
                args, args->dst.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclRecv(PTR_OFFSET(rbuf, displ * rdt_size),
                                    count * rdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

typedef struct {
    void *cpu_sbuf;
    void *staged_sbuf;
    uintptr_t sbuf_len;

    int first_peer_rank;
    void *first_peer_cpu_rbuf;
    uintptr_t first_peer_len;

    int last_peer_rank;
    uintptr_t last_peer_len;
} window_bounds_t;

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static void find_window_bounds(ucc_coll_task_t *coll_task, int round, window_bounds_t *win)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t *args = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team = TASK_TEAM(task);
    size_t sdt_size = ucc_dt_size(args->src.info.datatype);
    size_t rdt_size = ucc_dt_size(args->dst.info_v.datatype);

    /* initialize variables, so we don't accidentally use garbage
     * values */
    win->cpu_sbuf = NULL;
    win->staged_sbuf = NULL;
    win->sbuf_len = 0;
    win->first_peer_rank = -1;
    win->first_peer_cpu_rbuf = NULL;
    win->first_peer_len = 0;
    win->last_peer_rank = -1;
    win->last_peer_len = 0;

    uintptr_t window_start = round * UCC_TL_NCCL_SCRATCH_BUF_SIZE;
    uintptr_t window_end = window_start + UCC_TL_NCCL_SCRATCH_BUF_SIZE;


    /* sbuf setup */
    uintptr_t sbuf_start = 0;
    for (int peer = 0; peer < UCC_TL_TEAM_RANK(team); peer++) {
        sbuf_start += ucc_coll_args_get_count(args, args->dst.info_v.counts, peer) * rdt_size;
    }
    uintptr_t sbuf_end = sbuf_start + args->src.info.count * sdt_size;

    if (sbuf_end > window_start && sbuf_start < window_end) {
        uintptr_t sbuf_offset = 0;
        if (sbuf_start < window_start) {
            sbuf_offset = window_start - sbuf_start;
        }

        win->cpu_sbuf = PTR_OFFSET(args->src.info.buffer, sbuf_offset);
        if (sbuf_start <= window_start) {
            win->staged_sbuf = task->cpu_coll_scratch_buf;
        } else {
            win->staged_sbuf = PTR_OFFSET(task->cpu_coll_scratch_buf, sbuf_start - window_start);
        }
        win->sbuf_len = MIN(sbuf_end, window_end) - MAX(sbuf_start, window_start);
    }


    /* rbuf setup */
    uintptr_t offset = 0;
    int first_peer = 1;
    for (int peer = 0; peer < UCC_TL_TEAM_SIZE(team); peer++) {
        uintptr_t recv_size = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer) * rdt_size;

        if (recv_size == 0) {
            continue;
        } else if (offset + recv_size < window_start) {
            offset += recv_size;
            continue;
        } else if (offset >= window_end) {
            break;
        }

        recv_size = MIN(offset + recv_size, window_end) - MAX(offset, window_start);

        if (first_peer) {
            win->first_peer_rank = peer;
            uintptr_t displ = ucc_coll_args_get_displacement(args, args->dst.info_v.displacements, peer);
            win->first_peer_cpu_rbuf = PTR_OFFSET(args->dst.info_v.buffer, displ * rdt_size);
            win->first_peer_len = recv_size;

            first_peer = 0;
        }

        win->last_peer_rank = peer;
        win->last_peer_len = recv_size;

        offset += recv_size;
    }
}

static void CUDART_CB cpu_allgatherv_copy_in(void *data)
{
    ucc_coll_task_t *coll_task = (ucc_coll_task_t *) data;
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);

    window_bounds_t win;
    find_window_bounds(coll_task, task->cpu_coll_round, &win);

    if (win.sbuf_len != 0) {
        memcpy(win.staged_sbuf, win.cpu_sbuf, win.sbuf_len);
    }
}

static void CUDART_CB cpu_allgatherv_copy_out(void *data)
{
    ucc_coll_task_t *coll_task = (ucc_coll_task_t *) data;
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t *args = &TASK_ARGS(task);
    size_t rdt_size = ucc_dt_size(args->dst.info_v.datatype);

    window_bounds_t win;
    find_window_bounds(coll_task, task->cpu_coll_round, &win);

    void *rbuf = task->cpu_coll_scratch_buf;
    uintptr_t copied = 0;
    for (int peer = win.first_peer_rank; peer <= win.last_peer_rank; peer++) {
        uintptr_t recv_size;

        if (peer == win.first_peer_rank) {
            memcpy(win.first_peer_cpu_rbuf, rbuf, win.first_peer_len);
            copied += win.first_peer_len;
        } else if (peer == win.last_peer_rank) {
            size_t displ = ucc_coll_args_get_displacement(args, args->dst.info_v.displacements, peer);
            memcpy(PTR_OFFSET(args->dst.info_v.buffer, displ * rdt_size),
                   PTR_OFFSET(task->cpu_coll_scratch_buf, copied), win.last_peer_len);
            copied += win.last_peer_len;
        } else {
            uintptr_t copy_size = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer) * rdt_size;
            size_t displ = ucc_coll_args_get_displacement(args, args->dst.info_v.displacements, peer);
            memcpy(PTR_OFFSET(args->dst.info_v.buffer, displ * rdt_size),
                   PTR_OFFSET(task->cpu_coll_scratch_buf, copied), copy_size);
            copied += copy_size;
        }
    }

    task->cpu_coll_round++;

    uintptr_t total_bytes = 0;
    for (int peer = 0; peer < UCC_TL_TEAM_SIZE(team); peer++) {
        total_bytes += ucc_coll_args_get_count(args, args->dst.info_v.counts, peer) * rdt_size;
    }
    int num_rounds = total_bytes / UCC_TL_NCCL_SCRATCH_BUF_SIZE +
        !!(total_bytes % UCC_TL_NCCL_SCRATCH_BUF_SIZE);

    if (task->cpu_coll_round == num_rounds) {
        ucc_mpool_put(task->cpu_coll_scratch_buf);
        task->cpu_coll_scratch_buf = NULL;
    }
}

ucc_status_t ucc_tl_nccl_allgatherv_p2p_start_cpu(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context :
                                                       team->stream;
    ucc_status_t        status = UCC_OK;
    void               *sbuf;
    size_t sdt_size, rdt_size, count, displ;
    size_t sbuf_size;
    ucc_rank_t peer;

    task->super.status = UCC_INPROGRESS;
    sdt_size           = ucc_dt_size(args->src.info.datatype);
    rdt_size           = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgatherv_start", 0);

    ucc_tl_nccl_context_t *ctx = TASK_CTX(task);
    task->cpu_coll_scratch_buf = ucc_mpool_get(&ctx->cpu_staging_scratch_mp);
    if (ucc_unlikely(!task->cpu_coll_scratch_buf)) {
        status = UCC_ERR_NO_MEMORY;
        goto exit_coll;
    }
    task->cpu_coll_round = 0;

    uintptr_t total_bytes = 0;
    for (peer = 0; peer < size; peer++) {
        total_bytes += ucc_coll_args_get_count(args, args->dst.info_v.counts, peer) * rdt_size;
    }
    int num_rounds = total_bytes / UCC_TL_NCCL_SCRATCH_BUF_SIZE +
        !!(total_bytes % UCC_TL_NCCL_SCRATCH_BUF_SIZE);

    for (int i = 0; i < num_rounds; i++) {
        if (args->src.info.count != 0) {
            NCCLCHECK_GOTO(cudaLaunchHostFunc(stream, cpu_allgatherv_copy_in, (void *) coll_task),
                           exit_coll, status, UCC_TL_TEAM_LIB(team));
        }

        window_bounds_t win;
        find_window_bounds(coll_task, i, &win);

        NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
        if (win.sbuf_len != 0) {
            for (peer = 0; peer < size; peer++) {
                NCCLCHECK_GOTO(ncclSend(win.staged_sbuf, win.sbuf_len, ncclChar, peer,
                                        team->nccl_comm, stream),
                               exit_coll, status, UCC_TL_TEAM_LIB(team));
            }
        }

        uintptr_t offset = 0;
        for (peer = win.first_peer_rank; peer <= win.last_peer_rank; peer++) {
            uintptr_t recv_size;

            if (peer == win.first_peer_rank) {
                recv_size = win.first_peer_len;
            } else if (peer == win.last_peer_rank) {
                recv_size = win.last_peer_len;
            } else {
                recv_size = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer) * rdt_size;
            }

            NCCLCHECK_GOTO(ncclRecv(PTR_OFFSET(task->cpu_coll_scratch_buf, offset),
                                    recv_size, ncclChar, peer, team->nccl_comm, stream),
                           exit_coll, status, UCC_TL_TEAM_LIB(team));

            offset += recv_size;
        }
        NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));

        NCCLCHECK_GOTO(cudaLaunchHostFunc(stream, cpu_allgatherv_copy_out, (void *) coll_task),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
    }

    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_p2p_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *     team,
                                             ucc_coll_task_t **    task_h)
{
    ucc_tl_nccl_team_t *nccl_team = ucc_derived_of(team, ucc_tl_nccl_team_t);
    ucc_coll_args_t    *args      = &coll_args->args;
    ucc_status_t        status    = UCC_OK;
    ucc_tl_nccl_task_t *task;

    CHECK_INPLACE(*args, nccl_team);
    CHECK_USERDEFINED_DT(*args, nccl_team);
    task = ucc_tl_nccl_init_task(coll_args, team);
    if (!task) {
        return UCC_ERR_NO_MESSAGE;
    }
    if (args->src.info.mem_type == UCC_MEMORY_TYPE_HOST) {
        task->super.post = ucc_tl_nccl_allgatherv_p2p_start_cpu;
    } else {
        task->super.post = ucc_tl_nccl_allgatherv_p2p_start_gpu;
    }
    *task_h = &task->super;
out:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_bcopy_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task    = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team    = TASK_TEAM(task);
    ucc_rank_t          size    = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee      = coll_task->ee;
    cudaStream_t        stream  = (ee) ? (cudaStream_t) ee->ee_context :
                                                        team->stream;
    ucc_status_t        status  = UCC_OK;
    void               *sbuf    = args->src.info.buffer;
    void               *rbuf    = args->dst.info_v.buffer;
    void               *scratch = task->allgatherv_bcopy.scratch->addr;
    size_t              max_count, rdt_size, sdt_size, displ, scount, rcount;
    ucc_rank_t          peer;

    task->super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgatherv_start", 0);
    max_count = task->allgatherv_bcopy.max_count;
    scount    = args->src.info.count;
    rdt_size  = ucc_dt_size(args->dst.info_v.datatype);
    sdt_size  = ucc_dt_size(args->src.info.datatype);
    if (max_count * rdt_size > scount * sdt_size) {
        CUDA_CHECK_GOTO(cudaMemcpyAsync(PTR_OFFSET(scratch,
                                        max_count * rdt_size * size), sbuf,
                                        scount * sdt_size,
                                        cudaMemcpyDeviceToDevice, stream),
                        exit_coll, status);
        sbuf = PTR_OFFSET(scratch, max_count * rdt_size * size);
    }
    NCCLCHECK_GOTO(ncclAllGather(sbuf, scratch, max_count * rdt_size,
                                 ncclChar, team->nccl_comm, stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < size; peer++) {
        rcount = ucc_coll_args_get_count(args,
                                         args->dst.info_v.counts, peer);
        displ  = ucc_coll_args_get_displacement(args,
                                                args->dst.info_v.displacements,
                                                peer);
        CUDA_CHECK_GOTO(cudaMemcpyAsync(PTR_OFFSET(rbuf, displ * rdt_size),
                                        PTR_OFFSET(scratch,
                                                   peer * max_count * rdt_size),
                                        rcount * rdt_size,
                                        cudaMemcpyDeviceToDevice, stream),
                       exit_coll, status);
    }
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_bcopy_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);

    ucc_mc_free(task->allgatherv_bcopy.scratch);
    return ucc_tl_nccl_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_nccl_allgatherv_bcopy_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_nccl_team_t *nccl_team = ucc_derived_of(team, ucc_tl_nccl_team_t);
    ucc_coll_args_t    *args      = &coll_args->args;
    ucc_status_t        status    = UCC_OK;
    ucc_tl_nccl_task_t *task;
    size_t              max_count, sdt_size, rdt_size;
    ucc_rank_t          peer;

    CHECK_INPLACE(*args, nccl_team);
    CHECK_USERDEFINED_DT(*args, nccl_team);
    task = ucc_tl_nccl_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MESSAGE;
    }

    sdt_size = ucc_dt_size(args->src.info.datatype);
    rdt_size = ucc_dt_size(args->dst.info_v.datatype);
    max_count = ucc_coll_args_get_count(args, args->dst.info_v.counts, 0);
    for (peer = 1; peer < team->params.size; peer++) {
        max_count = ucc_max(ucc_coll_args_get_count(args,
                            args->dst.info_v.counts, peer), max_count);
    }
    task->allgatherv_bcopy.max_count = max_count;
    if (max_count * rdt_size > args->src.info.count * sdt_size) {
        status = ucc_mc_alloc(&task->allgatherv_bcopy.scratch,
                              (team->params.size + 1) * max_count *
                              ucc_dt_size(args->dst.info_v.datatype),
                              UCC_MEMORY_TYPE_CUDA);

    } else {
        status = ucc_mc_alloc(&task->allgatherv_bcopy.scratch, max_count *
                              team->params.size *
                              ucc_dt_size(args->dst.info_v.datatype),
                              UCC_MEMORY_TYPE_CUDA);
    }
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_nccl_free_task(task);
        return status;
    }
    task->super.post     = ucc_tl_nccl_allgatherv_bcopy_start;
    task->super.finalize = ucc_tl_nccl_allgatherv_bcopy_finalize;
    *task_h = &task->super;
out:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context :
                                                       team->stream;
    ucc_status_t        status = UCC_OK;
    void               *sbuf   = args->src.info.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)args->dst.info_v.buffer;
    size_t rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.status = UCC_INPROGRESS;
    rdt_size           = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgatherv_start", 0);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < size; peer++) {
        count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        displ = ucc_coll_args_get_displacement(args,
                                               args->dst.info_v.displacements,
                                               peer);
        NCCLCHECK_GOTO(ncclBroadcast(sbuf, PTR_OFFSET(rbuf, displ * rdt_size),
                                     count * rdt_size, ncclChar, peer,
                                     team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_bcast_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_nccl_team_t *nccl_team = ucc_derived_of(team, ucc_tl_nccl_team_t);
    ucc_coll_args_t    *args      = &coll_args->args;
    ucc_status_t        status    = UCC_OK;
    ucc_tl_nccl_task_t *task;

    CHECK_INPLACE(*args, nccl_team);
    CHECK_USERDEFINED_DT(*args, nccl_team);
    task = ucc_tl_nccl_init_task(coll_args, team);
    if (!task) {
        return UCC_ERR_NO_MESSAGE;
    }
    task->super.post = ucc_tl_nccl_allgatherv_bcast_start;
    *task_h = &task->super;
out:
    return status;
}
