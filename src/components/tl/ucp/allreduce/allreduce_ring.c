#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "tl_ucp_copy.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_dt_reduce.h"
#include "components/ec/ucc_ec.h"

/* Phase definitions */
#define RING_PHASE_COPY 0  /* Initial src->dst copy (non-inplace only) */
#define RING_PHASE_RS   1  /* Reduce-Scatter phase */
#define RING_PHASE_AG   2  /* Allgather phase */
#define RING_PHASE_DONE 3  /* Finished */

static inline void compute_rs_params(
    const int rank, const int step, const int tsize, const size_t chunk_sz,
    const size_t data_sz, size_t *send_off, size_t *send_n, size_t *recv_off,
    size_t *recv_n)
{
    int send_id = (rank - step + tsize) % tsize;
    int recv_id = (rank - step - 1 + tsize) % tsize;

    *send_off   = (size_t)send_id * chunk_sz;
    *send_n     = (*send_off >= data_sz) ? 0
                                         : ((chunk_sz < (data_sz - *send_off))
                                                ? chunk_sz
                                                : (data_sz - *send_off));
    *recv_off   = (size_t)recv_id * chunk_sz;
    *recv_n     = (*recv_off >= data_sz) ? 0
                                         : ((chunk_sz < (data_sz - *recv_off))
                                                ? chunk_sz
                                                : (data_sz - *recv_off));
}

static inline void compute_ag_params(
    const int rank, const int step, const int tsize, const size_t chunk_sz,
    const size_t data_sz, size_t *send_off, size_t *send_n, size_t *recv_off,
    size_t *recv_n)
{
    int send_id = (rank - step + 1 + tsize) %
                  tsize; /* chunk we currently own */
    int recv_id = (rank - step + tsize) % tsize;

    *send_off   = (size_t)send_id * chunk_sz;
    *send_n     = (*send_off >= data_sz) ? 0
                                         : ((chunk_sz < (data_sz - *send_off))
                                                ? chunk_sz
                                                : (data_sz - *send_off));
    *recv_off   = (size_t)recv_id * chunk_sz;
    *recv_n     = (*recv_off >= data_sz) ? 0
                                         : ((chunk_sz < (data_sz - *recv_off))
                                                ? chunk_sz
                                                : (data_sz - *recv_off));
}

static void post_p2p_ops(
    ucc_tl_ucp_task_t *task, void *send_buf, size_t send_n, void *recv_buf,
    size_t recv_n, ucc_rank_t send_to, ucc_rank_t recv_from,
    ucc_memory_type_t memt)
{
    if (send_n > 0) {
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(
                send_buf, send_n, memt, send_to, TASK_TEAM(task), task),
            task,
            err);
    }
    if (recv_n > 0) {
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(
                recv_buf, recv_n, memt, recv_from, TASK_TEAM(task), task),
            task,
            err);
    }
err:
    return;
}

static int progress_rs(
    ucc_tl_ucp_task_t *task, const int rank, const int tsize,
    const size_t data_sz, const size_t dt_size, void *rbuf,
    ucc_memory_type_t memt)
{
    int          step = task->allreduce_ring.step;
    size_t       send_off, recv_off, send_n, recv_n;
    ucc_status_t st;

    if (task->allreduce_ring.etask != NULL) {
        st = ucc_ee_executor_task_test(task->allreduce_ring.etask);
        if (st == UCC_INPROGRESS) {
            return 0; /* Still in progress */
        } else if (ucc_unlikely(st != UCC_OK)) {
            tl_error(
                UCC_TASK_LIB(task),
                "async reduction failed with %s",
                ucc_status_string(st));
            task->super.status = st;
            return 0;
        }

        ucc_ee_executor_task_finalize(task->allreduce_ring.etask);
        task->allreduce_ring.etask = NULL;

        task->allreduce_ring.step++;
        task->allreduce_ring.p2p_posted = 0;

        if (task->allreduce_ring.step == (tsize - 1)) {
            task->allreduce_ring.phase = RING_PHASE_AG;
            task->allreduce_ring.step  = 0;
        }
        return 1; // More progress is possible
    }

    compute_rs_params(
        rank,
        step,
        tsize,
        task->allreduce_ring.chunk_size,
        data_sz,
        &send_off,
        &send_n,
        &recv_off,
        &recv_n);

    if (send_n == 0 && recv_n == 0) {
        task->allreduce_ring.step++;
        task->allreduce_ring.p2p_posted = 0;

        if (task->allreduce_ring.step == (tsize - 1)) {
            task->allreduce_ring.phase = RING_PHASE_AG;
            task->allreduce_ring.step  = 0;
        }
        return 1; // More immediate progress is possible
    }

    /* Post the send/recv only once per step */
    if (!task->allreduce_ring.p2p_posted) {
        void      *send_buf = PTR_OFFSET(rbuf, send_off);
        void      *recv_buf = task->allreduce_ring.scratch;
        ucc_rank_t send_to  = ucc_ep_map_eval(
            task->subset.map, (rank + 1) % tsize);
        ucc_rank_t recv_from = ucc_ep_map_eval(
            task->subset.map, (rank - 1 + tsize) % tsize);

        post_p2p_ops(
            task, send_buf, send_n, recv_buf, recv_n, send_to, recv_from, memt);
        task->allreduce_ring.p2p_posted = 1;
        return 0; // Need to wait for completion
    }

    /* Test for completion once and yield if still in progress */
    ucc_status_t test_status = ucc_tl_ucp_test(task);
    if (test_status == UCC_INPROGRESS) {
        ucp_worker_progress(TASK_TEAM(task)->worker->ucp_worker);
        return 0;
    }
    if (test_status != UCC_OK) {
        tl_error(
            UCC_TASK_LIB(task),
            "P2P test failed with %s",
            ucc_status_string(test_status));
        task->super.status = test_status;
        return 0;
    }

    /* Process incoming chunk by reducing it into the proper location */
    if (recv_n > 0) {
        void  *final_buf = PTR_OFFSET(rbuf, recv_off);
        size_t count     = recv_n / dt_size;

        /* Ensure count is valid */
        if (count == 0) {
            tl_error(
                UCC_TASK_LIB(task),
                "Invalid count calculated: %zu (recv_n=%zu, dt_size=%zu)",
                count,
                recv_n,
                dt_size);
            task->super.status = UCC_ERR_INVALID_PARAM;
            return 0;
        }

        /* For all data types, correctly reduce the received data */
        st = ucc_dt_reduce(
            task->allreduce_ring.scratch,
            final_buf,
            final_buf,
            count,
            TASK_ARGS(task).dst.info.datatype,
            &TASK_ARGS(task),
            0,
            0,
            task->allreduce_ring.executor,
            &task->allreduce_ring.etask);

        if (ucc_unlikely(st != UCC_OK)) {
            tl_error(
                UCC_TASK_LIB(task),
                "dt_reduce failed with %s",
                ucc_status_string(st));
            task->super.status = st;
            return 0;
        }

        /* Check if reduction is asynchronous */
        if (task->allreduce_ring.etask != NULL) {
            return 0; // Wait for reduction to complete in next progress call
        }
    }

    /* Step complete */
    task->allreduce_ring.step++;
    task->allreduce_ring.p2p_posted = 0;

    /* If we've finished RS steps, switch to Allgather phase */
    if (task->allreduce_ring.step == (tsize - 1)) {
        task->allreduce_ring.phase = RING_PHASE_AG;
        task->allreduce_ring.step  = 0;
    }
    return 1; // More progress is possible
}

static int progress_ag(
    ucc_tl_ucp_task_t *task, const int rank, const int tsize,
    const size_t data_sz, void *rbuf, ucc_memory_type_t memt)
{
    int    step = task->allreduce_ring.step;
    size_t send_off, recv_off, send_n, recv_n;

    compute_ag_params(
        rank,
        step,
        tsize,
        task->allreduce_ring.chunk_size,
        data_sz,
        &send_off,
        &send_n,
        &recv_off,
        &recv_n);

    /* If no data to send/receive, advance immediately */
    if (send_n == 0 && recv_n == 0) {
        task->allreduce_ring.step++;
        task->allreduce_ring.p2p_posted = 0;

        /* If we've finished all AG steps, we're done */
        if (task->allreduce_ring.step == (tsize - 1)) {
            task->allreduce_ring.phase = RING_PHASE_DONE;
        }
        return 1; // More progress is possible
    }

    /* Post the send/recv only once per step */
    if (!task->allreduce_ring.p2p_posted) {
        void      *send_buf = PTR_OFFSET(rbuf, send_off);
        void      *recv_buf = PTR_OFFSET(rbuf, recv_off);
        ucc_rank_t send_to  = ucc_ep_map_eval(
            task->subset.map, (rank + 1) % tsize);
        ucc_rank_t recv_from = ucc_ep_map_eval(
            task->subset.map, (rank - 1 + tsize) % tsize);

        post_p2p_ops(
            task, send_buf, send_n, recv_buf, recv_n, send_to, recv_from, memt);
        task->allreduce_ring.p2p_posted = 1;
        return 0; // Need to wait for completion
    }

    /* Test for completion once and yield if still in progress */
    ucc_status_t test_status = ucc_tl_ucp_test(task);
    if (test_status == UCC_INPROGRESS) {
        ucp_worker_progress(TASK_TEAM(task)->worker->ucp_worker);
        return 0;
    }
    if (test_status != UCC_OK) {
        tl_error(
            UCC_TASK_LIB(task),
            "P2P test failed with %s",
            ucc_status_string(test_status));
        task->super.status = test_status;
        return 0;
    }

    /* Step complete */
    task->allreduce_ring.step++;
    task->allreduce_ring.p2p_posted = 0;

    /* If we've finished all AG steps, switch to Allgather phase */
    if (task->allreduce_ring.step == (tsize - 1)) {
        task->allreduce_ring.phase = RING_PHASE_DONE;
    }
    return 1;
}

void ucc_tl_ucp_allreduce_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t    *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_context_t *ctx     = TASK_CTX(task);
    const int             rank    = task->subset.myrank;
    const int             tsize   = task->subset.map.ep_num;
    void                 *sbuf    = TASK_ARGS(task).src.info.buffer;
    void                 *rbuf    = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t     memt    = TASK_ARGS(task).dst.info.mem_type;
    const size_t          count   = TASK_ARGS(task).dst.info.count;
    const ucc_datatype_t  dt      = TASK_ARGS(task).dst.info.datatype;
    const size_t          dt_size = ucc_dt_size(dt);
    const size_t          data_sz = count * dt_size;
    int                   can_make_progress = 1;
    ucc_status_t          status;

    /* Make sure we have a valid executor pointer */
    if (!task->allreduce_ring.executor) {
        status = ucc_coll_task_get_executor(
            &task->super, &task->allreduce_ring.executor);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(
                UCC_TASK_LIB(task),
                "failed to get executor for allreduce ring task");
            task->super.status = status;
            return;
        }
    }

    /* Handle async copy phase for non-inplace operations */
    if (task->allreduce_ring.phase == RING_PHASE_COPY) {
        if (task->allreduce_ring.copy_task != NULL) {
            /* Test for copy completion */
            status = ctx->copy.test(ctx, task->allreduce_ring.copy_task);
            if (status == UCC_INPROGRESS) {
                return; /* Still copying, yield */
            }
            ctx->copy.finalize(task->allreduce_ring.copy_task);
            task->allreduce_ring.copy_task = NULL;
            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(UCC_TASK_LIB(task), "async copy failed");
                task->super.status = status;
                return;
            }
            /* Copy complete, move to reduce-scatter phase */
            task->allreduce_ring.phase = RING_PHASE_RS;
        } else {
            /* Post the async copy */
            status = ctx->copy.post(
                rbuf, memt, NULL,  /* dst, dst_mtype, dst_memh */
                sbuf, memt, NULL,  /* src, src_mtype, src_memh */
                data_sz, task,
                &task->allreduce_ring.copy_task);
            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(UCC_TASK_LIB(task), "failed to post async copy");
                task->super.status = status;
                return;
            }
            return; /* Yield to let copy progress */
        }
    }

    /* Progress the algorithm until no more immediate progress can be made */
    can_make_progress = 1;
    while (can_make_progress) {
        can_make_progress = 0;

        if (task->allreduce_ring.phase == RING_PHASE_RS &&
            task->allreduce_ring.step < (tsize - 1)) {
            can_make_progress = progress_rs(
                task, rank, tsize, data_sz, dt_size, rbuf, memt);
        } else if (
            task->allreduce_ring.phase == RING_PHASE_AG &&
            task->allreduce_ring.step < (tsize - 1)) {
            can_make_progress = progress_ag(
                task, rank, tsize, data_sz, rbuf, memt);
        } else if (task->allreduce_ring.phase == RING_PHASE_DONE) {
            task->super.status = UCC_OK;
            UCC_TL_UCP_PROFILE_REQUEST_EVENT(
                coll_task, "ucp_allreduce_ring_done", 0);
            return;
        }
    }
}

ucc_status_t ucc_tl_ucp_allreduce_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_start", 0);

    /* For non-inplace, start with async copy phase; otherwise skip to RS */
    if (UCC_IS_INPLACE(TASK_ARGS(task))) {
        task->allreduce_ring.phase = RING_PHASE_RS;
    } else {
        task->allreduce_ring.phase = RING_PHASE_COPY;
    }
    task->allreduce_ring.step       = 0;
    task->allreduce_ring.p2p_posted = 0;
    task->allreduce_ring.copy_task  = NULL;

    if (task->allreduce_ring.etask != NULL) {
        ucc_ee_executor_task_finalize(task->allreduce_ring.etask);
        task->allreduce_ring.etask = NULL;
    }

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    status = ucc_coll_task_get_executor(
        &task->super, &task->allreduce_ring.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    return ucc_progress_queue_enqueue(
        UCC_TL_CORE_CTX(TASK_TEAM(task))->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allreduce_ring_init_common(ucc_tl_ucp_task_t *task)
{
    size_t             count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    size_t             dt_size   = ucc_dt_size(dt);
    size_t             data_size = count * dt_size;
    size_t             chunk_size;
    ucc_status_t       status;
    ucc_sbgp_t        *sbgp;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!(task->flags & UCC_TL_UCP_TASK_FLAG_SUBSET) &&
        team->cfg.use_reordering) {
        sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        task->subset.myrank = sbgp->group_rank;
        task->subset.map    = sbgp->map;
    }

    /* Adaptive chunk sizing: choose at least one 8 KiB MTU‑worth of elements
       to avoid single‑byte chunks for micro‑benchmarks. */
    size_t elems_per_chunk = ucc_div_round_up(count, task->subset.map.ep_num);
    size_t min_elems_per_chunk  = ucc_max((size_t)(8192 / dt_size), (size_t)1);
    size_t base_elems_per_chunk = ucc_max(min_elems_per_chunk, elems_per_chunk);
    chunk_size = task->allreduce_ring.chunk_size = base_elems_per_chunk *
                                                   dt_size;

    tl_debug(
        UCC_TASK_LIB(task),
        "Ring allreduce init: count=%zu, dt_size=%zu, data_size=%zu bytes, "
        "tsize=%lu, elems_per_chunk=%zu, chunk_size=%zu bytes",
        count,
        dt_size,
        data_size,
        task->subset.map.ep_num,
        base_elems_per_chunk,
        chunk_size);

    size_t
        scratch_size = chunk_size +
                       dt_size; // +dt_size for potential padding or alignment
    status = ucc_mc_alloc(
        &task->allreduce_ring.scratch_mc_header,
        scratch_size,
        TASK_ARGS(task).dst.info.mem_type);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate scratch buffer");
        return status;
    }

    task->allreduce_ring.scratch = task->allreduce_ring.scratch_mc_header->addr;

    task->allreduce_ring.step       = 0;
    task->allreduce_ring.phase      = RING_PHASE_RS;
    task->allreduce_ring.p2p_posted = 0;
    task->allreduce_ring.etask      = NULL;
    task->allreduce_ring.executor   = NULL;
    task->allreduce_ring.copy_task  = NULL;
    task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;

    task->super.post     = ucc_tl_ucp_allreduce_ring_start;
    task->super.progress = ucc_tl_ucp_allreduce_ring_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_ring_finalize;

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_ring_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task   = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_tl_ucp_allreduce_ring_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t    *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_context_t *ctx  = TASK_CTX(task);
    ucc_status_t          st, global_st = UCC_OK;

    ucc_assert(task->allreduce_ring.etask == NULL);

    /* Clean up any pending copy task */
    if (task->allreduce_ring.copy_task != NULL) {
        ctx->copy.finalize(task->allreduce_ring.copy_task);
        task->allreduce_ring.copy_task = NULL;
    }

    st = ucc_mc_free(task->allreduce_ring.scratch_mc_header);
    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to free scratch buffer");
        global_st = st;
    }

    st = ucc_tl_ucp_coll_finalize(&task->super);
    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed finalize collective");
        global_st = (global_st == UCC_OK) ? st : global_st;
    }

    return global_st;
}
