#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_dt_reduce.h"

void ucc_tl_ucp_allreduce_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         trank     = task->subset.myrank;
    ucc_rank_t         tsize     = (ucc_rank_t)task->subset.map.ep_num;
    void              *sbuf      = TASK_ARGS(task).src.info.buffer;
    void              *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  mem_type  = TASK_ARGS(task).dst.info.mem_type;
    size_t             count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    size_t             data_size = count * ucc_dt_size(dt);
    int                num_chunks = tsize; // Number of chunks equals number of ranks
    size_t             chunk_size, offset, remaining;
    ucc_rank_t         sendto, recvfrom;
    void              *recv_buf, *send_buf, *reduce_buf;
    ucc_status_t       status;
    int                step, chunk;

    // Divide data into chunks, rounding up to ensure we cover all data
    chunk_size = ucc_div_round_up(data_size, num_chunks);

    if (UCC_IS_INPLACE(TASK_ARGS(task))) {
        sbuf = rbuf;
    }

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    sendto   = ucc_ep_map_eval(task->subset.map, (trank + 1) % tsize);
    recvfrom = ucc_ep_map_eval(task->subset.map, (trank - 1 + tsize) % tsize);

    /* 
     * In the ring algorithm, each process sends/receives tsize-1 times
     * This is because after tsize-1 steps, each piece of data has traversed 
     * the entire ring and completed its reduction
     */
    while (task->allreduce_ring.step < tsize - 1) {
        step = task->allreduce_ring.step;
        
        /* Resume from the last processed chunk */
        for (chunk = task->allreduce_ring.chunk; chunk < num_chunks; chunk++) {
            offset = chunk * chunk_size;
            remaining = (chunk == num_chunks - 1) ? data_size - offset : chunk_size;

            send_buf  = (step == 0) ? PTR_OFFSET(sbuf, offset) : PTR_OFFSET(rbuf, offset);
            recv_buf  = PTR_OFFSET(task->allreduce_ring.scratch, offset);
            reduce_buf = PTR_OFFSET(rbuf, offset);

            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(send_buf, remaining, mem_type, sendto, team, task),
                task, out);
            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb(recv_buf, remaining, mem_type, recvfrom, team, task),
                task, out);

            /* Save current chunk position before testing progress */
            task->allreduce_ring.chunk = chunk;
            
            if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
                /* Return and resume from this chunk next time */
                return;
            }

            status = ucc_dt_reduce(send_buf, recv_buf, reduce_buf,
                                   remaining / ucc_dt_size(dt),
                                   dt, &TASK_ARGS(task), 0, 0,
                                   task->allreduce_ring.executor,
                                   &task->allreduce_ring.etask);
            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.status = status;
                return;
            }
        }

        task->allreduce_ring.step++;
        /* Reset chunk counter for the next step */
        task->allreduce_ring.chunk = 0;
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_done", 0);
}

ucc_status_t ucc_tl_ucp_allreduce_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allreduce_ring_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_sbgp_t        *sbgp;
    size_t             count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    size_t             data_size = count * ucc_dt_size(dt);
    ucc_status_t       status;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!(task->flags & UCC_TL_UCP_TASK_FLAG_SUBSET) && team->cfg.use_reordering) {
        sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        task->subset.myrank = sbgp->group_rank;
        task->subset.map    = sbgp->map;
    }

    /* Allocate scratch space for the receive buffer */
    status = ucc_mc_alloc(&task->allreduce_ring.scratch_mc_header,
                          data_size, TASK_ARGS(task).dst.info.mem_type);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate scratch buffer");
        return status;
    }
    task->allreduce_ring.scratch = task->allreduce_ring.scratch_mc_header->addr;

    task->allreduce_ring.step = 0;  /* Initialize step counter */
    task->allreduce_ring.chunk = 0;  /* Initialize chunk counter */
    
    task->super.post     = ucc_tl_ucp_allreduce_ring_start;
    task->super.progress = ucc_tl_ucp_allreduce_ring_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_ring_finalize;

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *     team,
                                            ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    task = ucc_tl_ucp_init_task(coll_args, team);
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
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t st, global_st;

    global_st = ucc_mc_free(task->allreduce_ring.scratch_mc_header);
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
