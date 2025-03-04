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
    size_t             chunk_size, offset, remaining;
    ucc_rank_t         sendto, recvfrom;
    void              *recv_buf, *send_buf, *reduce_buf;
    ucc_status_t       status;

    int num_chunks = tsize; // Use the number of ranks as the number of chunks (this is dynamic)
    chunk_size = (data_size + num_chunks - 1) / num_chunks; // Ensure chunks fit into data evenly

    if (UCC_IS_INPLACE(TASK_ARGS(task))) {
        sbuf = rbuf;
    }

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    sendto   = ucc_ep_map_eval(task->subset.map, (trank + 1) % tsize);
    recvfrom = ucc_ep_map_eval(task->subset.map, (trank - 1 + tsize) % tsize);

    while (task->tagged.send_posted < tsize - 1) {
        int step = task->tagged.send_posted;

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            offset = chunk * chunk_size;
            remaining = (chunk == num_chunks - 1) ? data_size - offset : chunk_size;

            send_buf  = (step == 0) ? sbuf + offset : rbuf + offset;
            recv_buf  = task->allreduce_ring.scratch + offset;
            reduce_buf = rbuf + offset;

            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(send_buf, remaining, mem_type, sendto, team, task),
                task, out);
            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb(recv_buf, remaining, mem_type, recvfrom, team, task),
                task, out);

            if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
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

        task->tagged.send_posted++;
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_done", 0);
}

ucc_status_t ucc_tl_ucp_allreduce_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    size_t             count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    size_t             data_size = count * ucc_dt_size(dt);
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    /* Allocate scratch space for the receive buffer */
    status = ucc_mc_alloc(&task->allreduce_ring.scratch_mc_header,
                          data_size, TASK_ARGS(task).dst.info.mem_type);
    task->allreduce_ring.scratch = task->allreduce_ring.scratch_mc_header->addr;
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate scratch buffer");
        return status;
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allreduce_ring_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_sbgp_t        *sbgp;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!(task->flags & UCC_TL_UCP_TASK_FLAG_SUBSET)) {
        if (team->cfg.use_reordering) {
            sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
            task->subset.myrank = sbgp->group_rank;
            task->subset.map    = sbgp->map;
        }
    }

    task->super.post     = ucc_tl_ucp_allreduce_ring_start;
    task->super.progress = ucc_tl_ucp_allreduce_ring_progress;

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
