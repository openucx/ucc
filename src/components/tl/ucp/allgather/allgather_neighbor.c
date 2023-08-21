#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"

static ucc_rank_t ucc_tl_ucp_allgather_neighbor_get_send_block(ucc_subset_t *subset,
                                                           ucc_rank_t trank,
                                                           ucc_rank_t tsize,
                                                           int step)
{
    return ucc_ep_map_eval(subset->map, (trank - step + tsize) % tsize);
}

static ucc_rank_t ucc_tl_ucp_allgather_neighbor_get_recv_block(ucc_subset_t *subset,
                                                           ucc_rank_t trank,
                                                           ucc_rank_t tsize,
                                                           int step)
{
    return ucc_ep_map_eval(subset->map, (trank - step - 1 + tsize) % tsize);
}

static ucc_rank_t get_recv_from_rank(ucc_rank_t rank, ucc_rank_t size, int i)
{
    int neighbor[2], offset_at_step[2], recv_data_from[2];
    int even_rank;
    even_rank = !(rank % 2);
    if (even_rank) {
        neighbor[0] = (rank + 1) % size;
        neighbor[1] = (rank - 1 + size) % size;
        recv_data_from[0] = rank;
        recv_data_from[1] = rank;
        offset_at_step[0] = (+2);
        offset_at_step[1] = (-2);
    } else {
        neighbor[0] = (rank - 1 + size) % size;
        neighbor[1] = (rank + 1) % size; 
        recv_data_from[0] = neighbor[0]; 
        recv_data_from[1] = neighbor[0]; 
        offset_at_step[0] = (-2);
        offset_at_step[1] = (+2);
    }
    const int i_parity = i % 2;
    return (recv_data_from[i_parity] + offset_at_step[i_parity] * ((i + 1) / 2) + size) % size;
}


ucc_status_t ucc_tl_ucp_allgather_neighbor_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *     team,
                                            ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_tl_ucp_allgather_neighbor_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
    }
    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allgather_neighbor_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    ucc_rank_t         tsize      = UCC_TL_TEAM_SIZE(team);
    ucc_sbgp_t *sbgp;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (tsize % 2)
    {
        tl_warn(UCC_TASK_LIB(task), "odd team size is not supported, switching to ring");
        return ucc_tl_ucp_allgather_ring_init_common(task);
    }
    
    if (!(task->flags & UCC_TL_UCP_TASK_FLAG_SUBSET)) {
        if (team->cfg.use_reordering) {
            sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
            task->subset.myrank = sbgp->group_rank;
            task->subset.map    = sbgp->map;
        }
    }

    task->allgather_neighbor.get_send_block = ucc_tl_ucp_allgather_neighbor_get_send_block;
    task->allgather_neighbor.get_recv_block = ucc_tl_ucp_allgather_neighbor_get_recv_block;
    task->super.post                    = ucc_tl_ucp_allgather_neighbor_start;
    task->super.progress                = ucc_tl_ucp_allgather_neighbor_progress;

    return UCC_OK;
}

/* Original implmenetation: https://github.com/open-mpi/ompi/blob/main/ompi/mca/coll/base/coll_base_allgather.c */
void  ucc_tl_ucp_allgather_neighbor_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    ucc_rank_t         trank      = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         tsize      = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         neighbor[2];
    void              *rbuf       = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  rmem       = TASK_ARGS(task).dst.info.mem_type;
    size_t             count      = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt         = TASK_ARGS(task).dst.info.datatype;
    size_t             data_size  = (count / tsize) * ucc_dt_size(dt);
    ucc_rank_t         i;
    int                even_rank;
    void              *tmprecv, *tmpsend;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    even_rank = !(trank % 2);
    if (even_rank) {
        neighbor[0] = (trank + 1) % tsize;
        neighbor[1] = (trank - 1 + tsize) % tsize;
    } else {
        neighbor[0] = (trank - 1 + tsize) % tsize;
        neighbor[1] = (trank + 1) % tsize;
    }

    while (task->tagged.send_posted < (tsize / 2)) {
        i = task->tagged.send_posted;
        const int i_parity = i % 2;
       
        tmprecv = PTR_OFFSET(rbuf, get_recv_from_rank(trank, tsize, i) * data_size);
        tmpsend = PTR_OFFSET(rbuf, get_recv_from_rank(trank, tsize, i - 1) * data_size);

        /* Sendreceive */
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(tmpsend, 2 * data_size, rmem, neighbor[i_parity], team, task),
            task, out);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(tmprecv, 2 * data_size, rmem, neighbor[i_parity], team, task),
            task, out);
        
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }
    }
    
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;

out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_neighbor_done", 0);
}

ucc_status_t ucc_tl_ucp_allgather_neighbor_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    size_t             count     = TASK_ARGS(task).dst.info.count;
    void              *sbuf      = TASK_ARGS(task).src.info.buffer;
    void              *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  smem      = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t  rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    ucc_rank_t         trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         tsize     = UCC_TL_TEAM_SIZE(team);
    size_t             data_size = (count / tsize) * ucc_dt_size(dt);
    ucc_status_t       status;
    ucc_rank_t         block;
    int                even_rank;
    ucc_rank_t         neighbor[2];
    void              *tmprecv, *tmpsend;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_neighbor_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        block = trank % tsize;
        status = ucc_mc_memcpy(PTR_OFFSET(rbuf, data_size * block),
                               sbuf, data_size, rmem, smem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    even_rank = !(trank % 2);
    if (even_rank) {
        neighbor[0] = (trank + 1) % tsize;
        neighbor[1] = (trank - 1 + tsize) % tsize;
    } else {
        neighbor[0] = (trank - 1 + tsize) % tsize;
        neighbor[1] = (trank + 1) % tsize;
    }

    tmprecv = PTR_OFFSET(rbuf, neighbor[0] * data_size);
    tmpsend = PTR_OFFSET(rbuf, trank * data_size);

    /* Sendreceive */
    UCPCHECK_GOTO(
        ucc_tl_ucp_send_nb(tmpsend, data_size, rmem, neighbor[0], team, task),
        task, out);
    UCPCHECK_GOTO(
        ucc_tl_ucp_recv_nb(tmprecv, data_size, rmem, neighbor[0], team, task),
        task, out);
out:
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}
