/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_helper.h"
#include "tl_mlx5_coll.h"
#include "tl_mlx5_mcast_rcache.h"
#include "tl_mlx5_mcast_progress.h"
#include "tl_mlx5_mcast_allgather.h"
#include <inttypes.h>

/* 32 here is the bit count of ib mcast packet's immediate data */
#define TL_MLX5_MCAST_IB_IMMEDIATE_PACKET_BIT_COUNT 32u

#define MCAST_GET_MAX_ALLGATHER_PACKET_COUNT(_max_count, _max_team, _max_counter)   \
do {                                                                                \
    _max_count = 2 << (TL_MLX5_MCAST_IB_IMMEDIATE_PACKET_BIT_COUNT -                \
                    ucc_ilog2(_max_team) -                                          \
                    ucc_ilog2(_max_counter));                                       \
} while (0);

#define MCAST_ALLGATHER_IN_PROGRESS(_req, _comm)                                             \
        (_req->to_send || _req->to_recv || _comm->pending_send ||                            \
         _comm->one_sided.rdma_read_in_progress || (NULL != _req->allgather_rkeys_req))      \

static inline ucc_status_t ucc_tl_mlx5_mcast_check_collective(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                              ucc_tl_mlx5_mcast_coll_req_t  *req)
{
    ucc_status_t status;

    ucc_assert(comm->one_sided.reliability_ready);

    if (comm->one_sided.rdma_read_in_progress) {
        return ucc_tl_mlx5_mcast_progress_one_sided_communication(comm, req);
    }

    /* check if remote rkey/address have arrived - applicable for sync design */
    if (req->allgather_rkeys_req != NULL) {
       status = comm->service_coll.coll_test(req->allgather_rkeys_req);
       if (status == UCC_OK) {
           ucc_assert(ONE_SIDED_SYNCHRONOUS_PROTO == req->one_sided_reliability_scheme);
           req->allgather_rkeys_req = NULL;
           tl_trace(comm->lib, "Allgather for remote_addr/rkey is completed");
       } else if (status < 0) {
           tl_error(comm->lib, "Allgather for remote_addr/rkey failed");
           return status;
       }
    }

    if (!req->to_send && !req->to_recv) {
        // all have been received, nothing to do
        return UCC_OK;

    } else if (req->to_send) {
        // it is not yet the time to start the reliability protocol
        return UCC_INPROGRESS;
    }

    if (!comm->timer) {
        if (comm->stalled < DROP_THRESHOLD) {
            comm->stalled++;
        } else {
            // kick the timer
            comm->timer = ucc_tl_mlx5_mcast_get_timer();
            comm->stalled = 0;
        }
    } else {
        if (comm->stalled < DROP_THRESHOLD || (NULL != req->allgather_rkeys_req)) {
            comm->stalled++;
        } else {
            // calcuate the current time and check if it's time to do RDMA READ
            if (ucc_tl_mlx5_mcast_get_timer() - comm->timer >=
                    comm->ctx->params.timeout) {
                tl_debug(comm->lib, "[REL] time out req->to_recv %d left out of total of %d packets",
                         req->to_recv, req->num_packets * comm->commsize);
                status = ucc_tl_mlx5_mcast_reliable_one_sided_get(comm, req, NULL);
                if (UCC_OK != status) {
                    return status;
                }
            } else {
                comm->stalled = 0;
            }
        }
    }

    return UCC_INPROGRESS;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_reset_reliablity(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm = req->comm;
    ucc_tl_mlx5_mcast_reg_t       *reg  = NULL;
    ucc_status_t                   status;

    ucc_assert(req->ag_counter == comm->allgather_comm.under_progress_counter);

    if (comm->one_sided.reliability_enabled && !comm->one_sided.reliability_ready) {
        /* initialize the structures needed by reliablity protocol */ 
        memset(comm->one_sided.recvd_pkts_tracker, 0, comm->commsize * sizeof(uint32_t));
        memset(comm->one_sided.remote_slot_info, ONE_SIDED_INVALID, comm->commsize * sizeof(uint32_t));
        /* local slots state */
        comm->one_sided.slots_state = ONE_SIDED_INVALID;

        if (ONE_SIDED_SYNCHRONOUS_PROTO == req->one_sided_reliability_scheme) {
            /* do nonblocking allgather over remote addresses/keys */
            if (!req->rreg) {
               /* register sbuf if it is not registered before */
               status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->ptr, req->length, &reg);
               if (UCC_OK != status) {
                    return status;
               }
               req->rreg = reg;
               req->mr   = reg->mr;
            }

            comm->one_sided.sendbuf_memkey_list[comm->rank].rkey        = req->mr->rkey;
            comm->one_sided.sendbuf_memkey_list[comm->rank].remote_addr = (uint64_t)req->ptr;

            tl_trace(comm->lib, "Allgather over sendbuf addresses/rkey: address %p rkey %d",
                     req->ptr, req->mr->rkey);

            status = comm->service_coll.allgather_post(comm->p2p_ctx, NULL /* in-place */,
                                                       comm->one_sided.sendbuf_memkey_list,
                                                       sizeof(ucc_tl_mlx5_mcast_slot_mem_info_t),
                                                       &req->allgather_rkeys_req);
            if (UCC_OK != status) {
                tl_error(comm->lib, "oob allgather failed during one-sided reliability reset of a collective call");
                return status;
            }
        }

        memset(comm->pending_recv_per_qp, 0, sizeof(int) * MAX_GROUP_COUNT);
        comm->one_sided.reliability_ready = 1;
    }

    return UCC_OK;
}

static inline void ucc_tl_mlx5_mcast_init_async_reliability_slots(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm = req->comm;
    char                          *dest;

    ucc_assert(req->ag_counter == comm->allgather_comm.under_progress_counter);

    if (ONE_SIDED_ASYNCHRONOUS_PROTO == req->one_sided_reliability_scheme &&
        ONE_SIDED_INVALID == comm->one_sided.slots_state) {
        /* copy the sendbuf and seqnum to the internal temp buf in case other processes need
         * to read from it */
        ucc_assert(req->length <= comm->one_sided.reliability_scheme_msg_threshold);
        dest = PTR_OFFSET(comm->one_sided.slots_buffer,
                          (req->ag_counter % ONE_SIDED_SLOTS_COUNT)
                          * comm->one_sided.slot_size);
    
        /* both user buffer and reliablity slots are on host */
        memcpy(PTR_OFFSET(dest, ONE_SIDED_SLOTS_INFO_SIZE), req->ptr, req->length);
        memcpy(dest, &req->ag_counter, ONE_SIDED_SLOTS_INFO_SIZE);

        comm->one_sided.slots_state = ONE_SIDED_VALID;
    }
}

static inline ucc_status_t ucc_tl_mlx5_mcast_do_staging_based_allgather(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_status_t                   status = UCC_OK;
    ucc_tl_mlx5_mcast_coll_comm_t *comm   = req->comm;
    const int                      zcopy  = req->proto != MCAST_PROTO_EAGER;
    int                            num_recvd;

    ucc_assert(req->to_recv >= 0 && req->to_send >= 0);

    status = ucc_tl_mlx5_mcast_reset_reliablity(req);
    if (UCC_OK != status) {
        return status;
    }

    if (req->to_send || req->to_recv) {
        ucc_assert(comm->allgather_comm.max_push_send >= comm->pending_send);
        if (req->to_send &&
            (comm->allgather_comm.max_push_send - comm->pending_send) > 0) {
            status = ucc_tl_mlx5_mcast_send_collective(comm, req, ucc_min(comm->allgather_comm.max_push_send -
                                                       comm->pending_send, req->to_send),
                                                       zcopy, UCC_COLL_TYPE_ALLGATHER, -1, SIZE_MAX);
            if (status < 0) {
                tl_error(comm->lib, "a failure happend during send packets");
                return status;
            }
        }

        ucc_tl_mlx5_mcast_init_async_reliability_slots(req);

        if (req->to_recv) {
            num_recvd = ucc_tl_mlx5_mcast_recv_collective(comm, req, req->to_recv, UCC_COLL_TYPE_ALLGATHER);
            if (num_recvd < 0) {
                tl_error(comm->lib, "a failure happend during cq polling");
                status = UCC_ERR_NO_MESSAGE;
                return status;
            }
        }
    }

    if (comm->pending_send) {
        if (ucc_tl_mlx5_mcast_poll_send(comm) < 0) {
            return UCC_ERR_NO_MESSAGE;
        }
    }

    if (comm->one_sided.reliability_enabled) {
        status = ucc_tl_mlx5_mcast_check_collective(comm, req);
        if (UCC_INPROGRESS != status && UCC_OK != status) {
            return status;
        }
    }

    if (MCAST_ALLGATHER_IN_PROGRESS(req, comm)) {
        return UCC_INPROGRESS;
    }

    if (ONE_SIDED_SYNCHRONOUS_PROTO == req->one_sided_reliability_scheme) {
        /* mcast operations are all done, now wait until all the processes 
         * are done with their mcast operations */
        if (!req->barrier_req) {
            // mcast operations are done and now go to barrier
           status = comm->service_coll.barrier_post(comm->p2p_ctx, &req->barrier_req);
           if (status != UCC_OK) {
               return status;
           }
           tl_trace(comm->lib, "mcast operations are done and now go to barrier");
        }

        status = comm->service_coll.coll_test(req->barrier_req);
        if (status == UCC_OK) {
            req->barrier_req = NULL;
            tl_trace(comm->lib, "barrier at the end of mcast allgather is completed");
        } else {
            return status;
        }
    }

    /* this task is completed */
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_allgather_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t *task      = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_team_t *mlx5_team = TASK_TEAM(task);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(mlx5_team)->pq, &task->super);
}

void ucc_tl_mlx5_mcast_allgather_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t           *task = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_mcast_coll_req_t *req  = task->coll_mcast.req_handle;

    ucc_assert(req != NULL);

    if (req->ag_counter != req->comm->allgather_comm.under_progress_counter) {
        /* it is not this task's turn for progress */
        ucc_assert(req->comm->allgather_comm.under_progress_counter < req->ag_counter);
        return;
    }

    coll_task->status = ucc_tl_mlx5_mcast_do_staging_based_allgather(req);
    if (UCC_OK != coll_task->status) {
        tl_error(UCC_TASK_LIB(task), "progress mcast allgather failed:%d", coll_task->status);
    }
}

ucc_status_t ucc_tl_mlx5_mcast_allgather_init(ucc_tl_mlx5_task_t *task)
{
    ucc_coll_task_t               *coll_task =  &(task->super);
    ucc_tl_mlx5_team_t            *mlx5_team = TASK_TEAM(task);
    ucc_tl_mlx5_mcast_team_t      *team      = mlx5_team->mcast;
    ucc_coll_args_t               *args      = &TASK_ARGS(task);
    ucc_datatype_t                 dt        = args->src.info.datatype;
    size_t                         count     = args->src.info.count;
    ucc_status_t                   status    = UCC_OK;
    size_t                         data_size = ucc_dt_size(dt) * count;
    void                          *sbuf      = args->src.info.buffer;
    void                          *rbuf      = args->dst.info.buffer;
    ucc_tl_mlx5_mcast_coll_comm_t *comm      = team->mcast_comm;
    ucc_tl_mlx5_mcast_reg_t       *reg       = NULL;
    ucc_rank_t                     max_team  = ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE;
    int                            max_ctr   = ONE_SIDED_MAX_ALLGATHER_COUNTER;
    ucc_tl_mlx5_mcast_coll_req_t  *req;


    task->coll_mcast.req_handle = NULL;

    tl_trace(comm->lib, "MCAST allgather init, sbuf %p, rbuf %p, size %ld, comm %d, "
             "comm_size %d, counter %d",
             sbuf, rbuf, data_size, comm->comm_id, comm->commsize, comm->allgather_comm.coll_counter);

    req = ucc_mpool_get(&comm->ctx->mcast_req_mp);
    if (!req) {
        tl_error(comm->lib, "failed to get a mcast req");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }
    memset(req, 0, sizeof(ucc_tl_mlx5_mcast_coll_req_t));

    req->comm   = comm;
    req->ptr    = sbuf;
    req->rptr   = rbuf;
    req->length = data_size;
    req->mr     = comm->pp_mr;
    req->rreg   = NULL;
    /* - zero copy protocol only provides zero copy design at sender side
     * - truly zero copy protocol provides zero copy design at receiver side as well
     * here we select the sender side protocol */
    req->proto  = (req->length < comm->max_eager) ? MCAST_PROTO_EAGER :
                                                    MCAST_PROTO_ZCOPY;

    assert(comm->commsize <= ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE);

    req->offset      = 0;
    req->num_packets = ucc_max(1, ucc_div_round_up(req->length, comm->max_per_packet));

    MCAST_GET_MAX_ALLGATHER_PACKET_COUNT(comm->allgather_comm.max_num_packets, max_team, max_ctr);

    if (comm->allgather_comm.max_num_packets < req->num_packets) {
        tl_warn(comm->lib,
                "msg size is %ld but max supported msg size of mcast allgather is %d",
                req->length, comm->allgather_comm.max_num_packets * comm->max_per_packet);
        status = UCC_ERR_NOT_SUPPORTED;
        goto failed;
    }

    req->last_pkt_len = req->length % comm->max_per_packet;

    ucc_assert(req->last_pkt_len > 0 && req->last_pkt_len <= comm->max_per_packet);

    if (req->proto == MCAST_PROTO_ZCOPY) {
        /* register the send buffer */
       status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->ptr, req->length, &reg);
       if (UCC_OK != status) {
           tl_error(comm->lib, "sendbuf registration failed");
           goto failed;
       }
       req->rreg = reg;
       req->mr   = reg->mr;
    }

    if (comm->one_sided.reliability_enabled) {
        req->one_sided_reliability_scheme = (req->length <
                comm->one_sided.reliability_scheme_msg_threshold) ?
                ONE_SIDED_ASYNCHRONOUS_PROTO : ONE_SIDED_SYNCHRONOUS_PROTO;
    } else {
        req->one_sided_reliability_scheme = ONE_SIDED_NO_RELIABILITY;
    }

    req->ag_counter = comm->allgather_comm.coll_counter;
    req->to_send    = req->num_packets;
    req->to_recv    = comm->commsize * req->num_packets;

    comm->allgather_comm.coll_counter++;

    task->coll_mcast.req_handle = req;
    coll_task->status           = UCC_OPERATION_INITIALIZED;
    task->super.post            = ucc_tl_mlx5_mcast_allgather_start;
    task->super.progress        = ucc_tl_mlx5_mcast_allgather_progress;
    return UCC_OK;

failed:
    tl_warn(UCC_TASK_LIB(task), "mcast init allgather failed:%d", status);
    if (req) {
        ucc_mpool_put(req);
    }
    return status;
}

