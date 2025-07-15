/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_hca_copy.h"
#include "tl_mlx5_mcast_helper.h"
#include "tl_mlx5_mcast_rcache.h"
#include <infiniband/verbs.h>

/* Create dedicated CQ and QP for HCA copy to avoid interference with mcast operations */
static ucc_status_t ucc_tl_mlx5_mcast_hca_copy_setup_resources(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct ibv_qp_init_attr    qp_attr;

    /* Create dedicated completion queue for HCA copy operations */
    if (!comm->hca_copy_cq) {
        comm->hca_copy_cq = ibv_create_cq(comm->ctx->ctx, 16, NULL, NULL, 0);
        if (!comm->hca_copy_cq) {
            tl_error(comm->lib, "failed to create HCA copy CQ, errno %d", errno);
            return UCC_ERR_NO_RESOURCE;
        }
        tl_debug(comm->lib, "created dedicated HCA copy CQ");
    }

    /* Create dedicated RC QP for HCA copy loopback */
    if (!comm->hca_copy_qp) {
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.qp_type                = IBV_QPT_RC;
        qp_attr.send_cq                = comm->hca_copy_cq;
        qp_attr.recv_cq                = comm->hca_copy_cq;
        qp_attr.sq_sig_all             = 1;  /* Signal all sends for HCA copy */
        qp_attr.cap.max_send_wr        = 16;
        qp_attr.cap.max_recv_wr        = 16;
        qp_attr.cap.max_send_sge       = 1;
        qp_attr.cap.max_recv_sge       = 1;
        qp_attr.cap.max_inline_data    = 0;

        comm->hca_copy_qp = ibv_create_qp(comm->ctx->pd, &qp_attr);
        if (!comm->hca_copy_qp) {
            tl_error(comm->lib, "failed to create HCA copy QP, errno %d", errno);
            ibv_destroy_cq(comm->hca_copy_cq);
            comm->hca_copy_cq = NULL;
            return UCC_ERR_NO_RESOURCE;
        }
        tl_debug(comm->lib, "created dedicated HCA copy QP");

        /* Initialize QP to INIT state */
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state        = IBV_QPS_INIT;
        attr.pkey_index      = 0;
        attr.port_num        = comm->ctx->ib_port;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

        if (ibv_modify_qp(comm->hca_copy_qp, &attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            tl_error(comm->lib, "failed to move HCA copy QP to INIT");
            ibv_destroy_qp(comm->hca_copy_qp);
            ibv_destroy_cq(comm->hca_copy_cq);
            comm->hca_copy_qp = NULL;
            comm->hca_copy_cq = NULL;
            return UCC_ERR_NO_RESOURCE;
        }

        /* Move to RTR state for loopback */
        memset(&attr, 0, sizeof(attr));
        attr.qp_state           = IBV_QPS_RTR;
        attr.path_mtu           = IBV_MTU_1024;
        attr.dest_qp_num        = comm->hca_copy_qp->qp_num;  /* Self-connection */
        attr.rq_psn             = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer      = 12;
        attr.ah_attr.is_global  = 0;
        attr.ah_attr.dlid       = comm->ctx->port_lid;  /* Self LID */
        attr.ah_attr.sl         = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num   = comm->ctx->ib_port;

        if (ibv_modify_qp(comm->hca_copy_qp, &attr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                         IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
            tl_error(comm->lib, "failed to move HCA copy QP to RTR");
            ibv_destroy_qp(comm->hca_copy_qp);
            ibv_destroy_cq(comm->hca_copy_cq);
            comm->hca_copy_qp = NULL;
            comm->hca_copy_cq = NULL;
            return UCC_ERR_NO_RESOURCE;
        }

        /* Move to RTS state */
        memset(&attr, 0, sizeof(attr));
        attr.qp_state      = IBV_QPS_RTS;
        attr.sq_psn        = 0;
        attr.timeout       = 14;
        attr.retry_cnt     = 7;
        attr.rnr_retry     = 7;
        attr.max_rd_atomic = 1;

        if (ibv_modify_qp(comm->hca_copy_qp, &attr,
                         IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                         IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC)) {
            tl_error(comm->lib, "failed to move HCA copy QP to RTS");
            ibv_destroy_qp(comm->hca_copy_qp);
            ibv_destroy_cq(comm->hca_copy_cq);
            comm->hca_copy_qp = NULL;
            comm->hca_copy_cq = NULL;
            return UCC_ERR_NO_RESOURCE;
        }

        tl_trace(comm->lib, "HCA copy QP ready for loopback RDMA operations");
    }

    return UCC_OK;
}

/* Completion handler for HCA copy RDMA operations */
static void ucc_tl_mlx5_mcast_hca_copy_completion(struct ibv_wc *wc, void *arg)
{
    ucc_tl_mlx5_mcast_hca_copy_task_t *task = (ucc_tl_mlx5_mcast_hca_copy_task_t *)arg;

    if (wc->status != IBV_WC_SUCCESS) {
        task->status = UCC_ERR_NO_MESSAGE;
    } else {
        task->status = UCC_OK;
    }

    task->completed = 1;
}

/* Post HCA copy operation - uses dedicated QP for true HCA-assisted copy
 * with rcache optimization */
ucc_status_t ucc_tl_mlx5_mcast_hca_copy_post(void *dst, ucc_memory_type_t dst_mtype,
                                             void *src, ucc_memory_type_t src_mtype,
                                             size_t size,
                                             ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                             ucc_tl_mlx5_mcast_hca_copy_task_t **copy_task)
{
    ucc_tl_mlx5_mcast_hca_copy_task_t *task;
    struct ibv_send_wr               *bad_wr;
    int                               ret;
    ucc_status_t                      status;

    if (!comm->one_sided.hca_copy_enabled) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* Only use HCA copy for CUDA memory */
    if (src_mtype != UCC_MEMORY_TYPE_CUDA && dst_mtype != UCC_MEMORY_TYPE_CUDA) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* Ensure dedicated HCA copy resources are set up */
    status = ucc_tl_mlx5_mcast_hca_copy_setup_resources(comm);
    if (status != UCC_OK) {
        return status;
    }

    task = ucc_malloc(sizeof(*task), "hca_copy_task");
    if (!task) {
        return UCC_ERR_NO_MEMORY;
    }

    task->dst         = dst;
    task->src         = src;
    task->size        = size;
    task->dst_mtype   = dst_mtype;
    task->src_mtype   = src_mtype;
    task->comm        = comm;
    task->target_rank = comm->rank;
    task->completed   = 0;
    task->status      = UCC_OK;
    task->src_reg     = NULL;
    task->dst_reg     = NULL;

    /* Use registration cache for better performance */
    status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, src, size, &task->src_reg);
    if (status != UCC_OK) {
        tl_error(comm->lib,
                 "failed to register source buffer via rcache for HCA copy");
        ucc_free(task);
        return status;
    }

    status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, dst, size, &task->dst_reg);
    if (status != UCC_OK) {
        tl_error(comm->lib,
                 "failed to register destination buffer via rcache for HCA copy");
        ucc_tl_mlx5_mcast_mem_deregister(comm->ctx, task->src_reg);
        ucc_free(task);
        return status;
    }

    /* Setup RDMA write work request for HCA-assisted loopback copy */
    task->rdma_sge.addr   = (uintptr_t)src;
    task->rdma_sge.length = size;
    task->rdma_sge.lkey   = ((struct ibv_mr*)task->src_reg->mr)->lkey;

    task->rdma_wr.wr_id      = (uintptr_t)task;
    task->rdma_wr.sg_list    = &task->rdma_sge;
    task->rdma_wr.num_sge    = 1;
    task->rdma_wr.opcode     = IBV_WR_RDMA_WRITE;
    task->rdma_wr.send_flags = IBV_SEND_SIGNALED;
    task->rdma_wr.next       = NULL;

    /* Set up RDMA write parameters for self-copy via HCA */
    task->rdma_wr.wr.rdma.remote_addr = (uintptr_t)dst;
    /* For loopback operations, use destination's lkey as rkey */
    task->rdma_wr.wr.rdma.rkey        = ((struct ibv_mr*)task->dst_reg->mr)->lkey;

    /* Use dedicated HCA copy QP */
    ret = ibv_post_send(comm->hca_copy_qp, &task->rdma_wr, &bad_wr);
    if (ret) {
        tl_error(comm->lib, "failed to post HCA copy RDMA write, errno %d", ret);
        ucc_tl_mlx5_mcast_mem_deregister(comm->ctx, task->src_reg);
        ucc_tl_mlx5_mcast_mem_deregister(comm->ctx, task->dst_reg);
        ucc_free(task);
        return UCC_ERR_NO_RESOURCE;
    }

    *copy_task = task;
    tl_trace(comm->lib,
             "posted HCA RDMA write operation via rcache: %p -> %p, size %zu",
             src, dst, size);
    return UCC_OK;
}

/* Test HCA copy completion using dedicated CQ */
ucc_status_t ucc_tl_mlx5_mcast_hca_copy_test(ucc_tl_mlx5_mcast_hca_copy_task_t *copy_task)
{
    struct ibv_wc wc;
    int           ne;

    if (!copy_task) {
        return UCC_ERR_INVALID_PARAM;
    }

    /* Check if already completed */
    if (copy_task->completed) {
        return copy_task->status;
    }

    /* Poll dedicated HCA copy completion queue */
    ne = ibv_poll_cq(copy_task->comm->hca_copy_cq, 1, &wc);
    if (ne < 0) {
        tl_error(copy_task->comm->lib, "failed to poll HCA copy CQ");
        return UCC_ERR_NO_MESSAGE;
    }

    if (ne > 0 && wc.wr_id == (uintptr_t)copy_task) {
        ucc_tl_mlx5_mcast_hca_copy_completion(&wc, copy_task);
    }

    /* Check completion */
    if (copy_task->completed) {
        if (copy_task->status == UCC_OK) {
            tl_trace(copy_task->comm->lib, "HCA copy RDMA write completed successfully");
        } else {
            tl_trace(copy_task->comm->lib, "HCA copy RDMA write completed with error");
        }
        return copy_task->status;
    }

    return UCC_INPROGRESS;
}

/* Finalize HCA copy task - uses rcache deregistration */
ucc_status_t ucc_tl_mlx5_mcast_hca_copy_finalize(ucc_tl_mlx5_mcast_hca_copy_task_t *copy_task)
{
    if (!copy_task) {
        return UCC_ERR_INVALID_PARAM;
    }

    /* Deregister memory regions via rcache */
    if (copy_task->src_reg) {
        ucc_tl_mlx5_mcast_mem_deregister(copy_task->comm->ctx, copy_task->src_reg);
    }
    if (copy_task->dst_reg) {
        ucc_tl_mlx5_mcast_mem_deregister(copy_task->comm->ctx, copy_task->dst_reg);
    }

    /* Free task */
    ucc_free(copy_task);

    return UCC_OK;
}

/* Helper function to choose between HCA copy and mc copy */
ucc_status_t ucc_tl_mlx5_mcast_memcpy(void *dst, ucc_memory_type_t dst_mtype,
                                      void *src, ucc_memory_type_t src_mtype,
                                      size_t size,
                                      ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_tl_mlx5_mcast_hca_copy_task_t *copy_task;
    ucc_status_t                       status;

    /* Use HCA copy if enabled and CUDA memory is involved */
    if (comm->one_sided.hca_copy_enabled &&
        (src_mtype == UCC_MEMORY_TYPE_CUDA || dst_mtype == UCC_MEMORY_TYPE_CUDA)) {

        tl_trace(comm->lib,
                 "using HCA-assisted copy with rcache for CUDA memory transfer (src: %s, dst: %s)",
                 src_mtype == UCC_MEMORY_TYPE_CUDA ? "CUDA" : "HOST",
                 dst_mtype == UCC_MEMORY_TYPE_CUDA ? "CUDA" : "HOST");

        /* Post HCA copy operation */
        status = ucc_tl_mlx5_mcast_hca_copy_post(dst, dst_mtype, src, src_mtype, size, comm, &copy_task);
        if (status != UCC_OK) {
            tl_warn(comm->lib, "HCA copy post failed, falling back to mc copy");
            return ucc_mc_memcpy(dst, src, size, dst_mtype, src_mtype);
        }

        /* Wait for completion */
        while ((status = ucc_tl_mlx5_mcast_hca_copy_test(copy_task)) == UCC_INPROGRESS) {
            /* Keep polling until completion */
        }

        /* Finalize the task */
        ucc_tl_mlx5_mcast_hca_copy_finalize(copy_task);

        if (status == UCC_OK) {
            tl_trace(comm->lib, "HCA copy completed successfully");
            return UCC_OK;
        } else {
            tl_warn(comm->lib, "HCA copy failed, falling back to mc copy");
            return ucc_mc_memcpy(dst, src, size, dst_mtype, src_mtype);
        }

    } else {
        /* Use regular mc copy for non-CUDA memory */
        return ucc_mc_memcpy(dst, src, size, dst_mtype, src_mtype);
    }
}
