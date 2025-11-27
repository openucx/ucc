/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "alltoall.h"
#include "alltoall_mkeys.h"

#include "tl_mlx5_ib.h"

#include "core/ucc_team.h"
#include <sys/shm.h>

struct rank_data {
    ucc_rank_t team_rank;
    ucc_rank_t sbgp_rank;
};

typedef struct net_exchage {
    struct rank_data    rd;
    alltoall_net_ctrl_t net_ctrl;
    uint32_t            port_lid;
    uint32_t            recv_mkey_rkey;
    uint32_t            qpn[1];
} net_exchange_t;

static int compare_rank_data(const void *a, const void *b)
{
    const struct rank_data *d1 = (const struct rank_data *)a;
    const struct rank_data *d2 = (const struct rank_data *)b;

    return d1->team_rank > d2->team_rank ? 1 : -1;
}

static ucc_status_t build_rank_map(ucc_tl_mlx5_alltoall_t *a2a,
                                   net_exchange_t         *global_data,
                                   size_t                  local_data_size)
{
    struct rank_data *data;
    net_exchange_t   *d;
    int               i;

    data = ucc_malloc(sizeof(*data) * a2a->net.net_size);
    if (!data) {
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < a2a->net.net_size; i++) {
        d                 = PTR_OFFSET(global_data, i * local_data_size);
        data[i].team_rank = d->rd.team_rank;
        data[i].sbgp_rank = d->rd.sbgp_rank;
    }

    a2a->net.rank_map = ucc_malloc(sizeof(int) * a2a->net.net_size);
    if (!a2a->net.rank_map) {
        ucc_free(data);
        return UCC_ERR_NO_MEMORY;
    }
    qsort(data, a2a->net.net_size, sizeof(*data), compare_rank_data);
    for (i = 0; i < a2a->net.net_size; i++) {
        a2a->net.rank_map[data[i].sbgp_rank] = i;
    }
    ucc_free(data);
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_team_init_alltoall(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t  *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    ucc_tl_mlx5_alltoall_t *a2a;
    ucc_sbgp_t             *node, *net;
    int                     i, j, node_size, ppn, team_size, nnodes;
    ucc_topo_t             *topo;

    team->a2a              = NULL;
    team->dm_ptr           = NULL;
    team->a2a_status.local = UCC_OK;

    topo      = team->topo;
    node      = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    net       = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);
    node_size = node->group_size;
    nnodes    = ucc_topo_nnodes(topo);
    team_size = UCC_TL_TEAM_SIZE(team);

    if (!ucc_topo_isoppn(topo)) {
        tl_debug(ctx->super.super.lib,
                 "disabling mlx5 a2a for team with non-uniform ppn, "
                 "min_ppn %d, max_ppn %d",
                 ucc_topo_min_ppn(topo), ucc_topo_max_ppn(topo));
        goto non_fatal_error;
    }
    ppn = ucc_topo_max_ppn(topo);

    if (net->status == UCC_SBGP_NOT_EXISTS) {
        tl_debug(ctx->super.super.lib,
                 "disabling mlx5 a2a for single node team");
        goto non_fatal_error;
    }

    if (nnodes == team_size) {
        tl_debug(ctx->super.super.lib,
                 "disabling mlx5 a2a for ppn=1 case, not supported so far");
        goto non_fatal_error;
    }


    for (i = 0; i < nnodes; i++) {
        for (j = 1; j < ppn; j++) {
            if (!ucc_team_ranks_on_same_node(i * ppn, i * ppn + j,
                                             UCC_TL_CORE_TEAM(team))) {
                tl_debug(ctx->super.super.lib,
                         "disabling mlx5 a2a for team with non contiguous "
                         "ranks-per-node placement");
                goto non_fatal_error;
            }
        }
    }

    team->a2a = ucc_calloc(1, sizeof(*team->a2a), "mlx5_a2a");
    if (!team->a2a) {
        return UCC_ERR_NO_MEMORY;
    }

    a2a                      = team->a2a;
    a2a->node_size           = node_size;
    a2a->pd                  = ctx->shared_pd;
    a2a->ctx                 = ctx->shared_ctx;
    a2a->ib_port             = ctx->ib_port;
    a2a->node.sbgp           = node;
    a2a->net.sbgp            = net;
    a2a->node.asr_rank       = MLX5_ASR_RANK;
    a2a->num_dci_qps         = UCC_TL_MLX5_TEAM_LIB(team)->cfg.num_dci_qps;
    a2a->sequence_number     = 1;
    a2a->net.atomic.counters = NULL;
    a2a->net.ctrl_mr         = NULL;
    a2a->net.remote_ctrl     = NULL;
    a2a->net.rank_map        = NULL;
    a2a->max_msg_size        = MAX_MSG_SIZE;
    a2a->max_num_of_columns  =
        ucc_div_round_up(node->group_size, 2 /* todo: there can be an estimation of
                                                    minimal possible block size */);

    if (a2a->node.asr_rank == node->group_rank) {
        team->a2a_status.local = ucc_tl_mlx5_dm_init(team);
        if (UCC_OK != team->a2a_status.local) {
            tl_debug(UCC_TL_TEAM_LIB(team), "failed to init device memory");
        }
    }

    return UCC_OK;

non_fatal_error:
    team->a2a_status.local = UCC_ERR_NOT_SUPPORTED;
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_team_test_alltoall_start(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t  *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    ucc_tl_mlx5_alltoall_t *a2a = team->a2a;
    size_t                  storage_size;

    if (team->a2a_status.global != UCC_OK) {
        tl_debug(ctx->super.super.lib, "global status in error state: %s",
                 ucc_status_string(team->a2a_status.global));

        ucc_tl_mlx5_dm_cleanup(team);
        if (a2a) {
            ucc_free(a2a);
            team->a2a = NULL;
        }
        ucc_tl_mlx5_topo_cleanup(team);
        return team->a2a_status.global;
    }

    if (a2a->node.asr_rank == a2a->node.sbgp->group_rank) {
        a2a->net.net_size = a2a->net.sbgp->group_size;
        storage_size      = OP_SEGMENT_SIZE(a2a) * MAX_OUTSTANDING_OPS;
        a2a->bcast_data.shmid =
            shmget(IPC_PRIVATE, storage_size, IPC_CREAT | 0600);
        if (a2a->bcast_data.shmid == -1) {
            tl_debug(ctx->super.super.lib,
                     "failed to allocate sysv shm segment for %zd bytes",
                     storage_size);
        } else {
            a2a->node.storage = shmat(a2a->bcast_data.shmid, NULL, 0);
            memset(a2a->node.storage, 0, storage_size);
            shmctl(a2a->bcast_data.shmid, IPC_RMID, NULL);
        }
        a2a->bcast_data.net_size = a2a->net.sbgp->group_size;
    }

    a2a->state = TL_MLX5_ALLTOALL_STATE_SHMID;

    team->a2a = a2a;
    return ucc_service_bcast(
        UCC_TL_CORE_TEAM(team), &a2a->bcast_data,
        sizeof(ucc_tl_mlx5_a2a_bcast_data_t), a2a->node.asr_rank,
        ucc_sbgp_to_subset(a2a->node.sbgp), &team->scoll_req);
}

static void ucc_tl_mlx5_alltoall_barrier_free(ucc_tl_mlx5_alltoall_t *a2a)
{
    ibv_dereg_mr(a2a->net.barrier.mr);
    ucc_free(a2a->net.barrier.flags);
}

static ucc_status_t ucc_tl_mlx5_alltoall_barrier_alloc(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t  *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    ucc_tl_mlx5_alltoall_t *a2a = team->a2a;
    size_t                  size;

    /* allocating net_size + 1 flags. Last one is used as local buf
       for barrier RDMA */
    size = (a2a->net.net_size + 1) * sizeof(*a2a->net.barrier.flags) *
           MAX_OUTSTANDING_OPS;
    a2a->net.barrier.flags = ucc_calloc(1, size, "barrier");
    if (!a2a->net.barrier.flags) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "failed to allocate %zd bytes for barrier flags array", size);
        return UCC_ERR_NO_MEMORY;
    }

    a2a->net.barrier.mr =
        ibv_reg_mr(ctx->shared_pd, a2a->net.barrier.flags, size,
                   IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);

    if (!a2a->net.barrier.mr) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "failed to register barrier flags array");
        ucc_free(a2a->net.barrier.flags);
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_team_test_alltoall_progress(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t     *ctx         = UCC_TL_MLX5_TEAM_CTX(team);
    ucc_tl_mlx5_alltoall_t    *a2a         = team->a2a;
    ucc_base_lib_t            *lib         = UCC_TL_TEAM_LIB(team);
    int                        i           = 0;
    net_exchange_t            *local_data  = NULL;
    ucc_rank_t                 node_size, node_rank;
    ucc_status_t               status;
    ucc_tl_mlx5_alltoall_op_t *op;
    int                        j, asr_cq_size, net_size, ret;
    struct ibv_port_attr       port_attr;
    size_t                     op_seg_size, local_data_size, umr_buf_size;
    net_exchange_t            *global_data, *remote_data;

    if (team->a2a_status.local < 0) {
        return team->a2a_status.local;
    }

    node_size   = a2a->node.sbgp->group_size;
    node_rank   = a2a->node.sbgp->group_rank;
    op_seg_size = OP_SEGMENT_SIZE(a2a);

    switch (a2a->state) {
    case TL_MLX5_ALLTOALL_STATE_SHMID:
        status = ucc_service_coll_test(team->scoll_req);
        if (status < 0) {
            tl_error(lib, "failure during service coll exchange: %s",
                    ucc_status_string(status));
            ucc_service_coll_finalize(team->scoll_req);
            return status;
        }
        if (UCC_INPROGRESS == status) {
            return status;
        }
        ucc_service_coll_finalize(team->scoll_req);

        if (a2a->bcast_data.shmid == -1) {
            tl_error(lib, "failed to allocate sysv shm segment");
            return UCC_ERR_NO_MEMORY;
        }
        a2a->net.net_size = a2a->bcast_data.net_size;
        if (a2a->node.asr_rank != node_rank) {
            // shmat already performed for asr above
            a2a->node.storage = shmat(a2a->bcast_data.shmid, NULL, 0);
        }
        if (a2a->node.storage == (void *)(-1)) {
            tl_error(lib, "failed to shmat seg %d", a2a->bcast_data.shmid);
            return UCC_ERR_NO_MEMORY;
        }
        for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
            op          = &a2a->node.ops[i];
            op->ctrl    = PTR_OFFSET(a2a->node.storage, op_seg_size * i);
            op->my_ctrl = PTR_OFFSET(
                op->ctrl, node_rank * sizeof(ucc_tl_mlx5_alltoall_ctrl_t));
        }

        if (UCC_TL_MLX5_TEAM_LIB(team)->cfg.block_size > MAX_BLOCK_SIZE) {
            tl_error(lib, "max block_size is %d", MAX_BLOCK_SIZE);
            return UCC_ERR_NO_MESSAGE;
        }
        a2a->requested_block_size = UCC_TL_MLX5_TEAM_LIB(team)->cfg.block_size;

        // Non-ASR ranks exit here
        if (a2a->node.asr_rank != node_rank) {
            return UCC_OK;
        }

        status = ucc_tl_mlx5_alltoall_barrier_alloc(team);
        if (UCC_OK != status) {
            goto err_barrier;
        }

        a2a->net.blocks_sent =
            ucc_malloc(sizeof(*a2a->net.blocks_sent) * a2a->net.net_size *
                            MAX_OUTSTANDING_OPS,
                        "blocks_sent");
        if (!a2a->net.blocks_sent) {
            tl_error(lib,
                        "failed to allocated %zd bytes for blocks_sent array",
                        sizeof(*a2a->net.blocks_sent) * a2a->net.net_size *
                            MAX_OUTSTANDING_OPS);
            status = UCC_ERR_NO_MEMORY;
            goto err_blocks_sent;
        }

        for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
            op              = &a2a->node.ops[i];
            op->blocks_sent = PTR_OFFSET(a2a->net.blocks_sent,
                                            sizeof(*a2a->net.blocks_sent) *
                                                a2a->net.net_size * i);
        }

        memset(a2a->previous_msg_size, 0, sizeof(a2a->previous_msg_size));

        status = ucc_tl_mlx5_alltoall_init_umr(team->a2a, lib);
        if (status != UCC_OK) {
            tl_error(lib, "failed to init UMR");
            goto err_umr;
        }
        net_size    = a2a->net.net_size;
        asr_cq_size = net_size *
                        (SQUARED(a2a->node.sbgp->group_size / 2 + 1) + 1) *
                        MAX_OUTSTANDING_OPS;
        a2a->net.cq =
            ibv_create_cq(ctx->shared_ctx, asr_cq_size, NULL, NULL, 0);
        if (!a2a->net.cq) {
            tl_error(lib, "failed to allocate ASR CQ");
            status = UCC_ERR_NO_MESSAGE;
            goto err_cq;
        }

        a2a->is_dc =
            (net_size >= UCC_TL_MLX5_TEAM_LIB(team)->cfg.dc_threshold);

        ibv_query_port(ctx->shared_ctx, ctx->ib_port, &port_attr);

        a2a->net.ctrl_mr = ibv_reg_mr(
            ctx->shared_pd, a2a->node.storage,
            op_seg_size * MAX_OUTSTANDING_OPS,
            IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE);
        if (!a2a->net.ctrl_mr) {
            tl_error(lib, "failed to register control data, errno %d",
                        errno);
            status = UCC_ERR_NO_MESSAGE;
            goto err_ctrl_mr;
        }
        a2a->net.remote_ctrl = ucc_calloc(sizeof(*a2a->net.remote_ctrl),
                                            net_size, "remote_ctrl");
        if (!a2a->net.remote_ctrl) {
            tl_error(lib, "failed to allocate %zd bytes for remote_ctrl",
                        sizeof(*a2a->net.remote_ctrl) * net_size);
            status = UCC_ERR_NO_MESSAGE;
            goto err_remote_ctrl;
        }

        status = ucc_tl_mlx5_init_mkeys(team, lib);
        if (status != UCC_OK) {
            tl_error(lib, "failed to init mkeys");
            goto err_mkeys;
        }
        // for each ASR - qp num, in addition to port lid,
        // ctrl segment rkey and address, receive mkey rkey
        local_data_size = sizeof(net_exchange_t);
        if (!a2a->is_dc) {
            /* need more space for net_size - 1 qpns */
            local_data_size += sizeof(uint32_t) * (net_size - 1);
        }

        local_data =
            ucc_malloc(local_data_size * (net_size + 1), "exchange_data");
        if (!local_data) {
            tl_error(lib, "failed to allocate %zd bytes for exchange data",
                        local_data_size * (net_size + 1));
            status = UCC_ERR_NO_MEMORY;
            goto err_local_data;
        }
        global_data = PTR_OFFSET(local_data, local_data_size);

        if (a2a->is_dc) {
            //SRQ
            struct ibv_srq_init_attr srq_attr;
            memset(&srq_attr, 0, sizeof(struct ibv_srq_init_attr));
            srq_attr.attr.max_wr  = 1;
            srq_attr.attr.max_sge = 1;

            a2a->net.srq = ibv_create_srq(ctx->shared_pd, &srq_attr);
            if (a2a->net.srq == NULL) {
                tl_error(lib, "failed to create SRQ");
                status = UCC_ERR_NO_MESSAGE;
                goto err_srq;
            }
            //DCI

            int tx_depth =
                (SQUARED(a2a->node.sbgp->group_size / 2 + 1) * 2 + 2) *
                MAX_OUTSTANDING_OPS *
                ucc_div_round_up(a2a->net.net_size, a2a->num_dci_qps);

            a2a->net.dcis = ucc_malloc(sizeof(ucc_tl_mlx5_dci_t) * a2a->num_dci_qps);
            if (!a2a->net.dcis) {
                tl_error(ctx->super.super.lib, "failed to allocate mem");
                status = UCC_ERR_NO_MEMORY;
                goto err_dcis;
            }

            for (i = 0; i < a2a->num_dci_qps; i++) {
                status = ucc_tl_mlx5_init_dci(
                    &a2a->net.dcis[i], ctx->shared_pd, ctx->shared_ctx,
                    a2a->net.cq, ctx->ib_port, tx_depth,
                    &UCC_TL_MLX5_TEAM_LIB(team)->cfg.qp_conf, lib);
                if (UCC_OK != status) {
                    goto err_init_dci;
                }
            }

            //DCT
            status = ucc_tl_mlx5_init_dct(
                ctx->shared_pd, ctx->shared_ctx, a2a->net.cq, a2a->net.srq,
                ctx->ib_port, &a2a->net.dct_qp, local_data->qpn,
                &UCC_TL_MLX5_TEAM_LIB(team)->cfg.qp_conf, lib);
            if (UCC_OK != status) {
                goto err_init_dct;
            }
        } else {
            a2a->net.rc_qps =
                ucc_malloc(sizeof(ucc_tl_mlx5_qp_t) * a2a->net.net_size);
            if (!a2a->net.rc_qps) {
                tl_error(lib, "failed to allocate asr qps array");
                status = UCC_ERR_NO_MEMORY;
                goto err_alloc_rc_qps;
            }
            int tx_depth =
                (SQUARED(a2a->node.sbgp->group_size / 2 + 1) * 2 + 2) *
                MAX_OUTSTANDING_OPS;
            for (i = 0; i < a2a->net.net_size; i++) {
                status = ucc_tl_mlx5_create_rc_qp(
                    ctx->shared_ctx, ctx->shared_pd, a2a->net.cq, tx_depth,
                    &a2a->net.rc_qps[i], &local_data->qpn[i], lib);
                if (UCC_OK != status) {
                    goto err_create_rc_qps;
                }
            }
        }

        local_data->port_lid              = port_attr.lid;
        local_data->recv_mkey_rkey        = a2a->node.team_recv_mkey->rkey;
        local_data->rd.team_rank          = UCC_TL_TEAM_RANK(team);
        local_data->rd.sbgp_rank          = a2a->net.sbgp->group_rank;
        local_data->net_ctrl.atomic.addr  = a2a->net.atomic.counters;
        local_data->net_ctrl.atomic.rkey  = a2a->net.atomic.mr->rkey;
        local_data->net_ctrl.barrier.addr = a2a->net.barrier.flags;
        local_data->net_ctrl.barrier.rkey = a2a->net.barrier.mr->rkey;
        status = ucc_service_allgather(UCC_TL_CORE_TEAM(team), local_data,
                                        global_data, local_data_size,
                                        ucc_sbgp_to_subset(a2a->net.sbgp),
                                        &team->scoll_req);
        if (UCC_OK != status) {
            tl_error(lib, "failed start service allgather");
            goto err_service_allgather_post;
        }
        team->scoll_req->data = local_data;
        a2a->state            = TL_MLX5_ALLTOALL_STATE_EXCHANGE_PROGRESS;

    case TL_MLX5_ALLTOALL_STATE_EXCHANGE_PROGRESS:
        status = ucc_service_coll_test(team->scoll_req);
        if (status < 0) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "failure during service coll exchange: %s",
                     ucc_status_string(status));
            ucc_service_coll_finalize(team->scoll_req);
            goto err_service_allgather_progress;
        }
        if (UCC_INPROGRESS == status) {
            return status;
        }
        ucc_assert(status == UCC_OK);
        a2a->state = TL_MLX5_ALLTOALL_STATE_EXCHANGE_DONE;

    case TL_MLX5_ALLTOALL_STATE_EXCHANGE_DONE:
        local_data = team->scoll_req->data;
        ucc_service_coll_finalize(team->scoll_req);

        net_size        = a2a->net.net_size;
        local_data_size = sizeof(net_exchange_t);
        if (!a2a->is_dc) {
            /* need more space for net_size - 1 qpns */
            local_data_size += sizeof(uint32_t) * (net_size - 1);
        }
        global_data = PTR_OFFSET(local_data, local_data_size);

        status = build_rank_map(a2a, global_data, local_data_size);
        if (status != UCC_OK) {
            tl_error(lib, "failed to build rank map");
            goto err_rank_map;
        }
        a2a->net.rkeys = ucc_malloc(sizeof(uint32_t) * net_size);
        if (!a2a->net.rkeys) {
            tl_error(lib, "failed to allocate %zd bytes for net rkeys",
                     sizeof(uint32_t) * net_size);
            status = UCC_ERR_NO_MEMORY;
            goto err_rkeys;
        }
        if (a2a->is_dc) {
            a2a->net.remote_dctns = ucc_malloc(sizeof(uint32_t) * net_size);
            if (!a2a->net.remote_dctns) {
                tl_error(lib, "failed to allocate %zd bytes for remote_dctns",
                         sizeof(uint32_t) * net_size);
                status = UCC_ERR_NO_MEMORY;
                goto err_remote_dctns;
            }
            a2a->net.ahs = ucc_malloc(sizeof(struct ibv_ah *) * net_size);
            if (!a2a->net.ahs) {
                tl_error(lib, "failed to allocate %zd bytes for net ahs",
                         sizeof(struct ibv_ah *) * net_size);
                status = UCC_ERR_NO_MEMORY;
                goto err_ahs;
            }
        }
        for (i = 0; i < net_size; i++) {
            remote_data = PTR_OFFSET(global_data, i * local_data_size);
            if (a2a->is_dc) {
                a2a->net.remote_dctns[i] = remote_data->qpn[0];
                status =
                    ucc_tl_mlx5_create_ah(ctx->shared_pd, remote_data->port_lid,
                                          ctx->ib_port, &a2a->net.ahs[i], lib);
                if (UCC_OK != status) {
                    tl_error(lib, "failed to create ah, %s",
                             ucc_status_string(status));
                    goto err_create_ah;
                }
            } else {
                status = ucc_tl_mlx5_qp_connect(
                    a2a->net.rc_qps[i].qp,
                    remote_data->qpn[a2a->net.sbgp->group_rank],
                    remote_data->port_lid, ctx->ib_port,
                    &UCC_TL_MLX5_TEAM_LIB(team)->cfg.qp_conf, lib);
                if (UCC_OK != status) {
                    tl_error(lib, "failed to connect rc qps, %s",
                             ucc_status_string(status));
                    goto err_qp_connect;
                }
            }
            a2a->net.remote_ctrl[i] = remote_data->net_ctrl;
            a2a->net.rkeys[i]       = remote_data->recv_mkey_rkey;
        }

        a2a->atomic_scratch_bf_mr =
            ibv_reg_mr(ctx->shared_pd, (void *)&a2a->atomic_scratch_bf,
                       sizeof(a2a->atomic_scratch_bf),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!a2a->atomic_scratch_bf_mr) {
            tl_error(lib, "failed to register atomic scratch buff (errno=%d)",
                     errno);
            status = UCC_ERR_NO_MESSAGE;
            goto err_atomic_atomic_scratch_bf_mr;
        }

        /* allocate buffer for noninline UMR registration, has to be 2KB aligned */
        umr_buf_size = ucc_align_up(
            sizeof(struct mlx5_wqe_umr_repeat_ent_seg) * (node_size + 1), 64);
        ret =
            ucc_posix_memalign(&a2a->node.umr_entries_buf, 2048, umr_buf_size);
        if (ret) {
            tl_error(lib,
                     "failed to allocate %zd bytes for noninline UMR buffer",
                     umr_buf_size);
            return UCC_ERR_NO_MEMORY;
        }
        a2a->node.umr_entries_mr = ibv_reg_mr(
            ctx->shared_pd, (void *)a2a->node.umr_entries_buf, umr_buf_size,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!a2a->node.umr_entries_mr) {
            tl_error(lib, "failed to register umr buff (errno=%d)", errno);
            status = UCC_ERR_NO_MESSAGE;
            goto err_umr_entries_mr;
        }
        break;
    }
    return UCC_OK;

err_umr_entries_mr:
    ibv_dereg_mr(a2a->atomic_scratch_bf_mr);
err_atomic_atomic_scratch_bf_mr:
    if (a2a->is_dc) {
err_create_ah:
        for (j = 0; j < i ; j++) {
            ibv_destroy_ah(a2a->net.ahs[j]);
        }
        ucc_free(a2a->net.ahs);
err_ahs:
        ucc_free(a2a->net.remote_dctns);
    }
err_qp_connect:
err_remote_dctns:
    ucc_free(a2a->net.rkeys);
err_rkeys:
    ucc_free(a2a->net.rank_map);
err_rank_map:
err_service_allgather_progress:
err_service_allgather_post:
    if (!a2a->is_dc) {
err_create_rc_qps:
        for (j = 0; j < i ; j++) {
            ibv_destroy_qp(a2a->net.rc_qps[j].qp);
        }
        ucc_free(a2a->net.rc_qps);
    } else {
        ibv_destroy_qp(a2a->net.dct_qp);
err_init_dct:
err_init_dci:
        for (j = 0; j < i; j++) {
            ibv_destroy_qp(a2a->net.dcis[j].dci_qp);
        }
        ucc_free(a2a->net.dcis);
err_dcis:
        ibv_destroy_srq(a2a->net.srq);
    }
err_alloc_rc_qps:
err_srq:
    if (local_data) {
        ucc_free(local_data);
    }
err_local_data:
    ucc_tl_mlx5_destroy_mkeys(a2a, 0, lib);
err_mkeys:
    ucc_free(a2a->net.remote_ctrl);
err_remote_ctrl:
    ibv_dereg_mr(a2a->net.ctrl_mr);
err_ctrl_mr:
    ibv_destroy_cq(a2a->net.cq);
err_cq:
    ucc_tl_mlx5_destroy_umr(a2a, lib);
err_umr:
    ucc_free(a2a->net.blocks_sent);
err_blocks_sent:
    ucc_tl_mlx5_alltoall_barrier_free(a2a);
err_barrier:
    return status;
}

void ucc_tl_mlx5_alltoall_cleanup(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_alltoall_t *a2a = team->a2a;
    ucc_base_lib_t *   lib = UCC_TL_TEAM_LIB(team);
    ucc_status_t       status;
    int                i;

    if (!a2a) {
        return;
    }

    if (-1 == shmdt(a2a->node.storage)) {
        tl_error(lib, "failed to shmdt %p, errno %d", a2a->node.storage, errno);
    }
    if (a2a->node.asr_rank == a2a->node.sbgp->group_rank) {
        status = ucc_tl_mlx5_destroy_umr(a2a, lib);
        if (status != UCC_OK) {
            tl_error(lib, "failed to destroy UMR");
        }
        ibv_dereg_mr(a2a->net.ctrl_mr);
        ucc_free(a2a->net.remote_ctrl);
        if (a2a->is_dc) {
            for (i = 0; i < a2a->num_dci_qps; i++) {
                ibv_destroy_qp(a2a->net.dcis[i].dci_qp);
            }
            ucc_free(a2a->net.dcis);
            ibv_destroy_qp(a2a->net.dct_qp);
            ibv_destroy_srq(a2a->net.srq);
            for (i = 0; i < a2a->net.net_size; i++) {
                ibv_destroy_ah(a2a->net.ahs[i]);
            }
        } else {
            for (i = 0; i < a2a->net.net_size; i++) {
                ibv_destroy_qp(a2a->net.rc_qps[i].qp);
            }
        }
        if (a2a->is_dc) {
            ucc_free(a2a->net.remote_dctns);
            ucc_free(a2a->net.ahs);
        } else {
            ucc_free(a2a->net.rc_qps);
        }
        if (ibv_destroy_cq(a2a->net.cq)) {
            tl_error(lib, "net cq destroy failed (errno=%d)", errno);
        }

        status = ucc_tl_mlx5_destroy_mkeys(a2a, 0, lib);
        if (status != UCC_OK) {
            tl_error(lib, "failed to destroy Mkeys");
        }
        ucc_free(a2a->net.rkeys);
        ibv_dereg_mr(a2a->atomic_scratch_bf_mr);
        ucc_free(a2a->net.rank_map);
        ibv_dereg_mr(a2a->node.umr_entries_mr);
        ucc_free(a2a->node.umr_entries_buf);

        ucc_free(a2a->net.blocks_sent);
        ucc_tl_mlx5_alltoall_barrier_free(a2a);
    }
    ucc_free(a2a);
}
