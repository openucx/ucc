/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"
#include "tl_mlx5_ib.h"
#include "tl_mlx5_mkeys.h"
#include "coll_score/ucc_coll_score.h"
#include <sys/shm.h>
#include "core/ucc_team.h"

static ucc_mpool_ops_t ucc_tl_mlx5_dm_ops;
static ucc_status_t ucc_tl_mlx5_dm_init(ucc_tl_mlx5_team_t *team);
static void ucc_tl_mlx5_dm_cleanup(ucc_tl_mlx5_team_t *team);

static void calc_block_size(ucc_tl_mlx5_team_t *team)
{
    int i;
    int block_size = ucc_min(team->node.sbgp->group_size, MAX_BLOCK_SIZE);
    int msg_len    = 1;
    for (i = 0; i < MLX5_NUM_OF_BLOCKS_SIZE_BINS; i++) {
        while ((block_size * block_size) * msg_len > MAX_TRANSPOSE_SIZE) {
            block_size -= 1;
        }
        team->blocks_sizes[i] = block_size;
        msg_len               = msg_len << 1;
    }
}

struct rank_data {
    int team_rank;
    int sbgp_rank;
};

typedef struct net_exchage {
    struct rank_data rd;
    net_ctrl_t       net_ctrl;
    uint32_t         port_lid;
    uint32_t         recv_mkey_rkey;
    uint32_t         qpn[1];
} net_exchange_t;

static int compare_rank_data(const void *a, const void *b)
{
    const struct rank_data *d1 = (const struct rank_data *)a;
    const struct rank_data *d2 = (const struct rank_data *)b;
    return d1->team_rank > d2->team_rank ? 1 : -1;
}

static void build_rank_map(ucc_tl_mlx5_team_t *team,
                           net_exchange_t *global_data, size_t local_data_size)
{
    struct rank_data *data    = ucc_malloc(sizeof(*data) * team->net.net_size);
    net_exchange_t *d;
    int               i;


    for (i = 0; i < team->net.net_size; i++) {
        d = PTR_OFFSET(global_data, i * local_data_size);
        data[i].team_rank = d->rd.team_rank;
        data[i].sbgp_rank = d->rd.sbgp_rank;
    }

    team->net.rank_map = ucc_malloc(sizeof(int) * team->net.net_size);
    qsort(data, team->net.net_size, sizeof(*data), compare_rank_data);
    for (i = 0; i < team->net.net_size; i++) {
        team->net.rank_map[data[i].sbgp_rank] = i;
    }
    ucc_free(data);
}

static inline int ucc_tl_mlx5_calc_max_block_size(void)
{
    int block_row_size = 0;

    while ((SQUARED(block_row_size + 1) * MAX_MSG_SIZE) <= MAX_TRANSPOSE_SIZE) {
        block_row_size += 1;
    }
    return block_row_size;
}

UCC_CLASS_INIT_FUNC(ucc_tl_mlx5_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_mlx5_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_mlx5_context_t);
    ucc_sbgp_t *  node, *net;
    size_t        storage_size;
    int           i, j, node_size, ppn, team_size, nnodes;
    ucc_topo_t         *topo;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);
    /* TODO: init based on ctx settings and on params: need to check
             if all the necessary ranks mappings are provided */

    node      = ucc_topo_get_sbgp(UCC_TL_CORE_TEAM(self)->topo, UCC_SBGP_NODE);
    node_size = node->group_size;
    net       = ucc_topo_get_sbgp(UCC_TL_CORE_TEAM(self)->topo,
                                  UCC_SBGP_NODE_LEADERS);
    topo      = UCC_TL_CORE_TEAM(self)->topo;
    nnodes    = ucc_topo_nnodes(topo);
    team_size = UCC_TL_TEAM_SIZE(self);
    if (net->status == UCC_SBGP_NOT_EXISTS) {
        tl_debug(tl_context->lib, "disabling mlx5 for single node team");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (nnodes == team_size) {
        tl_debug(tl_context->lib, "disabling mlx5 for ppn=1 case, not supported so far");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!topo ||
        ucc_topo_min_ppn(topo) != ucc_topo_max_ppn(topo)) {
        tl_debug(tl_context->lib,
                 "disabling mlx5 for team with non-uniform ppn, "
                 "min_ppn %d, max_ppn %d",
                 ucc_topo_min_ppn(topo), ucc_topo_max_ppn(topo));
        return UCC_ERR_NOT_SUPPORTED;
    }
    ppn       = ucc_topo_max_ppn(topo);

    self->node_size = node_size;
    ucc_assert(team_size  == ppn * nnodes);

    for (i = 0; i < nnodes; i++) {
        for (j = 1; j < ppn; j++) {
            if (!ucc_team_ranks_on_same_node(i * ppn, i * ppn + j,
                                             UCC_TL_CORE_TEAM(self))) {
                tl_debug(tl_context->lib,
                         "disabling mlx5 for team with non contiguous "
                         "ranks-per-node placement");
                return UCC_ERR_NOT_SUPPORTED;
            }
        }
    }


    tl_info(tl_context->lib, "posted tl team: %p", self);

    self->node.sbgp        = node;
    self->net.sbgp         = net;
    self->status           = UCC_INPROGRESS;
    self->node.asr_rank    = MLX5_ASR_RANK;
    self->num_dci_qps      = UCC_TL_MLX5_TEAM_LIB(self)->cfg.num_dci_qps;
    self->sequence_number  = 1;
    self->net.ctrl_mr      = NULL;
    self->net.remote_ctrl  = NULL;
    self->net.rank_map     = NULL;

    self->net.dcis = ucc_malloc(sizeof(struct dci) * self->num_dci_qps);
    if (!self->net.dcis) {
        tl_error(tl_context->lib,"failed to allocate mem");
        return UCC_ERR_NO_MEMORY;
    }

    memset(self->op_busy, 0, MAX_OUTSTANDING_OPS * sizeof(int));

    ucc_assert(self->net.sbgp->status == UCC_SBGP_ENABLED || node->group_rank != 0);

    self->max_msg_size = MAX_MSG_SIZE;

    self->max_num_of_columns =
        ucc_div_round_up(node->group_size, 2 /* todo: there can be an estimation of
                                                    minimal possible block size */);

    if (self->node.asr_rank == node->group_rank) {
        self->net.net_size = self->net.sbgp->group_size;
        storage_size       = OP_SEGMENT_SIZE(self) * MAX_OUTSTANDING_OPS;
        self->bcast_data.shmid =
            shmget(IPC_PRIVATE, storage_size, IPC_CREAT | 0600);
        if (self->bcast_data.shmid == -1) {
            tl_error(tl_context->lib,"failed to allocate sysv shm segment for %zd bytes",
                     storage_size);
        } else {
            self->node.storage = shmat(self->bcast_data.shmid, NULL, 0);
            memset(self->node.storage, 0, storage_size);
            shmctl(self->bcast_data.shmid, IPC_RMID, NULL);
        }
        self->bcast_data.net_size = self->net.sbgp->group_size;
        tmpnam(self->bcast_data.sock_path); //TODO switch to mkstemp
    }

    self->state = TL_MLX5_TEAM_STATE_SHMID;

    return ucc_service_bcast(UCC_TL_CORE_TEAM(self), &self->bcast_data,
                             sizeof(ucc_tl_mlx5_bcast_data_t), self->node.asr_rank,
                             ucc_sbgp_to_subset(node), &self->scoll_req);
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_mlx5_team_t)
{

    ucc_status_t status = UCC_OK;
    int           i;

    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);

    if (-1 == shmdt(self->node.storage)) {
        tl_error(UCC_TL_TEAM_LIB(self), "failed to shmdt %p, errno %d",
                 self->node.storage, errno);
    }
    if (self->node.asr_rank == self->node.sbgp->group_rank) {
        status = ucc_tl_mlx5_destroy_umr(&self->net, UCC_TL_TEAM_LIB(self));
        if (status != UCC_OK) {
            tl_error(UCC_TL_TEAM_LIB(self), "failed to destroy UMR");
        }
        ibv_dereg_mr(self->net.ctrl_mr);
        ucc_free(self->net.remote_ctrl);
        if (self->is_dc) {
            for (i = 0; i < self->num_dci_qps; i++) {
                ibv_destroy_qp(self->net.dcis[i].dci_qp);
            }
            ibv_destroy_qp(self->net.dct_qp);
            ibv_destroy_srq(self->net.srq);
            for (i = 0; i < self->net.net_size; i++) {
                ibv_destroy_ah(self->net.ahs[i]);
            }
        } else {
            for (i = 0; i < self->net.net_size; i++) {
                ibv_destroy_qp(self->net.rc_qps[i]);
            }
        }
        if (self->is_dc) {
            ucc_free(self->net.remote_dctns);
            ucc_free(self->net.ahs);
        } else {
            ucc_free(self->net.rc_qps);
        }
        if (ibv_destroy_cq(self->net.cq)) {
            tl_error(UCC_TL_TEAM_LIB(self), "net cq destroy failed (errno=%d)",
                     errno);
        }

        status = ucc_tl_mlx5_destroy_mkeys(self, 0);
        if (status != UCC_OK) {
            tl_error(UCC_TL_TEAM_LIB(self), "failed to destroy Mkeys");
        }
        ucc_free(self->net.rkeys);
        ibv_dereg_mr(self->dummy_bf_mr);
        ucc_free(self->work_completion);
        ucc_free(self->net.rank_map);
        ibv_dereg_mr(self->node.umr_entries_mr);
        ucc_free(self->node.umr_entries_buf);

        ucc_free(self->net.blocks_sent);
        ibv_dereg_mr(self->net.barrier.mr);
        ucc_free(self->net.barrier.flags);
        ibv_dereg_mr(self->net.atomic.mr);
        ucc_free(self->net.atomic.counters);
        ucc_tl_mlx5_dm_cleanup(self);
    }
    ucc_free(self->net.dcis);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mlx5_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_mlx5_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_mlx5_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_mlx5_team_t)(tl_team);
    return UCC_OK;
}

            /* local_data_size = ((team->is_dc ? 1 : net_size) * sizeof(uint32_t)) + */
                /* sizeof(uint32_t) + 2 * sizeof(uint32_t) + */
                /* sizeof(void *) + sizeof(struct rank_data); */


static ucc_status_t tl_mlx5_alloc_atomic(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    size_t                 size;

    size = sizeof(*team->net.atomic.counters) * MAX_OUTSTANDING_OPS;
    team->net.atomic.counters = ucc_malloc(size, "atomic");
    if (!team->net.atomic.counters) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "failed to allocate %zd bytes for atomic counters array",
                 size);
        return UCC_ERR_NO_MEMORY;
    }

    team->net.atomic.mr = ibv_reg_mr(ctx->shared_pd, team->net.atomic.counters, size,
                                     IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE);

    if (!team->net.atomic.mr) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to register atomic couters array");
        ucc_free(team->net.atomic.counters);
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static ucc_status_t tl_mlx5_alloc_barrier(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    size_t                 size;

    /* allocating net_size + 1 flags. Last one is used as local buf
       for barrier RDMA */
    size = (team->net.net_size + 1) * sizeof(*team->net.barrier.flags) *
        MAX_OUTSTANDING_OPS;
    team->net.barrier.flags = ucc_calloc(1, size, "barrier");
    if (!team->net.barrier.flags) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "failed to allocate %zd bytes for barrier flags array",
                 size);
        return UCC_ERR_NO_MEMORY;
    }

    team->net.barrier.mr = ibv_reg_mr(ctx->shared_pd, team->net.barrier.flags, size,
                                     IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);

    if (!team->net.atomic.mr) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to register atomic couters array");
        ucc_free(team->net.atomic.counters);
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_team_create_test(ucc_base_team_t *tl_team)
{

    ucc_tl_mlx5_team_t *   team = ucc_derived_of(tl_team, ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    ucc_rank_t             node_size = team->node.sbgp->group_size;
    ucc_rank_t             node_rank = team->node.sbgp->group_rank;
    ucc_status_t status;
    ucc_tl_mlx5_op_t *op;
    int i, asr_cq_size, net_size;
    struct ibv_port_attr    port_attr;
    size_t                  local_data_size, umr_buf_size, op_seg_size;
    net_exchange_t *local_data, *global_data, *remote_data;

    status = ucc_service_coll_test(team->scoll_req);
    if (status < 0) {
        tl_error(tl_team->context->lib, "failure during service coll exchange: %s",
                 ucc_status_string(status));
        return status;
    }
    if (UCC_INPROGRESS == status) {
        return status;
    }
    local_data = team->scoll_req->data;
    ucc_service_coll_finalize(team->scoll_req);
    op_seg_size = OP_SEGMENT_SIZE(team);

    switch (team->state) {
    case TL_MLX5_TEAM_STATE_SHMID:
        if (team->bcast_data.shmid == -1) {
            tl_error(tl_team->context->lib,
                     "failed to allocate sysv shm segment");
            return UCC_ERR_NO_MEMORY;
        }
        team->net.net_size = team->bcast_data.net_size;
        if (team->node.asr_rank != node_rank) {
            // shmat already performed for asr above
            team->node.storage = shmat(team->bcast_data.shmid, NULL, 0);
        }
        if (team->node.storage == (void *)(-1)) {
            tl_error(tl_team->context->lib, "failed to shmat seg %d",
                     team->bcast_data.shmid);
            return UCC_ERR_NO_MEMORY;
        }
        for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
            op = &team->node.ops[i];
            op->ctrl     = PTR_OFFSET(team->node.storage, op_seg_size * i);
            op->my_ctrl  = PTR_OFFSET(op->ctrl, node_rank * sizeof(ucc_tl_mlx5_ctrl_t));
        }


        calc_block_size(team);
        if (UCC_TL_MLX5_TEAM_LIB(team)->cfg.block_size > MAX_BLOCK_SIZE) {
            tl_error(tl_team->context->lib, "Max Block size is %d", MAX_BLOCK_SIZE);
            return UCC_ERR_NO_MESSAGE;
        }
        team->requested_block_size = UCC_TL_MLX5_TEAM_LIB(team)->cfg.block_size;
        if (team->node.asr_rank == node_rank) {
            status = tl_mlx5_alloc_atomic(team);
            if (UCC_OK != status) {
                goto err;
            }

            status = tl_mlx5_alloc_barrier(team);
            if (UCC_OK != status) {
                goto err;
            }

            team->net.blocks_sent = ucc_malloc(sizeof(*team->net.blocks_sent) *
                                               team->net.net_size * MAX_OUTSTANDING_OPS,
                                               "blocks_sent");
            if (!team->net.blocks_sent) {
                tl_error(tl_team->context->lib, "failed to allocated %zd bytes for blocks_sent array",
                         sizeof(*team->net.blocks_sent) * team->net.net_size * MAX_OUTSTANDING_OPS);
                status = UCC_ERR_NO_MEMORY;
                goto err;
            }

            for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
                op->blocks_sent = PTR_OFFSET(team->net.blocks_sent,
                                             sizeof(*team->net.blocks_sent) * team->net.net_size * i);
            }

            for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
                team->previous_msg_size[i] = 0;
            }

            status = ucc_tl_mlx5_init_umr(ctx, &team->net);
            if (status != UCC_OK) {
                tl_error(tl_team->context->lib, "failed to init UMR");
                return status;
            }
            net_size = team->net.net_size;
            asr_cq_size =
                net_size * (SQUARED(team->node.sbgp->group_size / 2 + 1) + 1) * MAX_OUTSTANDING_OPS;
            team->net.cq =
                ibv_create_cq(ctx->shared_ctx, asr_cq_size, NULL, NULL, 0);
            if (!team->net.cq) {
                tl_error(tl_team->context->lib, "failed to allocate ASR CQ");
                return UCC_ERR_NO_MESSAGE;
                    }

            team->is_dc = (UCC_TL_MLX5_TEAM_LIB(team)->cfg.rc_dc == 2)
                ? ((net_size > RC_DC_LIMIT) ? 1 : 0)
                : UCC_TL_MLX5_TEAM_LIB(team)->cfg.rc_dc;

            ibv_query_port(ctx->ib_ctx, ctx->ib_port, &port_attr);

            team->net.ctrl_mr =
                ibv_reg_mr(ctx->shared_pd, team->node.storage,
                           op_seg_size * MAX_OUTSTANDING_OPS,
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE);
            if (!team->net.ctrl_mr) {
                tl_error(tl_team->context->lib,
                         "failed to register control data, errno %d", errno);
                return UCC_ERR_NO_MESSAGE;
            }
            team->net.remote_ctrl =
                ucc_calloc(sizeof(*team->net.remote_ctrl), net_size, "remote_ctrl");
            if (!team->net.remote_ctrl) {
                tl_error(tl_team->context->lib, "failed to allocate %zd bytes for remote_ctrl",
                         sizeof(*team->net.remote_ctrl) * net_size);
            }

            status = ucc_tl_mlx5_init_mkeys(team);
            if (status != UCC_OK) {
                tl_error(tl_team->context->lib, "failed to init mkeys");
                return status;
            }

            // for each ASR - qp num, in addition to port lid,
            // ctrl segment rkey and address, recieve mkey rkey
            /* local_data_size = ((team->is_dc ? 1 : net_size) * sizeof(uint32_t)) + */
                /* sizeof(uint32_t) + 2 * sizeof(uint32_t) + */
                /* sizeof(void *) + sizeof(struct rank_data); */
            local_data_size = sizeof(net_exchange_t);
            if (!team->is_dc) {
                /* need more space for net_size - 1 qpns */
                local_data_size += sizeof(uint32_t) * (net_size - 1);
            }

            local_data = ucc_malloc(local_data_size * (net_size + 1), "exchange_data");
            if (!local_data) {
                tl_error(tl_team->context->lib, "failed to allocate %zd bytes for exchange data",
                         local_data_size * (net_size + 1));
                return UCC_ERR_NO_MEMORY;
            }
            global_data = PTR_OFFSET(local_data, local_data_size);

            if (team->is_dc) {
                status = ucc_tl_mlx5_init_dc_qps_and_connect(team, local_data->qpn,
                                                             ctx->ib_port);
                if (UCC_OK != status) {
                    tl_error(tl_team->context->lib, "failed to init DC QPs");
                    goto free_data;
                }
            } else {
                status = ucc_tl_mlx5_create_rc_qps(team, local_data->qpn);
                if (UCC_OK != status) {
                    tl_error(tl_team->context->lib, "failed to init RC QPs");
                    goto free_data;
                }
            }

            local_data->port_lid = port_attr.lid;
            local_data->recv_mkey_rkey = team->node.team_recv_mkey->rkey;
            local_data->rd.team_rank = UCC_TL_TEAM_RANK(team);
            local_data->rd.sbgp_rank = team->net.sbgp->group_rank;
            local_data->net_ctrl.atomic.addr = team->net.atomic.counters;
            local_data->net_ctrl.atomic.rkey = team->net.atomic.mr->rkey;
            local_data->net_ctrl.barrier.addr = team->net.barrier.flags;
            local_data->net_ctrl.barrier.rkey = team->net.barrier.mr->rkey;
            status = ucc_service_allgather(UCC_TL_CORE_TEAM(team), local_data, global_data,
                                           local_data_size, ucc_sbgp_to_subset(team->net.sbgp),
                                           &team->scoll_req);
            if (UCC_OK != status) {
                tl_error(tl_team->context->lib, "failed start service allgather");
                goto free_data;
            }
            team->scoll_req->data = local_data;
            team->state = TL_MLX5_TEAM_STATE_EXCHANGE;
            return UCC_INPROGRESS;
        }
    case TL_MLX5_TEAM_STATE_EXCHANGE:
        if (team->node.asr_rank != node_rank) {
            break;
        }
        net_size = team->net.net_size;
        local_data_size = sizeof(net_exchange_t);
        if (!team->is_dc) {
            /* need more space for net_size - 1 qpns */
            local_data_size += sizeof(uint32_t) * (net_size - 1);
        }

        global_data = PTR_OFFSET(local_data, local_data_size);

        build_rank_map(team, global_data, local_data_size);
        team->net.rkeys = ucc_malloc(sizeof(uint32_t) * net_size);
        if (!team->net.rkeys) {
            tl_error(tl_team->context->lib, "failed to allocate %zd bytes for net rkeys",
                     sizeof(uint32_t) * net_size);
            return UCC_ERR_NO_MEMORY;
        }
        if (team->is_dc) {
            team->net.remote_dctns = ucc_malloc(sizeof(uint32_t) * net_size);
            if (!team->net.remote_dctns) {
                tl_error(tl_team->context->lib, "failed to allocate %zd bytes for remote_dctns",
                         sizeof(uint32_t) * net_size);
                return UCC_ERR_NO_MEMORY;
            }
            team->net.ahs = ucc_malloc(sizeof(struct ibv_ah *) * net_size);
            if (!team->net.ahs) {
                tl_error(tl_team->context->lib, "failed to allocate %zd bytes for net ahs",
                         sizeof(struct ibv_ah*) * net_size);
                return UCC_ERR_NO_MEMORY;
            }
        }
        for (i = 0; i < net_size; i++) {
            remote_data = PTR_OFFSET(global_data, i * local_data_size);
            if (team->is_dc) {
                team->net.remote_dctns[i] = remote_data->qpn[0];
                status = ucc_tl_mlx5_create_ah(&team->net.ahs[i], remote_data->port_lid,
                                               ctx->ib_port, team);
                if (UCC_OK != status) {
                    tl_error(tl_team->context->lib, "failed to create ah, %s",
                             ucc_status_string(status));
                    return status;
                }
            } else {
                status = ucc_tl_mlx5_qp_connect(team->net.rc_qps[i],
                                       remote_data->qpn[team->net.sbgp->group_rank],
                                       remote_data->port_lid, ctx->ib_port,tl_team->context->lib);
                if (UCC_OK != status) {
                    tl_error(tl_team->context->lib, "failed to connect rc qps, %s",
                             ucc_status_string(status));
                    return status;
                }
            }
            team->net.remote_ctrl[i] = remote_data->net_ctrl;
            team->net.rkeys[i]       = remote_data->recv_mkey_rkey;
        }

        team->dummy_bf_mr =
            ibv_reg_mr(ctx->shared_pd, (void *)&team->dummy_atomic_buff,
                       sizeof(team->dummy_atomic_buff),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!team->dummy_bf_mr) {
            tl_error(tl_team->context->lib,
                     "failed to register dummy buff (errno=%d)", errno);
            return UCC_ERR_NO_MESSAGE;
        }

        team->work_completion = ucc_malloc(sizeof(struct ibv_wc) *
                                           ucc_min(net_size, MIN_POLL_WC));
        if (!team->work_completion) {
            tl_error(tl_team->context->lib, "failed to allocate %zd bytes for wc array",
                     sizeof(struct ibv_wc) * net_size);
            return UCC_ERR_NO_MEMORY;
        }
        memset(team->cq_completions, 0, sizeof(team->cq_completions));
        /* allocate buffer for noninline UMR registration, has to be 2KB aligned */
        umr_buf_size = ucc_align_up(sizeof(struct mlx5_wqe_umr_repeat_ent_seg) *
                                    (node_size + 1), 64);
        ucc_posix_memalign(&team->node.umr_entries_buf, 2048, umr_buf_size);
        if (!team->node.umr_entries_buf) {
            tl_error(tl_team->context->lib,
                     "failed to allocate %zd bytes for noninline UMR buffer",
                     umr_buf_size);
            return UCC_ERR_NO_MEMORY;
        }
        team->node.umr_entries_mr = ibv_reg_mr(
            ctx->shared_pd, (void *)team->node.umr_entries_buf,
            umr_buf_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!team->node.umr_entries_mr) {
            tl_error(tl_team->context->lib,
                     "failed to register umr buff (errno=%d)", errno);
            return UCC_ERR_NO_MESSAGE;
        }

        /* MEMIC alloc, todo move to CTX */
        status = ucc_tl_mlx5_dm_init(team);
        if (UCC_OK != status) {
            return status;
        }
        break;
    }

    tl_info(tl_team->context->lib, "initialized tl team: %p", team);
    team->status = UCC_OK;

    return UCC_OK;

free_data:
    ucc_free(local_data);
err:
    return status;
}

static void ucc_tl_mlx5_dm_cleanup(ucc_tl_mlx5_team_t *team)
{
    ibv_dereg_mr(team->dm_mr);
    if (UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host) {
        ucc_free(team->dm_ptr);
    }  else {
        ibv_free_dm(team->dm_ptr);
    }
    ucc_mpool_cleanup(&team->dm_pool, 1);
}

static ucc_status_t ucc_tl_mlx5_dm_init(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    size_t memic_chunk         = UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_buf_size;
    size_t n_memic_chunks      = UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_buf_num;
    int    dm_host             = UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host;
    struct ibv_device_attr_ex attr;
    struct ibv_alloc_dm_attr  dm_attr;
    int max_n_chunks, chunks_to_alloc, i;
    ucc_status_t status;

    if (dm_host) {
    max_n_chunks = 8;
    chunks_to_alloc = (n_memic_chunks == UCC_ULUNITS_AUTO) ? max_n_chunks :
        n_memic_chunks;
    dm_attr.length = chunks_to_alloc * memic_chunk ;
    team->dm_ptr = ucc_malloc(dm_attr.length, "memic_host");
    if (!team->dm_ptr) {
        tl_error(UCC_TL_TEAM_LIB(team), " memic_host allocation failed");
        return UCC_ERR_NO_MEMORY;
    }
    team->dm_mr = ibv_reg_mr(ctx->shared_pd, team->dm_ptr, dm_attr.length,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

    } else {
        attr.comp_mask = 0;
        if(ibv_query_device_ex(ctx->ib_ctx, NULL, &attr)){
            tl_error(UCC_TL_TEAM_LIB(team),
                     "failed to query device (errno=%d)", errno);
            return UCC_ERR_NO_MESSAGE;
        }
        if (!attr.max_dm_size) {
            tl_error(UCC_TL_TEAM_LIB(team), "device doesn't support dm allocation");
            return UCC_ERR_NO_MESSAGE;
        }
        memset(&dm_attr, 0, sizeof(dm_attr));
        max_n_chunks = attr.max_dm_size / memic_chunk;
        chunks_to_alloc = (n_memic_chunks == UCC_ULUNITS_AUTO) ? max_n_chunks :
            n_memic_chunks;

        for(i = chunks_to_alloc; i > 0; i--) {
            dm_attr.length = i * memic_chunk;
            team->dm_ptr = ibv_alloc_dm(ctx->ib_ctx, &dm_attr);
            if (team->dm_ptr) {
                break;
            }
        }
        if (!team->dm_ptr) {
            tl_error(UCC_TL_TEAM_LIB(team), "dev mem allocation failed, attr.max %zd, errno %d",
                     attr.max_dm_size, errno);
            return UCC_ERR_NO_MESSAGE;
        }
        if (n_memic_chunks != UCC_ULUNITS_AUTO &&
            i != n_memic_chunks) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "couldn't allocate memic chunks, required %zd allocated %d, max %d",
                     n_memic_chunks, i, max_n_chunks);
            return UCC_ERR_NO_MESSAGE;

        }
        n_memic_chunks = i;
        team->dm_mr = ibv_reg_dm_mr(ctx->shared_pd, team->dm_ptr, 0, dm_attr.length,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_ZERO_BASED);
    }
    if(!team->dm_mr){
        tl_error(UCC_TL_TEAM_LIB(team),"failed to reg memic");
        return UCC_ERR_NO_MESSAGE;
    }

    team->oob_req = NULL;
    status = ucc_mpool_init(
        &team->dm_pool, 0, sizeof(ucc_tl_mlx5_dm_chunk_t), 0,
        UCC_CACHE_LINE_SIZE, n_memic_chunks, n_memic_chunks,
        &ucc_tl_mlx5_dm_ops, UCC_THREAD_MULTIPLE, "mlx5 dm pool");
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "failed to init dm pool");
        return status;
    }
    return UCC_OK;
}


static ucc_status_t ucc_tl_mlx5_dm_chunk_alloc(ucc_mpool_t *mp, //NOLINT
                                            size_t *size_p,
                                            void **chunk_p)
{
    *chunk_p = ucc_malloc(*size_p, "mlx5 dm");
    if (!*chunk_p) {
        return UCC_ERR_NO_MEMORY;
    }
    return UCC_OK;
}

static void ucc_tl_mlx5_dm_chunk_init(ucc_mpool_t *mp, //NOLINT
                                   void *obj, void *chunk) //NOLINT
{
    ucc_tl_mlx5_dm_chunk_t *c = (ucc_tl_mlx5_dm_chunk_t *)obj;
    ucc_tl_mlx5_team_t *team = ucc_container_of(mp, ucc_tl_mlx5_team_t, dm_pool);
    const size_t memic_chunk = 8192;
    c->offset = (ptrdiff_t) team->oob_req;
    team->oob_req = PTR_OFFSET(team->oob_req, memic_chunk);
}

static void ucc_tl_mlx5_dm_chunk_release(ucc_mpool_t *mp, void *chunk) //NOLINT
{
    ucc_free(chunk);
}

static ucc_mpool_ops_t ucc_tl_mlx5_dm_ops = {.chunk_alloc   = ucc_tl_mlx5_dm_chunk_alloc,
                                             .chunk_release = ucc_tl_mlx5_dm_chunk_release,
                                             .obj_init      = ucc_tl_mlx5_dm_chunk_init,
                                             .obj_cleanup   = NULL};
