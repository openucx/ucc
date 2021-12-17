/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mhba.h"
#include "tl_mhba_ib.h"
#include "tl_mhba_mkeys.h"
#include "coll_score/ucc_coll_score.h"
#include <sys/shm.h>
#include "core/ucc_team.h"

static void calc_block_size(ucc_tl_mhba_team_t *team)
{
    int i;
    int block_size = ucc_min(team->node.sbgp->group_size, MAX_BLOCK_SIZE);
    int msg_len    = 1;
    for (i = 0; i < MHBA_NUM_OF_BLOCKS_SIZE_BINS; i++) {
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

static int compare_rank_data(const void *a, const void *b)
{
    const struct rank_data *d1 = (const struct rank_data *)a;
    const struct rank_data *d2 = (const struct rank_data *)b;
    return d1->team_rank > d2->team_rank ? 1 : -1;
}

static void build_rank_map(ucc_tl_mhba_team_t *team,
                           uint32_t *global_data, size_t local_data_size)
{
    struct rank_data *data    = ucc_malloc(sizeof(*data) * team->net.net_size);
    uint32_t *d;
    int               i;


    for (i = 0; i < team->net.net_size; i++) {
        d = PTR_OFFSET(global_data, i * local_data_size);
        data[i].team_rank =
            ((struct rank_data*)(&d[(team->is_dc ? 1 : team->net.net_size) + 5]))->team_rank;
        data[i].sbgp_rank =
            ((struct rank_data*)(&d[(team->is_dc ? 1 : team->net.net_size) + 5]))->sbgp_rank;
    }

    team->net.rank_map = ucc_malloc(sizeof(int) * team->net.net_size);
    qsort(data, team->net.net_size, sizeof(*data), compare_rank_data);
    for (i = 0; i < team->net.net_size; i++) {
        team->net.rank_map[data[i].sbgp_rank] = i;
    }
    ucc_free(data);
}

static inline int ucc_tl_mhba_calc_max_block_size(void)
{
    int block_row_size = 0;

    while ((SQUARED(block_row_size + 1) * MAX_MSG_SIZE) <= MAX_TRANSPOSE_SIZE) {
        block_row_size += 1;
    }
    return block_row_size;
}

UCC_CLASS_INIT_FUNC(ucc_tl_mhba_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_mhba_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_mhba_context_t);
    ucc_sbgp_t *  node, *net;
    size_t        storage_size;
    int i,j,node_size;
    void *ctrl_addr;
    ucc_tl_mhba_ctrl_t *rank_ctrl;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);
    /* TODO: init based on ctx settings and on params: need to check
             if all the necessary ranks mappings are provided */

    node      = ucc_topo_get_sbgp(UCC_TL_CORE_TEAM(self)->topo, UCC_SBGP_NODE);
    node_size = node->group_size;
    net       = ucc_topo_get_sbgp(UCC_TL_CORE_TEAM(self)->topo,
                                  UCC_SBGP_NODE_LEADERS);

    if (net->status == UCC_SBGP_NOT_EXISTS) {
        tl_debug(tl_context->lib, "disabling mhba for single node team");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (net->group_size == UCC_TL_TEAM_SIZE(self)) {
        tl_debug(tl_context->lib, "disabling mhba for ppn=1 case, not supported so far");
        return UCC_ERR_NOT_SUPPORTED;
    }

    tl_info(tl_context->lib, "posted tl team: %p", self);

    self->node.sbgp        = node;
    self->net.sbgp         = net;
    self->status           = UCC_INPROGRESS;
    self->node.asr_rank    = MHBA_ASR_RANK;
    self->transpose        = UCC_TL_MHBA_TEAM_LIB(self)->cfg.transpose;
    self->num_dci_qps      = UCC_TL_MHBA_TEAM_LIB(self)->cfg.num_dci_qps;
    self->sequence_number  = 0;
    self->net.ctrl_mr      = NULL;
    self->net.remote_ctrl  = NULL;
    self->net.rank_map     = NULL;
    self->transpose_buf_mr = NULL;
    self->transpose_buf    = NULL;

    self->net.dcis = ucc_malloc(sizeof(struct dci) * self->num_dci_qps);
    if (!self->net.dcis) {
        tl_error(tl_context->lib,"failed to allocate mem");
        return UCC_ERR_NO_MEMORY;
    }

    memset(self->op_busy, 0, MAX_OUTSTANDING_OPS * sizeof(int));

    ucc_assert(self->net.sbgp->status == UCC_SBGP_ENABLED || node->group_rank != 0);

    self->max_msg_size = MAX_MSG_SIZE;

    self->max_num_of_columns =
    		ucc_div_round_up(node->group_size, ucc_tl_mhba_calc_max_block_size());

    storage_size =
        (MHBA_CTRL_SIZE + (2 * MHBA_DATA_SIZE * self->max_num_of_columns)) *
            node_size * MAX_OUTSTANDING_OPS +
        MHBA_CTRL_SIZE * MAX_OUTSTANDING_OPS;

    if (self->node.asr_rank == node->group_rank) {
        self->bcast_data.shmid =
            shmget(IPC_PRIVATE, storage_size, IPC_CREAT | 0600);
        if (self->bcast_data.shmid == -1) {
            tl_error(tl_context->lib,"failed to allocate sysv shm segment for %zd bytes",
                     storage_size);
        } else {
            self->node.storage = shmat(self->bcast_data.shmid, NULL, 0);
            for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
                ctrl_addr = self->node.storage +
                    MHBA_CTRL_SIZE * MAX_OUTSTANDING_OPS +
                    MHBA_CTRL_SIZE * node_size * i;
                for (j = 0; j < node->group_size; j++) {
                    rank_ctrl = PTR_OFFSET(ctrl_addr, j * MHBA_CTRL_SIZE);
                    memset(rank_ctrl, 0, MHBA_CTRL_SIZE);
                    rank_ctrl->seq_num = -1; // because sequence number begin from 0
                }
            }
            shmctl(self->bcast_data.shmid, IPC_RMID, NULL);
        }
        self->bcast_data.net_size = self->net.sbgp->group_size;
        tmpnam(self->bcast_data.sock_path); //TODO switch to mkstemp
    }

    self->state = TL_MHBA_TEAM_STATE_SHMID;

    return ucc_service_bcast(UCC_TL_CORE_TEAM(self), &self->bcast_data,
                             sizeof(ucc_tl_mhba_bcast_data_t), self->node.asr_rank,
                             ucc_sbgp_to_subset(node), &self->scoll_req);
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_mhba_team_t)
{

    ucc_status_t status = UCC_OK;
    int           i;

    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);

    if (-1 == shmdt(self->node.storage)) {
        tl_error(UCC_TL_TEAM_LIB(self), "failed to shmdt %p, errno %d",
                 self->node.storage, errno);
    }
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        ucc_tl_mhba_op_t *op = &self->node.ops[i];
        ucc_free(op->recv_umr_data);
        ucc_free(op->send_umr_data);
        ucc_free(op->my_send_umr_data);
        ucc_free(op->my_recv_umr_data);
    }
    if (self->node.asr_rank == self->node.sbgp->group_rank) {
        status = ucc_tl_mhba_destroy_umr(&self->node,UCC_TL_TEAM_LIB(self));
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

        status = ucc_tl_mhba_destroy_mkeys(self, 0);
        if (status != UCC_OK) {
            tl_error(UCC_TL_TEAM_LIB(self), "failed to destroy Mkeys");
        }
        ucc_free(self->net.rkeys);
        ibv_dereg_mr(self->dummy_bf_mr);
        ucc_free(self->work_completion);
        ucc_free(self->net.rank_map);
        if (self->transpose) {
            ibv_dereg_mr(self->transpose_buf_mr);
            ucc_free(self->transpose_buf);
        }
        ibv_dereg_mr(self->node.umr_entries_mr);
        ucc_free(self->node.umr_entries_buf);
    }
    ucc_free(self->net.dcis);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mhba_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_mhba_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_mhba_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_mhba_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_mhba_team_create_test(ucc_base_team_t *tl_team)
{

    ucc_tl_mhba_team_t *   team = ucc_derived_of(tl_team, ucc_tl_mhba_team_t);
    ucc_tl_mhba_context_t *ctx = UCC_TL_MHBA_TEAM_CTX(team);
    int                    n_cols = team->max_num_of_columns;
    ucc_rank_t             node_size = team->node.sbgp->group_size;
    ucc_status_t status;
    ucc_tl_mhba_op_t *op;
    int i, j, asr_cq_size, net_size;
    struct ibv_port_attr    port_attr;
    size_t                  local_data_size, umr_buf_size;
    uint32_t               *local_data, *global_data, *remote_data;

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

    switch (team->state) {
    case TL_MHBA_TEAM_STATE_SHMID:
        if (team->bcast_data.shmid == -1) {
            tl_error(tl_team->context->lib,
                     "failed to allocate sysv shm segment");
            return UCC_ERR_NO_MEMORY;
        }
        team->net.net_size = team->bcast_data.net_size;
        if (team->node.asr_rank != team->node.sbgp->group_rank) {
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

            op->my_recv_umr_data = ucc_malloc(sizeof(void *) * n_cols);
            op->my_send_umr_data = ucc_malloc(sizeof(void *) * n_cols);
            op->send_umr_data = ucc_malloc(sizeof(void *) * n_cols);
            op->recv_umr_data = ucc_malloc(sizeof(void *) * n_cols);
            if (!op->my_recv_umr_data || !op->my_send_umr_data ||
                !op->send_umr_data || !op->recv_umr_data) {
                tl_error(tl_team->context->lib, "failed to allocate umr data");
                return UCC_ERR_NO_MEMORY;
            }

            op->ctrl = team->node.storage + MHBA_CTRL_SIZE * MAX_OUTSTANDING_OPS +
                MHBA_CTRL_SIZE * node_size * i;
            op->my_ctrl = PTR_OFFSET(op->ctrl, team->node.sbgp->group_rank * MHBA_CTRL_SIZE);
            for (j = 0; j < n_cols; j++) {
                op->send_umr_data[j] = PTR_OFFSET(team->node.storage,
                                                  (node_size + 1) * MHBA_CTRL_SIZE *
                                                  MAX_OUTSTANDING_OPS +
                                                  i * MHBA_DATA_SIZE * n_cols *
                                                  node_size +
                                                  j * MHBA_DATA_SIZE * node_size);
                op->my_send_umr_data[j] = PTR_OFFSET(op->send_umr_data[j],
                                                     team->node.sbgp->group_rank * MHBA_DATA_SIZE);
                op->recv_umr_data[j] = PTR_OFFSET(op->send_umr_data[j],
                                                  MHBA_DATA_SIZE * n_cols * node_size *
                                                  MAX_OUTSTANDING_OPS);
                op->my_recv_umr_data[j] = PTR_OFFSET(op->recv_umr_data[j],
                                                     team->node.sbgp->group_rank * MHBA_DATA_SIZE);
            }
        }

        calc_block_size(team);
        if (UCC_TL_MHBA_TEAM_LIB(team)->cfg.block_size > MAX_BLOCK_SIZE) {
            tl_error(tl_team->context->lib, "Max Block size is %d", MAX_BLOCK_SIZE);
            return UCC_ERR_NO_MESSAGE;
        }
        team->requested_block_size = UCC_TL_MHBA_TEAM_LIB(team)->cfg.block_size;
        if (team->node.asr_rank == team->node.sbgp->group_rank) {
            for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
                team->previous_msg_size[i] = 0;
            }
            if (team->transpose) {
                team->transpose_buf =
                    ucc_malloc(UCC_TL_MHBA_TEAM_LIB(team)->cfg.transpose_buf_size);
                if (!team->transpose_buf) {
                    tl_error(tl_team->context->lib,
                             "failed to allocate %zd bytes for transpose buf",
                             UCC_TL_MHBA_TEAM_LIB(team)->cfg.transpose_buf_size);
                    return UCC_ERR_NO_MEMORY;
                }
                team->transpose_buf_mr =
                    ibv_reg_mr(ctx->shared_pd, team->transpose_buf,
                               UCC_TL_MHBA_TEAM_LIB(team)->cfg.transpose_buf_size,
                               IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
                if (!team->transpose_buf_mr) {
                    tl_error(tl_team->context->lib,
                             "failed to register transpose buff, errno %d", errno);
                    return UCC_ERR_NO_MESSAGE;
                        }
            }
//        build_rank_map(team); TODO
            status = ucc_tl_mhba_init_umr(ctx, &team->node);
            if (status != UCC_OK) {
                tl_error(tl_team->context->lib, "failed to init UMR");
                return status;
            }
            net_size = team->net.net_size;
            asr_cq_size = net_size * MAX_OUTSTANDING_OPS;
            team->net.cq =
                ibv_create_cq(ctx->shared_ctx, asr_cq_size, NULL, NULL, 0);
            if (!team->net.cq) {
                tl_error(tl_team->context->lib, "failed to allocate ASR CQ");
                return UCC_ERR_NO_MESSAGE;
                    }

            team->is_dc = (UCC_TL_MHBA_TEAM_LIB(team)->cfg.rc_dc == 2)
                ? ((net_size > RC_DC_LIMIT) ? 1 : 0)
                : UCC_TL_MHBA_TEAM_LIB(team)->cfg.rc_dc;

            ibv_query_port(ctx->ib_ctx, ctx->ib_port, &port_attr);

            team->net.ctrl_mr =
                ibv_reg_mr(ctx->shared_pd, team->node.storage,
                           MHBA_CTRL_SIZE * MAX_OUTSTANDING_OPS,
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

            status = ucc_tl_mhba_init_mkeys(team);
            if (status != UCC_OK) {
                tl_error(tl_team->context->lib, "failed to init mkeys");
                return status;
            }

            // for each ASR - qp num, in addition to port lid,
            // ctrl segment rkey and address, recieve mkey rkey
            local_data_size = ((team->is_dc ? 1 : net_size) * sizeof(uint32_t)) +
                sizeof(uint32_t) + 2 * sizeof(uint32_t) +
                sizeof(void *) + sizeof(struct rank_data);
            local_data = ucc_malloc(local_data_size * (net_size + 1), "exchange_data");
            if (!local_data) {
                tl_error(tl_team->context->lib, "failed to allocate %zd bytes for exchange data",
                         local_data_size * (net_size + 1));
                return UCC_ERR_NO_MEMORY;
            }
            global_data = PTR_OFFSET(local_data, local_data_size);

            if (team->is_dc) {
                status = ucc_tl_mhba_init_dc_qps_and_connect(team, local_data,
                                                             ctx->ib_port);
                if (UCC_OK != status) {
                    tl_error(tl_team->context->lib, "failed to init DC QPs");
                    goto free_data;
                }
            } else {
                status = ucc_tl_mhba_create_rc_qps(team, local_data);
                if (UCC_OK != status) {
                    tl_error(tl_team->context->lib, "failed to init RC QPs");
                    goto free_data;
                }
            }

            local_data[(team->is_dc ? 1 : net_size)]     = port_attr.lid;
            local_data[(team->is_dc ? 1 : net_size) + 1] = team->net.ctrl_mr->rkey;
            *((uint64_t *)&local_data[(team->is_dc ? 1 : net_size) + 2]) =
                (uint64_t)(uintptr_t)team->net.ctrl_mr->addr;
            local_data[(team->is_dc ? 1 : net_size) + 4] =
                team->node.team_recv_mkey->rkey;
            ((struct rank_data*)(&local_data[(team->is_dc ? 1 : net_size) + 5]))->team_rank =
                UCC_TL_TEAM_RANK(team);
            ((struct rank_data*)(&local_data[(team->is_dc ? 1 : net_size) + 5]))->sbgp_rank =
                team->net.sbgp->group_rank;

            status = ucc_service_allgather(UCC_TL_CORE_TEAM(team), local_data, global_data,
                                           local_data_size, ucc_sbgp_to_subset(team->net.sbgp),
                                           &team->scoll_req);
            if (UCC_OK != status) {
                tl_error(tl_team->context->lib, "failed start service allgather");
                goto free_data;
            }
            team->scoll_req->data = local_data;
            team->state = TL_MHBA_TEAM_STATE_EXCHANGE;
            return UCC_INPROGRESS;
        }
    case TL_MHBA_TEAM_STATE_EXCHANGE:
        if (team->node.asr_rank != team->node.sbgp->group_rank) {
            break;
        }
        net_size = team->net.net_size;
        local_data_size = ((team->is_dc ? 1 : net_size) * sizeof(uint32_t)) +
            sizeof(uint32_t) + 2 * sizeof(uint32_t) +
            sizeof(void *) + sizeof(struct rank_data);
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
                team->net.remote_dctns[i] = remote_data[0];
                status = ucc_tl_mhba_create_ah(&team->net.ahs[i], remote_data[1],
                                               ctx->ib_port, team);
                if (UCC_OK != status) {
                    tl_error(tl_team->context->lib, "failed to create ah, %s",
                             ucc_status_string(status));
                    return status;
                }
            } else {
                status = ucc_tl_mhba_qp_connect(team->net.rc_qps[i],
                                       remote_data[team->net.sbgp->group_rank],
                                       remote_data[net_size], ctx->ib_port,tl_team->context->lib);
                if (UCC_OK != status) {
                    tl_error(tl_team->context->lib, "failed to connect rc qps, %s",
                             ucc_status_string(status));
                    return status;
                }
            }
            team->net.remote_ctrl[i].rkey =
                remote_data[(team->is_dc ? 1 : net_size) + 1];
            team->net.remote_ctrl[i].addr = (void *)(uintptr_t)(
                *((uint64_t *)&remote_data[(team->is_dc ? 1 : net_size) + 2]));
            team->net.rkeys[i] = remote_data[(team->is_dc ? 1 : net_size) + 4];
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

        team->work_completion = ucc_malloc(sizeof(struct ibv_wc) * net_size);
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
        break;
    }

    tl_info(tl_team->context->lib, "initialized tl team: %p", team);
    team->status = UCC_OK;

    return UCC_OK;

free_data:
    ucc_free(local_data);
    return status;
}
