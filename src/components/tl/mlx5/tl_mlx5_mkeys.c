/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"
#include "tl_mlx5_ib.h"
#include "tl_mlx5_coll.h"
#include "tl_mlx5_mkeys.h"
#include <inttypes.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

static ucc_status_t create_umr_qp(struct ibv_qp **qp,
                                  struct ibv_cq  *cq,
                                  ucc_tl_mlx5_context_t *     ctx)
{
    struct ibv_qp_init_attr_ex umr_init_attr_ex;
    struct mlx5dv_qp_init_attr umr_mlx5dv_qp_attr;
    struct ibv_port_attr       port_attr;
    ucc_status_t               status = UCC_OK;
    struct ibv_qp_ex    *qp_ex;

    tl_debug(UCC_TL_CTX_LIB(ctx), "Create UMR QP");

    memset(&umr_mlx5dv_qp_attr, 0, sizeof(umr_mlx5dv_qp_attr));
    memset(&umr_init_attr_ex, 0, sizeof(umr_init_attr_ex));

    umr_mlx5dv_qp_attr.comp_mask =
        MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    umr_mlx5dv_qp_attr.create_flags = 0;
    umr_mlx5dv_qp_attr.send_ops_flags =
        MLX5DV_QP_EX_WITH_MR_LIST | MLX5DV_QP_EX_WITH_MR_INTERLEAVED | MLX5DV_QP_EX_WITH_RAW_WQE;

    umr_init_attr_ex.send_cq          = cq;
    umr_init_attr_ex.recv_cq          = cq;
    umr_init_attr_ex.cap.max_send_wr  = 1;
    umr_init_attr_ex.cap.max_recv_wr  = 1;
    umr_init_attr_ex.cap.max_send_sge = 1;
    umr_init_attr_ex.cap.max_recv_sge = 1;
    // `max_inline_data` determines the WQE size that the QP will support.
    // The 'max_inline_data' should be modified only when the number of
    // arrays to interleave is greater than 3.
    //TODO query the devices what is max supported
    umr_init_attr_ex.cap.max_inline_data =
        828; // the max number possible, Sergey Gorenko's email
    umr_init_attr_ex.qp_type = IBV_QPT_RC;
    umr_init_attr_ex.comp_mask =
        IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    umr_init_attr_ex.pd = ctx->shared_pd;
    umr_init_attr_ex.send_ops_flags |= IBV_QP_EX_WITH_SEND;
    *qp = mlx5dv_create_qp(ctx->shared_ctx, &umr_init_attr_ex,
                                   &umr_mlx5dv_qp_attr);
    if (*qp == NULL) {
        tl_error(UCC_TL_CTX_LIB(ctx), "UMR QP creation failed");
        return UCC_ERR_NO_MESSAGE;
    }
    tl_debug(UCC_TL_CTX_LIB(ctx),"UMR QP created. Returned with cap.max_inline_data = %d",
             umr_init_attr_ex.cap.max_inline_data);

    qp_ex = ibv_qp_to_qp_ex(*qp);
    if (qp_ex == NULL) {
        tl_error(UCC_TL_CTX_LIB(ctx), "UMR qp_ex creation failed");
        status = UCC_ERR_NO_MESSAGE;
        goto failure;
    }

    // Turning on the IBV_SEND_SIGNALED option, will cause the reported work comletion to be with MLX5DV_WC_UMR opcode.
    // The option IBV_SEND_INLINE is required by the current API.
    qp_ex->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    if (ibv_query_port(ctx->ib_ctx, ctx->ib_port, &port_attr)) {
        tl_error(UCC_TL_CTX_LIB(ctx), "Couldn't get port info (errno=%d)",
                 errno);
        status = UCC_ERR_NO_MESSAGE;
        goto failure;
    }
    tl_debug(UCC_TL_CTX_LIB(ctx), "Connect UMR QP to itself");
    status = ucc_tl_mlx5_qp_connect(*qp, (*qp)->qp_num,
                                    port_attr.lid, ctx->ib_port, &UCC_TL_CTX_LIB(ctx)->super.super);
    if (status != UCC_OK) {
        goto failure;
    }

    return UCC_OK;

failure:
    if (ibv_destroy_qp(*qp)) {
        tl_error(UCC_TL_CTX_LIB(ctx), "UMR qp destroy failed (errno=%d)",
                 errno);
    }
    return status;
}

/**
 * Create and connect UMR qp & cq.
 * @param ctx mlx5 team context
 * @param node struct of the current process's node
 */
ucc_status_t ucc_tl_mlx5_init_umr(ucc_tl_mlx5_context_t *ctx,
                                  ucc_tl_mlx5_net_t *   net)
{
    ucc_status_t status = UCC_OK;
    tl_debug(UCC_TL_CTX_LIB(ctx), "Create UMR CQ");
    net->umr_cq = ibv_create_cq(ctx->shared_ctx, UMR_CQ_SIZE, NULL, NULL, 0);
    if (net->umr_cq == NULL) {
        tl_error(UCC_TL_CTX_LIB(ctx), "UMR CQ creation failed");
        return UCC_ERR_NO_MESSAGE;
    }
    status = create_umr_qp(&net->umr_qp, net->umr_cq, ctx);
    if (status != UCC_OK) {
        goto err;
    }
    return status;

err:
    if (ibv_destroy_cq(net->umr_cq)) {
        tl_error(UCC_TL_CTX_LIB(ctx), "UMR cq destroy failed (errno=%d)",
                 errno);
    }
    return status;
}

static ucc_status_t create_master_key(ucc_tl_mlx5_node_t * node,
                                      struct mlx5dv_mkey **mkey_ptr,
                                      int                  num_of_entries,
                                      ucc_tl_mlx5_context_t *  ctx)
{
    struct mlx5dv_mkey *         mkey;
    struct mlx5dv_mkey_init_attr umr_mkey_init_attr;
    memset(&umr_mkey_init_attr, 0, sizeof(umr_mkey_init_attr));
    umr_mkey_init_attr.pd           = ctx->shared_pd;
    umr_mkey_init_attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
    // Defines how many entries the Mkey will support.
    // In case the MKey is used as "strided-KLM based MKey", the number
    // of entries that is needed is increased by one because one entry is
    // consumed by the "strided header" (see mlx5dv_wr_post manual).
    umr_mkey_init_attr.max_entries = num_of_entries;
    mkey                           = mlx5dv_create_mkey(&umr_mkey_init_attr);
    if (mkey == NULL) {
        tl_error(ctx->super.super.lib, "MasterMKey creation failed (errno=%d)", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    tl_debug(ctx->super.super.lib, "umr_master_key_dv_mkey: lkey=0x%x, with %d entries",
             mkey->lkey, num_of_entries);
    *mkey_ptr = mkey;
    return UCC_OK;
}

static ucc_status_t poll_umr_cq(struct ibv_cq *cq,
                                ucc_tl_mlx5_lib_t *lib)
{
    struct ibv_wc wc;
    int           ret = 0;
    while (!ret) {
        ret = ibv_poll_cq(cq, 1, &wc);
        if (ret < 0) {
            tl_error(lib, "ibv_poll_cq() failed for UMR execution");
            return UCC_ERR_NO_MESSAGE;
        } else if (ret > 0) {
            if (wc.status != IBV_WC_SUCCESS) {
                tl_error(lib,
                         "umr cq returned incorrect completion: status "
                         "%s",
                         ibv_wc_status_str(wc.status));
                return UCC_ERR_NO_MESSAGE;
            }
        }
    }
    tl_debug(lib, "Successfully executed the UMR WQE");
    return UCC_OK;
}

// Execute the UMR WQE for populating the UMR's MasterMKey
static ucc_status_t populate_non_strided_mkey(ucc_tl_mlx5_team_t *team,
                                              int mem_access_flags,
                                              struct mlx5dv_mkey *mkey,
                                              void *              mkey_entries)
{
    ucc_tl_mlx5_net_t *net = &team->net;
    struct ibv_qp_ex *qp_ex = ibv_qp_to_qp_ex(net->umr_qp);
    struct mlx5dv_qp_ex *mqp = mlx5dv_qp_ex_from_ibv_qp_ex(qp_ex);
    ucc_status_t        status;

    ibv_wr_start(qp_ex);
    qp_ex->wr_id = 1; // First (and only) WR
    mlx5dv_wr_mr_list(mqp, mkey, mem_access_flags,
                      MAX_OUTSTANDING_OPS * team->max_num_of_columns,
                      (struct ibv_sge *)mkey_entries);
    tl_debug(
        UCC_TL_MLX5_TEAM_LIB(team),
        "Execute the UMR WQE for populating the team MasterMKeys lkey 0x%x",
        mkey->lkey);

    if (ibv_wr_complete(qp_ex)) {
        tl_error(UCC_TL_MLX5_TEAM_LIB(team), "UMR WQE failed (errno=%d)",
                 errno);
        return UCC_ERR_NO_MESSAGE;
    }
    status = poll_umr_cq(net->umr_cq, UCC_TL_MLX5_TEAM_LIB(team));
    if (status != UCC_OK) {
        return status;
    }
    return UCC_OK;
}

// Execute the UMR WQE for populating the UMR's MasterMKey
static ucc_status_t populate_strided_mkey(ucc_tl_mlx5_team_t *team,
                                          int                 mem_access_flags,
                                          struct mlx5dv_mkey *mkey,
                                          void *mkey_entries, int repeat_count)
{
    ucc_status_t        status;
    ucc_tl_mlx5_net_t *net = &team->net;
    ucc_tl_mlx5_node_t *node = &team->node;

    tl_debug(UCC_TL_MLX5_TEAM_LIB(team),
             "Execute the UMR WQE for populating the send/recv "
             "MasterMKey lkey 0x%x",
             mkey->lkey);


    ucc_tl_mlx5_post_umr(net->umr_qp, mkey, mem_access_flags, repeat_count,
                         node->sbgp->group_size, (struct mlx5dv_mr_interleaved *)mkey_entries,
                         node->umr_entries_mr->lkey, node->umr_entries_buf);

    status = poll_umr_cq(net->umr_cq, UCC_TL_MLX5_TEAM_LIB(team));
    if (status != UCC_OK) {
        return status;
    }
    return UCC_OK;
}

static ucc_status_t create_and_populate_recv_team_mkey(ucc_tl_mlx5_team_t *team)
{
    ucc_status_t        status;
    ucc_tl_mlx5_node_t *node = &team->node;
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    int                 i, j;
    status = create_master_key(node, &node->team_recv_mkey,
                               MAX_OUTSTANDING_OPS * team->max_num_of_columns, ctx);
    if (status != UCC_OK) {
        return status;
    }
    struct ibv_sge *team_mkey_klm_entries = (struct ibv_sge *)calloc(
        MAX_OUTSTANDING_OPS * team->max_num_of_columns, sizeof(struct ibv_sge));
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        for (j = 0; j < team->max_num_of_columns; j++) {
            team_mkey_klm_entries[(i * team->max_num_of_columns) + j].addr = 0;
            //length could be minimized for all mkeys beside the first, but no need because address space is big enough
            team_mkey_klm_entries[(i * team->max_num_of_columns) + j].length =
                node->sbgp->group_size * team->max_msg_size * UCC_TL_TEAM_SIZE(team);
            //todo check lkey or rkey
            team_mkey_klm_entries[(i * team->max_num_of_columns) + j].lkey =
                node->ops[i].recv_mkeys[j]->rkey;
        }
    }
    status = populate_non_strided_mkey(
        team, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
        node->team_recv_mkey, team_mkey_klm_entries);
    if (status != UCC_OK) {
        tl_error(UCC_TL_MLX5_TEAM_LIB(team), "Failed to populate team mkey");
        if (mlx5dv_destroy_mkey(node->team_recv_mkey)) {
            tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                     "mkey destroy failed(errno=%d)", errno);
        }
        return status;
    }
    ucc_free(team_mkey_klm_entries);
    return UCC_OK;
}

/**
 * Create mkeys for all outstanding AlltoAll ops in each rank. Creats team mkey, and execute the team mkey's
 * population WQE
 * @param node struct of the current process's node
 * @param team_size number of processes in team
 */
ucc_status_t ucc_tl_mlx5_init_mkeys(ucc_tl_mlx5_team_t *team)
{
    ucc_status_t        status;
    ucc_tl_mlx5_node_t *node = &team->node;
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    int                 i, j;
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        node->ops[i].send_mkeys = (struct mlx5dv_mkey **)ucc_malloc(
            sizeof(struct mlx5dv_mkey *) * team->max_num_of_columns);
        if (!node->ops[i].send_mkeys) {
            tl_error(UCC_TL_MLX5_TEAM_LIB(team), "Failed to malloc");
            ucc_tl_mlx5_destroy_mkeys(team, 1);
            return UCC_ERR_NO_MEMORY;
        }
        node->ops[i].recv_mkeys = (struct mlx5dv_mkey **)ucc_malloc(
            sizeof(struct mlx5dv_mkey *) * team->max_num_of_columns);
        if (!node->ops[i].recv_mkeys) {
            tl_error(UCC_TL_MLX5_TEAM_LIB(team), "Failed to malloc");
            ucc_tl_mlx5_destroy_mkeys(team, 1);
            return UCC_ERR_NO_MEMORY;
        }
        for (j = 0; j < team->max_num_of_columns; j++) {
            status = create_master_key(node, &node->ops[i].send_mkeys[j],
                                       node->sbgp->group_size + 1, ctx);
            if (status != UCC_OK) {
                tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                         "create send masterkey[%d,%d] failed", i, j);
                ucc_tl_mlx5_destroy_mkeys(team, 1);
                return status;
            }
            status = create_master_key(node, &node->ops[i].recv_mkeys[j],
                                       node->sbgp->group_size + 1, ctx);
            if (status != UCC_OK) {
                tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                         "create recv masterkey[%d,%d] failed", i, j);
                ucc_tl_mlx5_destroy_mkeys(team, 1);
                return status;
            }
        }
    }
    status = create_and_populate_recv_team_mkey(team);
    if (status != UCC_OK) {
        tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                 "create recv top masterkey failed");
        ucc_tl_mlx5_destroy_mkeys(team, 1);
        return status;
    }
    return UCC_OK;
}

/**
 * Execute UMR WQE to populate mkey of specific AlltoAll operation, after the mkey entries were already updated
 * @param team struct of the current team
 * @param req current AlltoAll operation request
 */
ucc_status_t ucc_tl_mlx5_populate_send_recv_mkeys(ucc_tl_mlx5_team_t *    team,
                                                  ucc_tl_mlx5_schedule_t *req)
{
    int                 send_mem_access_flags = 0;
    ucc_tl_mlx5_node_t *node                  = &team->node;
    int                 recv_mem_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    int          i;
    ucc_status_t status;
    int          repeat_count = req->num_of_blocks_columns
                                    ? team->net.sbgp->group_size
                                    : UCC_TL_TEAM_SIZE(team) / req->block_size;
    int n_mkeys = req->num_of_blocks_columns ? req->num_of_blocks_columns : 1;
    if (ucc_tl_mlx5_get_my_ctrl(team, req->seq_index)->mkey_cache_flag &
        UCC_MLX5_NEED_SEND_MKEY_UPDATE) {
        for (i = 0; i < n_mkeys; i++) {
            status = populate_strided_mkey(
                team, send_mem_access_flags,
                node->ops[req->seq_index].send_mkeys[i],
                SEND_UMR_DATA(req, team, i), repeat_count);
            if (status != UCC_OK) {
                tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                         "Failed to populate send umr[%d,%d]", req->seq_index,
                         i);
                return status;
            }
        }
    }
    if (ucc_tl_mlx5_get_my_ctrl(team, req->seq_index)->mkey_cache_flag &
        UCC_MLX5_NEED_RECV_MKEY_UPDATE) {
        for (i = 0; i < n_mkeys; i++) {
            status = populate_strided_mkey(
                team, recv_mem_access_flags,
                node->ops[req->seq_index].recv_mkeys[i],
                RECV_UMR_DATA(req, team, i), repeat_count);
            if (status != UCC_OK) {
                tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                         "Failed to populate recv umr[%d,%d]", req->seq_index,
                         i);
                return status;
            }
        }
    }
    return UCC_OK;
}

static void update_mkey_entry(ucc_tl_mlx5_node_t *    node,
                              ucc_tl_mlx5_schedule_t *req, int direction_send,
                              ucc_tl_mlx5_lib_t *lib)
{
    struct mlx5dv_mr_interleaved *mkey_entry;
    ucc_tl_mlx5_team_t *    team =
        ucc_derived_of(req->super.super.team, ucc_tl_mlx5_team_t);

    struct ibv_mr *buff = direction_send ? req->send_rcache_region_p->mr
                                         : req->recv_rcache_region_p->mr;
    int            i;
    if (!req->num_of_blocks_columns) {
        mkey_entry =
            (struct mlx5dv_mr_interleaved
                 *)(direction_send
                    ? MY_SEND_UMR_DATA(req, team, 0)
                    : MY_RECV_UMR_DATA(req, team, 0));
        mkey_entry->addr        = (uintptr_t)buff->addr;
        mkey_entry->bytes_count = req->block_size * req->msg_size;
        mkey_entry->bytes_skip  = 0;
        mkey_entry->lkey        = direction_send ? buff->lkey : buff->rkey;
//        tl_debug(lib,
//                 "%s MasterMKey Strided KLM entries[%d]: addr = %"PRIx64", "
//                 "bytes_count = %d, bytes_skip = %d,lkey=%"PRIx64"",
//                 direction_send ? "send" : "recv", node->sbgp->group_rank,
//                 mkey_entry->addr, mkey_entry->bytes_count,
//                 mkey_entry->bytes_skip, mkey_entry->lkey);
    } else {
        for (i = 0; i < req->num_of_blocks_columns; i++) {
            mkey_entry =
                (struct mlx5dv_mr_interleaved
                     *)(direction_send
                    ? MY_SEND_UMR_DATA(req, team, i)
                    : MY_RECV_UMR_DATA(req, team, i));
            mkey_entry->addr =
                (uintptr_t)buff->addr + i * (req->block_size * req->msg_size);
            mkey_entry->bytes_count =
                (i == (req->num_of_blocks_columns - 1))
                    ? ((node->sbgp->group_size % req->block_size) *
                       req->msg_size)
                    : (req->block_size * req->msg_size);
            mkey_entry->bytes_skip =
                (i == (req->num_of_blocks_columns - 1))
                    ? ((node->sbgp->group_size -
                        (node->sbgp->group_size % req->block_size)) *
                       req->msg_size)
                    : ((node->sbgp->group_size - req->block_size) *
                       req->msg_size);
            mkey_entry->lkey = direction_send ? buff->lkey : buff->rkey;
//            tl_debug(lib,
//                     "%s MasterMKey Strided KLM entries[%d,%d]: addr = %"PRIx64", "
//                     "bytes_count = %d, bytes_skip = %d,lkey=%"PRIx64"",
//                     direction_send ? "send" : "recv", node->sbgp->group_rank,
//                     i, mkey_entry->addr, mkey_entry->bytes_count,
//                     mkey_entry->bytes_skip, mkey_entry->lkey);
        }
    }
}

/**
 * Update the UMR klm entry (ranks send & receive buffers) for specific AlltoAll operation
 * @param node struct of the current process's node
 * @param req AlltoAll operation request object
 */
ucc_status_t ucc_tl_mlx5_update_mkeys_entries(ucc_tl_mlx5_node_t *    node,
                                              ucc_tl_mlx5_schedule_t *req,
                                              ucc_tl_mlx5_lib_t *     lib)
{
    update_mkey_entry(node, req, 1, lib);
    update_mkey_entry(node, req, 0, lib);
    return UCC_OK;
}

/**
 * Clean UMR qp & cq
 */
ucc_status_t ucc_tl_mlx5_destroy_umr(ucc_tl_mlx5_net_t *net,
		ucc_base_lib_t * lib)
{
    if (ibv_destroy_qp(net->umr_qp)) {
        tl_error(lib, "umr qp destroy failed (errno=%d)", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    if (ibv_destroy_cq(net->umr_cq)) {
        tl_error(lib, "umr cq destroy failed (errno=%d)", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

/**
 * Clean all mkeys -  operation mkeys, team mkeys
 * @param node struct of the current process's node
 * @param error_mode boolean - ordinary destroy or destroy due to an earlier error
 */
ucc_status_t ucc_tl_mlx5_destroy_mkeys(ucc_tl_mlx5_team_t *team, int error_mode)
{
    int                 i, j;
    ucc_tl_mlx5_node_t *node   = &team->node;
    ucc_status_t        status = UCC_OK;
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        for (j = 0; j < team->max_num_of_columns; j++) {
            if (mlx5dv_destroy_mkey(node->ops[i].send_mkeys[j])) {
                if (!error_mode) {
                    tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                             "mkey destroy failed(errno=%d)", errno);
                    status = UCC_ERR_NO_MESSAGE;
                }
            }
            if (mlx5dv_destroy_mkey(node->ops[i].recv_mkeys[j])) {
                if (!error_mode) {
                    tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                             "mkey destroy failed(errno=%d)", errno);
                    status = UCC_ERR_NO_MESSAGE;
                }
            }
        }
        ucc_free(node->ops[i].send_mkeys);
        ucc_free(node->ops[i].recv_mkeys);
    }
    if (mlx5dv_destroy_mkey(node->team_recv_mkey)) {
        if (!error_mode) {
            tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                     "mkey destroy failed(errno=%d)", errno);
            status = UCC_ERR_NO_MESSAGE;
        }
    }
    return status;
}
