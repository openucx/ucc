/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"
#include "tl_mlx5_ib.h"
#include "tl_mlx5_coll.h"
#include "tl_mlx5_wqe.h"

#include "alltoall/alltoall.h"
#include "alltoall/alltoall_mkeys.h"

#include <inttypes.h>

/**
 * Create and connect UMR qp & cq.
 */
ucc_status_t ucc_tl_mlx5_alltoall_init_umr(ucc_tl_mlx5_alltoall_t *a2a,
                                           ucc_base_lib_t         *lib)
{
    ucc_tl_mlx5_lib_config_t cfg = ucc_derived_of(lib, ucc_tl_mlx5_lib_t)->cfg;
    ucc_status_t status = UCC_OK;

    a2a->net.umr_cq = ibv_create_cq(a2a->ctx, UMR_CQ_SIZE, NULL, NULL, 0);
    if (a2a->net.umr_cq == NULL) {
        tl_error(lib, "failed to create UMR (errno %d)", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    status = ucc_tl_mlx5_create_umr_qp(a2a->ctx, a2a->pd, a2a->net.umr_cq,
                                       a2a->ib_port, &a2a->net.umr_qp,
                                       &cfg.qp_conf, lib);
    if (status != UCC_OK) {
        goto err;
    }
    return status;

err:
    if (ibv_destroy_cq(a2a->net.umr_cq)) {
        tl_error(lib, "failed to destroy UMR CQ (errno=%d)", errno);
    }
    return status;
}

static ucc_status_t create_master_key(int num_of_entries, struct ibv_pd *pd,
                                      struct mlx5dv_mkey **mkey_ptr,
                                      ucc_base_lib_t *     lib)
{
    struct mlx5dv_mkey *         mkey;
    struct mlx5dv_mkey_init_attr umr_mkey_init_attr;

    memset(&umr_mkey_init_attr, 0, sizeof(umr_mkey_init_attr));
    umr_mkey_init_attr.pd           = pd;
    umr_mkey_init_attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
    // Defines how many entries the Mkey will support.
    // In case the MKey is used as "strided-KLM based MKey", the number
    // of entries that is needed is increased by one because one entry is
    // consumed by the "strided header" (see mlx5dv_wr_post manual).
    umr_mkey_init_attr.max_entries = num_of_entries;
    mkey                           = mlx5dv_create_mkey(&umr_mkey_init_attr);
    if (mkey == NULL) {
        tl_error(lib, "MasterMKey creation failed (errno=%d)", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    tl_trace(lib, "umr_master_key_dv_mkey: lkey=0x%x, with %d entries",
             mkey->lkey, num_of_entries);
    *mkey_ptr = mkey;
    return UCC_OK;
}

static ucc_status_t poll_umr_cq(struct ibv_cq *cq, ucc_base_lib_t *lib)
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
    return UCC_OK;
}

// Execute the UMR WQE for populating the UMR's MasterMKey
static ucc_status_t populate_non_strided_mkey(struct ibv_qp *umr_qp,
                                              struct ibv_cq *umr_cq,
                                              int            mem_access_flags,
                                              struct mlx5dv_mkey *mkey,
                                              void *mkey_entries, int n_entries,
                                              ucc_base_lib_t *lib)
{
    struct ibv_qp_ex *   qp_ex = ibv_qp_to_qp_ex(umr_qp);
    struct mlx5dv_qp_ex *mqp   = mlx5dv_qp_ex_from_ibv_qp_ex(qp_ex);
    ucc_status_t         status;

    ibv_wr_start(qp_ex);
    qp_ex->wr_id = 1; // First (and only) WR
    mlx5dv_wr_mr_list(mqp, mkey, mem_access_flags, n_entries,
                      (struct ibv_sge *)mkey_entries);

    if (ibv_wr_complete(qp_ex)) {
        tl_error(lib, "UMR WQE failed (errno=%d)", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    status = poll_umr_cq(umr_cq, lib);
    if (status != UCC_OK) {
        return status;
    }
    return UCC_OK;
}

// Execute the UMR WQE for populating the UMR's MasterMKey
static ucc_status_t populate_strided_mkey(ucc_tl_mlx5_alltoall_t *a2a,
                                          int mem_access_flags,
                                          struct mlx5dv_mkey *mkey,
                                          void *mkey_entries, int repeat_count,
                                          ucc_base_lib_t *lib)
{
    ucc_status_t            status;
    ucc_tl_mlx5_alltoall_net_t * net  = &a2a->net;
    ucc_tl_mlx5_alltoall_node_t *node = &a2a->node;

    ucc_tl_mlx5_post_umr(net->umr_qp, mkey, mem_access_flags, repeat_count,
                         node->sbgp->group_size,
                         (struct mlx5dv_mr_interleaved *)mkey_entries,
                         node->umr_entries_mr->lkey, node->umr_entries_buf);

    status = poll_umr_cq(net->umr_cq, lib);
    if (status != UCC_OK) {
        tl_error(lib, "failed to populate strided UMR mkey (errno=%d)", errno);
        return status;
    }
    return UCC_OK;
}

static ucc_status_t create_and_populate_recv_team_mkey(ucc_tl_mlx5_team_t *team,
                                                       ucc_base_lib_t     *lib)
{
    int                          team_size = UCC_TL_TEAM_SIZE(team);
    ucc_tl_mlx5_alltoall_t      *a2a       = team->a2a;
    ucc_tl_mlx5_alltoall_node_t *node      = &a2a->node;
    int                          mnc       = a2a->max_num_of_columns;
    ucc_status_t                 status    = UCC_OK;
    int                          i, j, index;

    status = create_master_key(MAX_OUTSTANDING_OPS * mnc, a2a->pd,
                               &node->team_recv_mkey, lib);
    if (status != UCC_OK) {
        return status;
    }
    struct ibv_sge *team_mkey_klm_entries = (struct ibv_sge *)calloc(
        MAX_OUTSTANDING_OPS * mnc, sizeof(struct ibv_sge));

    if (!team_mkey_klm_entries) {
        tl_error(lib, "failed to allocate team_mkey_klm_entries");
        status = UCC_ERR_NO_MEMORY;
        goto err_calloc;
    }

    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        for (j = 0; j < mnc; j++) {
            index = i * mnc + j;
            team_mkey_klm_entries[index].addr = 0;
            //length could be minimized for all mkeys beside the first, but no need because address space is big enough
            team_mkey_klm_entries[index].length =
                node->sbgp->group_size * a2a->max_msg_size * team_size;
            team_mkey_klm_entries[index].lkey =
                node->ops[i].recv_mkeys[j]->rkey;
        }
    }
    status = populate_non_strided_mkey(
        a2a->net.umr_qp, a2a->net.umr_cq,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE, node->team_recv_mkey,
        team_mkey_klm_entries, MAX_OUTSTANDING_OPS * mnc, lib);
    if (status != UCC_OK) {
        tl_error(a2a, "failed to populate team mkey");
        goto err_mkey;
    }
    ucc_free(team_mkey_klm_entries);
    return status;

err_mkey:
    ucc_free(team_mkey_klm_entries);
err_calloc:
    if (mlx5dv_destroy_mkey(node->team_recv_mkey)) {
        tl_error(lib, "mkey destroy failed(errno=%d)", errno);
    }
    return status;
}

/**
 * Create mkeys for all outstanding AlltoAll ops in each rank. Creats team mkey, and execute the team mkey's
 * population WQE
 */
ucc_status_t ucc_tl_mlx5_init_mkeys(ucc_tl_mlx5_team_t *team,
                                    ucc_base_lib_t     *lib)
{
    ucc_tl_mlx5_alltoall_t      *a2a  = team->a2a;
    ucc_tl_mlx5_alltoall_node_t *node = &a2a->node;
    int                          i, j, k, l;
    ucc_status_t                 status;

    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        node->ops[i].send_mkeys = (struct mlx5dv_mkey **)ucc_malloc(
            sizeof(struct mlx5dv_mkey *) * a2a->max_num_of_columns);
        if (!node->ops[i].send_mkeys) {
            tl_error(lib, "failed to malloc");
            goto err_malloc;
        }
        node->ops[i].recv_mkeys = (struct mlx5dv_mkey **)ucc_malloc(
            sizeof(struct mlx5dv_mkey *) * a2a->max_num_of_columns);
        if (!node->ops[i].recv_mkeys) {
            tl_error(lib, "failed to malloc");
            ucc_free(node->ops[i].send_mkeys);
            goto err_malloc;
        }
        for (j = 0; j < a2a->max_num_of_columns; j++) {
            status = create_master_key(node->sbgp->group_size + 1, a2a->pd,
                                       &node->ops[i].send_mkeys[j], lib);
            if (status != UCC_OK) {
                tl_error(lib, "failed to create send masterkey [%d,%d]", i, j);
                goto err_create_mkey;
            }
            status = create_master_key(node->sbgp->group_size + 1, a2a->pd,
                                       &node->ops[i].recv_mkeys[j], lib);
            if (status != UCC_OK) {
                tl_error(lib, "failed to create recv masterkey [%d,%d]", i, j);
                if (!mlx5dv_destroy_mkey(node->ops[i].send_mkeys[j])) {
                    tl_error(lib, "mkey destroy failed(errno=%d)", errno);
                }
                goto err_create_mkey;
            }
        }
    }

    status = create_and_populate_recv_team_mkey(team, lib);
    if (status != UCC_OK) {
        tl_error(lib, "failed to create recv top masterkey");
        goto err_malloc;
    }

    return UCC_OK;

err_create_mkey:
    for (l = 0; l < j; l++) {
        if (!mlx5dv_destroy_mkey(node->ops[i].recv_mkeys[l])) {
            tl_error(lib, "mkey destroy failed(errno=%d)", errno);
        }
    }
    ucc_free(node->ops[i].recv_mkeys);
    ucc_free(node->ops[i].send_mkeys);
err_malloc:
    for (k = 0; k < i; k++) {
        for (l = 0; l < a2a->max_num_of_columns; l++) {
            if (!mlx5dv_destroy_mkey(node->ops[k].send_mkeys[l])) {
                tl_error(lib, "mkey destroy failed(errno=%d)", errno);
            }
            if (!mlx5dv_destroy_mkey(node->ops[k].recv_mkeys[l])) {
                tl_error(lib, "mkey destroy failed(errno=%d)", errno);
            }
        }
        ucc_free(node->ops[k].send_mkeys);
        ucc_free(node->ops[k].recv_mkeys);
    }
    return UCC_ERR_NO_MESSAGE;
}

/**
 * Execute UMR WQE to populate mkey of specific AlltoAll operation, after the mkey entries were already updated
 * @param team struct of the current team
 * @param req current AlltoAll operation request
 */
ucc_status_t ucc_tl_mlx5_populate_send_recv_mkeys(ucc_tl_mlx5_team_t *    team,
                                                  ucc_tl_mlx5_schedule_t *req)
{
    int                     send_mem_access_flags = 0;
    ucc_tl_mlx5_alltoall_t *     a2a                   = team->a2a;
    ucc_tl_mlx5_alltoall_node_t *node                  = &a2a->node;
    int                     recv_mem_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    int          nbc          = req->alltoall.num_of_blocks_columns;
    int          seq_index    = req->alltoall.seq_index;
    int          n_mkeys      = nbc ? nbc : 1;
    int          repeat_count;
    int          i;
    ucc_status_t status;

    if (ucc_tl_mlx5_get_my_ctrl(a2a, seq_index)->mkey_cache_flag &
        UCC_MLX5_NEED_SEND_MKEY_UPDATE) {
        repeat_count = nbc ? a2a->net.sbgp->group_size
                           : UCC_TL_TEAM_SIZE(team) / req->alltoall.block_width;
        for (i = 0; i < n_mkeys; i++) {
            status = populate_strided_mkey(a2a, send_mem_access_flags,
                                           node->ops[seq_index].send_mkeys[i],
                                           SEND_UMR_DATA(req, a2a, i),
                                           repeat_count, UCC_TL_TEAM_LIB(team));
            if (status != UCC_OK) {
                tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                         "Failed to populate send umr[%d,%d]", seq_index, i);
                return status;
            }
        }
    }
    if (ucc_tl_mlx5_get_my_ctrl(a2a, seq_index)->mkey_cache_flag &
        UCC_MLX5_NEED_RECV_MKEY_UPDATE) {
        repeat_count =
            nbc ? a2a->net.sbgp->group_size
                : UCC_TL_TEAM_SIZE(team) / req->alltoall.block_height;
        for (i = 0; i < n_mkeys; i++) {
            status = populate_strided_mkey(a2a, recv_mem_access_flags,
                                           node->ops[seq_index].recv_mkeys[i],
                                           RECV_UMR_DATA(req, a2a, i),
                                           repeat_count, UCC_TL_TEAM_LIB(team));
            if (status != UCC_OK) {
                tl_error(UCC_TL_MLX5_TEAM_LIB(team),
                         "Failed to populate recv umr[%d,%d]", seq_index, i);
                return status;
            }
        }
    }
    return UCC_OK;
}

static void update_mkey_entry(ucc_tl_mlx5_alltoall_t *a2a,
                              ucc_tl_mlx5_schedule_t *req, int direction_send)
{
    ucc_tl_mlx5_alltoall_node_t  *node         = &a2a->node;
    int                           block_height = req->alltoall.block_height;
    int                           block_width  = req->alltoall.block_width;
    size_t                        msg_size     = req->alltoall.msg_size;
    int                           nbc  = req->alltoall.num_of_blocks_columns;
    struct ibv_mr                *buff = direction_send
                                             ? req->alltoall.send_rcache_region_p->reg.mr
                                             : req->alltoall.recv_rcache_region_p->reg.mr;
    struct mlx5dv_mr_interleaved *mkey_entry;
    int                           i;

    if (!nbc) {
        mkey_entry = (umr_t *)(direction_send ? MY_SEND_UMR_DATA(req, a2a, 0)
                                              : MY_RECV_UMR_DATA(req, a2a, 0));
        mkey_entry->addr        = (uintptr_t)buff->addr;
        mkey_entry->bytes_count =
            (direction_send ? block_width : block_height) * msg_size;
        mkey_entry->bytes_skip  = 0;
        mkey_entry->lkey        = direction_send ? buff->lkey : buff->rkey;
    } else {
        for (i = 0; i < nbc; i++) {
            ucc_assert(block_height == block_width);
            mkey_entry =
                (umr_t *)(direction_send ? MY_SEND_UMR_DATA(req, a2a, i)
                                         : MY_RECV_UMR_DATA(req, a2a, i));
            mkey_entry->addr =
                (uintptr_t)buff->addr + i * (block_height * msg_size);
            mkey_entry->bytes_count =
                (i == (nbc - 1))
                    ? ((node->sbgp->group_size % block_height) * msg_size)
                    : (block_height * msg_size);
            mkey_entry->bytes_skip =
                (i == (nbc - 1))
                    ? ((node->sbgp->group_size -
                        (node->sbgp->group_size % block_height)) *
                       msg_size)
                    : ((node->sbgp->group_size - block_height) * msg_size);
            mkey_entry->lkey = direction_send ? buff->lkey : buff->rkey;
        }
    }
}

/**
 * Update the UMR klm entry (ranks send & receive buffers) for specific AlltoAll operation
 * @param node struct of the current process's node
 * @param req AlltoAll operation request object
 */
ucc_status_t ucc_tl_mlx5_update_mkeys_entries(ucc_tl_mlx5_alltoall_t *a2a,
                                              ucc_tl_mlx5_schedule_t *req,
                                              int                     flag)
{
    if (flag & UCC_MLX5_NEED_SEND_MKEY_UPDATE) {
        update_mkey_entry(a2a, req, 1);
    }
    if (flag & UCC_MLX5_NEED_RECV_MKEY_UPDATE) {
        update_mkey_entry(a2a, req, 0);
    }
    return UCC_OK;
}

/**
 * Clean UMR qp & cq
 */
ucc_status_t ucc_tl_mlx5_destroy_umr(ucc_tl_mlx5_alltoall_t *a2a,
                                     ucc_base_lib_t         *lib)
{
    if (ibv_destroy_qp(a2a->net.umr_qp)) {
        tl_error(lib, "umr qp destroy failed (errno=%d)", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    if (ibv_destroy_cq(a2a->net.umr_cq)) {
        tl_error(lib, "umr cq destroy failed (errno=%d)", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

/**
 * Clean all mkeys -  operation mkeys, team mkeys
 */
ucc_status_t ucc_tl_mlx5_destroy_mkeys(ucc_tl_mlx5_alltoall_t *a2a,
                                       int error_mode, ucc_base_lib_t *lib)
{
    int                     i, j;
    ucc_tl_mlx5_alltoall_node_t *node   = &a2a->node;
    ucc_status_t            status = UCC_OK;

    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        for (j = 0; j < a2a->max_num_of_columns; j++) {
            if (mlx5dv_destroy_mkey(node->ops[i].send_mkeys[j])) {
                if (!error_mode) {
                    tl_error(lib, "mkey destroy failed(errno=%d)", errno);
                    status = UCC_ERR_NO_MESSAGE;
                }
            }
            if (mlx5dv_destroy_mkey(node->ops[i].recv_mkeys[j])) {
                if (!error_mode) {
                    tl_error(lib, "mkey destroy failed(errno=%d)", errno);
                    status = UCC_ERR_NO_MESSAGE;
                }
            }
        }
        ucc_free(node->ops[i].send_mkeys);
        ucc_free(node->ops[i].recv_mkeys);
    }
    if (mlx5dv_destroy_mkey(node->team_recv_mkey)) {
        if (!error_mode) {
            tl_error(lib, "mkey destroy failed(errno=%d)", errno);
            status = UCC_ERR_NO_MESSAGE;
        }
    }
    return status;
}
