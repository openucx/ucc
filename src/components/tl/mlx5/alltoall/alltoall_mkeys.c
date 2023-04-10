/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
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
 * @param ctx mlx5 team context
 * @param node struct of the current process's node
 */
ucc_status_t ucc_tl_mlx5_a2a_init_umr(ucc_tl_mlx5_a2a_t *a2a,
                                      ucc_base_lib_t *   lib)
{
    ucc_tl_mlx5_lib_config_t cfg = ucc_derived_of(lib, ucc_tl_mlx5_lib_t)->cfg;
    ucc_status_t status = UCC_OK;

    a2a->net.umr_cq = ibv_create_cq(a2a->ctx, UMR_CQ_SIZE, NULL, NULL, 0);
    if (a2a->net.umr_cq == NULL) {
        tl_error(lib, "failed to create UMR (errno %d)", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    status = ucc_tl_mlx5_create_umr_qp(a2a->ctx, a2a->pd, a2a->net.umr_cq,
                                       a2a->ib_port, &a2a->net.umr_qp, &cfg.qp_conf, lib);
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
static ucc_status_t populate_strided_mkey(ucc_tl_mlx5_a2a_t * a2a,
                                          int                 mem_access_flags,
                                          struct mlx5dv_mkey *mkey,
                                          void *mkey_entries, int repeat_count,
                                          ucc_base_lib_t *lib)
{
    ucc_status_t            status;
    ucc_tl_mlx5_a2a_net_t * net  = &a2a->net;
    ucc_tl_mlx5_a2a_node_t *node = &a2a->node;

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

static ucc_status_t create_and_populate_recv_team_mkey(ucc_tl_mlx5_a2a_t *a2a,
                                                       ucc_base_lib_t *   lib)
{
    ucc_status_t            status;
    ucc_tl_mlx5_a2a_node_t *node = &a2a->node;
    int                     mnc  = a2a->max_num_of_columns;

    int i, j;

    status = create_master_key(MAX_OUTSTANDING_OPS * mnc, a2a->pd,
                               &node->team_recv_mkey, lib);
    if (status != UCC_OK) {
        return status;
    }
    struct ibv_sge *team_mkey_klm_entries = (struct ibv_sge *)calloc(
        MAX_OUTSTANDING_OPS * mnc, sizeof(struct ibv_sge));

    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        for (j = 0; j < mnc; j++) {
            team_mkey_klm_entries[(i * mnc) + j].addr = 0;
            //length could be minimized for all mkeys beside the first, but no need because address space is big enough
            team_mkey_klm_entries[(i * mnc) + j].length =
                node->sbgp->group_size * a2a->max_msg_size * a2a->team_size;
            //todo check lkey or rkey
            team_mkey_klm_entries[(i * mnc) + j].lkey =
                node->ops[i].recv_mkeys[j]->rkey;
        }
    }
    status = populate_non_strided_mkey(
        a2a->net.umr_qp, a2a->net.umr_cq,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE, node->team_recv_mkey,
        team_mkey_klm_entries, MAX_OUTSTANDING_OPS * mnc, lib);
    if (status != UCC_OK) {
        tl_error(a2a, "failed to populate team mkey");
        if (mlx5dv_destroy_mkey(node->team_recv_mkey)) {
            tl_error(lib, "mkey destroy failed(errno=%d)", errno);
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
ucc_status_t ucc_tl_mlx5_init_mkeys(ucc_tl_mlx5_a2a_t *a2a, ucc_base_lib_t *lib)
{
    ucc_status_t            status;
    ucc_tl_mlx5_a2a_node_t *node = &a2a->node;
    int                     i, j;

    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        node->ops[i].send_mkeys = (struct mlx5dv_mkey **)ucc_malloc(
            sizeof(struct mlx5dv_mkey *) * a2a->max_num_of_columns);
        if (!node->ops[i].send_mkeys) {
            tl_error(lib, "failed to malloc");
            ucc_tl_mlx5_destroy_mkeys(a2a, 1, lib);
            return UCC_ERR_NO_MEMORY;
        }
        node->ops[i].recv_mkeys = (struct mlx5dv_mkey **)ucc_malloc(
            sizeof(struct mlx5dv_mkey *) * a2a->max_num_of_columns);
        if (!node->ops[i].recv_mkeys) {
            tl_error(lib, "failed to malloc");
            ucc_tl_mlx5_destroy_mkeys(a2a, 1, lib);
            return UCC_ERR_NO_MEMORY;
        }
        for (j = 0; j < a2a->max_num_of_columns; j++) {
            status = create_master_key(node->sbgp->group_size + 1, a2a->pd,
                                       &node->ops[i].send_mkeys[j], lib);
            if (status != UCC_OK) {
                tl_error(lib, " failed to create send masterkey [%d,%d]", i, j);
                ucc_tl_mlx5_destroy_mkeys(a2a, 1, lib);
                return status;
            }
            status = create_master_key(node->sbgp->group_size + 1, a2a->pd,
                                       &node->ops[i].recv_mkeys[j], lib);
            if (status != UCC_OK) {
                tl_error(lib, "failed to create recv masterkey [%d,%d]", i, j);
                ucc_tl_mlx5_destroy_mkeys(a2a, 1, lib);
                return status;
            }
        }
    }
    status = create_and_populate_recv_team_mkey(a2a, lib);
    if (status != UCC_OK) {
        tl_error(lib, "failed to create recv top masterkey");
        ucc_tl_mlx5_destroy_mkeys(a2a, 1, lib);
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
    int                     send_mem_access_flags = 0;
    ucc_tl_mlx5_a2a_t *     a2a                   = team->a2a;
    ucc_tl_mlx5_a2a_node_t *node                  = &a2a->node;
    int                     recv_mem_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    int          nbc          = req->alltoall.num_of_blocks_columns;
    int          seq_index    = req->alltoall.seq_index;
    int          repeat_count = nbc ? a2a->net.sbgp->group_size
                                    : UCC_TL_TEAM_SIZE(team) / req->alltoall.block_size;
    int          n_mkeys      = nbc ? nbc : 1;
    int          i;
    ucc_status_t status;

    if (ucc_tl_mlx5_get_my_ctrl(a2a, seq_index)->mkey_cache_flag &
        UCC_MLX5_NEED_SEND_MKEY_UPDATE) {
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

static void update_mkey_entry(ucc_tl_mlx5_a2a_t *     a2a,
                              ucc_tl_mlx5_schedule_t *req, int direction_send)
{
    ucc_tl_mlx5_a2a_node_t *      node       = &a2a->node;
    int                           block_size = req->alltoall.block_size;
    size_t                        msg_size   = req->alltoall.msg_size;
    int                           nbc  = req->alltoall.num_of_blocks_columns;
    struct ibv_mr *               buff = direction_send
                                             ? req->alltoall.send_rcache_region_p->mr
                                             : req->alltoall.recv_rcache_region_p->mr;
    struct mlx5dv_mr_interleaved *mkey_entry;
    int                           i;

    if (!nbc) {
        mkey_entry = (umr_t *)(direction_send ? MY_SEND_UMR_DATA(req, a2a, 0)
                                              : MY_RECV_UMR_DATA(req, a2a, 0));
        mkey_entry->addr        = (uintptr_t)buff->addr;
        mkey_entry->bytes_count = block_size * msg_size;
        mkey_entry->bytes_skip  = 0;
        mkey_entry->lkey        = direction_send ? buff->lkey : buff->rkey;
        //        tl_debug(lib,
        //                 "%s MasterMKey Strided KLM entries[%d]: addr = %"PRIx64", "
        //                 "bytes_count = %d, bytes_skip = %d,lkey=%"PRIx64"",
        //                 direction_send ? "send" : "recv", node->sbgp->group_rank,
        //                 mkey_entry->addr, mkey_entry->bytes_count,
        //                 mkey_entry->bytes_skip, mkey_entry->lkey);
    } else {
        for (i = 0; i < nbc; i++) {
            mkey_entry =
                (umr_t *)(direction_send ? MY_SEND_UMR_DATA(req, a2a, i)
                                         : MY_RECV_UMR_DATA(req, a2a, i));
            mkey_entry->addr =
                (uintptr_t)buff->addr + i * (block_size * msg_size);
            mkey_entry->bytes_count =
                (i == (nbc - 1))
                    ? ((node->sbgp->group_size % block_size) * msg_size)
                    : (block_size * msg_size);
            mkey_entry->bytes_skip =
                (i == (nbc - 1))
                    ? ((node->sbgp->group_size -
                        (node->sbgp->group_size % block_size)) *
                       msg_size)
                    : ((node->sbgp->group_size - block_size) * msg_size);
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
ucc_status_t ucc_tl_mlx5_update_mkeys_entries(ucc_tl_mlx5_a2a_t *     a2a,
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
ucc_status_t ucc_tl_mlx5_destroy_umr(ucc_tl_mlx5_a2a_t *a2a,
                                     ucc_base_lib_t *   lib)
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
 * @param node struct of the current process's node
 * @param error_mode boolean - ordinary destroy or destroy due to an earlier error
 */
ucc_status_t ucc_tl_mlx5_destroy_mkeys(ucc_tl_mlx5_a2a_t *a2a, int error_mode,
                                       ucc_base_lib_t *lib)
{
    int                     i, j;
    ucc_tl_mlx5_a2a_node_t *node   = &a2a->node;
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
