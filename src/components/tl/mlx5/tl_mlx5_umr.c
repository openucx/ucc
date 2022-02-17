/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_ib.h"
#include "utils/arch/cpu.h"

#define SQ_WQE_SHIFT 6
#define DS_SIZE      16 // size of a single Data Segment in the WQE

/* UMR pointer to KLMs/MTTs/RepeatBlock and BSFs location (when inline = 0) */
struct mlx5_wqe_umr_pointer_seg {
    __be32 reserved;
    __be32 mkey;
    __be64 address;
};

static inline uint8_t get_umr_mr_flags(uint32_t acc)
{
    return ((acc & IBV_ACCESS_REMOTE_ATOMIC
                 ? MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_ATOMIC
                 : 0) |
            (acc & IBV_ACCESS_REMOTE_WRITE
                 ? MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_REMOTE_WRITE
                 : 0) |
            (acc & IBV_ACCESS_REMOTE_READ
                 ? MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_REMOTE_READ
                 : 0) |
            (acc & IBV_ACCESS_LOCAL_WRITE
                 ? MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_LOCAL_WRITE
                 : 0));
}

#define MLX5_OPCODE_LOCAL_MMO 0x32

typedef struct transpose_seg {
    __be32 element_size; /* 8 bit value */
    __be16 num_rows; /* 7 bit value */
    __be16 num_cols; /* 7 bit value */
    __be64 padding;
} transpose_seg_t;

/* External API to expose the non-inline UMR registration */
ucc_status_t ucc_tl_mlx5_post_transpose(struct ibv_qp *qp, uint32_t src_mr_lkey, uint32_t dst_mr_key,
                                        uintptr_t src_mkey_addr, uintptr_t dst_addr,
                                        uint32_t element_size, uint16_t ncols, uint16_t nrows)
{

    uint32_t                  opcode = MLX5_OPCODE_LOCAL_MMO;
    uint32_t                  opmode = 0x0; //TRanspose
    uint32_t                  n_ds   = 4;
    char                      wqe_desc[n_ds * DS_SIZE];
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_data_seg *data;
    transpose_seg_t          *tseg;
    struct ibv_qp_ex *qp_ex = ibv_qp_to_qp_ex(qp);
    struct mlx5dv_qp_ex *mqp = mlx5dv_qp_ex_from_ibv_qp_ex(qp_ex);

    memset(wqe_desc, 0, n_ds * DS_SIZE);
    /* SET CTRL SEG */

    ctrl = (void*)wqe_desc;
    uint8_t fm_ce_se =  MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE;

    mlx5dv_set_ctrl_seg(ctrl, /* pi */ 0x0, opcode, opmode,
                        qp->qp_num, fm_ce_se, n_ds, 0x0, 0x0);

    /* SET TRANSPOSE SEG */
    tseg  = PTR_OFFSET(ctrl, DS_SIZE);
    tseg->element_size = htobe32(element_size);
    tseg->num_rows = htobe16(nrows);
    tseg->num_cols = htobe16(ncols);

    /* SET SRC DATA SEG */
    data = PTR_OFFSET(tseg, DS_SIZE);
    mlx5dv_set_data_seg(data, ncols * nrows * element_size, src_mr_lkey, src_mkey_addr);

    /* SET DST DATA SEG */
    data = PTR_OFFSET(data, DS_SIZE);
    mlx5dv_set_data_seg(data, ncols * nrows * element_size, dst_mr_key, dst_addr);
    mlx5dv_wr_raw_wqe(mqp, wqe_desc);
    return UCC_OK;
}

/* The strided block format is as the following:
 * | repeat_block | entry_block | entry_block |...| entry_block |
 * While the repeat entry contains details on the list of the block_entries.
 */
static void umr_pointer_seg_init(uint32_t repeat_count,
    uint16_t num_interleaved, struct mlx5dv_mr_interleaved *data, struct mlx5_wqe_umr_pointer_seg *     pseg,
    uint32_t ptr_mkey, void *ptr_address, int *xlat_size, uint64_t *reglen)
{
    struct mlx5_wqe_umr_repeat_block_seg *rb;
    struct mlx5_wqe_umr_repeat_ent_seg *  eb;
    uint64_t                              byte_count = 0;
    int                                   i;

    /* set pointer segment */
    pseg->mkey    = htobe32(ptr_mkey);
    pseg->address = htobe64((uint64_t)ptr_address);

    /* set actual repeated and entry blocks segments */
    rb               = ptr_address;
    rb->op           = htobe32(0x400); // PRM header entry - repeated blocks
    rb->reserved     = 0;
    rb->num_ent      = htobe16(num_interleaved);
    rb->repeat_count = htobe32(repeat_count);
    eb               = rb->entries;

    /*
	 * ------------------------------------------------------------
	 * | repeat_block | entry_block | entry_block |...| entry_block
	 * ------------------------------------------------------------
	 */
    for (i = 0; i < num_interleaved; i++, eb++) {
        byte_count += data[i].bytes_count;
        eb->va         = htobe64(data[i].addr);
        eb->byte_count = htobe16(data[i].bytes_count);
        eb->stride     = htobe16(data[i].bytes_count + data[i].bytes_skip);
        eb->memkey     = htobe32(data[i].lkey);
    }

    rb->byte_count = htobe32(byte_count);
    *reglen        = byte_count * repeat_count;
    *xlat_size     = (num_interleaved + 1) * sizeof(*eb);
}

ucc_status_t ucc_tl_mlx5_post_umr(
    struct ibv_qp *qp, struct mlx5dv_mkey *dv_mkey,
    uint32_t access_flags, uint32_t repeat_count, uint16_t num_entries,
    struct mlx5dv_mr_interleaved *data, uint32_t ptr_mkey, void *ptr_address)
{
    uint32_t                  opcode = MLX5_OPCODE_UMR;
    uint32_t                  opmode = 0x0;
    uint32_t                  n_ds   =
        (sizeof(struct mlx5_wqe_ctrl_seg) + sizeof(struct mlx5_wqe_umr_ctrl_seg) +
         sizeof(struct mlx5_wqe_mkey_context_seg) + sizeof(struct mlx5_wqe_umr_pointer_seg)) / DS_SIZE;
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_umr_ctrl_seg *    umr_ctrl_seg;
    struct mlx5_wqe_mkey_context_seg *mk_seg;
    struct mlx5_wqe_umr_pointer_seg  *pseg;
    struct ibv_qp_ex *qp_ex = ibv_qp_to_qp_ex(qp);
    struct mlx5dv_qp_ex *mqp = mlx5dv_qp_ex_from_ibv_qp_ex(qp_ex);

    char                      wqe_desc[n_ds * DS_SIZE];
    int                               xlat_size;
    uint64_t                          reglen = 0;

    ibv_wr_start(qp_ex);
    memset(wqe_desc, 0, n_ds * DS_SIZE);

    ctrl = (void*)wqe_desc;
    uint8_t fm_ce_se =  MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE | MLX5_WQE_CTRL_CQ_UPDATE;

    mlx5dv_set_ctrl_seg(ctrl, /* pi */ 0x0, opcode, opmode,
                        qp->qp_num, fm_ce_se, n_ds, 0x0,
                        htobe32(dv_mkey->lkey));

    umr_ctrl_seg = PTR_OFFSET(ctrl, sizeof(*ctrl));
    umr_ctrl_seg->mkey_mask =
        htobe64(MLX5_WQE_UMR_CTRL_MKEY_MASK_LEN |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_LOCAL_WRITE |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_REMOTE_READ |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_REMOTE_WRITE |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_ATOMIC |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_FREE);

    mk_seg = PTR_OFFSET(umr_ctrl_seg, sizeof(*umr_ctrl_seg));
    mk_seg->access_flags = get_umr_mr_flags(access_flags);
    mk_seg->qpn_mkey     = htobe32(0xffffff00 | (dv_mkey->lkey & 0xff));

    pseg = PTR_OFFSET(mk_seg, sizeof(*mk_seg));

    umr_pointer_seg_init(repeat_count, num_entries, data ,pseg, ptr_mkey, ptr_address,
                         &xlat_size, &reglen);

    mk_seg->len                 = htobe64(reglen);
    umr_ctrl_seg->klm_octowords = htobe16(ucc_align_up(xlat_size, 64) / DS_SIZE);
    mlx5dv_wr_raw_wqe(mqp, wqe_desc);
    ibv_wr_complete(qp_ex);
    return UCC_OK;
}
