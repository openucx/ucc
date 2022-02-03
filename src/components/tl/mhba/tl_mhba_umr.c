/*
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tl_mhba_ib.h"
#include "utils/arch/cpu.h"

#define SQ_WQE_SHIFT 6
#define DS_SIZE      16 // size of a single Data Segment in the WQE

/* UMR pointer to KLMs/MTTs/RepeatBlock and BSFs location (when inline = 0) */
struct mlx5_wqe_umr_pointer_seg {
    __be32 reserved;
    __be32 mkey;
    __be64 address;
};

ucc_status_t
ucc_tl_mhba_ibv_qp_to_mlx5dv_qp(struct ibv_qp *                 umr_qp,
                                struct ucc_tl_mhba_internal_qp *mqp, ucc_tl_mhba_lib_t *lib)
{
    struct mlx5dv_obj dv_obj;
    memset((void *)&dv_obj, 0, sizeof(struct mlx5dv_obj));
    dv_obj.qp.in  = umr_qp;
    dv_obj.qp.out = &mqp->qp;
    mqp->qp_num   = umr_qp->qp_num;
    if (mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_QP)) {
        tl_error(lib, "mlx5dv_init failed - errno %d", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    mqp->sq_cur_post = 0;
    mqp->sq_qend     = mqp->qp.sq.buf + (mqp->qp.sq.wqe_cnt << SQ_WQE_SHIFT);
    mqp->fm_cache    = 0;
    mqp->sq_start    = mqp->qp.sq.buf;
    mqp->offset      = 0;
    ucs_spinlock_init(&mqp->qp_spinlock, 0);
    return UCC_OK;
}

ucc_status_t ucc_tl_mhba_destroy_mlxdv_qp(struct ucc_tl_mhba_internal_qp *mqp)
{
    ucs_spinlock_destroy(&mqp->qp_spinlock);
    return UCC_OK;
}

static inline void post_send_db(struct ucc_tl_mhba_internal_qp *mqp, int nreq,
                                void *ctrl)
{
    if (ucs_unlikely(!nreq))
        return;

    /*
     * Make sure that descriptors are written before
     * updating doorbell record and ringing the doorbell
     */
    ucc_memory_bus_fence();
    mqp->qp.dbrec[MLX5_SND_DBR] = htobe32(mqp->sq_cur_post & 0xffff);

    /* Make sure that the doorbell write happens before the memcpy
	 * to WC memory below
	 */
    ucc_memory_cpu_fence();

    *(__be64 *)PTR_OFFSET(mqp->qp.bf.reg, mqp->offset) = *(__be64 *)ctrl;

    ucc_memory_bus_store_fence();

    mqp->offset ^= mqp->qp.bf.size;
}

void ucc_tl_mhba_wr_start(struct ucc_tl_mhba_internal_qp *mqp)
{
    ucs_spin_lock(&mqp->qp_spinlock);
    mqp->nreq = 0;
}

void ucc_tl_mhba_wr_complete(struct ucc_tl_mhba_internal_qp *mqp)
{
    post_send_db(mqp, mqp->nreq, mqp->cur_ctrl);
    ucs_spin_unlock(&mqp->qp_spinlock);
}

static inline void common_wqe_init(struct ibv_qp_ex *              ibqp,
                                   struct ucc_tl_mhba_internal_qp *mqp,
                                   int opcode)
{
    struct mlx5_wqe_ctrl_seg *ctrl;
    uint8_t                   fence;
    uint32_t                  idx;

    idx = mqp->sq_cur_post & (mqp->qp.sq.wqe_cnt - 1);

    ctrl = mqp->sq_start + (idx << MLX5_SEND_WQE_SHIFT);
    *(uint32_t *)((void *)(ptrdiff_t)ctrl + 8) = 0;

    fence =
        (ibqp->wr_flags & IBV_SEND_FENCE) ? MLX5_WQE_CTRL_FENCE : mqp->fm_cache;
    mqp->fm_cache = 0;

    // if any Fence issue - this section has been changed
    ctrl->fm_ce_se =
        fence |
        (ibqp->wr_flags & IBV_SEND_SIGNALED ? MLX5_WQE_CTRL_CQ_UPDATE : 0) |
        (ibqp->wr_flags & IBV_SEND_SOLICITED ? MLX5_WQE_CTRL_SOLICITED : 0);

    ctrl->opmod_idx_opcode =
        htobe32(((mqp->sq_cur_post & 0xffff) << 8) | opcode);

    mqp->cur_ctrl = ctrl;
}

static inline void common_wqe_finilize(struct ucc_tl_mhba_internal_qp *mqp)
{
    mqp->cur_ctrl->qpn_ds = htobe32(mqp->cur_size | (mqp->qp_num << 8));

    mqp->sq_cur_post += ucc_div_round_up(mqp->cur_size, 4);
}

/* The strided block format is as the following:
 * | repeat_block | entry_block | entry_block |...| entry_block |
 * While the repeat entry contains details on the list of the block_entries.
 */
static void umr_strided_seg_create_noninline(
    struct ucc_tl_mhba_internal_qp *mqp, uint32_t repeat_count,
    uint16_t num_interleaved, struct mlx5dv_mr_interleaved *data, void *seg,
    void *qend, uint32_t ptr_mkey, void *ptr_address, int *wqe_size,
    int *xlat_size, uint64_t *reglen)
{
    struct mlx5_wqe_umr_pointer_seg *     pseg;
    struct mlx5_wqe_umr_repeat_block_seg *rb;
    struct mlx5_wqe_umr_repeat_ent_seg *  eb;
    uint64_t                              byte_count = 0;
    int                                   i;

    /* set pointer segment */
    pseg          = seg;
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
    *wqe_size      = sizeof(struct mlx5_wqe_umr_pointer_seg);
    *xlat_size     = (num_interleaved + 1) * sizeof(*eb);
}

static inline bool check_comp_mask(uint64_t input, uint64_t supported)
{
    return (input & ~supported) == 0;
}

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

/* External API to expose the non-inline UMR registration */
ucc_status_t ucc_tl_mhba_send_wr_mr_noninline(
    struct ucc_tl_mhba_internal_qp *mqp, struct mlx5dv_mkey *dv_mkey,
    uint32_t access_flags, uint32_t repeat_count, uint16_t num_entries,
    struct mlx5dv_mr_interleaved *data, uint32_t ptr_mkey, void *ptr_address,
    struct ibv_qp_ex *ibqp)
{
    struct mlx5_wqe_umr_ctrl_seg *    umr_ctrl_seg;
    struct mlx5_wqe_mkey_context_seg *mk_seg;
    int                               xlat_size;
    int                               size;
    uint64_t                          reglen = 0;
    void *                            qend   = mqp->sq_qend;
    void *                            seg;

    if (ucs_unlikely(!check_comp_mask(
            access_flags, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_ATOMIC))) {
//        tl_error("Un-supported UMR flags");
        return UCC_ERR_NO_MESSAGE;
    }

    common_wqe_init(ibqp, mqp, MLX5_OPCODE_UMR);
    mqp->cur_size      = sizeof(struct mlx5_wqe_ctrl_seg) / DS_SIZE;
    mqp->cur_ctrl->imm = htobe32(dv_mkey->lkey);
    seg                = umr_ctrl_seg =
        (void *)mqp->cur_ctrl + sizeof(struct mlx5_wqe_ctrl_seg);

    memset(umr_ctrl_seg, 0, sizeof(*umr_ctrl_seg));
    umr_ctrl_seg->mkey_mask =
        htobe64(MLX5_WQE_UMR_CTRL_MKEY_MASK_LEN |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_LOCAL_WRITE |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_REMOTE_READ |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_REMOTE_WRITE |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_ATOMIC |
                MLX5_WQE_UMR_CTRL_MKEY_MASK_FREE);

    seg += sizeof(struct mlx5_wqe_umr_ctrl_seg);
    mqp->cur_size += sizeof(struct mlx5_wqe_umr_ctrl_seg) / DS_SIZE;

    if (ucs_unlikely(seg == qend))
        seg = mqp->sq_start;

    mk_seg = seg;
    memset(mk_seg, 0, sizeof(*mk_seg));
    mk_seg->access_flags = get_umr_mr_flags(access_flags);
    mk_seg->qpn_mkey     = htobe32(0xffffff00 | (dv_mkey->lkey & 0xff));

    seg += sizeof(struct mlx5_wqe_mkey_context_seg);
    mqp->cur_size += (sizeof(struct mlx5_wqe_mkey_context_seg) / DS_SIZE);

    if (ucs_unlikely(seg == qend))
        seg = mqp->sq_start;

    umr_strided_seg_create_noninline(mqp, repeat_count, num_entries, data, seg,
                                     qend, ptr_mkey, ptr_address, &size,
                                     &xlat_size, &reglen);

    mk_seg->len                 = htobe64(reglen);
    umr_ctrl_seg->klm_octowords = htobe16(ucc_align_up(xlat_size, 64) / DS_SIZE);
    mqp->cur_size += size / DS_SIZE;

    mqp->fm_cache = MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE;
    mqp->nreq++;

    common_wqe_finilize(mqp);
    return UCC_OK;
}

#define MLX5_OPCODE_LOCAL_MMO 0x32

typedef struct transpose_seg {
    __be32 element_size; /* 8 bit value */
    __be16 num_rows; /* 7 bit value */
    __be16 num_cols; /* 7 bit value */
    __be64 padding;
} transpose_seg_t;

/* External API to expose the non-inline UMR registration */
ucc_status_t ucc_tl_mhba_post_transpose(struct ibv_qp *qp, uint32_t src_mr_lkey, uint32_t dst_mr_key,
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

    ibv_wr_start(qp_ex);
    memset(wqe_desc, 0, n_ds * DS_SIZE);
    /* SET CTRL SEG */

    ctrl = (void*)wqe_desc;
    uint8_t fm_ce_se = MLX5_WQE_CTRL_FENCE;
    // MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE ?

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
    if (ibv_wr_complete(qp_ex)) {
        fprintf(stderr, "failed to post transpose wqe");
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}
