/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_DPU_OFFLOAD_H_
#define UCC_TL_UCP_DPU_OFFLOAD_H_

#include "tl_ucp.h"
#include "schedule/ucc_schedule_pipelined.h"
#include "components/mc/base/ucc_mc_base.h"
#include "components/ec/ucc_ec.h"
#include "tl_ucp_tag.h"


#define ALLREDUCE_PACKED_KEY_MAX_LEN 1024

typedef struct ucc_tl_ucp_allreduce_sw_global_work_buf_info {
    void *packed_src_memh;
    void *packed_dst_memh;
} ucc_tl_ucp_allreduce_sw_global_work_buf_info_t;

struct ucc_tl_ucp_allreduce_sw_export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void         *packed_memh;
    void         *packed_key;
    size_t        packed_key_len;
};

typedef struct ucc_tl_ucp_allreduce_sw_host_allgather {
    void *src_buf;
    void *dst_buf;
    char  packed_src_key[ALLREDUCE_PACKED_KEY_MAX_LEN];
    char  packed_dst_key[ALLREDUCE_PACKED_KEY_MAX_LEN];
} ucc_tl_ucp_allreduce_sw_host_allgather_t;

typedef struct ucc_tl_ucp_dpu_offload_buf_info {
    ucp_rkey_h                                *src_rkeys; //unpacked
    ucp_rkey_h                                *dst_rkeys; //unpacked
    void                                     **sbufs;
    void                                     **rbufs;
    struct ucc_tl_ucp_allreduce_sw_export_buf *src_ebuf;
    struct ucc_tl_ucp_allreduce_sw_export_buf *dst_ebuf;
} ucc_tl_ucp_dpu_offload_buf_info_t;

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_register(
    ucp_context_h ucp_context, ucc_tl_ucp_team_t *tl_team,
    struct ucc_tl_ucp_allreduce_sw_export_buf *ebuf, void *packed_memh);


#endif
