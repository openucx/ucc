/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLREDUCE_SW_H_
#define ALLREDUCE_SW_H_

#include "tl_ucp_coll.h"

#define ALLREDUCE_PACKED_KEY_MAX_LEN 1024

typedef struct ucc_tl_ucp_allreduce_sw_global_work_buf_info {
    void *packed_src_memh;
    void *packed_dst_memh;
} ucc_tl_ucp_allreduce_sw_global_work_buf_info_t;

typedef enum ucc_tl_ucp_allreduce_sw_buf_state {
    FREE,
    RECVING,
    REDUCING,
    REDUCED,
    SENDING,
    IDLE,
} ucc_tl_ucp_allreduce_sw_buf_state_t;

typedef struct ucc_tl_ucp_allreduce_sw_buf {
    void                               *buf;
    ucc_tl_ucp_allreduce_sw_buf_state_t state;
    ucs_status_ptr_t                    ucp_req;
    size_t                              count;
    size_t                              bytes;
} ucc_tl_ucp_allreduce_sw_buf_t;

typedef struct ucc_tl_ucp_allreduce_sw_pipeline {
    ucc_tl_ucp_allreduce_sw_buf_t  accbuf;
    ucc_tl_ucp_allreduce_sw_buf_t *getbuf;
    ucs_status_ptr_t              *put_requests;
    size_t                         buffer_size;
    size_t                         num_buffers;
    size_t                         avail_buffs;
    size_t                         my_count;
    size_t                         my_offset;
    size_t                         count_issued;
    size_t                         count_received;
    size_t                         count_reduced;
    size_t                         count_serviced;
    size_t                         get_idx;
    size_t                         red_idx;
    ucc_rank_t                     src_rank;
    ucc_rank_t                     dst_rank;
    int                            done_get;
    int                            done_red;
    int                            done_put;
    int                            posted_put;
} ucc_tl_ucp_allreduce_sw_pipeline_t;

struct ucc_tl_ucp_allreduce_sw_export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void         *packed_memh;
    size_t        packed_memh_len;
    void         *packed_key;
    size_t        packed_key_len;
    uint64_t      memh_id;
};

typedef struct ucc_tl_ucp_allreduce_sw_host_allgather {
    void *src_buf;
    void *dst_buf;
    char  packed_src_key[ALLREDUCE_PACKED_KEY_MAX_LEN];
    char  packed_dst_key[ALLREDUCE_PACKED_KEY_MAX_LEN];
} ucc_tl_ucp_allreduce_sw_host_allgather_t;

#endif
