/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLREDUCE_H_
#define ALLREDUCE_H_
#include "tl_ucp_coll.h"

enum {
    UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL,
    UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL,
    UCC_TL_UCP_ALLREDUCE_ALG_SLIDING_WINDOW,
    UCC_TL_UCP_ALLREDUCE_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_allreduce_algs[UCC_TL_UCP_ALLREDUCE_ALG_LAST + 1];
ucc_status_t ucc_tl_ucp_allreduce_init(ucc_tl_ucp_task_t *task);

#define UCC_TL_UCP_ALLREDUCE_DEFAULT_ALG_SELECT_STR                            \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1"

#define CHECK_SAME_MEMTYPE(_args, _team)                                       \
    do {                                                                       \
        if (!UCC_IS_INPLACE(_args) &&                                          \
            (_args.src.info.mem_type != _args.dst.info.mem_type)) {            \
            tl_error(UCC_TL_TEAM_LIB(_team),                                   \
                     "asymmetric src/dst memory types are not supported yet"); \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while (0)

#define ALLREDUCE_TASK_CHECK(_args, _team)                                     \
    CHECK_SAME_MEMTYPE((_args), (_team));

#define ALLREDUCE_PACKED_KEY_MAX_LEN 1024

typedef struct ucc_tl_ucp_allreduce_sw_global_work_buf_info {
    void *packed_src_memh;
    void *packed_dst_memh;
} ucc_tl_ucp_allreduce_sw_global_work_buf_info;

typedef enum ucc_tl_ucp_allreduce_sw_buf_state
{
    FREE,
    RECVING,
    REDUCING,
    REDUCED,
    SENDING,
    IDLE,
} ucc_tl_ucp_allreduce_sw_buf_state;

typedef struct ucc_tl_ucp_allreduce_sw_buf {
    void *                            buf;
    ucc_tl_ucp_allreduce_sw_buf_state state;
    ucs_status_ptr_t                  ucp_req;
    size_t                            count;
    size_t                            bytes;
} ucc_tl_ucp_allreduce_sw_buf;

typedef struct ucc_tl_ucp_allreduce_sw_pipeline {
    ucc_tl_ucp_allreduce_sw_buf  accbuf;
    ucc_tl_ucp_allreduce_sw_buf *getbuf;
    ucs_status_ptr_t *           put_requests;
    size_t                       buffer_size;
    size_t                       num_buffers;
    size_t                       avail_buffs;
    size_t                       my_count;
    size_t                       my_offset;
    size_t                       count_issued;
    size_t                       count_received;
    size_t                       count_reduced;
    size_t                       count_serviced;
    size_t                       get_idx;
    size_t                       red_idx;
    ucc_rank_t                   src_rank;
    ucc_rank_t                   dst_rank;
    int                          done_get;
    int                          done_red;
    int                          done_put;
    int                          posted_put;
} ucc_tl_ucp_allreduce_sw_pipeline;

struct ucc_tl_ucp_allreduce_sw_export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void *        packed_memh;
    size_t        packed_memh_len;
    void *        packed_key;
    size_t        packed_key_len;
    uint64_t      memh_id;
};

typedef struct ucc_tl_ucp_allreduce_sw_host_allgather {
    void *src_buf;
    void *dst_buf;
    char  packed_src_key[ALLREDUCE_PACKED_KEY_MAX_LEN];
    char  packed_dst_key[ALLREDUCE_PACKED_KEY_MAX_LEN];
} ucc_tl_ucp_allreduce_sw_host_allgather;

ucc_status_t ucc_tl_ucp_allreduce_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h);

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *     team,
                                         ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_ucp_allreduce_knomial_init_common(ucc_tl_ucp_task_t *task);

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_alloc_pipe(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_tl_ucp_task_t *   task);

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_task_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *     team,
                                              ucc_tl_ucp_task_t *   task);

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_allgather_info_finalize(
    ucc_service_coll_req_t *scoll_req, ucc_tl_ucp_task_t *sw_task);

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_free_gwbi(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_ucp_allreduce_knomial_start(ucc_coll_task_t *task);

void ucc_tl_ucp_allreduce_knomial_progress(ucc_coll_task_t *task);

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_start(ucc_coll_task_t *coll_task);

void ucc_tl_ucp_allreduce_sliding_window_progress(ucc_coll_task_t *task);

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allreduce_knomial_finalize(ucc_coll_task_t *task);

ucc_status_t
ucc_tl_ucp_allreduce_sra_knomial_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t *     team,
                                      ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_start(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_progress(ucc_coll_task_t *task);

static inline int ucc_tl_ucp_allreduce_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_ALLREDUCE_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_allreduce_algs[i].name)) {
            break;
        }
    }
    return i;
}
#endif
