/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLREDUCE_SW_H_
#define ALLREDUCE_SW_H_

#include "tl_ucp_coll.h"
#include "tl_ucp_dpu_offload.h"

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

void
ucc_tl_ucp_allreduce_sliding_window_free_task(ucc_coll_task_t *coll_task);

void
ucc_tl_ucp_allreduce_sliding_window_free_pipe(ucc_coll_task_t *coll_task);

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_alloc_pipe(ucc_base_team_t   *team,
                                               ucc_tl_ucp_task_t *task);

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_task_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t      *team,
                                              ucc_tl_ucp_task_t    *task);

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_allgather_info_finalize(
                                    ucc_tl_ucp_task_t *sw_task);

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_rdma_task_post(ucc_coll_task_t *coll_task);

void ucc_tl_ucp_allreduce_sliding_window_rdma_progress(ucc_coll_task_t *task);

#endif
