/**
 * Copyright(c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "../allgather/allgather.h"
#include "../barrier/barrier.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"

static inline void
ucc_tl_ucp_allreduce_sliding_window_reset_buf(ucc_tl_ucp_allreduce_sw_buf *buf)
{
}

static inline void ucc_tl_ucp_allreduce_sliding_window_reset_pipeline(
    ucc_tl_ucp_allreduce_sw_pipeline *pipe, ucc_rank_t rank,
    size_t put_window_size)
{
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_start(ucc_coll_task_t *coll_task)
{
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_finalize(ucc_coll_task_t *coll_task)
{
    return UCC_OK;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_reduction(
    ucc_coll_task_t *coll_task, ucc_tl_ucp_allreduce_sw_buf *accbuf,
    ucc_tl_ucp_allreduce_sw_buf *getbuf)
{
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_test_reduction(ucc_tl_ucp_task_t *task)
{
}

static inline ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_req_test(ucs_status_ptr_t   request,
                                             ucc_tl_ucp_task_t *task)
{
    return UCC_OK;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_allgather_info_test(
    ucc_coll_task_t *coll_task)
{
}

static inline void ucc_tl_ucp_allreduce_sliding_window_allgather_free_rkeys(
    ucc_coll_task_t *coll_task)
{
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_barrier(ucc_coll_task_t *coll_task)
{
}

void ucc_tl_ucp_allreduce_sliding_window_progress(ucc_coll_task_t *coll_task)
{
}
