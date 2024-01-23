/**
 * Copyright(c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "allreduce_sliding_window.h"
#include "../allgather/allgather.h"
#include "../barrier/barrier.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"


static inline void //NOLINT
ucc_tl_ucp_allreduce_sliding_window_reset_buf(ucc_tl_ucp_allreduce_sw_buf_t __attribute__((unused))  *buf) //NOLINT
{
}

static inline void ucc_tl_ucp_allreduce_sliding_window_reset_pipeline( //NOLINT
    ucc_tl_ucp_allreduce_sw_pipeline_t __attribute__((unused)) *pipe, ucc_rank_t __attribute__((unused)) rank, //NOLINT
    size_t __attribute__((unused)) put_window_size) //NOLINT
{
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_start(ucc_coll_task_t __attribute__((unused)) *coll_task) //NOLINT
{
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_finalize(ucc_coll_task_t __attribute__((unused)) *coll_task) //NOLINT
{
    return UCC_OK;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_reduction(
    ucc_coll_task_t __attribute__((unused)) *coll_task, ucc_tl_ucp_allreduce_sw_buf_t __attribute__((unused)) *accbuf,//NOLINT
    ucc_tl_ucp_allreduce_sw_buf_t __attribute__((unused)) *getbuf)//NOLINT
{
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_test_reduction(ucc_tl_ucp_task_t __attribute__((unused)) *task)//NOLINT
{
}

static inline ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_req_test(ucs_status_ptr_t __attribute__((unused))  request,//NOLINT
                                             ucc_tl_ucp_task_t __attribute__((unused)) *task)//NOLINT
{
    return UCC_OK;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_allgather_info_test(//NOLINT
    ucc_coll_task_t __attribute__((unused)) *coll_task)//NOLINT
{
}

static inline void ucc_tl_ucp_allreduce_sliding_window_allgather_free_rkeys(//NOLINT
    ucc_coll_task_t __attribute__((unused)) *coll_task)//NOLINT
{
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_barrier(ucc_coll_task_t __attribute__((unused)) *coll_task)//NOLINT
{
}

void ucc_tl_ucp_allreduce_sliding_window_progress(ucc_coll_task_t *coll_task)//NOLINT
{
    ucs_status_ptr_t request = 0;
    ucc_tl_ucp_task_t *task = NULL;
    ucc_tl_ucp_allreduce_sw_buf_t *accbuf = NULL;
    ucc_tl_ucp_allreduce_sw_buf_t *getbuf = NULL;
    ucc_tl_ucp_allreduce_sw_pipeline_t *pipe = NULL;

    // suppress "function unused" Werrors
    ucc_tl_ucp_allreduce_sliding_window_barrier(coll_task);
    ucc_tl_ucp_allreduce_sliding_window_allgather_free_rkeys(coll_task);
    ucc_tl_ucp_allreduce_sliding_window_allgather_info_test(coll_task);
    ucc_tl_ucp_allreduce_sliding_window_req_test(request, task);
    ucc_tl_ucp_allreduce_sliding_window_test_reduction(task);
    ucc_tl_ucp_allreduce_sliding_window_reduction(coll_task, accbuf, getbuf);
    ucc_tl_ucp_allreduce_sliding_window_finalize(coll_task);
    ucc_tl_ucp_allreduce_sliding_window_start(coll_task);
    ucc_tl_ucp_allreduce_sliding_window_reset_pipeline(pipe, 0, 0);
    ucc_tl_ucp_allreduce_sliding_window_reset_buf(accbuf);
}
