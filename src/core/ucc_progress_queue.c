/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_progress_queue.h"

ucc_status_t ucc_pq_st_init(ucc_progress_queue_t **pq);
ucc_status_t ucc_pq_mt_init(ucc_progress_queue_t **pq, uint32_t lock_free_progress_q);

ucc_status_t ucc_progress_queue_init(ucc_progress_queue_t **pq,
                                     ucc_thread_mode_t      tm,
                                     uint32_t lock_free_progress_q)
{
    if (tm == UCC_THREAD_SINGLE) {
        return ucc_pq_st_init(pq);
    } else { // TODO also for UCC_THREAD_FUNNELED?
        return ucc_pq_mt_init(pq, lock_free_progress_q);
    }
}

void ucc_progress_queue_finalize(ucc_progress_queue_t *pq)
{
    return pq->finalize(pq);
}
