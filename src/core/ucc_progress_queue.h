/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_PROGRESS_QUEUE_H_
#define UCC_PROGRESS_QUEUE_H_

#include "ucc/api/ucc.h"
#include "schedule/ucc_schedule.h"

typedef struct ucc_progress_queue ucc_progress_queue_t;
struct ucc_progress_queue {
    void (*enqueue)(ucc_progress_queue_t *pq, ucc_coll_task_t *task);
    void (*dequeue)(ucc_progress_queue_t *pq, ucc_coll_task_t **task);
    int  (*progress)(ucc_progress_queue_t *pq);
    void (*finalize)(ucc_progress_queue_t *pq);
};

ucc_status_t ucc_progress_queue_init(ucc_progress_queue_t **pq,
                                     ucc_thread_mode_t tm,
                                     uint32_t lock_free_progress_q);

static inline void ucc_progress_enqueue(ucc_progress_queue_t *pq,
                                        ucc_coll_task_t *task)
{
    pq->enqueue(pq, task);
}

static inline ucc_status_t ucc_progress_queue_enqueue(ucc_progress_queue_t *pq,
                                                      ucc_coll_task_t *task)
{
    task->progress(task);
    if (task->status != UCC_INPROGRESS) {
        /* task completed immediately, don't add it to the progress queue */
        return ucc_task_complete(task);
    }
    /* set user visible status */
    task->super.status = UCC_INPROGRESS;
    pq->enqueue(pq, task);
    return UCC_OK;
}

static inline int ucc_progress_queue(ucc_progress_queue_t *pq)
{
    return pq->progress(pq);
}

void ucc_progress_queue_finalize(ucc_progress_queue_t *pq);

#endif
