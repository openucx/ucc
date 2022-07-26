/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_progress_queue.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_time.h"
#include "utils/ucc_spinlock.h"
#include "utils/ucc_list.h"
#include "utils/ucc_lock_free_queue.h"
#include "utils/ucc_coll_utils.h"

typedef struct ucc_pq_mt {
    ucc_progress_queue_t super;
    ucc_lf_queue_t       lf_queue;
} ucc_pq_mt_t;

typedef struct ucc_pq_mt_locked {
    ucc_progress_queue_t super;
    ucc_spinlock_t       queue_lock;
    ucc_list_link_t      queue;
} ucc_pq_mt_locked_t;

static void ucc_pq_locked_mt_enqueue(ucc_progress_queue_t *pq,
                                     ucc_coll_task_t *     task)
{
    ucc_pq_mt_locked_t *pq_mt = ucc_derived_of(pq, ucc_pq_mt_locked_t);

    ucc_spin_lock(&pq_mt->queue_lock);
    ucc_list_add_tail(&pq_mt->queue, &task->list_elem);
    ucc_spin_unlock(&pq_mt->queue_lock);
}

static void ucc_pq_mt_enqueue(ucc_progress_queue_t *pq, ucc_coll_task_t *task)
{
    ucc_pq_mt_t *pq_mt      = ucc_derived_of(pq, ucc_pq_mt_t);

    ucc_lf_queue_enqueue(&pq_mt->lf_queue, &task->lf_elem);
}

static void ucc_pq_locked_mt_dequeue(ucc_progress_queue_t *pq,
                                     ucc_coll_task_t **    popped_task)
{
    ucc_pq_mt_locked_t *pq_mt = ucc_derived_of(pq, ucc_pq_mt_locked_t);
    *popped_task              = NULL;

    ucc_spin_lock(&pq_mt->queue_lock);
    if (!ucc_list_is_empty(&pq_mt->queue)) {
        *popped_task =
            ucc_list_extract_head(&pq_mt->queue, ucc_coll_task_t, list_elem);
    }
    ucc_spin_unlock(&pq_mt->queue_lock);
}

static void ucc_pq_mt_dequeue(ucc_progress_queue_t *pq,
                              ucc_coll_task_t **    popped_task)
{
    ucc_pq_mt_t *pq_mt  = ucc_derived_of(pq, ucc_pq_mt_t);
    ucc_lf_queue_elem_t *elem   = ucc_lf_queue_dequeue(&pq_mt->lf_queue, 1);
    *popped_task =
        elem ? ucc_container_of(elem, ucc_coll_task_t, lf_elem) : NULL;
}

static int ucc_pq_mt_progress(ucc_progress_queue_t *pq)
{
    int              n_progressed =  0;
    double           timestamp    = -1;
    ucc_coll_task_t *task;
    ucc_status_t     status;

    pq->dequeue(pq, &task);
    if (task) {
        if (task->progress) {
            task->progress(task);
        }
        if (UCC_INPROGRESS == task->status) {
            if (UCC_COLL_TIMEOUT_REQUIRED(task)) {
                if (timestamp < 0) {
                    timestamp = ucc_get_time();
                }
                if (ucc_unlikely(timestamp - task->start_time >
                                 task->bargs.args.timeout)) {
                    task->status = UCC_ERR_TIMED_OUT;
                    ucc_task_complete(task);
                    return UCC_ERR_TIMED_OUT;
                }
            }

            pq->enqueue(pq, task);
            return n_progressed;
        }
        n_progressed++;
        if (ucc_unlikely(0 > (status = ucc_task_complete(task)))) {
            return status;
        }
    }
    return n_progressed;
}

static void ucc_pq_locked_mt_finalize(ucc_progress_queue_t *pq)
{
    ucc_pq_mt_locked_t *pq_mt = ucc_derived_of(pq, ucc_pq_mt_locked_t);
    ucc_spinlock_destroy(&pq_mt->queue_lock);
    ucc_free(pq_mt);
}

static void ucc_pq_mt_finalize(ucc_progress_queue_t *pq)
{
    ucc_pq_mt_t *pq_mt = ucc_derived_of(pq, ucc_pq_mt_t);
    ucc_lf_queue_destroy(&pq_mt->lf_queue);
    ucc_free(pq_mt);
}

ucc_status_t ucc_pq_mt_init(ucc_progress_queue_t **pq,
                            uint32_t lock_free_progress_q)
{
    if (lock_free_progress_q) {
        ucc_pq_mt_t *pq_mt = ucc_malloc(sizeof(*pq_mt), "pq_mt");
        if (!pq_mt) {
            ucc_error("failed to allocate %zd bytes for pq_mt", sizeof(*pq_mt));
            return UCC_ERR_NO_MEMORY;
        }
        ucc_lf_queue_init(&pq_mt->lf_queue);
        pq_mt->super.enqueue    = ucc_pq_mt_enqueue;
        pq_mt->super.dequeue    = ucc_pq_mt_dequeue;
        pq_mt->super.progress   = ucc_pq_mt_progress;
        pq_mt->super.finalize   = ucc_pq_mt_finalize;
        *pq                     = &pq_mt->super;
    } else {
        ucc_pq_mt_locked_t *pq_mt = ucc_malloc(sizeof(*pq_mt), "pq_mt");
        if (!pq_mt) {
            ucc_error("failed to allocate %zd bytes for pq_mt", sizeof(*pq_mt));
            return UCC_ERR_NO_MEMORY;
        }
        ucc_spinlock_init(&pq_mt->queue_lock, 0);
        ucc_list_head_init(&pq_mt->queue);
        pq_mt->super.enqueue  = ucc_pq_locked_mt_enqueue;
        pq_mt->super.dequeue  = ucc_pq_locked_mt_dequeue;
        pq_mt->super.progress = ucc_pq_mt_progress;
        pq_mt->super.finalize = ucc_pq_locked_mt_finalize;
        *pq                   = &pq_mt->super;
    }
    return UCC_OK;
}
