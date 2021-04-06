/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_progress_queue.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_spinlock.h"
#include "utils/ucc_atomic.h"

// Number of tasks in a single lock free pool - could be changed, but in tests performed great
#define LINE_SIZE 8
#define NUM_POOLS 2 // We need exactly two pools

typedef struct ucc_pq_mt {
    ucc_progress_queue_t super;
    ucc_spinlock_t       locked_queue_lock[NUM_POOLS];
    ucc_coll_task_t *    tasks[NUM_POOLS][LINE_SIZE];
    uint8_t              which_pool;
    ucc_list_link_t      locked_queue[NUM_POOLS];
} ucc_pq_mt_t; // TODO the struct isn't a queue because not maintaining order, maybe another name

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
    int          which_pool = task->was_progressed ^ (pq_mt->which_pool & 1);
    int          i;
    for (i = 0; i < LINE_SIZE; i++) {
        if (ucc_atomic_bool_cswap64((uint64_t *)&(pq_mt->tasks[which_pool][i]),
                                    0, (uint64_t)task)) {
            return;
        }
    }

    ucc_spin_lock(&pq_mt->locked_queue_lock[which_pool]);
    ucc_list_add_tail(&pq_mt->locked_queue[which_pool], &task->list_elem);
    ucc_spin_unlock(&pq_mt->locked_queue_lock[which_pool]);
}

static void ucc_pq_locked_mt_dequeue(ucc_progress_queue_t *pq,
                                     ucc_coll_task_t **    popped_task,
                                     int                   is_first_call) //NOLINT 
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
                              ucc_coll_task_t **popped_task, int is_first_call)
{
    ucc_pq_mt_t *pq_mt  = ucc_derived_of(pq, ucc_pq_mt_t);
    // Save value in the beginning of the function
    int curr_which_pool = pq_mt->which_pool;
    int which_pool      = curr_which_pool & 1; // turn from even/odd -> bool
    int              i;
    for (i = 0; i < LINE_SIZE; i++) {
        *popped_task = pq_mt->tasks[which_pool][i];
        if (*popped_task) {
            if (ucc_atomic_bool_cswap64(
                    (uint64_t *)&(pq_mt->tasks[which_pool][i]),
                    (uint64_t)*popped_task, 0)) {
                (*popped_task)->was_progressed = 1;
                return;
            }
        }
    }
    *popped_task = NULL;
    ucc_spin_lock(&pq_mt->locked_queue_lock[which_pool]);
    if (!ucc_list_is_empty(&pq_mt->locked_queue[which_pool])) {
        *popped_task = ucc_list_extract_head(&pq_mt->locked_queue[which_pool],
                                             ucc_coll_task_t, list_elem);
        (*popped_task)->was_progressed = 1;
    }
    ucc_spin_unlock(&pq_mt->locked_queue_lock[which_pool]);
    ucc_atomic_cswap8(&pq_mt->which_pool, curr_which_pool, curr_which_pool + 1);
    if (is_first_call) {
        // TODO maybe only when which_pool increase is OK
        pq->dequeue(pq, popped_task, 0);
    }
}

static int ucc_pq_mt_progress(ucc_progress_queue_t *pq)
{
    int              n_progressed = 0;
    ucc_coll_task_t *task;
    ucc_status_t     status;
    pq->dequeue(pq, &task, 1);
    if (task) {
        if (task->progress) {
            status = task->progress(task);
            if (status < 0) {
                return status;
            }
        }
        if (UCC_INPROGRESS == task->super.status) {
            pq->enqueue(pq, task);
            return n_progressed;
        }
        n_progressed++;
        if (0 > (status = ucc_task_complete(task))) {
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
    ucc_spinlock_destroy(&pq_mt->locked_queue_lock[0]);
    ucc_spinlock_destroy(&pq_mt->locked_queue_lock[1]);
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
        memset(&pq_mt->tasks, 0,
               NUM_POOLS * LINE_SIZE * sizeof(ucc_coll_task_t *));
        ucc_spinlock_init(&pq_mt->locked_queue_lock[0], 0);
        ucc_spinlock_init(&pq_mt->locked_queue_lock[1], 0);
        ucc_list_head_init(&pq_mt->locked_queue[0]);
        ucc_list_head_init(&pq_mt->locked_queue[1]);
        pq_mt->which_pool       = 0;
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
