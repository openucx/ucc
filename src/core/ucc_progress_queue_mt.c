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
    ucc_spinlock_t       locked_queue_lock;
    ucc_coll_task_t *    tasks[NUM_POOLS][LINE_SIZE];
    uint8_t              which_pool;
    ucc_list_link_t      locked_queue;
    uint32_t             tasks_counters[NUM_POOLS];
} ucc_pq_mt_t; // TODO the struct isn't a queue because not maintaining order, maybe another name

static void ucc_pq_mt_enqueue(ucc_progress_queue_t *pq, ucc_coll_task_t *task)
{
    ucc_pq_mt_t *pq_mt = ucc_derived_of(pq, ucc_pq_mt_t);

    ucc_spin_lock(&pq_mt->locked_queue_lock);
    ucc_list_add_tail(&pq_mt->locked_queue, &task->list_elem);
    ucc_spin_unlock(&pq_mt->locked_queue_lock);
}

static void ucc_pq_mt_enqueue_opt(ucc_progress_queue_t *pq, ucc_coll_task_t *task)
{
    ucc_pq_mt_t *pq_mt      = ucc_derived_of(pq, ucc_pq_mt_t);
    int          which_pool = task->was_progressed ^ (pq_mt->which_pool & 1);
    int          i;
    for (i = 0; i < LINE_SIZE; i++) {
        if (ucc_atomic_bool_cswap64((uint64_t *)&(pq_mt->tasks[which_pool][i]),
                                    0, (uint64_t)task)) {
            ucc_atomic_add32(&pq_mt->tasks_counters[which_pool], 1);
            return;
        }
    }

    ucc_pq_mt_enqueue(pq, task);
}

static void ucc_pq_mt_dequeue(ucc_progress_queue_t *pq,
                              ucc_coll_task_t     **popped_task_ptr,
                              int                   is_first_call) //NOLINT
{
    ucc_pq_mt_t *pq_mt           = ucc_derived_of(pq, ucc_pq_mt_t);
    ucc_coll_task_t *popped_task = NULL;

    ucc_spin_lock(&pq_mt->locked_queue_lock);
    if (!ucc_list_is_empty(&pq_mt->locked_queue)) {
        popped_task = ucc_list_extract_head(&pq_mt->locked_queue,
                                            ucc_coll_task_t, list_elem);
        popped_task->was_progressed = 1;
    }
    ucc_spin_unlock(&pq_mt->locked_queue_lock);
    *popped_task_ptr = popped_task;
}

static void ucc_pq_mt_dequeue_opt(ucc_progress_queue_t *pq,
                                  ucc_coll_task_t     **popped_task_ptr,
                                  int                   is_first_call)
{
    ucc_pq_mt_t *pq_mt  = ucc_derived_of(pq, ucc_pq_mt_t);
    // Save value in the beginning of the function
    int curr_which_pool = pq_mt->which_pool;
    int which_pool      = curr_which_pool & 1; // turn from even/odd -> bool
    ucc_coll_task_t *popped_task = NULL;
    int              i;
    if (pq_mt->tasks_counters[which_pool]) {
        for (i = 0; i < LINE_SIZE; i++) {
            popped_task = pq_mt->tasks[which_pool][i];
            if (popped_task) {
                if (ucc_atomic_bool_cswap64(
                        (uint64_t *)&(pq_mt->tasks[which_pool][i]),
                        (uint64_t)popped_task, 0)) {
                    ucc_atomic_sub32(&pq_mt->tasks_counters[which_pool], 1);
                    *popped_task_ptr            = popped_task;
                    popped_task->was_progressed = 1;
                    return;
                }
            }
        }
    }

    if (is_first_call) {
        ucc_atomic_cswap8(&pq_mt->which_pool, curr_which_pool,
                          curr_which_pool + 1);
        pq->dequeue(pq, popped_task_ptr, 0);
        return;
    }

    ucc_pq_mt_dequeue(pq, popped_task_ptr, 0);
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
        if (UCC_OK == task->super.status) {
            n_progressed++;
            // TODO need to decide for st/mt progress should it return #progressed/#completed tasks,
            // and document accordingly
            status = ucc_event_manager_notify(task, UCC_EVENT_COMPLETED);
            if (status != UCC_OK) {
                return status;
            }
            if (task->flags & UCC_COLL_TASK_FLAG_INTERNAL) {
                task->finalize(task);
            }
        } else {
            pq->enqueue(pq, task);
        }
    }
    return n_progressed;
}

static void ucc_pq_mt_finalize(ucc_progress_queue_t *pq)
{
    ucc_pq_mt_t *pq_mt = ucc_derived_of(pq, ucc_pq_mt_t);
    ucc_spinlock_destroy(&pq_mt->locked_queue_lock);
    ucc_free(pq_mt);
}

ucc_status_t ucc_pq_mt_init(ucc_progress_queue_t **pq,
                            uint32_t lock_free_progress_q)
{
    ucc_pq_mt_t *pq_mt = ucc_malloc(sizeof(*pq_mt), "pq_mt");
    if (!pq_mt) {
        ucc_error("failed to allocate %zd bytes for pq_mt", sizeof(*pq_mt));
        return UCC_ERR_NO_MEMORY;
    }
    memset(&pq_mt->tasks, 0, NUM_POOLS * LINE_SIZE * sizeof(ucc_coll_task_t *));
    ucc_spinlock_init(&pq_mt->locked_queue_lock, 0);
    ucc_list_head_init(&pq_mt->locked_queue);
    pq_mt->which_pool        = 0;
    pq_mt->tasks_counters[0] = 0;
    pq_mt->tasks_counters[1] = 0;

    if (lock_free_progress_q) {
        pq_mt->super.enqueue    = ucc_pq_mt_enqueue_opt;
        pq_mt->super.dequeue    = ucc_pq_mt_dequeue_opt;
    } else {
        pq_mt->super.enqueue    = ucc_pq_mt_enqueue;
        pq_mt->super.dequeue    = ucc_pq_mt_dequeue;
    }
    pq_mt->super.progress    = ucc_pq_mt_progress;
    pq_mt->super.finalize    = ucc_pq_mt_finalize;
    *pq                      = &pq_mt->super;
    return UCC_OK;
}
