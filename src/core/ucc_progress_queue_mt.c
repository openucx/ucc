/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_progress_queue.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_spinlock.h"
#include "utils/ucc_atomic.h"

typedef struct ucc_pq_mt {
    ucc_progress_queue_t super;
    ucc_spinlock_t       locked_queue_lock;
    ucc_coll_task_t***   tasks;
    uint8_t              which_pool;
    ucc_list_link_t      locked_queue;
    uint32_t             tasks_countrs[2];
} ucc_pq_mt_t; //todo the struct isn't a queue because not maintaining order, maybe another name

#define LINE_SIZE 8
#define NUM_POOLS 2

static void ucc_pq_mt_enqueue(ucc_progress_queue_t *pq, ucc_coll_task_t *task) {
    ucc_pq_mt_t *pq_mt = ucc_derived_of(pq, ucc_pq_mt_t);
    int i;
    int which_pool = task->was_progressed ^(pq_mt->which_pool & 1);
    for (i = 0; i < LINE_SIZE; i++) {
        if (ucc_atomic_bool_cswap64((uint64_t *) &(pq_mt->tasks[which_pool][i]), 0, (uint64_t) task)) {
            ucc_atomic_add32(&pq_mt->tasks_countrs[which_pool], 1);
            return;
        }
    }
    ucc_spin_lock(&pq_mt->locked_queue_lock);
    ucc_list_add_tail(&pq_mt->locked_queue, &task->list_elem);
    ucc_spin_unlock(&pq_mt->locked_queue_lock);
}

static void ucc_pq_mt_dequeue(ucc_pq_mt_t *pq_mt, ucc_coll_task_t **popped_task_ptr, int is_first_call) {
    int i, curr_which_pool, which_pool;
    curr_which_pool = pq_mt->which_pool; // extract value in the beginning of the function
    which_pool = curr_which_pool & 1; // turn from even/odd -> bool
    ucc_coll_task_t *popped_task = NULL;
    if (pq_mt->tasks_countrs[which_pool]) {
        for (i = 0; i < LINE_SIZE; i++) {
            popped_task = pq_mt->tasks[which_pool][i];
            if (popped_task) {
                if (ucc_atomic_bool_cswap64((uint64_t *) &(pq_mt->tasks[which_pool][i]), (uint64_t) popped_task, 0)) {
                    ucc_atomic_sub32(&pq_mt->tasks_countrs[which_pool], 1);
                    *popped_task_ptr = popped_task;
                    popped_task->was_progressed = 1;
                    return;
                } else {
                    i = -1;
                    //break statement - was here on XCCL, seems like a mistake - we want the pop to give at least 1 task if exist
                }
            }
        }
    }
    if (is_first_call) {
        ucc_atomic_cswap8(&pq_mt->which_pool, curr_which_pool, curr_which_pool + 1);
        ucc_pq_mt_dequeue(pq_mt, popped_task_ptr, 0);
        return;
    }
    popped_task = NULL;
    ucc_spin_lock(&pq_mt->locked_queue_lock);
    if (!ucc_list_is_empty(&pq_mt->locked_queue)) {
        popped_task = ucc_list_extract_head(&pq_mt->locked_queue, ucc_coll_task_t, list_elem);
    }
    ucc_spin_unlock(&pq_mt->locked_queue_lock);
    if (popped_task) {
        popped_task->was_progressed = 1;
    }
    *popped_task_ptr = popped_task;
}

ucc_status_t ucc_pq_mt_progress(ucc_progress_queue_t *pq) {
    ucc_pq_mt_t     *pq_mt        = ucc_derived_of(pq, ucc_pq_mt_t);
    ucc_coll_task_t *task;
    ucc_status_t     status;
    ucc_pq_mt_dequeue(pq_mt, &task, 1);
    if (task) {
        if (task->progress) {
            status = task->progress(task);
            if (status != UCC_OK) { //todo in st its status < 0
                return status;
            }
        }
        if (UCC_OK == task->super.status) {
            return ucc_event_manager_notify(&task->em, UCC_EVENT_COMPLETED);
        } else {
            ucc_pq_mt_enqueue(pq, task);
        }
    }
    return UCC_OK;
}

static void ucc_pq_mt_finalize(ucc_progress_queue_t *pq) {
    ucc_pq_mt_t *pq_mt = ucc_derived_of(pq, ucc_pq_mt_t);
    int i;
    for (i = 0; i < NUM_POOLS; i++) {
        ucc_free(pq_mt->tasks[i]);
    }
    ucc_free(pq_mt->tasks);
    ucc_spinlock_destroy(&pq_mt->locked_queue_lock);
    ucc_free(pq_mt);
}

ucc_status_t ucc_pq_mt_init(ucc_progress_queue_t **pq) {
    ucc_pq_mt_t *pq_mt = ucc_malloc(sizeof(*pq_mt), "pq_mt");
    if (!pq_mt) {
        ucc_error("failed to allocate %zd bytes for pq_mt", sizeof(*pq_mt));
        return UCC_ERR_NO_MEMORY;
    }
    pq_mt->tasks = (ucc_coll_task_t ***) ucc_calloc(NUM_POOLS, sizeof(ucc_coll_task_t **));
    if (!pq_mt->tasks) {
        ucc_error("failed to allocate %zd bytes for pq_mt->tasks", NUM_POOLS * sizeof(ucc_coll_task_t **));
        return UCC_ERR_NO_MEMORY;
    }
    pq_mt->tasks[0] = (ucc_coll_task_t **) ucc_calloc(LINE_SIZE, sizeof(ucc_coll_task_t *));
    if (!pq_mt->tasks[0]) {
        ucc_error("failed to allocate %zd bytes for pq_mt->tasks[0]", LINE_SIZE * sizeof(ucc_coll_task_t *));
        return UCC_ERR_NO_MEMORY;
    }
    pq_mt->tasks[1] = (ucc_coll_task_t **) ucc_calloc(LINE_SIZE, sizeof(ucc_coll_task_t *));
    if (!pq_mt->tasks[1]) {
        ucc_error("failed to allocate %zd bytes for pq_mt->tasks[1]", LINE_SIZE * sizeof(ucc_coll_task_t *));
        return UCC_ERR_NO_MEMORY;
    }
    ucc_spinlock_init(&pq_mt->locked_queue_lock, 0);
    ucc_list_head_init(&pq_mt->locked_queue);
    pq_mt->which_pool = 0;
    pq_mt->tasks_countrs[0] = 0;
    pq_mt->tasks_countrs[1] = 0;
    pq_mt->super.enqueue  = ucc_pq_mt_enqueue;
    pq_mt->super.progress = ucc_pq_mt_progress;
    pq_mt->super.finalize = ucc_pq_mt_finalize;
    *pq                   = &pq_mt->super;
    return UCC_OK;
}
