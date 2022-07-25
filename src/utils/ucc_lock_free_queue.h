/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_LOCKFREE_QUEUE_H_
#define UCC_LOCKFREE_QUEUE_H_

#include "utils/ucc_spinlock.h"
#include "utils/ucc_atomic.h"
#include "utils/ucc_list.h"
#include <string.h>

/* This data structure is thread safe */

// Number of elements in a single lock free pool - could be changed, but in tests performed great
#define LINE_SIZE 8
#define NUM_POOLS 2

typedef struct ucc_lf_queue_elem {
    uint8_t         was_queued;
    ucc_list_link_t locked_list_elem;
} ucc_lf_queue_elem_t;

typedef struct ucc_lf_queue {
    ucc_spinlock_t       locked_queue_lock[NUM_POOLS];
    ucc_lf_queue_elem_t *elements[NUM_POOLS][LINE_SIZE];
    uint8_t              which_pool;
    ucc_list_link_t      locked_queue[NUM_POOLS];
} ucc_lf_queue_t;

static inline void ucc_lf_queue_init_elem(ucc_lf_queue_elem_t *elem){
    elem->was_queued = 0;
}

static inline void ucc_lf_queue_enqueue(ucc_lf_queue_t *     queue,
                                        ucc_lf_queue_elem_t *elem)
{
    uint8_t which_pool = elem->was_queued ^ (queue->which_pool & 1);
    int     i;
    for (i = 0; i < LINE_SIZE; i++) {
        if (ucc_atomic_bool_cswap64(
                (uint64_t *)&(queue->elements[which_pool][i]), 0LL,
                (uint64_t)elem)) {
            return;
        }
    }
    ucc_spin_lock(&queue->locked_queue_lock[which_pool]);
    ucc_list_add_tail(&queue->locked_queue[which_pool],
                      &elem->locked_list_elem);
    ucc_spin_unlock(&queue->locked_queue_lock[which_pool]);
}

static inline ucc_lf_queue_elem_t *ucc_lf_queue_dequeue(ucc_lf_queue_t *queue,
                                                        int is_first_call)
{
    // Save value in the beginning of the function
    uint8_t curr_which_pool = queue->which_pool;
    uint8_t which_pool      = curr_which_pool & 1; // turn from even/odd -> bool
    int     i;
    ucc_lf_queue_elem_t *elem;
    for (i = 0; i < LINE_SIZE; i++) {
        elem = queue->elements[which_pool][i];
        if (elem) {
            if (ucc_atomic_bool_cswap64(
                    (uint64_t *)&(queue->elements[which_pool][i]),
                    (uint64_t)elem, 0LL)) {
                elem->was_queued = 1;
                return elem;
            }
        }
    }
    elem = NULL;
    ucc_spin_lock(&queue->locked_queue_lock[which_pool]);
    if (!ucc_list_is_empty(&queue->locked_queue[which_pool])) {
        elem = ucc_list_extract_head(&queue->locked_queue[which_pool],
                                     ucc_lf_queue_elem_t, locked_list_elem);
        elem->was_queued = 1;
    }
    ucc_spin_unlock(&queue->locked_queue_lock[which_pool]);
    if (!elem) {
        if (ucc_atomic_bool_cswap8(&queue->which_pool, curr_which_pool,
                curr_which_pool + 1) && is_first_call) {
            return ucc_lf_queue_dequeue(queue, 0);
        }
    }
    return elem;
}

static inline void ucc_lf_queue_destroy(ucc_lf_queue_t *queue)
{
    ucc_spinlock_destroy(&queue->locked_queue_lock[0]);
    ucc_spinlock_destroy(&queue->locked_queue_lock[1]);
}

static inline void ucc_lf_queue_init(ucc_lf_queue_t *queue)
{
    memset(&queue->elements, 0,
           NUM_POOLS * LINE_SIZE * sizeof(ucc_lf_queue_elem_t *));
    ucc_spinlock_init(&queue->locked_queue_lock[0], 0);
    ucc_spinlock_init(&queue->locked_queue_lock[1], 0);
    ucc_list_head_init(&queue->locked_queue[0]);
    ucc_list_head_init(&queue->locked_queue[1]);
    queue->which_pool = 0;
}

#endif
