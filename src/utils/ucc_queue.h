/**
* Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
*
* See file LICENSE for terms.
*/

#ifndef UCC_QUEUE_H_
#define UCC_QUEUE_H_

#include <stddef.h>
#include <assert.h>

typedef struct ucc_queue_elem ucc_queue_elem_t;
typedef struct ucc_queue_head ucc_queue_head_t;
typedef ucc_queue_elem_t**    ucc_queue_iter_t;


/**
 * Queue element type.
 */
struct ucc_queue_elem {
    ucc_queue_elem_t    *next;
};


/**
 * Queue type.
 */
struct ucc_queue_head {
    ucc_queue_elem_t    *head;
    ucc_queue_elem_t    **ptail;
};

/**
 * Initialize a queue.
 *
 * @param queue  Queue to initialize.
 */
static inline void ucc_queue_head_init(ucc_queue_head_t *queue)
{
#ifdef __clang_analyzer__
    queue->head  = (ucc_queue_elem_t*)(void*)queue;
#endif
    queue->ptail = &queue->head;
}

/**
 * @return Queue length.
 */
static inline size_t ucc_queue_length(ucc_queue_head_t *queue)
{
    ucc_queue_elem_t **pelem;
    size_t length;

    length = 0;
    for (pelem = &queue->head; pelem != queue->ptail; pelem = &(*pelem)->next) {
        ++length;
    }
    return length;
}

/**
 * @return Whether the queue is empty.
 */
static inline int ucc_queue_is_empty(ucc_queue_head_t *queue)
{
    return queue->ptail == &queue->head;
}

/**
 * Enqueue an element to the tail of the queue.
 *
 * @param queue  Queue to add to.
 * @param elem   Element to add.
 */
static inline void ucc_queue_push(ucc_queue_head_t *queue, ucc_queue_elem_t *elem)
{
    *queue->ptail = elem;
    queue->ptail = &elem->next;
#if UCC_ENABLE_ASSERT
    elem->next = NULL; /* For sanity check below */
#endif
}

/**
 * Add an element to the head of the queue.
 *
 * @param queue  Queue to add to.
 * @param elem   Element to add.
 */
static inline void ucc_queue_push_head(ucc_queue_head_t *queue,
                                       ucc_queue_elem_t *elem)
{
    elem->next = queue->head;
    queue->head = elem;
    if (queue->ptail == &queue->head) {
        queue->ptail = &elem->next;
    }
}

/**
 * Dequeue an element from the head of the queue, assuming the queue is not empty.
 *
 * @param queue  Non-empty queue to pull from.
 * @return  Element from the head of the queue.
 */
static inline ucc_queue_elem_t *ucc_queue_pull_non_empty(ucc_queue_head_t *queue)
{
    ucc_queue_elem_t *elem;

    elem = queue->head;
    queue->head = elem->next;
    if (queue->ptail == &elem->next) {
        queue->ptail = &queue->head;
    }
    return elem;
}

/**
 * Delete an element.
 * The element must be valid when deleting it.
 * After the call, iter points to the next element, and the element may be released.
 */
static inline void ucc_queue_del_iter(ucc_queue_head_t *queue, ucc_queue_iter_t iter)
{
    assert((iter != NULL) && (*iter != NULL));

    if (queue->ptail == &(*iter)->next) {
        queue->ptail = iter; /* deleting the last element */
        *iter = NULL;        /* make *ptail point to NULL */
    } else {
        *iter = (*iter)->next;
    }

}

/**
 * Dequeue an element from the head of the queue.
 *
 * @param queue  Queue to pull from.
 * @return  Element from the head of the queue, or NULL if the queue is empty.
 */
static inline ucc_queue_elem_t *ucc_queue_pull(ucc_queue_head_t *queue)
{
    if (ucc_queue_is_empty(queue))
        return NULL;
    return ucc_queue_pull_non_empty(queue);
}

/**
 * Insert all elements from one queue to another queue, leaving the first queue
 * empty.
 *
 * @param queue     Queue to push elements to.
 * @param new_elems Queue of elements to add.
 */
static inline void ucc_queue_splice(ucc_queue_head_t *queue,
                                    ucc_queue_head_t *new_elems)
{
    if (!ucc_queue_is_empty(new_elems)) {
        *queue->ptail = new_elems->head;
        queue->ptail = new_elems->ptail;
        new_elems->ptail = &new_elems->head;
    }
}

/**
 * Convenience macro to pull from a non-empty queue and return the containing element.
 *
 * @param queue   Non-empty queue to pull from.
 * @param type    Container element type.
 * @param member  Queue element member inside the container.
 *
 * @return Pulled element.
 */
#define ucc_queue_pull_elem_non_empty(queue, type, member) \
    ucc_container_of(ucc_queue_pull_non_empty(queue), type, member)

/**
 * Convenience macro to get the head element of a non-empty queue.
 *
 * @param queue   Non-empty queue whose head element to get.
 * @param type    Container element type.
 * @param member  Queue element member inside the container.
 *
 * @return Head element.
 */
#define ucc_queue_head_elem_non_empty(queue, type, member) \
    ucc_container_of((queue)->head, type, member)

/**
 * Convenience macro to get the tail element of a non-empty queue.
 *
 * @param queue   Non-empty queue whose head element to get.
 * @param type    Container element type.
 * @param member  Queue element member inside the container.
 *
 * @return Head element.
 */
#define ucc_queue_tail_elem_non_empty(queue, type, member) \
    ucc_container_of((queue)->ptail, type, member)

/**
 * Iterate over queue elements. The queue must not be modified during the iteration.
 *
 * @param elem    Variable which will hold point to the element in the queue.
 * @param queue   Queue to iterate on.
 * @param member  Member inside 'elem' which is the queue link.
 */
#define ucc_queue_for_each(elem, queue, member) \
    /* we set `ptail` field to queue address to not subtract NULL pointer */ \
    for (*(queue)->ptail = (ucc_queue_elem_t*)(void*)(queue), \
             elem = ucc_container_of((queue)->head, typeof(*elem), member); \
         (UCC_PTR_BYTE_OFFSET(elem, ucc_offsetof(typeof(*elem), member)) != \
             (void*)(queue)); \
         elem = ucc_container_of(elem->member.next, typeof(*elem), member))

/**
 * Iterate over queue elements. The current element may be safely removed from
 * the queue using ucc_queue_del_iter().
 *
 * @param elem    Variable which will hold point to the element in the queue.
 * @param iter    Iterator variable. May be passed to ucc_queue_del_iter().
 * @param queue   Queue to iterate on.
 * @param member  Member inside 'elem' which is the queue link.
 */
#define ucc_queue_for_each_safe(elem, iter, queue, member) \
    for (iter = &(queue)->head, \
         elem = ucc_container_of(*iter, typeof(*elem), member); \
          iter != (queue)->ptail; \
          iter = (*iter == &elem->member) ? &(*iter)->next : iter, \
            elem = ucc_container_of(*iter, typeof(*elem), member))

/**
 * Iterate and extract elements from the queue while a condition is true.
 *
 * @param elem    Variable which will hold point to the element in the queue.
 * @param queue   Queue to iterate on.
 * @param member  Member inside 'elem' which is the queue link.
 * @param cond    Condition to continue iterating.
 *
 * TODO optimize
 */
#define ucc_queue_for_each_extract(elem, queue, member, cond) \
    for (elem = ucc_container_of((queue)->head, typeof(*elem), member); \
         \
         !ucc_queue_is_empty(queue) && (cond) && ucc_queue_pull_non_empty(queue); \
         \
         elem = ucc_container_of((queue)->head, typeof(*elem), member))


/*
 * Queue iteration
 */

static inline ucc_queue_iter_t ucc_queue_iter_begin(ucc_queue_head_t *q)
{
    return &q->head;
}

static inline ucc_queue_iter_t ucc_queue_iter_next(ucc_queue_iter_t i)
{
    return  &(*i)->next;
}

static inline int ucc_queue_iter_end(ucc_queue_head_t *q, ucc_queue_iter_t i)
{
    return i == q->ptail;
}

static inline void ucc_queue_remove(ucc_queue_head_t *queue, ucc_queue_elem_t *elem)
{
    ucc_queue_iter_t iter = ucc_queue_iter_begin(queue);

    while (!ucc_queue_iter_end(queue, iter)) {
        if (*iter == elem) {
            ucc_queue_del_iter(queue, iter);
            return;
        }
        iter = ucc_queue_iter_next(iter);
    }
}

#define ucc_queue_iter_elem(elem, iter, member) \
    ucc_container_of(*iter, typeof(*elem), member)

#endif
