/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_progress_queue.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"

typedef struct ucc_pq_st {
    ucc_progress_queue_t super;
    ucc_list_link_t      list;
} ucc_pq_st_t;

static int ucc_pq_st_progress(ucc_progress_queue_t *pq)
{
    ucc_pq_st_t     *pq_st        = ucc_derived_of(pq, ucc_pq_st_t);
    int              n_progressed = 0;
    ucc_coll_task_t *task, *tmp;
    ucc_status_t     status;

    ucc_list_for_each_safe(task, tmp, &pq_st->list, list_elem) {
        if (task->progress) { //TODO maybe dummy empty progress fn is better than branch?
            status = task->progress(task);
            if (status < 0) {
                return status;
            }
        }
        if (UCC_OK == task->super.status) {
            n_progressed++;
            status = ucc_event_manager_notify(&task->em, UCC_EVENT_COMPLETED);
            if (status != UCC_OK) {
                return status;
            }
            ucc_list_del(&task->list_elem);
        }
    }
    return n_progressed;
}

static void ucc_pq_st_enqueue(ucc_progress_queue_t *pq, ucc_coll_task_t *task)
{
    ucc_pq_st_t *pq_st = ucc_derived_of(pq, ucc_pq_st_t);
    ucc_list_add_tail(&pq_st->list, &task->list_elem);
}

static void ucc_pq_st_finalize(ucc_progress_queue_t *pq)
{
    ucc_pq_st_t *pq_st = ucc_derived_of(pq, ucc_pq_st_t);
    ucc_free(pq_st);
}

ucc_status_t ucc_pq_st_init(ucc_progress_queue_t **pq)
{
    ucc_pq_st_t *pq_st = ucc_malloc(sizeof(*pq_st), "pq_st");
    if (!pq_st) {
        ucc_error("failed to allocate %zd bytes for pq_st", sizeof(*pq_st));
        return UCC_ERR_NO_MEMORY;
    }
    ucc_list_head_init(&pq_st->list);
    pq_st->super.enqueue  = ucc_pq_st_enqueue;
    pq_st->super.progress = ucc_pq_st_progress;
    pq_st->super.finalize = ucc_pq_st_finalize;
    *pq                   = &pq_st->super;
    return UCC_OK;
}
