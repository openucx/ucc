/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_SCHEDULE_H_
#define UCC_SCHEDULE_H_

#include "ucc/api/ucc.h"
#include "utils/ucc_list.h"
#include "utils/ucc_log.h"
#include "utils/ucc_lock_free_queue.h"

#define MAX_LISTENERS 4

typedef enum {
    UCC_EVENT_COMPLETED = 0,
    UCC_EVENT_SCHEDULE_STARTED,
    UCC_EVENT_ERROR,
    UCC_EVENT_LAST
} ucc_event_t;

typedef struct ucc_coll_task ucc_coll_task_t;

typedef ucc_status_t (*ucc_task_event_handler_p)(ucc_coll_task_t *parent,
                                                 ucc_coll_task_t *task);
typedef ucc_status_t (*ucc_coll_post_fn_t)(ucc_coll_task_t *task);
typedef ucc_status_t (*ucc_coll_triggered_post_fn_t)(ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *task);
typedef ucc_status_t (*ucc_coll_finalize_fn_t)(ucc_coll_task_t *task);

typedef struct ucc_event_manager {
    ucc_coll_task_t *listeners[UCC_EVENT_LAST][MAX_LISTENERS];
    int              listeners_size[UCC_EVENT_LAST];
} ucc_event_manager_t;

enum {
    UCC_COLL_TASK_FLAG_INTERNAL = UCC_BIT(0),
    UCC_COLL_TASK_FLAG_CB       = UCC_BIT(1)
};

typedef struct ucc_coll_task {
    ucc_coll_req_t               super;
    uint32_t                     flags;
    ucc_coll_post_fn_t           post;
    ucc_coll_triggered_post_fn_t triggered_post;
    ucc_coll_finalize_fn_t       finalize;
    ucc_coll_callback_t          cb;
    ucc_event_manager_t          em;
    ucc_task_event_handler_p     handlers[UCC_EVENT_LAST];
    ucc_status_t               (*progress)(struct ucc_coll_task *self);
    struct ucc_schedule         *schedule;
    ucc_ee_h                     ee;
    ucc_ev_t                    *ev;
    void                        *ee_task;
    ucc_coll_task_t             *triggered_task;
    union {
        /* used for st & locked mt progress queue */
        ucc_list_link_t              list_elem;
        /* used for lf mt progress queue */
        ucc_lf_queue_elem_t          lf_elem;
    };
} ucc_coll_task_t;

typedef struct ucc_context ucc_context_t;
typedef struct ucc_schedule {
    ucc_coll_task_t super;
    int             n_completed_tasks;
    int             n_tasks;
    ucc_context_t  *ctx;
} ucc_schedule_t;

ucc_status_t ucc_event_manager_init(ucc_event_manager_t *em);
ucc_status_t ucc_coll_task_init(ucc_coll_task_t *task);
void ucc_event_manager_subscribe(ucc_event_manager_t *em, ucc_event_t event,
                                 ucc_coll_task_t *task);
ucc_status_t ucc_event_manager_notify(ucc_coll_task_t *parent_task,
                                      ucc_event_t event);
ucc_status_t ucc_schedule_init(ucc_schedule_t *schedule, ucc_context_t *ctx);
void ucc_schedule_add_task(ucc_schedule_t *schedule, ucc_coll_task_t *task);
ucc_status_t ucc_schedule_start(ucc_schedule_t *schedule);
ucc_status_t ucc_task_start_handler(ucc_coll_task_t *parent,
                                    ucc_coll_task_t *task);

static inline ucc_status_t ucc_task_error(ucc_coll_task_t *task)
{
    ucc_status_t status = task->super.status;

    ucc_assert(status < 0);
    ucc_error("failure in task %p, %s", task, ucc_status_string(status));
    return ucc_event_manager_notify(task, UCC_EVENT_ERROR);
}

static inline ucc_status_t ucc_task_complete(ucc_coll_task_t *task)
{
    ucc_status_t status = task->super.status;
    ucc_assert((status == UCC_OK) || (status < 0));
    if (ucc_likely(status == UCC_OK)) {
        status = ucc_event_manager_notify(task, UCC_EVENT_COMPLETED);
    } else {
        /* error in task status */
        ucc_error("failure in task %p, %s", task,
                  ucc_status_string(task->super.status));
    }

    if (task->flags & UCC_COLL_TASK_FLAG_CB) {
        task->cb.cb(task->cb.data, status);
    }

    if (task->flags & UCC_COLL_TASK_FLAG_INTERNAL) {
        task->finalize(task);
    }
    return status;
}
#endif
