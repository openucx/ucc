/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_SCHEDULE_H_
#define UCC_SCHEDULE_H_

#include "ucc/api/ucc.h"
#include "utils/ucc_list.h"
#define MAX_LISTENERS 4

typedef enum {
    UCC_EVENT_COMPLETED = 0,
    UCC_EVENT_SCHEDULE_STARTED,
    UCC_EVENT_LAST
} ucc_event_t;

typedef struct ucc_coll_task ucc_coll_task_t;

typedef ucc_status_t (*ucc_task_event_handler_p)(ucc_coll_task_t *task);
typedef ucc_status_t (*ucc_coll_post_fn_t)(ucc_coll_task_t *task);
typedef ucc_status_t (*ucc_coll_triggered_post_fn_t)(ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *task);
typedef ucc_status_t (*ucc_coll_finalize_fn_t)(ucc_coll_task_t *task);

typedef struct ucc_event_manager {
    ucc_coll_task_t *listeners[UCC_EVENT_LAST][MAX_LISTENERS];
    int              listeners_size[UCC_EVENT_LAST];
} ucc_event_manager_t;

enum {
    UCC_COLL_TASK_FLAG_INTERNAL = UCC_BIT(0)
};

typedef struct ucc_coll_task {
    ucc_coll_req_t             super;
    uint32_t                   flags;
    ucc_coll_post_fn_t         post;
    ucc_coll_triggered_post_fn_t triggered_post;
    ucc_coll_finalize_fn_t     finalize;
    ucc_event_manager_t        em;
    ucc_task_event_handler_p   handlers[UCC_EVENT_LAST];
    ucc_status_t             (*progress)(struct ucc_coll_task *self);
    struct ucc_schedule       *schedule;
    ucc_ee_h                   ee;
    ucc_ev_t                   *ev;
    void                       *ee_task;
    /* used for progress queue */
    ucc_list_link_t            list_elem;
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
ucc_status_t ucc_event_manager_notify(ucc_event_manager_t *em,
                                      ucc_event_t event);
ucc_status_t ucc_schedule_init(ucc_schedule_t *schedule, ucc_context_t *ctx);
void ucc_schedule_add_task(ucc_schedule_t *schedule, ucc_coll_task_t *task);
ucc_status_t ucc_schedule_start(ucc_schedule_t *schedule);

#endif
