/**
 * Copyright (C) Mellanox Technologies Ltd. 2021-2022.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_SCHEDULE_H_
#define UCC_SCHEDULE_H_

#include "ucc/api/ucc.h"
#include "utils/ucc_list.h"
#include "utils/ucc_log.h"
#include "utils/ucc_lock_free_queue.h"
#include "utils/ucc_coll_utils.h"
#include "components/base/ucc_base_iface.h"
#include "components/ec/ucc_ec.h"

#define MAX_LISTENERS 4

typedef enum {
    UCC_EVENT_COMPLETED = 0,
    UCC_EVENT_SCHEDULE_STARTED,
    UCC_EVENT_TASK_STARTED,
    UCC_EVENT_ERROR,
    UCC_EVENT_LAST
} ucc_event_t;

typedef struct ucc_coll_task ucc_coll_task_t;

typedef struct ucc_schedule ucc_schedule_t;

typedef struct ucc_base_team ucc_base_team_t;

typedef ucc_status_t (*ucc_coll_post_fn_t)(ucc_coll_task_t *task);

typedef ucc_status_t (*ucc_coll_progress_fn_t)(ucc_coll_task_t *task);

typedef ucc_status_t (*ucc_coll_finalize_fn_t)(ucc_coll_task_t *task);

typedef ucc_status_t (*ucc_task_event_handler_p)(ucc_coll_task_t *parent,
                                                 ucc_coll_task_t *task);

/* triggered post setup function will be launched before starting executor */
typedef ucc_status_t (*ucc_coll_triggered_post_setup_fn_t)(ucc_coll_task_t *task);

typedef ucc_status_t (*ucc_coll_triggered_post_fn_t)(ucc_ee_h ee,
                                                     ucc_ev_t *ev,
                                                     ucc_coll_task_t *task);

typedef struct ucc_em_listener {
    ucc_coll_task_t          *task;
    ucc_task_event_handler_p  handler;
    ucc_event_t               event;
} ucc_em_listener_t;

typedef struct ucc_event_manager {
    ucc_em_listener_t listeners[MAX_LISTENERS];
} ucc_event_manager_t;

enum {
    UCC_COLL_TASK_FLAG_INTERNAL      = UCC_BIT(0),
    UCC_COLL_TASK_FLAG_CB            = UCC_BIT(1),
    /* executor is required for collective*/
    UCC_COLL_TASK_FLAG_EXECUTOR      = UCC_BIT(2),
    /* user visible task */
    UCC_COLL_TASK_FLAG_TOP_LEVEL     = UCC_BIT(3),
    /* stop executor in task complete*/
    UCC_COLL_TASK_FLAG_EXECUTOR_STOP = UCC_BIT(4)
};

typedef struct ucc_coll_task {
    ucc_coll_req_t                     super;
    ucc_event_manager_t                em;
    ucc_base_coll_args_t               bargs;
    ucc_base_team_t                   *team; //CL/TL team pointer
    ucc_schedule_t                    *schedule;
    uint32_t                           flags;
    ucc_coll_post_fn_t                 post;
    ucc_coll_triggered_post_setup_fn_t triggered_post_setup;
    ucc_coll_triggered_post_fn_t       triggered_post;
    ucc_coll_progress_fn_t             progress;
    ucc_coll_finalize_fn_t             finalize;
    ucc_coll_callback_t                cb;
    ucc_ee_h                           ee;
    ucc_ev_t                          *ev;
    void                              *ee_task;
    ucc_coll_task_t                   *triggered_task;
    ucc_ee_executor_t                 *executor;
    union {
        /* used for st & locked mt progress queue */
        ucc_list_link_t              list_elem;
        /* used for lf mt progress queue */
        ucc_lf_queue_elem_t          lf_elem;
    };
    uint8_t  n_deps;
    uint8_t  n_deps_satisfied;
    uint8_t  n_deps_base;
    double   start_time; /* timestamp of the start time:
                            either post or triggered_post */
    uint32_t seq_num;
} ucc_coll_task_t;

typedef struct ucc_context ucc_context_t;

#define UCC_SCHEDULE_MAX_TASKS 8

typedef struct ucc_schedule {
    ucc_coll_task_t  super;
    int              n_completed_tasks;
    int              n_tasks;
    ucc_context_t   *ctx;
    ucc_coll_task_t *tasks[UCC_SCHEDULE_MAX_TASKS];
} ucc_schedule_t;

ucc_status_t ucc_event_manager_init(ucc_event_manager_t *em);

ucc_status_t ucc_coll_task_init(ucc_coll_task_t *task,
                                ucc_base_coll_args_t *args,
                                ucc_base_team_t *team);

ucc_status_t ucc_coll_task_get_executor(ucc_coll_task_t *task,
                                        ucc_ee_executor_t **exec);

void ucc_event_manager_subscribe(ucc_event_manager_t *em, ucc_event_t event,
                                 ucc_coll_task_t *task,
                                 ucc_task_event_handler_p handler);

ucc_status_t ucc_event_manager_notify(ucc_coll_task_t *parent_task,
                                      ucc_event_t event);

ucc_status_t ucc_schedule_init(ucc_schedule_t *schedule,
                               ucc_base_coll_args_t *bargs,
                               ucc_base_team_t *team);

void ucc_schedule_add_task(ucc_schedule_t *schedule, ucc_coll_task_t *task);

ucc_status_t ucc_schedule_start(ucc_schedule_t *schedule);

ucc_status_t ucc_task_start_handler(ucc_coll_task_t *parent,
                                    ucc_coll_task_t *task);
ucc_status_t ucc_schedule_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_dependency_handler(ucc_coll_task_t *parent, /* NOLINT */
                                    ucc_coll_task_t *task);

ucc_status_t ucc_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                ucc_coll_task_t *task);

static inline ucc_status_t ucc_task_complete(ucc_coll_task_t *task)
{
    ucc_status_t status = task->super.status;

    ucc_trace("task complete %p", task);
    ucc_assert((status == UCC_OK) || (status < 0));
    if (ucc_likely(status == UCC_OK)) {
        status = ucc_event_manager_notify(task, UCC_EVENT_COMPLETED);
    } else {
        /* error in task status */
        if (UCC_ERR_TIMED_OUT == status) {
            char coll_str[256];
            ucc_coll_str(task, coll_str, sizeof(coll_str));
            ucc_warn("timeout %g sec has expired on %s",
                     task->bargs.args.timeout, coll_str);
        } else {
            ucc_error("failure in task %p, %s", task,
                      ucc_status_string(task->super.status));
        }
        ucc_assert(task->super.status < 0);
        ucc_event_manager_notify(task, UCC_EVENT_ERROR);
        status = task->super.status;
    }

    if (task->executor && (task->flags & UCC_COLL_TASK_FLAG_EXECUTOR_STOP)) {
        status = ucc_ee_executor_stop(task->executor);
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_error("failed to stop executor %s", ucc_status_string(status));
        }
    }

    if (task->flags & UCC_COLL_TASK_FLAG_CB) {
        task->cb.cb(task->cb.data, status);
    }

    if (task->flags & UCC_COLL_TASK_FLAG_INTERNAL) {
        task->finalize(task);
    }

    return status;
}

static inline void ucc_task_subscribe_dep(ucc_coll_task_t *target,
                                          ucc_coll_task_t *subscriber,
                                          ucc_event_t      event)
{
    ucc_event_manager_subscribe(&target->em, event, subscriber,
                                ucc_dependency_handler);
    subscriber->n_deps++;
}

#define UCC_TASK_LIB(_task) (((ucc_coll_task_t *)_task)->team->context->lib)
#define UCC_TASK_CORE_CTX(_task)                                               \
    (((ucc_coll_task_t *)_task)->team->context->ucc_context)

#endif
