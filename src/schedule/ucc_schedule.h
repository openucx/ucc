/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
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
#include "components/mc/ucc_mc.h"

#define MAX_LISTENERS 4

typedef enum {
    UCC_EVENT_COMPLETED = 0,
    UCC_EVENT_SCHEDULE_STARTED,
    UCC_EVENT_TASK_STARTED,
    UCC_EVENT_COMPLETED_SCHEDULE, /*< Event is used to notify SCHEDULE that
                                    one of its task has completed */
    UCC_EVENT_ERROR,
    UCC_EVENT_LAST
} ucc_event_t;

typedef struct ucc_coll_task ucc_coll_task_t;

typedef struct ucc_schedule ucc_schedule_t;

typedef struct ucc_base_team ucc_base_team_t;

typedef ucc_status_t (*ucc_coll_post_fn_t)(ucc_coll_task_t *task);

typedef void (*ucc_coll_progress_fn_t)(ucc_coll_task_t *task);

typedef ucc_status_t (*ucc_coll_finalize_fn_t)(ucc_coll_task_t *task);

typedef ucc_status_t (*ucc_task_event_handler_p)(ucc_coll_task_t *parent,
                                                 ucc_coll_task_t *task);

/* triggered post setup function will be launched before starting executor */
typedef ucc_status_t (*ucc_coll_triggered_post_setup_fn_t)(ucc_coll_task_t *task);

typedef ucc_status_t (*ucc_coll_triggered_post_fn_t)(ucc_ee_h ee, ucc_ev_t *ev,
                                                     ucc_coll_task_t *task);

typedef struct ucc_em_listener {
    ucc_coll_task_t          *task;
    ucc_task_event_handler_p  handler;
    ucc_event_t               event;
} ucc_em_listener_t;

typedef struct ucc_event_manager {
    ucc_list_link_t   list_elem;
    unsigned          n_listeners;
    ucc_em_listener_t listeners[MAX_LISTENERS];
} ucc_event_manager_t;

enum {
    UCC_COLL_TASK_FLAG_CB                    = UCC_BIT(0),
    /* executor is required for collective*/
    UCC_COLL_TASK_FLAG_EXECUTOR              = UCC_BIT(1),
    /* user visible task */
    UCC_COLL_TASK_FLAG_TOP_LEVEL             = UCC_BIT(2),
    /* stop executor in task complete*/
    UCC_COLL_TASK_FLAG_EXECUTOR_STOP         = UCC_BIT(3),
    /* destroy executor in task complete */
    UCC_COLL_TASK_FLAG_EXECUTOR_DESTROY      = UCC_BIT(4),
    /* if set task can be casted to scheulde */
    UCC_COLL_TASK_FLAG_IS_SCHEDULE           = UCC_BIT(5),
    /* if set task can be casted to scheulde */
    UCC_COLL_TASK_FLAG_IS_PIPELINED_SCHEDULE = UCC_BIT(6),

};

typedef struct ucc_coll_task {
    ucc_coll_req_t                     super;
    /**
     *  Task internal status, TLs and CLs should use it to track collective
     *  state. super.status is visible to user and should be updated only
     *  by core level to avoid potential races
     */
    ucc_status_t                       status;
    ucc_list_link_t                    em_list;
    ucc_base_coll_args_t               bargs;
    ucc_base_team_t                   *team; /* CL/TL team pointer */
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
    ucc_coll_task_t                   *triggered_task;
    ucc_ee_executor_t                 *executor;
    union {
        /* used for st & locked mt progress queue */
        ucc_list_link_t                list_elem;
        /* used for lf mt progress queue */
        ucc_lf_queue_elem_t            lf_elem;
    };
    uint32_t                           n_deps;
    uint32_t                           n_deps_satisfied;
    uint32_t                           n_deps_base;
    /* timestamp of the start time: either post or triggered_post */
    double                             start_time;
    uint32_t                           seq_num;
} ucc_coll_task_t;

extern struct ucc_mpool_ops ucc_coll_task_mpool_ops;
typedef struct ucc_context ucc_context_t;

#define UCC_SCHEDULE_MAX_TASKS 8

typedef struct ucc_schedule {
    ucc_coll_task_t  super;
    uint32_t         n_completed_tasks;
    uint32_t         n_tasks;
    ucc_context_t   *ctx;
    ucc_coll_task_t *tasks[UCC_SCHEDULE_MAX_TASKS];
} ucc_schedule_t;

void ucc_coll_task_construct(ucc_coll_task_t *task);

void ucc_coll_task_destruct(ucc_coll_task_t *task);

ucc_status_t ucc_coll_task_init(ucc_coll_task_t *task,
                                ucc_base_coll_args_t *args,
                                ucc_base_team_t *team);

ucc_status_t ucc_coll_task_get_executor(ucc_coll_task_t *task,
                                        ucc_ee_executor_t **exec);

ucc_status_t ucc_event_manager_subscribe(ucc_coll_task_t *parent_task,
                                         ucc_event_t event,
                                         ucc_coll_task_t *task,
                                         ucc_task_event_handler_p handler);

ucc_status_t ucc_event_manager_notify(ucc_coll_task_t *parent_task,
                                      ucc_event_t event);

ucc_status_t ucc_schedule_init(ucc_schedule_t *schedule,
                               ucc_base_coll_args_t *bargs,
                               ucc_base_team_t *team);

ucc_status_t ucc_schedule_add_task(ucc_schedule_t *schedule,
                                   ucc_coll_task_t *task);

ucc_status_t ucc_schedule_start(ucc_coll_task_t *task);

ucc_status_t ucc_task_start_handler(ucc_coll_task_t *parent,
                                    ucc_coll_task_t *task);
ucc_status_t ucc_schedule_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_dependency_handler(ucc_coll_task_t *parent,
                                    ucc_coll_task_t *task);

ucc_status_t ucc_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                ucc_coll_task_t *task);

static inline ucc_status_t ucc_task_complete(ucc_coll_task_t *task)
{
    ucc_status_t        status    = task->status;
    ucc_coll_callback_t cb        = task->cb;
    int                 has_cb    = task->flags & UCC_COLL_TASK_FLAG_CB;
    int                 has_sched = task->schedule != NULL;

    ucc_assert((status == UCC_OK) || (status < 0));

    /* If task is part of a schedule then it can be
       released during ucc_event_manager_notify(EVENT_COMPLETED_SCHEDULE) below.
       Sequence: notify => schedule->n_completed_tasks++ =>
       schedule->super.status = UCC_OK => user releases schedule from another
       thread => schedule_finalize => schedule finalizes all the tasks.
       After that the task ptr should not be accessed.
       This is why notification of schedule is done separately in the end of
       this function. Internal implementation must make sure that tasks
       with schedules are not released during a callback (if set). */

    if (ucc_likely(status == UCC_OK)) {
        ucc_buffer_info_asymmetric_memtype_t *save = &task->bargs.asymmetric_save_info;
        if (save->scratch &&
            task->bargs.args.coll_type != UCC_COLL_TYPE_SCATTERV &&
            task->bargs.args.coll_type != UCC_COLL_TYPE_SCATTER) {
            status = ucc_copy_asymmetric_buffer(task);
            if (status != UCC_OK) {
                ucc_error("failure copying out asymmetric buffer: %s",
                          ucc_status_string(status));
            }
        }
        status = ucc_event_manager_notify(task, UCC_EVENT_COMPLETED);
    } else {
        /* error in task status */
        if (UCC_ERR_TIMED_OUT == status) {
            char coll_str[256];
            ucc_coll_str(task, coll_str, sizeof(coll_str),
                         UCC_LOG_LEVEL_DEBUG);
            ucc_warn("timeout %g sec. has expired on %s",
                     task->bargs.args.timeout, coll_str);
        } else {
            ucc_error("failure in task %p, %s", task,
                      ucc_status_string(task->status));
        }
        ucc_event_manager_notify(task, UCC_EVENT_ERROR);
    }

    if ((task->executor) && (task->flags & UCC_COLL_TASK_FLAG_EXECUTOR_STOP)) {
        status = ucc_ee_executor_stop(task->executor);
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_error("failed to stop executor %s", ucc_status_string(status));
        }
    }

    if ((task->executor) && (task->flags & UCC_COLL_TASK_FLAG_EXECUTOR_DESTROY)) {
        status = ucc_ee_executor_finalize(task->executor);
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_error("failed to finalize executor %s",
                      ucc_status_string(status));
        }
        task->executor = NULL;
    }

    task->super.status = status;
    if (has_cb) {
        cb.cb(cb.data, status);
    }

    if (has_sched && status == UCC_OK) {
        status = ucc_event_manager_notify(task, UCC_EVENT_COMPLETED_SCHEDULE);
    }

    return status;
}

static inline ucc_status_t ucc_task_subscribe_dep(ucc_coll_task_t *target,
                                                  ucc_coll_task_t *subscriber,
                                                  ucc_event_t event)
{
    ucc_status_t status =
    ucc_event_manager_subscribe(target, event, subscriber,
                                ucc_dependency_handler);
    subscriber->n_deps++;
    return status;
}

#define UCC_TASK_LIB(_task) (((ucc_coll_task_t *)_task)->team->context->lib)
#define UCC_TASK_CORE_CTX(_task)                                               \
    (((ucc_coll_task_t *)_task)->team->context->ucc_context)

#define UCC_TASK_THREAD_MODE(_task) (UCC_TASK_CORE_CTX(_task)->thread_mode)
#endif
