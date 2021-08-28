/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "ucc_schedule.h"
#include "utils/ucc_compiler_def.h"
#include "components/base/ucc_base_iface.h"

ucc_status_t ucc_event_manager_init(ucc_event_manager_t *em)
{
    int i;
    for (i = 0; i < MAX_LISTENERS; i++) {
        em->listeners[i].task = NULL;
    }
    return UCC_OK;
}

void ucc_event_manager_subscribe(ucc_event_manager_t *em, ucc_event_t event,
                                 ucc_coll_task_t *task,
                                 ucc_task_event_handler_p handler)
{
    int i;
    for (i = 0; i < MAX_LISTENERS; i++) {
        if (!em->listeners[i].task) {
            em->listeners[i].task    = task;
            em->listeners[i].event   = event;
            em->listeners[i].handler = handler;
            break;
        }
    }
    ucc_assert(i < MAX_LISTENERS);
}

ucc_status_t ucc_coll_task_init(ucc_coll_task_t *task, ucc_coll_args_t *args,
                                ucc_base_team_t *team)
{
    task->super.status = UCC_OPERATION_INITIALIZED;
    task->ee           = NULL;
    task->flags        = 0;
    task->team         = team;
    if (args) {
        memcpy(&task->args, args, sizeof(*args));
    }
    ucc_lf_queue_init_elem(&task->lf_elem);
    return ucc_event_manager_init(&task->em);
}

static ucc_status_t
ucc_task_error_handler(ucc_coll_task_t *parent_task,
                       ucc_coll_task_t *task)
{
    ucc_event_manager_t *em = &parent_task->em;
    ucc_coll_task_t     *listener;
    int                 i;

    task->super.status = parent_task->super.status;
    for (i = 0; i < MAX_LISTENERS; i++) {
        listener = em->listeners[i].task;
        if (listener &&
            listener->super.status != parent_task->super.status) {
            /* status has not been propagated yet */
            ucc_task_error_handler(task, listener);
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_event_manager_notify(ucc_coll_task_t *parent_task,
                                      ucc_event_t event)
{
    ucc_event_manager_t *em = &parent_task->em;
    ucc_coll_task_t     *task;
    ucc_status_t        status;
    int                 i;

    for (i = 0; i < MAX_LISTENERS; i++) {
        task = em->listeners[i].task;
        if (task) {
            if (UCC_EVENT_ERROR == event) {
                ucc_task_error_handler(parent_task, task);
            } else if (em->listeners[i].event == event) {
                status = em->listeners[i].handler(parent_task, task);
                if (ucc_unlikely(status != UCC_OK)) {
                    return status;
                }
            }
        }
    }
    return UCC_OK;
}

static ucc_status_t
ucc_schedule_completed_handler(ucc_coll_task_t *parent_task, //NOLINT
                               ucc_coll_task_t *task)
{
    ucc_schedule_t *self = ucc_container_of(task, ucc_schedule_t, super);
    self->n_completed_tasks += 1;
    if (self->n_completed_tasks == self->n_tasks) {
        self->super.super.status = UCC_OK;
        ucc_task_complete(&self->super);
    }
    return UCC_OK;
}

ucc_status_t ucc_schedule_init(ucc_schedule_t *schedule, ucc_coll_args_t *args,
                               ucc_base_team_t *team)
{
    ucc_status_t status;

    status            = ucc_coll_task_init(&schedule->super, args, team);
    schedule->ctx     = team->context->ucc_context;
    schedule->n_tasks = 0;
    return status;
}

void ucc_schedule_add_task(ucc_schedule_t *schedule, ucc_coll_task_t *task)
{
    ucc_event_manager_subscribe(&task->em, UCC_EVENT_COMPLETED,
                            &schedule->super, ucc_schedule_completed_handler);
    task->schedule                       = schedule;
    schedule->tasks[schedule->n_tasks++] = task;
}

ucc_status_t ucc_schedule_start(ucc_schedule_t *schedule)
{
    schedule->n_completed_tasks  = 0;
    schedule->super.super.status = UCC_INPROGRESS;
    return ucc_event_manager_notify(&schedule->super,
                                    UCC_EVENT_SCHEDULE_STARTED);
}

ucc_status_t ucc_task_start_handler(ucc_coll_task_t *parent, /* NOLINT */
                                    ucc_coll_task_t *task)
{
    return task->post(task);
}

ucc_status_t ucc_schedule_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule       = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status_overall = UCC_OK;
    ucc_status_t    status;
    int             i;

    for (i = 0; i < schedule->n_tasks; i++) {
        if (schedule->tasks[i]->finalize) {
            status = schedule->tasks[i]->finalize(schedule->tasks[i]);
            if (UCC_OK != status) {
                status_overall = status;
            }
        }
    }
    return status_overall;
}
