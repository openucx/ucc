/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "ucc_schedule.h"
#include "utils/ucc_compiler_def.h"

ucc_status_t ucc_event_manager_init(ucc_event_manager_t *em)
{
    int i;
    for (i = 0; i < UCC_EVENT_LAST; i++) {
        em->listeners_size[i] = 0;
    }
    return UCC_OK;
}

void ucc_event_manager_subscribe(ucc_event_manager_t *em, ucc_event_t event,
                                 ucc_coll_task_t *task)
{
    ucc_assert(em->listeners_size[event] < MAX_LISTENERS);
    em->listeners[event][em->listeners_size[event]] = task;
    em->listeners_size[event]++;
}

ucc_status_t ucc_coll_task_init(ucc_coll_task_t *task)
{
    task->super.status   = UCC_OPERATION_INITIALIZED;
    task->ee             = NULL;
    task->flags          = 0;
    ucc_lf_queue_init_elem(&task->lf_elem);
    return ucc_event_manager_init(&task->em);
}

ucc_status_t ucc_event_manager_notify(ucc_coll_task_t *parent_task,
                                      ucc_event_t event)
{
    ucc_event_manager_t *em = &parent_task->em;
    ucc_coll_task_t     *task;
    ucc_status_t        status;
    int                 i;

    for (i = 0; i < em->listeners_size[event]; i++) {
        task   = em->listeners[event][i];
        status = task->handlers[event](parent_task, task);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
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

static ucc_status_t
ucc_schedule_error_handler(ucc_coll_task_t *parent_task, //NOLINT
                           ucc_coll_task_t *task)
{
    task->super.status = parent_task->super.status;
    return ucc_event_manager_notify(task, UCC_EVENT_ERROR);
}

ucc_status_t ucc_schedule_init(ucc_schedule_t *schedule, ucc_context_t *ctx)
{
    ucc_status_t status;
    status = ucc_coll_task_init(&schedule->super);
    schedule->super.handlers[UCC_EVENT_COMPLETED] =
        ucc_schedule_completed_handler;
    schedule->super.handlers[UCC_EVENT_ERROR] =
        ucc_schedule_error_handler;
    schedule->n_completed_tasks = 0;
    schedule->ctx               = ctx;
    schedule->n_tasks           = 0;
    return status;
}

void ucc_schedule_add_task(ucc_schedule_t *schedule, ucc_coll_task_t *task)
{
    ucc_event_manager_subscribe(&task->em, UCC_EVENT_COMPLETED,
                                &schedule->super);
    ucc_event_manager_subscribe(&task->em, UCC_EVENT_ERROR,
                                &schedule->super);
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
