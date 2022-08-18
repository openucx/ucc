/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#include "ucc_schedule.h"
#include "utils/ucc_compiler_def.h"
#include "components/base/ucc_base_iface.h"
#include "coll_score/ucc_coll_score.h"
#include "core/ucc_context.h"

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

ucc_status_t ucc_coll_task_init(ucc_coll_task_t *task,
                                ucc_base_coll_args_t *bargs,
                                ucc_base_team_t *team)
{
    task->flags                = 0;
    task->ee                   = NULL;
    task->team                 = team;
    task->n_deps               = 0;
    task->n_deps_satisfied     = 0;
    task->bargs.args.mask      = 0;
    task->schedule             = NULL;
    task->executor             = NULL;
    task->super.status         = UCC_OPERATION_INITIALIZED;
    task->triggered_post_setup = NULL;
    if (bargs) {
        memcpy(&task->bargs, bargs, sizeof(*bargs));
    }
    ucc_lf_queue_init_elem(&task->lf_elem);
    return ucc_event_manager_init(&task->em);
}

ucc_status_t ucc_coll_task_get_executor(ucc_coll_task_t *task,
                                        ucc_ee_executor_t **exec)
{
    ucc_status_t st = UCC_OK;

    if (task->executor == NULL) {
        if (ucc_unlikely(!task->schedule)) {
            ucc_error("executor wasn't initialized for the collective");
            return UCC_ERR_INVALID_PARAM;
        }
        st = ucc_coll_task_get_executor(&task->schedule->super,
                                        &task->executor);
    }

    *exec = task->executor;
    return st;
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
            if (ucc_unlikely(event == UCC_EVENT_ERROR)) {
                ucc_task_error_handler(parent_task, task);
                continue;
            }
            if (em->listeners[i].event == event) {
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
    uint32_t        n_completed_tasks;

    n_completed_tasks = ucc_atomic_fadd32(&self->n_completed_tasks, 1);

    if (n_completed_tasks + 1 == self->n_tasks) {
        self->super.status = UCC_OK;
        ucc_task_complete(&self->super);
    }
    return UCC_OK;
}

ucc_status_t ucc_schedule_init(ucc_schedule_t *schedule,
                               ucc_base_coll_args_t *bargs,
                               ucc_base_team_t *team)
{
    ucc_status_t status;

    status            = ucc_coll_task_init(&schedule->super, bargs, team);
    schedule->ctx     = team->context->ucc_context;
    schedule->n_tasks = 0;
    return status;
}

void ucc_schedule_add_task(ucc_schedule_t *schedule, ucc_coll_task_t *task)
{
    ucc_event_manager_subscribe(&task->em, UCC_EVENT_COMPLETED_SCHEDULE,
                                &schedule->super,
                                ucc_schedule_completed_handler);
    task->schedule                       = schedule;
    schedule->tasks[schedule->n_tasks++] = task;
    if (task->flags & UCC_COLL_TASK_FLAG_EXECUTOR) {
        schedule->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    }
}

ucc_status_t ucc_schedule_start(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);

    schedule->n_completed_tasks  = 0;
    schedule->super.status       = UCC_INPROGRESS;
    schedule->super.super.status = UCC_INPROGRESS;
    return ucc_event_manager_notify(&schedule->super,
                                    UCC_EVENT_SCHEDULE_STARTED);
}

ucc_status_t ucc_task_start_handler(ucc_coll_task_t *parent,
                                    ucc_coll_task_t *task)
{
    task->start_time = parent->start_time;
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
