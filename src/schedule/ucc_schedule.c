/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#include "ucc_schedule.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_mpool.h"
#include "components/base/ucc_base_iface.h"
#include "coll_score/ucc_coll_score.h"
#include "core/ucc_context.h"

static void ucc_coll_task_mpool_obj_init(ucc_mpool_t *mp, void *obj, //NOLINT
                                         void *chunk) //NOLINT
{
    ucc_coll_task_t *task = obj;

    ucc_coll_task_construct(task);
}

static void ucc_coll_task_mpool_obj_cleanup(ucc_mpool_t *mp, void *obj) //NOLINT
{
    ucc_coll_task_t *task = obj;

    ucc_coll_task_destruct(task);
}

struct ucc_mpool_ops ucc_coll_task_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_coll_task_mpool_obj_init,
    .obj_cleanup   = ucc_coll_task_mpool_obj_cleanup
};

static ucc_status_t ucc_event_manager_init(ucc_coll_task_t *task)
{
    ucc_event_manager_t *em;

    ucc_list_for_each(em, &task->em_list, list_elem) {
        em->n_listeners = 0;
    }
    return UCC_OK;
}

ucc_status_t ucc_event_manager_subscribe(ucc_coll_task_t *parent_task,
                                         ucc_event_t event, ucc_coll_task_t *task,
                                         ucc_task_event_handler_p handler)
{
    ucc_event_manager_t *em;

    ucc_list_for_each(em, &parent_task->em_list, list_elem) {
        if (em->n_listeners < MAX_LISTENERS) {
        set:
            em->listeners[em->n_listeners].task    = task;
            em->listeners[em->n_listeners].event   = event;
            em->listeners[em->n_listeners].handler = handler;
            em->n_listeners++;
            return UCC_OK;
        }
    }
    em              = ucc_malloc(sizeof(*em), "em");
    if (ucc_unlikely(!em)) {
        ucc_error("failed to allocate %zd bytes for event_manager", sizeof(*em));
        return UCC_ERR_NO_MEMORY;
    }
    em->n_listeners = 0;
    ucc_list_add_tail(&parent_task->em_list, &em->list_elem);
    goto set;
}

void ucc_coll_task_construct(ucc_coll_task_t *task)
{
    ucc_list_head_init(&task->em_list);
}

void ucc_coll_task_destruct(ucc_coll_task_t *task)
{
    ucc_event_manager_t *em, *m;

    ucc_list_for_each_safe(em, m, &task->em_list, list_elem) {
        ucc_list_del(&em->list_elem);
        free(em);
    }
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
    task->triggered_post       = ucc_triggered_post;
    if (bargs) {
        memcpy(&task->bargs, bargs, sizeof(*bargs));
    }
    ucc_lf_queue_init_elem(&task->lf_elem);
    return ucc_event_manager_init(task);
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

static ucc_status_t ucc_task_error_handler(ucc_coll_task_t *parent_task,
                                           ucc_coll_task_t *task)
{
    ucc_event_manager_t *em;
    ucc_coll_task_t     *listener;
    int                  i;

    task->super.status = parent_task->super.status;

    ucc_list_for_each(em, &parent_task->em_list, list_elem) {
        for (i = 0; i < em->n_listeners; i++) {
            listener = em->listeners[i].task;
            if (listener->super.status != parent_task->super.status) {
                /* status has not been propagated yet */
                ucc_task_error_handler(task, listener);
            }
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_event_manager_notify(ucc_coll_task_t *parent_task,
                                      ucc_event_t event)
{
    ucc_event_manager_t *em;
    ucc_coll_task_t     *task;
    ucc_status_t         status;
    int                  i;

    ucc_list_for_each(em, &parent_task->em_list, list_elem) {
        for (i = 0; i < em->n_listeners; i++) {
            task = em->listeners[i].task;
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

    status                = ucc_coll_task_init(&schedule->super, bargs, team);
    schedule->super.flags |= UCC_COLL_TASK_FLAG_IS_SCHEDULE;
    schedule->ctx         = team->context->ucc_context;
    schedule->n_tasks     = 0;
    return status;
}

ucc_status_t ucc_schedule_add_task(ucc_schedule_t *schedule,
                                   ucc_coll_task_t *task)
{
    ucc_status_t status;

    status = ucc_event_manager_subscribe(task, UCC_EVENT_COMPLETED_SCHEDULE,
                                         &schedule->super,
                                         ucc_schedule_completed_handler);
    task->schedule                       = schedule;
    schedule->tasks[schedule->n_tasks++] = task;
    if (task->flags & UCC_COLL_TASK_FLAG_EXECUTOR) {
        schedule->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    }
    return status;
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
