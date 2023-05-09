/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_schedule.h"
#include "ucc_schedule_pipelined.h"
#include "coll_score/ucc_coll_score.h"
#include "core/ucc_context.h"

const char* ucc_pipeline_order_names[] = {
    [UCC_PIPELINE_PARALLEL]   = "parallel",
    [UCC_PIPELINE_ORDERED]    = "ordered",
    [UCC_PIPELINE_SEQUENTIAL] = "sequential",
    [UCC_PIPELINE_LAST]       =  NULL
};

static ucc_status_t ucc_frag_start_handler(ucc_coll_task_t *parent,
                                           ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *schedule = ucc_derived_of(parent,
                                                        ucc_schedule_pipelined_t);
    ucc_schedule_t           *frag     = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t st;

    task->start_time = parent->start_time;
    if (schedule->frag_setup) {
        st = schedule->frag_setup(schedule, frag, schedule->n_frags_started);
        if (ucc_unlikely(UCC_OK != st)) {
            ucc_error("failed to setup fragment %d of pipelined schedule",
                      schedule->n_frags_started);
            return st;
        }
    }

    schedule->next_frag_to_post = (schedule->next_frag_to_post + 1) %
                                  schedule->n_frags;
    ucc_trace_req("sched %p started frag %p frag_num %d next_to_post %d",
                  schedule, frag, schedule->n_frags_started,
                  schedule->next_frag_to_post);
    schedule->n_frags_started++;
    schedule->n_frags_in_pipeline++;
    return task->post(task);
}

static ucc_status_t
ucc_schedule_pipelined_completed_handler(ucc_coll_task_t *parent_task,
                                         ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *schedule =
        ucc_container_of(task, ucc_schedule_pipelined_t, super);
    ucc_schedule_t *frag = ucc_derived_of(parent_task, ucc_schedule_t);
    int             i;

    if (UCC_TASK_THREAD_MODE(task) == UCC_THREAD_MULTIPLE) {
        ucc_recursive_spin_lock(&schedule->lock);
    }

    schedule->super.n_completed_tasks += 1;
    schedule->n_frags_in_pipeline--;
    ucc_trace_req(
        "sched %p completed frag %p, n_completed %d, n_started %d, n_total %d",
        schedule, frag, schedule->super.n_completed_tasks,
        schedule->n_frags_started, schedule->super.n_tasks);
    if (schedule->super.n_completed_tasks == schedule->super.n_tasks) {
        schedule->super.super.status = UCC_OK;
        if (UCC_TASK_THREAD_MODE(task) == UCC_THREAD_MULTIPLE) {
            ucc_recursive_spin_unlock(&schedule->lock);
        }
        ucc_task_complete(task);
        return UCC_OK;
    }
    while ((schedule->super.n_completed_tasks + schedule->n_frags_in_pipeline <
            schedule->super.n_tasks) &&
           (frag->super.status == UCC_OK)) {
        /* need to post more frags*/
        if (frag == schedule->frags[schedule->next_frag_to_post]) {
            ucc_trace_req("sched %p restarting frag %d %p", schedule,
                          schedule->next_frag_to_post, frag);
            frag->super.status = UCC_OPERATION_INITIALIZED;
            frag->n_completed_tasks  = 0;
            for (i = 0; i < frag->n_tasks; i++) {
                frag->tasks[i]->n_deps += frag->tasks[i]->n_deps_base;
                frag->tasks[i]->status = UCC_OPERATION_INITIALIZED;
            }
            ucc_frag_start_handler(&schedule->super.super, &frag->super);
        }
        frag = schedule->frags[schedule->next_frag_to_post];
        if (&frag->super == parent_task) {
            break;
        }
    }
    if (UCC_TASK_THREAD_MODE(task) == UCC_THREAD_MULTIPLE) {
        ucc_recursive_spin_unlock(&schedule->lock);
    }
    return UCC_OK;
}

ucc_status_t ucc_schedule_pipelined_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *schedule_p =
        ucc_derived_of(task, ucc_schedule_pipelined_t);
    ucc_schedule_t **frags = schedule_p->frags;
    int              i;

    ucc_trace_req("schedule pipelined %p is complete", schedule_p);
    for (i = 0; i < schedule_p->n_frags; i++) {
        schedule_p->frags[i]->super.finalize(&frags[i]->super);
    }

    if (UCC_TASK_THREAD_MODE(task) == UCC_THREAD_MULTIPLE) {
        ucc_recursive_spinlock_destroy(&schedule_p->lock);
    }

    return UCC_OK;
}

ucc_status_t ucc_schedule_pipelined_post(ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *schedule_p =
        ucc_derived_of(task, ucc_schedule_pipelined_t);
    ucc_schedule_t **frags = schedule_p->frags;
    int              i, j;

    schedule_p->super.super.super.status = UCC_OPERATION_INITIALIZED;
    schedule_p->super.n_completed_tasks  = 0;
    schedule_p->n_frags_started          = 0;
    schedule_p->next_frag_to_post        = 0;
    schedule_p->n_frags_in_pipeline      = 0;

    for (i = 0; i < schedule_p->n_frags; i++) {
        frags[i]->n_completed_tasks  = 0;
        frags[i]->super.super.status = UCC_OPERATION_INITIALIZED;
        for (j = 0; j < frags[0]->n_tasks; j++) {
            frags[i]->tasks[j]->n_deps = frags[i]->tasks[j]->n_deps_base;
            frags[i]->tasks[j]->n_deps_satisfied = 0;
            frags[i]->tasks[j]->super.status     = UCC_OPERATION_INITIALIZED;
            if (i == 0 && schedule_p->n_frags > 1 &&
                UCC_PIPELINE_PARALLEL != schedule_p->order) {
                frags[0]->tasks[j]->n_deps_satisfied++;
            }
        }
    }

    return ucc_schedule_start(task);
}

ucc_status_t ucc_schedule_pipelined_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *team,
                                         ucc_schedule_frag_init_fn_t frag_init,
                                         ucc_schedule_frag_setup_fn_t frag_setup,
                                         int n_frags, int n_frags_total,
                                         ucc_pipeline_order_t order,
                                         ucc_schedule_pipelined_t *schedule)
{
    ucc_event_t      task_dependency_event = UCC_EVENT_LAST;
    int              i, j;
    ucc_status_t     status;
    ucc_schedule_t **frags;

    if (ucc_unlikely(n_frags > UCC_SCHEDULE_PIPELINED_MAX_FRAGS)) {
        ucc_error("n_frags %d exceeds max limit of %d",
                  n_frags, UCC_SCHEDULE_PIPELINED_MAX_FRAGS);
        return UCC_ERR_INVALID_PARAM;
    }

    if (n_frags > 1) {
        /* determine dependency between frags */
        switch (order) {
            case UCC_PIPELINE_PARALLEL:
                /* no dependency between tasks of different fragments */
                task_dependency_event = UCC_EVENT_LAST;
                break;
            case UCC_PIPELINE_ORDERED:
                /* next fragment starts when previous has started */
                task_dependency_event = UCC_EVENT_TASK_STARTED;
                break;
            case UCC_PIPELINE_SEQUENTIAL:
                /* next fragment starts when previous has completed */
                task_dependency_event = UCC_EVENT_COMPLETED;
                break;
            default:
                return UCC_ERR_INVALID_PARAM;
        }
    }

    status = ucc_schedule_init(&schedule->super, coll_args, team);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("failed to init pipelined schedule");
        return status;
    }

    if (UCC_TASK_THREAD_MODE(&schedule->super.super) == UCC_THREAD_MULTIPLE) {
        ucc_recursive_spinlock_init(&schedule->lock, 0);
    }

    schedule->super.super.flags    |= UCC_COLL_TASK_FLAG_IS_PIPELINED_SCHEDULE;
    schedule->super.n_tasks        = n_frags_total;
    schedule->n_frags              = n_frags;
    schedule->order                = order;
    schedule->frag_setup           = frag_setup;
    schedule->next_frag_to_post    = 0;
    schedule->n_frags_in_pipeline  = 0;
    schedule->super.super.finalize = ucc_schedule_pipelined_finalize;
    schedule->super.super.post     = ucc_schedule_pipelined_post;
    frags                          = schedule->frags;
    for (i = 0; i < n_frags; i++) {
        status = frag_init(coll_args, schedule, team, &frags[i]);
        if (ucc_unlikely(UCC_OK != status)) {
            ucc_error("failed to initialize fragment for pipeline");
            goto err;
        }
        frags[i]->super.schedule = &schedule->super;
        if (frags[i]->super.flags & UCC_COLL_TASK_FLAG_EXECUTOR) {
            schedule->super.super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
        }
        frags[i]->super.status       = UCC_OPERATION_INITIALIZED;
        frags[i]->super.super.status = UCC_OPERATION_INITIALIZED;
    }

    for (i = 0; i < n_frags; i++) {
        for (j = 0; j < frags[i]->n_tasks; j++) {
            frags[i]->tasks[j]->n_deps_base = frags[i]->tasks[j]->n_deps;
            if (task_dependency_event != UCC_EVENT_LAST) {
                UCC_CHECK_GOTO(
                    ucc_event_manager_subscribe(
                        frags[(i + n_frags - 1) % n_frags]->tasks[j],
                        task_dependency_event, frags[i]->tasks[j],
                        ucc_dependency_handler),
                    err, status);
                frags[i]->tasks[j]->n_deps_base++;
            }
        }
        UCC_CHECK_GOTO(ucc_event_manager_subscribe(
                           &schedule->super.super, UCC_EVENT_SCHEDULE_STARTED,
                           &frags[i]->super, ucc_frag_start_handler),
                       err, status);
        UCC_CHECK_GOTO(ucc_event_manager_subscribe(
                           &frags[i]->super, UCC_EVENT_COMPLETED_SCHEDULE,
                           &schedule->super.super,
                           ucc_schedule_pipelined_completed_handler),
                       err, status);
    }
    return UCC_OK;
err:
    for (i = i - 1; i >= 0; i--) {
        frags[i]->super.finalize(&frags[i]->super);
    }
    return status;
}

ucc_status_t ucc_dependency_handler(ucc_coll_task_t *parent,
                                    ucc_coll_task_t *task)
{
    ucc_status_t status;
    uint8_t      n_deps_satisfied;

    n_deps_satisfied = ucc_atomic_fadd8(&task->n_deps_satisfied, 1);

    ucc_trace_req("task %p, n_deps %d, satisfied %d", task, task->n_deps,
                  n_deps_satisfied);
    if (task->n_deps == n_deps_satisfied + 1) {
        task->start_time = parent->start_time;
        status = task->post(task);
        if (status >= 0) {
            ucc_event_manager_notify(task, UCC_EVENT_TASK_STARTED);
        }
        return status;
    }

    return UCC_OK;
}
