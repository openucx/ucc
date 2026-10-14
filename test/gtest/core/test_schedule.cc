/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include "schedule/ucc_schedule.h"
#include "schedule/ucc_schedule_pipelined.h"
#include "core/ucc_context.h"
#include "utils/ucc_spinlock.h"
}

class test_coll_task : public ucc_coll_task_t {
public:
    test_coll_task() {
        ucc_coll_task_construct(this);
        EXPECT_EQ(UCC_OK, ucc_coll_task_init((ucc_coll_task_t *)this, NULL, NULL));
    }
    ~test_coll_task() {
        ucc_coll_task_destruct(this);
    }
};

typedef std::tuple<test_coll_task*, int> rst_t;


class test_schedule : public test_coll_task, public ucc::test
{
public:
    std::vector<rst_t> rst;
    static ucc_status_t handler_1(ucc_coll_task_t *parent,
                                  ucc_coll_task_t *task) {
        test_schedule *ts = (test_schedule*)task;
        ts->rst.push_back(rst_t((test_coll_task*)parent, 1));
        return UCC_OK;
    }
    static ucc_status_t handler_2(ucc_coll_task_t *parent,
                                  ucc_coll_task_t *task) {
        test_schedule *ts = (test_schedule*)task;
        ts->rst.push_back(rst_t((test_coll_task*)parent, 2));
        return UCC_OK;
    }
};

/* Tasks subscribes on 2 tasks to EVENT_COMPLETED with the same
   handler */
UCC_TEST_F(test_schedule, single_handler)
{
    std::vector<test_coll_task> tasks(2);

    for (auto &t :  tasks) {
        ucc_event_manager_subscribe(&t, UCC_EVENT_COMPLETED,
                                    (ucc_coll_task_t*)this,
                                    test_schedule::handler_1);
    }

    for (auto &t :  tasks) {
        EXPECT_EQ(UCC_OK, ucc_event_manager_notify(&t, UCC_EVENT_COMPLETED));
    }
    EXPECT_EQ(2, rst.size());
    EXPECT_EQ(true, (std::get<0>(rst[0]) == &tasks[0]) &&
              (std::get<1>(rst[0]) == 1));
    EXPECT_EQ(true, (std::get<0>(rst[1]) == &tasks[1]) &&
              (std::get<1>(rst[1]) == 1));
}

/* Tasks subscribes on 2 tasks to EVENT_COMPLETED with 2 different
   handlers */
UCC_TEST_F(test_schedule, different_handlers)
{
    std::vector<test_coll_task> tasks(2);

    ucc_event_manager_subscribe(&tasks[0], UCC_EVENT_COMPLETED,
                                (ucc_coll_task_t*)this,
                                test_schedule::handler_1);
    ucc_event_manager_subscribe(&tasks[1], UCC_EVENT_COMPLETED,
                                (ucc_coll_task_t*)this,
                                test_schedule::handler_2);

    for (auto &t :  tasks) {
        EXPECT_EQ(UCC_OK, ucc_event_manager_notify(&t, UCC_EVENT_COMPLETED));
    }

    EXPECT_EQ(2, rst.size());
    EXPECT_EQ(true, (std::get<0>(rst[0]) == &tasks[0]) &&
              (std::get<1>(rst[0]) == 1));
    EXPECT_EQ(true, (std::get<0>(rst[1]) == &tasks[1]) &&
              (std::get<1>(rst[1]) == 2));
}

/* Tasks subscribes to multiple tasks exceeding MAX_LISTENERS */
UCC_TEST_F(test_schedule, multiple)
{
    const int n_subscribers = 16;
    std::vector<test_coll_task> tasks(n_subscribers);

    for (int i = 0; i < n_subscribers; i++) {
        ucc_event_manager_subscribe(&tasks[i], UCC_EVENT_COMPLETED,
                                    (ucc_coll_task_t*)this,
                                    ((i % 2) == 0 ? test_schedule::handler_1
                                     : test_schedule::handler_2));
    }

    for (auto &t :  tasks) {
        EXPECT_EQ(UCC_OK, ucc_event_manager_notify(&t, UCC_EVENT_COMPLETED));
    }

    EXPECT_EQ(n_subscribers, rst.size());
    for (int i = 0; i < n_subscribers; i++) {
        EXPECT_EQ(true, (std::get<0>(rst[i]) == &tasks[i]) &&
                  (std::get<1>(rst[i]) == ((i % 2) + 1)));
    }
}

/* fragments complete inside their post, so every restart nests the handler */
typedef struct test_pipelined_schedule {
    ucc_schedule_pipelined_t super;
    int                      fail_frag_num;
    int                      mt;
    int                      n_completions;
    ucc_status_t             status;
    int                      lock_held;
} test_pipelined_schedule_t;

static void test_pipelined_completion_cb(void *data, ucc_status_t status)
{
    test_pipelined_schedule_t *t = (test_pipelined_schedule_t *)data;

    t->n_completions++;
    t->status = status;
    if (t->mt) {
        t->lock_held = ucc_recursive_spin_is_owner(
            &t->super.lock, pthread_self());
    }
}

static ucc_status_t test_pipelined_frag_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *frag = ucc_derived_of(task, ucc_schedule_t);

    for (int i = 0; i < frag->n_tasks; i++) {
        ucc_coll_task_destruct(frag->tasks[i]);
        free(frag->tasks[i]);
    }
    ucc_coll_task_destruct(&frag->super);
    free(frag);
    return UCC_OK;
}

static ucc_status_t test_pipelined_frag_init(
    ucc_base_coll_args_t *coll_args, ucc_schedule_pipelined_t *,
    ucc_base_team_t *team, ucc_schedule_t **frag_p)
{
    ucc_schedule_t  *frag = (ucc_schedule_t *)calloc(1, sizeof(*frag));
    ucc_coll_task_t *task = (ucc_coll_task_t *)calloc(1, sizeof(*task));

    ucc_coll_task_construct(&frag->super);
    ucc_coll_task_construct(task);
    EXPECT_EQ(UCC_OK, ucc_schedule_init(frag, coll_args, team));
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(task, coll_args, team));
    EXPECT_EQ(UCC_OK, ucc_schedule_add_task(frag, task));
    EXPECT_EQ(
        UCC_OK,
        ucc_task_subscribe_dep(&frag->super, task, UCC_EVENT_SCHEDULE_STARTED));
    frag->super.post     = ucc_schedule_start;
    frag->super.finalize = test_pipelined_frag_finalize;
    *frag_p              = frag;
    return UCC_OK;
}

static ucc_status_t test_pipelined_frag_setup(
    ucc_schedule_pipelined_t *sp, ucc_schedule_t *, int frag_num)
{
    test_pipelined_schedule_t *t = (test_pipelined_schedule_t *)sp;

    return (frag_num == t->fail_frag_num) ? UCC_ERR_NO_RESOURCE : UCC_OK;
}

class test_schedule_pipelined : public ucc::test {
  public:
    ucc_context_t        core_ctx;
    ucc_base_context_t   base_ctx;
    ucc_base_team_t      team;
    ucc_base_coll_args_t bargs;

    test_schedule_pipelined() : core_ctx(), base_ctx(), team(), bargs()
    {
        base_ctx.ucc_context = &core_ctx;
        team.context         = &base_ctx;
    }

    void init(
        test_pipelined_schedule_t *t, ucc_thread_mode_t tm, int depth,
        int n_frags_total, int fail_frag_num)
    {
        ucc_coll_task_t *task = &t->super.super.super;

        *t                    = test_pipelined_schedule_t();
        t->fail_frag_num      = fail_frag_num;
        t->mt                 = (tm == UCC_THREAD_MULTIPLE);
        core_ctx.thread_mode  = tm;
        ucc_coll_task_construct(task);
        ASSERT_EQ(
            UCC_OK,
            ucc_schedule_pipelined_init(
                &bargs,
                &team,
                test_pipelined_frag_init,
                test_pipelined_frag_setup,
                depth,
                n_frags_total,
                UCC_PIPELINE_PARALLEL,
                &t->super));
        task->cb.cb   = test_pipelined_completion_cb;
        task->cb.data = t;
        task->flags |= UCC_COLL_TASK_FLAG_CB;
    }

    void finalize(test_pipelined_schedule_t *t)
    {
        EXPECT_EQ(
            UCC_OK, ucc_schedule_pipelined_finalize(&t->super.super.super));
        ucc_coll_task_destruct(&t->super.super.super);
    }
};

/* schedule completes once, with the lock released */
UCC_TEST_F(test_schedule_pipelined, nested_completion)
{
    const int                 n_frags_total = 8;
    test_pipelined_schedule_t t;

    for (auto tm : {UCC_THREAD_SINGLE, UCC_THREAD_MULTIPLE}) {
        init(&t, tm, 2, n_frags_total, -1);
        for (int i = 1; i <= 2; i++) {
            ucc_schedule_pipelined_post(&t.super.super.super);
            EXPECT_EQ(i, t.n_completions);
            EXPECT_EQ(UCC_OK, t.status);
            EXPECT_EQ(UCC_OK, t.super.super.super.super.status);
            EXPECT_EQ(n_frags_total, t.super.n_frags_started);
            EXPECT_EQ(0, t.lock_held);
        }
        finalize(&t);
    }
}

/* nested restart fails: schedule fails once, with the lock released */
UCC_TEST_F(test_schedule_pipelined, nested_failure)
{
    const int                 fail_frag_num = 4;
    test_pipelined_schedule_t t;

    for (auto tm : {UCC_THREAD_SINGLE, UCC_THREAD_MULTIPLE}) {
        init(&t, tm, 2, 8, fail_frag_num);
        ucc_schedule_pipelined_post(&t.super.super.super);
        EXPECT_EQ(1, t.n_completions);
        EXPECT_EQ(UCC_ERR_NO_RESOURCE, t.status);
        EXPECT_EQ(UCC_ERR_NO_RESOURCE, t.super.super.super.super.status);
        EXPECT_EQ(fail_frag_num, t.super.n_frags_started);
        EXPECT_EQ(0, t.lock_held);
        finalize(&t);
    }
}
