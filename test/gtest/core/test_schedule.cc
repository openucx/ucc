/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include <atomic>
#include <common/test.h>

extern "C" {
#include "schedule/ucc_schedule.h"
#include "schedule/ucc_schedule.h"
#include "schedule/ucc_schedule_pipelined.h"
#include "core/ucc_context.h"
#include "components/base/ucc_base_iface.h"
#include "utils/ucc_malloc.h"
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
    /* The fault-injection callbacks (subscribe fault cb, lock observer) are
       process-wide globals that schedule code reads from any thread. The
       fault_* tests install both; restore them to NULL after every test so
       later, unrelated schedules in this binary never invoke test
       instrumentation. */
    void TearDown() override {
        ucc_event_manager_set_subscribe_fault_cb(NULL);
        ucc_schedule_pipelined_set_lock_observer(NULL);
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


/* ------------------------------------------------------------------ */
/* Minimal synthetic team/context hierarchy                           */
/* ------------------------------------------------------------------ */

static ucc_base_team_t *create_test_team(int thread_mode)
{
    ucc_context_t      *ctx;
    ucc_base_context_t *base_ctx;
    ucc_base_team_t    *team;

    ctx = (ucc_context_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }
    ctx->thread_mode = (ucc_thread_mode_t)thread_mode;

    base_ctx = (ucc_base_context_t *)calloc(1, sizeof(*base_ctx));
    if (!base_ctx) {
        free(ctx);
        return NULL;
    }
    base_ctx->ucc_context = ctx;

    team = (ucc_base_team_t *)calloc(1, sizeof(*team));
    if (!team) {
        free(base_ctx);
        free(ctx);
        return NULL;
    }
    team->context = base_ctx;
    return team;
}

static void destroy_test_team(ucc_base_team_t *team)
{
    if (!team) {
        return;
    }
    free(team->context->ucc_context);
    free(team->context);
    free(team);
}

/* ------------------------------------------------------------------ */
/* Failure injection and finalization tracking                        */
/* ------------------------------------------------------------------ */

static std::atomic<int> g_frag_init_count; /* successful frag_init calls */
static std::atomic<int> g_total_finalizes; /* total finalize() invocations */
static int
    g_fail_after_frag_idx; /* fail at this fragment init index (-1 = never) */

/* Finalize wrappers that increment the global counter. The task finalizer
 * frees the synthetic task; the schedule finalizer delegates to the real
 * ucc_schedule_finalize() (which finalizes and frees its tasks). */
static ucc_status_t count_task_finalize(ucc_coll_task_t *task)
{
    g_total_finalizes++;
    ucc_coll_task_destruct(task);
    free(task);
    return UCC_OK;
}

static ucc_status_t count_schedule_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *frag = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;
    g_total_finalizes++;
    status = ucc_schedule_finalize(task);
    ucc_coll_task_destruct(task);
    free(frag);
    return status;
}

/* Fragments with controlled init failure injection. */
static ucc_status_t mock_frag_init(
    ucc_base_coll_args_t *coll_args, ucc_schedule_pipelined_t *schedule_p,
    ucc_base_team_t *team, ucc_schedule_t **frag_p)
{
    int              my_idx = std::atomic_load(&g_frag_init_count);
    ucc_status_t     status;
    ucc_schedule_t  *frag;
    ucc_coll_task_t *task;

    (void)schedule_p;

    /* Inject failure: if we should fail at this init index, return error. */
    if (my_idx == g_fail_after_frag_idx) {
        return UCC_ERR_NO_MEMORY;
    }
    std::atomic_fetch_add(&g_frag_init_count, 1);

    frag = (ucc_schedule_t *)calloc(1, sizeof(*frag));
    if (!frag) {
        return UCC_ERR_NO_MEMORY;
    }

    ucc_coll_task_construct(&frag->super);
    status = ucc_schedule_init(frag, coll_args, team);
    if (status != UCC_OK) {
        free(frag);
        return status;
    }

    /* Add one synthetic task per fragment. */
    task = (ucc_coll_task_t *)calloc(1, sizeof(*task));
    if (!task) {
        ucc_schedule_finalize(&frag->super);
        free(frag);
        return UCC_ERR_NO_MEMORY;
    }
    task->post = [](ucc_coll_task_t *t) -> ucc_status_t {
        (void)t;
        return UCC_OK;
    };
    task->finalize = count_task_finalize;
    ucc_coll_task_construct(task);
    frag->tasks[frag->n_tasks++] = task;

    frag->super.finalize = count_schedule_finalize;
    *frag_p              = frag;
    return UCC_OK;
}

/* ------------------------------------------------------------------ */
/* Tests                                                              */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Test: successful depth-16 initialization (verifies new MAX_FRAGS=16)   */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipelined_init_depth_16_success)
{
    ucc_base_coll_args_t     bargs = {0};
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_status_t             status;

    g_frag_init_count.store(0);
    g_total_finalizes.store(0);
    g_fail_after_frag_idx = -1; /* no failure */

    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    memset(&bargs, 0, sizeof(bargs));
    bargs.args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    bargs.args.src.info.count    = 1024;
    bargs.args.src.info.datatype = UCC_DT_INT32;
    bargs.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    bargs.args.dst.info.count    = 1024;
    bargs.args.dst.info.datatype = UCC_DT_INT32;
    bargs.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }

    /* Request depth-16 pipeline (matches UCC_SCHEDULE_PIPELINED_MAX_FRAGS). */
    status = ucc_schedule_pipelined_init(
        &bargs,
        team,
        mock_frag_init,
        NULL,
        16,
        16,
        UCC_PIPELINE_PARALLEL,
        &sched);
    EXPECT_EQ(UCC_OK, status) << "Depth-16 init should succeed";

    /* All 16 fragments should have been finalized by the pipelined_finalize. */
    /* Trigger explicit finalization to exercise cleanup path. */
    if (sched.super.super.finalize) {
        sched.super.super.finalize(&sched.super.super);
    }
    destroy_test_team(team);
}

/* ------------------------------------------------------------------ */
/* Test: init failure at first fragment (index 0) — no leak              */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipelined_init_failure_at_zero)
{
    ucc_base_coll_args_t     bargs = {0};
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_status_t             status;

    g_frag_init_count.store(0);
    g_total_finalizes.store(0);
    g_fail_after_frag_idx = 0; /* fail on first frag init */

    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    memset(&bargs, 0, sizeof(bargs));
    bargs.args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    bargs.args.src.info.count    = 1024;
    bargs.args.src.info.datatype = UCC_DT_INT32;
    bargs.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    bargs.args.dst.info.count    = 1024;
    bargs.args.dst.info.datatype = UCC_DT_INT32;
    bargs.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }

    /* Request n_frags=8 so that failure at index 0 is well-tested. */
    status = ucc_schedule_pipelined_init(
        &bargs,
        team,
        mock_frag_init,
        NULL,
        8,
        8,
        UCC_PIPELINE_PARALLEL,
        &sched);
    EXPECT_NE(UCC_OK, status)
        << "Init should fail when frag 0 is forced to error";

    /* Zero fragments were initialized (g_frag_init_count stayed at 0),
       so zero finalizers should have been called. */
    EXPECT_EQ(0, std::atomic_load(&g_total_finalizes))
        << "No fragments init'd => no finalizers";

    destroy_test_team(team);
}

/* ------------------------------------------------------------------ */
/* Test: init failure at middle fragment (index 3 of 8)                  */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipelined_init_failure_at_middle)
{
    ucc_base_coll_args_t     bargs = {0};
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_status_t             status;

    g_frag_init_count.store(0);
    g_total_finalizes.store(0);
    g_fail_after_frag_idx         = 3; /* fail on 4th frag (index 3) */

    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    memset(&bargs, 0, sizeof(bargs));
    bargs.args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    bargs.args.src.info.count    = 1024;
    bargs.args.src.info.datatype = UCC_DT_INT32;
    bargs.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    bargs.args.dst.info.count    = 1024;
    bargs.args.dst.info.datatype = UCC_DT_INT32;
    bargs.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }

    status = ucc_schedule_pipelined_init(
        &bargs,
        team,
        mock_frag_init,
        NULL,
        8,
        8,
        UCC_PIPELINE_PARALLEL,
        &sched);
    EXPECT_NE(UCC_OK, status) << "Init should fail at frag index 3";

    /* Fragments 0-2 were initialized (g_frag_init_count = 3).
       Each fragment has 1 task, and count_finalize is set for both the
       schedule AND its task. When finalize is called on the schedule, it
       finalizes all tasks too, so we get n_frags * 2 finalizations. */
    EXPECT_EQ(6, std::atomic_load(&g_total_finalizes))
        << "3 fragments with 1 task each => 3*2=6 finalizers (schedule + task)";

    destroy_test_team(team);
}

/* ------------------------------------------------------------------ */
/* Test: init failure at last fragment (index 7 of 8)                    */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipelined_init_failure_at_last)
{
    ucc_base_coll_args_t     bargs = {0};
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_status_t             status;

    g_frag_init_count.store(0);
    g_total_finalizes.store(0);
    g_fail_after_frag_idx         = 7; /* fail on last frag (index 7 of 8) */

    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    memset(&bargs, 0, sizeof(bargs));
    /* bargs.mask intentionally zero — ucc_coll_task_init copies struct via memcpy. */
    bargs.args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    bargs.args.src.info.count    = 1024;
    bargs.args.src.info.datatype = UCC_DT_INT32;
    bargs.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    bargs.args.dst.info.count    = 1024;
    bargs.args.dst.info.datatype = UCC_DT_INT32;
    bargs.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }

    status = ucc_schedule_pipelined_init(
        &bargs,
        team,
        mock_frag_init,
        NULL,
        8,
        8,
        UCC_PIPELINE_PARALLEL,
        &sched);
    EXPECT_NE(UCC_OK, status) << "Init should fail at last fragment";
    /* Fragments 0-6 were initialized (g_frag_init_count = 7).
       Each fragment has 1 task, and count_finalize is set for both the
       schedule AND its task. When finalize is called on the schedule, it
       finalizes all tasks too, so we get n_frags * 2 finalizations. */
    EXPECT_EQ(14, std::atomic_load(&g_total_finalizes))
        << "7 fragments with 1 task each => 7*2=14 finalizers (schedule + "
           "task)";

    destroy_test_team(team);
}
/* ------------------------------------------------------------------ */
/* Test: subscription failure scenario (simulated via frag init)       */
/* This verifies that when errors occur after fragments are initialized,*/
/* the error unwind path correctly finalizes all successfully created   */
/* fragments. Event subscription failures would hit this same path.     */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipelined_init_failure_middle_of_two)
{
    ucc_base_coll_args_t     bargs = {0};
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_status_t             status;

    g_frag_init_count.store(0);
    g_total_finalizes.store(0);
    g_fail_after_frag_idx = 1; /* fail on second frag (index 1 of 2) */

    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    memset(&bargs, 0, sizeof(bargs));
    bargs.args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    bargs.args.src.info.count    = 1024;
    bargs.args.src.info.datatype = UCC_DT_INT32;
    bargs.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    bargs.args.dst.info.count    = 1024;
    bargs.args.dst.info.datatype = UCC_DT_INT32;
    bargs.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }

    /* Request n_frags=2 - minimal case where subscription would occur */
    status = ucc_schedule_pipelined_init(
        &bargs,
        team,
        mock_frag_init,
        NULL,
        2,
        2,
        UCC_PIPELINE_PARALLEL,
        &sched);
    EXPECT_NE(UCC_OK, status) << "Init should fail at frag index 1";
    /* Exactly 1 fragment was initialized (index 0).
       Each fragment has 1 task, and count_finalize is set for both the
       schedule AND its task. When finalize is called on the schedule, it
       finalizes all tasks too, so we get n_frags * 2 finalizations. */
    EXPECT_EQ(2, std::atomic_load(&g_total_finalizes))
        << "1 fragment with 1 task => 1*2=2 finalizers (schedule + task)";

    destroy_test_team(team);
}
/* ------------------------------------------------------------------ */
/* Test: ORDERED pipeline with mid-initialization failure               */
/* Verifies error unwind works correctly when task dependencies exist.  */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipelined_ordered_failure_at_three)
{
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_base_coll_args_t     bargs = {0};
    ucc_status_t             status;

    g_frag_init_count.store(0);
    g_total_finalizes.store(0);
    g_fail_after_frag_idx = 3; /* fail on 4th frag (index 3 of 6) */

    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    memset(&bargs, 0, sizeof(bargs));
    bargs.args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    bargs.args.src.info.count    = 2048;
    bargs.args.src.info.datatype = UCC_DT_INT64;
    bargs.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    bargs.args.dst.info.count    = 2048;
    bargs.args.dst.info.datatype = UCC_DT_INT64;
    bargs.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }

    /* ORDERED pipeline: fragments have task-level dependencies. */
    status = ucc_schedule_pipelined_init(
        &bargs, team, mock_frag_init, NULL, 6, 6, UCC_PIPELINE_ORDERED, &sched);
    EXPECT_NE(UCC_OK, status) << "Init should fail at frag index 3";
    /* Exactly 3 fragments were initialized (indices 0,1,2).
       Each fragment has 1 task, and count_finalize is set for both the
       schedule AND its task. When finalize is called on the schedule, it
       finalizes all tasks too, so we get n_frags * 2 finalizations. */
    EXPECT_EQ(6, std::atomic_load(&g_total_finalizes))
        << "3 fragments with 1 task each => 3*2=6 finalizers (schedule + task)";

    destroy_test_team(team);
}

/* ------------------------------------------------------------------ */
/* Test: n_frags=0 must be rejected at the init entry guard               */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipelined_nfrags_zero_rejected)
{
    ucc_base_team_t *team = NULL;
    ucc_status_t     status;

    /* n_frags=0 means disabled pipeline - should be rejected for active use */
    status = ucc_schedule_pipelined_init(
        NULL, team, NULL, NULL, 0, 1, UCC_PIPELINE_PARALLEL, NULL);
    EXPECT_EQ(UCC_ERR_INVALID_PARAM, status)
        << "n_frags=0 must be rejected for active pipeline";
}

/* ------------------------------------------------------------------ */
/* Test: n_frags_total < n_frags should be rejected                    */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipelined_nfrags_total_lt_nfrags_rejected)
{
    ucc_base_team_t *team;
    ucc_status_t     status;

    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }

    /* n_frags_total=3 < n_frags=5 is inconsistent - should fail */
    status = ucc_schedule_pipelined_init(
        NULL, team, NULL, NULL, 5, 3, UCC_PIPELINE_PARALLEL, NULL);
    EXPECT_EQ(UCC_ERR_INVALID_PARAM, status)
        << "n_frags_total < n_frags must be rejected";

    destroy_test_team(team);
}
/* ------------------------------------------------------------------ */
/* Test: n_frags_total=0 should be rejected                            */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipelined_nfrags_total_zero_rejected)
{
    ucc_base_team_t *team;
    ucc_status_t     status;

    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }

    /* n_frags_total=0 means no fragments - should be rejected */
    status = ucc_schedule_pipelined_init(
        NULL, team, NULL, NULL, 1, 0, UCC_PIPELINE_PARALLEL, NULL);
    EXPECT_EQ(UCC_ERR_INVALID_PARAM, status)
        << "n_frags_total=0 must be rejected";

    destroy_test_team(team);
}

UCC_TEST_F(test_schedule, pipelined_invalid_order_rejected)
{
    ucc_base_coll_args_t     bargs = {0};
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_status_t             status;

    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    memset(&bargs, 0, sizeof(bargs));
    bargs.args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    bargs.args.src.info.count    = 2048;
    bargs.args.src.info.datatype = UCC_DT_INT64;
    bargs.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    bargs.args.dst.info.count    = 2048;
    bargs.args.dst.info.datatype = UCC_DT_INT64;
    bargs.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }

    /* Invalid order value - should fail at order validation */
    status = ucc_schedule_pipelined_init(
        &bargs,
        team,
        mock_frag_init,
        NULL,
        2,
        2,
        (ucc_pipeline_order_t)99,
        &sched);
    EXPECT_NE(UCC_OK, status) << "Invalid order must be rejected";

    destroy_test_team(team);
}

/* ------------------------------------------------------------------ */
/* Test: direct pdepth=0 rejection via ucc_pipeline_nfrags_pdepth()    */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipeline_params_pdepth_zero_rejected)
{
    ucc_pipeline_params_t params         = {0};
    int                   n_frags        = 77;
    int                   pipeline_depth = 88;
    ucc_status_t          status;

    /* Configure an active pipeline (n_frags > 0) with pdepth=0 */
    params.n_frags   = 4; /* Active: requesting multiple fragments */
    params.pdepth    = 0; /* Invalid: zero depth will cause hang */
    params.frag_size = 1024;
    params.threshold = SIZE_MAX;

    status = ucc_pipeline_nfrags_pdepth(
        &params, 2048, &n_frags, &pipeline_depth);
    EXPECT_EQ(UCC_ERR_INVALID_PARAM, status)
        << "Active pipeline with pdepth=0 must be rejected by helper";
    EXPECT_EQ(77, n_frags) << "failure must not publish fragment count";
    EXPECT_EQ(88, pipeline_depth) << "failure must not publish usable depth";
}

/* ------------------------------------------------------------------ */
/* Test: direct frag_size=0 rejection when msgsize > threshold         */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipeline_params_frag_size_zero_rejected)
{
    ucc_pipeline_params_t params = {0};
    int                   n_frags, pipeline_depth;
    ucc_status_t          status;

    /* Configure active pipeline with frag_size=0 and message exceeding threshold */
    params.n_frags   = 4;
    params.pdepth    = 8;
    params.frag_size = 0;    /* Invalid: would cause division by zero */
    params.threshold = 1024; /* Message size will exceed this */

    status           = ucc_pipeline_nfrags_pdepth(
        &params, 2048, &n_frags, &pipeline_depth);
    EXPECT_EQ(UCC_ERR_INVALID_PARAM, status)
        << "frag_size=0 with msgsize>threshold must be rejected before "
           "division";
}

/* ------------------------------------------------------------------ */
/* Test: frag_size=0 is OK when msgsize <= threshold (monolithic path)  */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipeline_params_frag_size_zero_ok_monolithic)
{
    ucc_pipeline_params_t params = {0};
    int                   n_frags, pipeline_depth;
    ucc_status_t          status;

    /* frag_size=0 is acceptable for monolithic (small message) path */
    params.n_frags   = 1;
    params.pdepth    = 1;
    params.frag_size = 0;    /* OK: no pipelining needed */
    params.threshold = 4096; /* Message size is below threshold */

    status           = ucc_pipeline_nfrags_pdepth(
        &params, 2048, &n_frags, &pipeline_depth);
    EXPECT_EQ(UCC_OK, status)
        << "frag_size=0 should be OK when msgsize <= threshold";
    EXPECT_EQ(1, n_frags) << "Monolithic path uses single fragment";
    EXPECT_EQ(1, pipeline_depth) << "Single fragment gives depth 1";
}

/* ------------------------------------------------------------------ */
/* Test: valid depth-1 operation succeeds                              */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipeline_params_valid_depth_one)
{
    ucc_pipeline_params_t params = {0};
    int                   n_frags, pipeline_depth;
    ucc_status_t          status;

    params.n_frags   = 1;
    params.pdepth    = 1;
    params.frag_size = 512;
    params.threshold = 1024;

    status           = ucc_pipeline_nfrags_pdepth(
        &params, 2048, &n_frags, &pipeline_depth);
    EXPECT_EQ(UCC_OK, status) << "Valid depth-1 pipeline should succeed";
    EXPECT_GE(n_frags, 1) << "Should compute at least one fragment";
    EXPECT_EQ(1, pipeline_depth) << "Depth limited to 1";
}

/* ------------------------------------------------------------------ */
/* Test: disabled pipeline (n_frags=0) passes helper validation        */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, pipeline_params_disabled_nfrags_zero_ok)
{
    ucc_pipeline_params_t params = {0};
    int                   n_frags, pipeline_depth;
    ucc_status_t          status;

    /* n_frags=0 means disabled/monolithic-only - pdepth check skipped */
    params.n_frags   = 0; /* Disabled: no pipelining */
    params.pdepth    = 0; /* Irrelevant when disabled */
    params.frag_size = 0;
    params.threshold = SIZE_MAX;

    status           = ucc_pipeline_nfrags_pdepth(
        &params, 2048, &n_frags, &pipeline_depth);
    EXPECT_EQ(UCC_OK, status)
        << "Disabled pipeline (n_frags=0) should pass helper validation";
    EXPECT_EQ(1, n_frags);
    EXPECT_EQ(1, pipeline_depth);
}

/* ========================================================================== */
/* Complete helper boundary and pipelined-init guard coverage.       */
/* ========================================================================== */

UCC_TEST_F(test_schedule, pipeline_params_exact_threshold_boundaries)
{
    ucc_pipeline_params_t params = {0};
    int                   n_frags, pipeline_depth;

    params.n_frags   = 4;
    params.pdepth    = 3;
    params.frag_size = 256;
    params.threshold = 1024;

    ASSERT_EQ(
        UCC_OK,
        ucc_pipeline_nfrags_pdepth(&params, 1023, &n_frags, &pipeline_depth));
    EXPECT_EQ(1, n_frags);
    EXPECT_EQ(1, pipeline_depth);

    ASSERT_EQ(
        UCC_OK,
        ucc_pipeline_nfrags_pdepth(&params, 1024, &n_frags, &pipeline_depth));
    EXPECT_EQ(1, n_frags);
    EXPECT_EQ(1, pipeline_depth);

    ASSERT_EQ(
        UCC_OK,
        ucc_pipeline_nfrags_pdepth(&params, 1025, &n_frags, &pipeline_depth));
    EXPECT_EQ(5, n_frags);
    EXPECT_EQ(3, pipeline_depth);
}

UCC_TEST_F(test_schedule, pipeline_params_zero_frag_size_threshold_boundary)
{
    ucc_pipeline_params_t params  = {0};
    int                   n_frags = -1, pipeline_depth = -1;

    params.n_frags   = 2;
    params.pdepth    = 2;
    params.frag_size = 0;
    params.threshold = 1024;

    ASSERT_EQ(
        UCC_OK,
        ucc_pipeline_nfrags_pdepth(&params, 1024, &n_frags, &pipeline_depth));
    EXPECT_EQ(1, n_frags);
    EXPECT_EQ(1, pipeline_depth);

    n_frags        = 77;
    pipeline_depth = 88;
    EXPECT_EQ(
        UCC_ERR_INVALID_PARAM,
        ucc_pipeline_nfrags_pdepth(&params, 1025, &n_frags, &pipeline_depth));
    EXPECT_EQ(77, n_frags) << "failure must not publish fragment count";
    EXPECT_EQ(88, pipeline_depth) << "failure must not publish usable depth";
}

UCC_TEST_F(test_schedule, pipeline_params_explicit_multifragment)
{
    ucc_pipeline_params_t params = {0};
    int                   n_frags, pipeline_depth;

    params.n_frags   = 8;
    params.pdepth    = 4;
    params.frag_size = 1024;
    params.threshold = 0;

    ASSERT_EQ(
        UCC_OK,
        ucc_pipeline_nfrags_pdepth(&params, 4096, &n_frags, &pipeline_depth));
    EXPECT_EQ(8, n_frags);
    EXPECT_EQ(4, pipeline_depth);
}

UCC_TEST_F(test_schedule, pipelined_negative_depth_rejected)
{
    EXPECT_EQ(
        UCC_ERR_INVALID_PARAM,
        ucc_schedule_pipelined_init(
            NULL, NULL, NULL, NULL, -1, 1, UCC_PIPELINE_PARALLEL, NULL));
}

UCC_TEST_F(test_schedule, pipelined_invalid_order_depth_one_rejected)
{
    EXPECT_EQ(
        UCC_ERR_INVALID_PARAM,
        ucc_schedule_pipelined_init(
            NULL, NULL, NULL, NULL, 1, 1, (ucc_pipeline_order_t)99, NULL));
}

UCC_TEST_F(test_schedule, pipelined_all_valid_orders_at_boundary)
{
    const ucc_pipeline_order_t orders[] = {
        UCC_PIPELINE_PARALLEL, UCC_PIPELINE_ORDERED, UCC_PIPELINE_SEQUENTIAL};

    for (auto order : orders) {
        ucc_base_team_t         *team;
        ucc_schedule_pipelined_t sched;
        ucc_base_coll_args_t     bargs = {0};

        memset(&sched, 0, sizeof(sched));
        ucc_coll_task_construct(&sched.super.super);
        g_frag_init_count.store(0);
        g_total_finalizes.store(0);
        g_fail_after_frag_idx = -1;
        team = create_test_team(UCC_THREAD_SINGLE);
        if (team == nullptr) {
            ADD_FAILURE() << "Failed to create test team";
            return;
        }
        ASSERT_EQ(
            UCC_OK,
            ucc_schedule_pipelined_init(
                &bargs, team, mock_frag_init, NULL, 1, 1, order, &sched));
        EXPECT_EQ(1, g_frag_init_count.load());
        ASSERT_NE(nullptr, sched.super.super.finalize);
        EXPECT_EQ(UCC_OK, sched.super.super.finalize(&sched.super.super));
        EXPECT_EQ(2, g_total_finalizes.load());
        destroy_test_team(team);
    }
}

UCC_TEST_F(test_schedule, pipelined_depth_above_limit_clamped)
{
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_base_coll_args_t     bargs = {0};

    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    g_frag_init_count.store(0);
    g_total_finalizes.store(0);
    g_fail_after_frag_idx = -1;
    team = create_test_team(UCC_THREAD_SINGLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }
    ASSERT_EQ(
        UCC_OK,
        ucc_schedule_pipelined_init(
            &bargs,
            team,
            mock_frag_init,
            NULL,
            UCC_SCHEDULE_PIPELINED_MAX_FRAGS + 1,
            UCC_SCHEDULE_PIPELINED_MAX_FRAGS + 1,
            UCC_PIPELINE_PARALLEL,
            &sched));
    EXPECT_EQ(UCC_SCHEDULE_PIPELINED_MAX_FRAGS, sched.n_frags);
    EXPECT_EQ(UCC_SCHEDULE_PIPELINED_MAX_FRAGS, g_frag_init_count.load());
    ASSERT_NE(nullptr, sched.super.super.finalize);
    EXPECT_EQ(UCC_OK, sched.super.super.finalize(&sched.super.super));
    EXPECT_EQ(2 * UCC_SCHEDULE_PIPELINED_MAX_FRAGS, g_total_finalizes.load());
    destroy_test_team(team);
}

/* Real ucc_event_manager_subscribe() failure/unwind oracles. */
static std::atomic<int> g_fault_subscribe_attempt(0);
static std::atomic<int> g_fault_subscribe_fail_at(-1);
static std::atomic<int> g_fault_frag_inits(0);
static std::atomic<int> g_fault_task_finalizes(0);
static std::atomic<int> g_fault_frag_finalizes(0);
static std::atomic<int> g_fault_pool_returns(0);
static std::atomic<int> g_fault_live_listeners_at_finalize(0);
static std::atomic<int> g_fault_lock_inits(0);
static std::atomic<int> g_fault_lock_destroys(0);
static int              g_fault_frag_fail_at = -1;

static ucc_status_t     fault_subscribe_cb(void)
{
    int attempt = g_fault_subscribe_attempt.fetch_add(1);
    return attempt == g_fault_subscribe_fail_at.load() ? UCC_ERR_NO_MEMORY
                                                    : UCC_OK;
}

static void fault_lock_observe(int initialized)
{
    initialized ? g_fault_lock_inits.fetch_add(1)
                : g_fault_lock_destroys.fetch_add(1);
}

static unsigned fault_listener_count(ucc_coll_task_t *task)
{
    ucc_event_manager_t *em;
    unsigned             count = 0;
    ucc_list_for_each (em, &task->em_list, list_elem) {
        count += em->n_listeners;
    }
    return count;
}

static ucc_status_t fault_task_finalize(ucc_coll_task_t *task)
{
    g_fault_live_listeners_at_finalize += fault_listener_count(task);
    g_fault_task_finalizes++;
    g_fault_pool_returns++;
    ucc_coll_task_destruct(task);
    free(task);
    return UCC_OK;
}

static ucc_status_t fault_frag_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *frag = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;
    g_fault_live_listeners_at_finalize += fault_listener_count(task);
    g_fault_frag_finalizes++;
    status = ucc_schedule_finalize(task);
    g_fault_pool_returns++;
    ucc_coll_task_destruct(task);
    free(frag);
    return status;
}

static ucc_status_t fault_frag_init(
    ucc_base_coll_args_t *args, ucc_schedule_pipelined_t *,
    ucc_base_team_t *team, ucc_schedule_t **frag_p)
{
    int              index = g_fault_frag_inits.load();
    ucc_schedule_t  *frag;
    ucc_coll_task_t *task;
    if (index == g_fault_frag_fail_at) {
        return UCC_ERR_NO_MEMORY;
    }
    frag = (ucc_schedule_t *)calloc(1, sizeof(*frag));
    task = (ucc_coll_task_t *)calloc(1, sizeof(*task));
    if (!frag || !task) {
        free(frag);
        free(task);
        return UCC_ERR_NO_MEMORY;
    }
    ucc_coll_task_construct(&frag->super);
    EXPECT_EQ(UCC_OK, ucc_schedule_init(frag, args, team));
    ucc_coll_task_construct(task);
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(task, args, team));
    task->finalize               = fault_task_finalize;
    frag->tasks[frag->n_tasks++] = task;
    frag->super.finalize         = fault_frag_finalize;
    *frag_p                      = frag;
    g_fault_frag_inits++;
    return UCC_OK;
}

static void fault_reset(int subscribe_fail, int frag_fail)
{
    g_fault_subscribe_attempt          = 0;
    g_fault_subscribe_fail_at          = subscribe_fail;
    g_fault_frag_inits                 = 0;
    g_fault_task_finalizes             = 0;
    g_fault_frag_finalizes             = 0;
    g_fault_pool_returns               = 0;
    g_fault_live_listeners_at_finalize = 0;
    g_fault_lock_inits                 = 0;
    g_fault_lock_destroys              = 0;
    g_fault_frag_fail_at               = frag_fail;
    ucc_event_manager_set_subscribe_fault_cb(fault_subscribe_cb);
    ucc_schedule_pipelined_set_lock_observer(fault_lock_observe);
}

static void fault_set_args(ucc_base_coll_args_t *args)
{
    memset(args, 0, sizeof(*args));
    args->args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    args->args.src.info.datatype = UCC_DT_INT32;
    args->args.dst.info.datatype = UCC_DT_INT32;
    args->args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    args->args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
}

static void fault_expect_subscription_unwind(
    ucc_pipeline_order_t order, int fail_at, int thread_mode)
{
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_base_coll_args_t     args;
    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    fault_set_args(&args);
    team = create_test_team(thread_mode);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }
    fault_reset(fail_at, -1);
    EXPECT_EQ(
        UCC_ERR_NO_MEMORY,
        ucc_schedule_pipelined_init(
            &args, team, fault_frag_init, NULL, 6, 6, order, &sched));
    EXPECT_EQ(fail_at + 1, g_fault_subscribe_attempt.load());
    EXPECT_EQ(6, g_fault_frag_inits.load());
    EXPECT_EQ(6, g_fault_task_finalizes.load());
    EXPECT_EQ(6, g_fault_frag_finalizes.load());
    EXPECT_EQ(12, g_fault_pool_returns.load());
    EXPECT_EQ(0u, fault_listener_count(&sched.super.super));
    EXPECT_EQ(0, g_fault_live_listeners_at_finalize.load());
    EXPECT_EQ(
        thread_mode == UCC_THREAD_MULTIPLE ? 1 : 0, g_fault_lock_inits.load());
    EXPECT_EQ(g_fault_lock_inits.load(), g_fault_lock_destroys.load());
    ucc_coll_task_destruct(&sched.super.super);
    destroy_test_team(team);
}

UCC_TEST_F(test_schedule, pipelined_real_subscription_failure_first)
{
    fault_expect_subscription_unwind(
        UCC_PIPELINE_SEQUENTIAL, 0, UCC_THREAD_SINGLE);
}

UCC_TEST_F(test_schedule, pipelined_real_subscription_failure_later)
{
    /* Attempt seven is after a dependency and both lifecycle subscription
     * kinds have already been installed, and crosses MAX_LISTENERS. */
    fault_expect_subscription_unwind(
        UCC_PIPELINE_ORDERED, 7, UCC_THREAD_MULTIPLE);
}

UCC_TEST_F(test_schedule, pipelined_depth_16_exact_lifecycle)
{
    ucc_base_team_t         *team;
    ucc_schedule_pipelined_t sched;
    ucc_base_coll_args_t     args;
    memset(&sched, 0, sizeof(sched));
    ucc_coll_task_construct(&sched.super.super);
    fault_set_args(&args);
    team = create_test_team(UCC_THREAD_MULTIPLE);
    if (team == nullptr) {
        ADD_FAILURE() << "Failed to create test team";
        return;
    }
    fault_reset(-1, -1);
    ASSERT_EQ(
        UCC_OK,
        ucc_schedule_pipelined_init(
            &args,
            team,
            fault_frag_init,
            NULL,
            16,
            16,
            UCC_PIPELINE_PARALLEL,
            &sched));
    EXPECT_EQ(16, g_fault_frag_inits.load());
    EXPECT_EQ(32, g_fault_subscribe_attempt.load());
    EXPECT_EQ(UCC_OK, sched.super.super.finalize(&sched.super.super));
    EXPECT_EQ(16, g_fault_task_finalizes.load());
    EXPECT_EQ(16, g_fault_frag_finalizes.load());
    EXPECT_EQ(32, g_fault_pool_returns.load());
    EXPECT_EQ(0, g_fault_live_listeners_at_finalize.load());
    EXPECT_EQ(1, g_fault_lock_inits.load());
    EXPECT_EQ(1, g_fault_lock_destroys.load());
    ucc_coll_task_destruct(&sched.super.super);
    destroy_test_team(team);
}

UCC_TEST_F(test_schedule, pipelined_fragment_failure_exact_unwind)
{
    const int failures[] = {0, 3, 7};
    for (int fail_at : failures) {
        ucc_base_team_t         *team;
        ucc_schedule_pipelined_t sched;
        ucc_base_coll_args_t     args;
        memset(&sched, 0, sizeof(sched));
        ucc_coll_task_construct(&sched.super.super);
        fault_set_args(&args);
        team = create_test_team(UCC_THREAD_SINGLE);
        if (team == nullptr) {
            ADD_FAILURE() << "Failed to create test team";
            return;
        }
        fault_reset(-1, fail_at);
        EXPECT_EQ(
            UCC_ERR_NO_MEMORY,
            ucc_schedule_pipelined_init(
                &args,
                team,
                fault_frag_init,
                NULL,
                8,
                8,
                UCC_PIPELINE_PARALLEL,
                &sched));
        EXPECT_EQ(fail_at, g_fault_frag_inits.load());
        EXPECT_EQ(fail_at, g_fault_task_finalizes.load());
        EXPECT_EQ(fail_at, g_fault_frag_finalizes.load());
        EXPECT_EQ(2 * fail_at, g_fault_pool_returns.load());
        EXPECT_EQ(0, g_fault_live_listeners_at_finalize.load());
        ucc_coll_task_destruct(&sched.super.super);
        destroy_test_team(team);
    }
}

/* ------------------------------------------------------------------ */
/* EM node counting helper                                            */
/* ------------------------------------------------------------------ */

/* We cannot easily intercept free() per-test, so we count by
 * iterating em_list before and after each init cycle. */
static int count_em_nodes(ucc_coll_task_t *task)
{
    int                  count = 0;
    ucc_event_manager_t *em;
    ucc_list_for_each (em, &task->em_list, list_elem) {
        count++;
    }
    return count;
}

/* ------------------------------------------------------------------ */
/* Test: single-node EM survives task reinitialization                */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, em_reuse_single_node)
{
    ucc_coll_task_t task;
    ucc_coll_task_t listener_task;
    int             em_nodes_before, em_nodes_after;

    /* Construct + init the task */
    ucc_coll_task_construct(&task);
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(&task, NULL, NULL));

    /* Construct a listener task for subscriptions */
    ucc_coll_task_construct(&listener_task);
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(&listener_task, NULL, NULL));

    /* Subscribe 4 listeners — fills one EM node (MAX_LISTENERS=4) */
    for (int i = 0; i < MAX_LISTENERS; i++) {
        EXPECT_EQ(
            UCC_OK,
            ucc_event_manager_subscribe(
                &task,
                UCC_EVENT_COMPLETED,
                &listener_task,
                ucc_task_start_handler));
    }

    /* Verify one EM node with MAX_LISTENERS listeners */
    em_nodes_before = count_em_nodes(&task);
    EXPECT_EQ(1, em_nodes_before);

    /* Reinitialize the task (simulates pooled-task reuse) */
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(&task, NULL, NULL));

    /* EM node must still be reachable */
    em_nodes_after = count_em_nodes(&task);
    EXPECT_EQ(em_nodes_before, em_nodes_after)
        << "EM node lost after reinit — list head was reset";

    /* Listener counts must be zeroed */
    {
        ucc_event_manager_t *em;
        ucc_list_for_each (em, &task.em_list, list_elem) {
            EXPECT_EQ(0u, em->n_listeners);
        }
    }

    ucc_coll_task_destruct(&listener_task);
    ucc_coll_task_destruct(&task);
}

/* ------------------------------------------------------------------ */
/* Test: multi-node EM survives multiple reuse cycles                 */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, em_reuse_multi_node_cycles)
{
    const int       n_cycles      = 5;
    /* Subscribe more than MAX_LISTENERS so multiple EM nodes are allocated. */
    const int       n_subscribers = 2 * MAX_LISTENERS + 1; /* 9 → 3 EM nodes */

    ucc_coll_task_t task;
    ucc_coll_task_t listener_task;
    ucc_event_manager_t *first_em = NULL;

    ucc_coll_task_construct(&task);
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(&task, NULL, NULL));

    ucc_coll_task_construct(&listener_task);
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(&listener_task, NULL, NULL));

    for (int cycle = 0; cycle < n_cycles; cycle++) {
        int em_count = 0;

        /* Subscribe enough to allocate new EM nodes on top of existing ones */
        for (int i = 0; i < n_subscribers; i++) {
            EXPECT_EQ(
                UCC_OK,
                ucc_event_manager_subscribe(
                    &task,
                    UCC_EVENT_COMPLETED,
                    &listener_task,
                    ucc_task_start_handler));
        }

        /* Count EM nodes */
        em_count = count_em_nodes(&task);
        EXPECT_GT(em_count, 1)
            << "Multiple EM nodes expected after " << n_subscribers
            << " subscriptions (MAX_LISTENERS=" << MAX_LISTENERS << ")";

        /* Save pointer to first EM node for reachability check */
        {
            ucc_event_manager_t *em;
            ucc_list_for_each (em, &task.em_list, list_elem) {
                if (first_em == NULL) {
                    first_em = em;
                }
                break;
            }
        }

        /* Reinitialize the task (simulates pooled-task reuse) */
        EXPECT_EQ(UCC_OK, ucc_coll_task_init(&task, NULL, NULL));

        /* Verify EM nodes still reachable */
        em_count = count_em_nodes(&task);
        EXPECT_GT(em_count, 1)
            << "Cycle " << cycle << ": EM nodes lost after reinit";

        /* All listener counts must be zero */
        {
            ucc_event_manager_t *em;
            ucc_list_for_each (em, &task.em_list, list_elem) {
                EXPECT_EQ(0u, em->n_listeners)
                    << "Cycle " << cycle << ": stale listeners remain";
            }
        }
    }

    ucc_coll_task_destruct(&listener_task);
    ucc_coll_task_destruct(&task);
}

/* ------------------------------------------------------------------ */
/* Test: EM nodes freed exactly once on destruct                      */
/* ------------------------------------------------------------------ */
UCC_TEST_F(test_schedule, em_nodes_freed_exactly_once)
{
    const int n_subscribers     = 3 * MAX_LISTENERS + 2; /* 14 → 4 EM nodes */
    const int expected_em_nodes = (n_subscribers + MAX_LISTENERS - 1) /
                                  MAX_LISTENERS;

    ucc_coll_task_t task;
    ucc_coll_task_t listener_task;
    int             em_count;

    ucc_coll_task_construct(&task);
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(&task, NULL, NULL));

    ucc_coll_task_construct(&listener_task);
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(&listener_task, NULL, NULL));

    /* Subscribe to create multiple EM nodes */
    for (int i = 0; i < n_subscribers; i++) {
        EXPECT_EQ(
            UCC_OK,
            ucc_event_manager_subscribe(
                &task,
                UCC_EVENT_COMPLETED,
                &listener_task,
                ucc_task_start_handler));
    }

    /* Verify expected number of EM nodes */
    em_count = count_em_nodes(&task);
    EXPECT_EQ(expected_em_nodes, em_count)
        << "Expected " << expected_em_nodes << " EM nodes for " << n_subscribers
        << " subscribers";

    /* Reinit and verify nodes retained */
    EXPECT_EQ(UCC_OK, ucc_coll_task_init(&task, NULL, NULL));
    em_count = count_em_nodes(&task);
    EXPECT_EQ(expected_em_nodes, em_count)
        << "EM nodes should be retained across reinit";

    /* Subscribe again (should reuse existing nodes, not allocate new ones) */
    for (int i = 0; i < MAX_LISTENERS; i++) {
        EXPECT_EQ(
            UCC_OK,
            ucc_event_manager_subscribe(
                &task,
                UCC_EVENT_COMPLETED,
                &listener_task,
                ucc_task_start_handler));
    }
    em_count = count_em_nodes(&task);
    EXPECT_EQ(expected_em_nodes, em_count)
        << "No new EM nodes should be allocated after reinit";

    /* Destruct — should free all EM nodes without crash */
    ucc_coll_task_destruct(&task);
    ucc_coll_task_destruct(&listener_task);
}
