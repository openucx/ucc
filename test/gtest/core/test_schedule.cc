/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include "schedule/ucc_schedule.h"
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
