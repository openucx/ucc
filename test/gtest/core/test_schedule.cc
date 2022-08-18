/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include "schedule/ucc_schedule.h"
}
typedef std::tuple<ucc_coll_task_t*, int> rst_t;
class test_schedule : public ucc_coll_task_t, public ucc::test
{
public:
    std::vector<rst_t> rst;
    static ucc_status_t handler_1(ucc_coll_task_t *parent,
                                  ucc_coll_task_t *task) {
        test_schedule *ts = (test_schedule*)task;
        ts->rst.push_back(rst_t(parent, 1));
        return UCC_OK;
    }
    static ucc_status_t handler_2(ucc_coll_task_t *parent,
                                  ucc_coll_task_t *task) {
        test_schedule *ts = (test_schedule*)task;
        ts->rst.push_back(rst_t(parent, 2));
        return UCC_OK;
    }
};

/* Tasks subscribes on 2 tasks to EVENT_COMPLETED with the same
   handler */
UCC_TEST_F(test_schedule, single_handler)
{
    std::vector<ucc_coll_task_t> tasks(2);

    EXPECT_EQ(UCC_OK, ucc_coll_task_init((ucc_coll_task_t *)this, NULL, NULL));
    for (auto &t :  tasks) {
        EXPECT_EQ(UCC_OK, ucc_coll_task_init(&t, NULL, NULL));
        ucc_event_manager_subscribe(&t.em, UCC_EVENT_COMPLETED,
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
    std::vector<ucc_coll_task_t> tasks(2);

    EXPECT_EQ(UCC_OK, ucc_coll_task_init((ucc_coll_task_t *)this, NULL, NULL));
    for (auto &t :  tasks) {
        EXPECT_EQ(UCC_OK, ucc_coll_task_init(&t, NULL, NULL));
    }
    ucc_event_manager_subscribe(&tasks[0].em, UCC_EVENT_COMPLETED,
                                (ucc_coll_task_t*)this,
                                test_schedule::handler_1);
    ucc_event_manager_subscribe(&tasks[1].em, UCC_EVENT_COMPLETED,
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
