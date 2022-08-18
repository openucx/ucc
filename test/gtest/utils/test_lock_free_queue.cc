
/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

extern "C" {
#include "utils/ucc_lock_free_queue.h"
#include "utils/ucc_atomic.h"
#include "utils/ucc_malloc.h"
#include <pthread.h>
#include <stdio.h>
}
#include <common/test.h>
#include <vector>

#define NUM_ITERS 5000000

typedef struct ucc_test_queue {
    ucc_lf_queue_t      lf_queue;
    int64_t             test_sum;
    uint32_t            elems_num;
    uint32_t            active_producers_threads;
    uint32_t            memory_err;
} ucc_test_queue_t;

void *producer_thread(void *arg)
{
    ucc_test_queue_t *test = (ucc_test_queue_t *)arg;
    for (int j = 0; j < NUM_ITERS; j++) {
        ucc_lf_queue_elem_t *elem =
            (ucc_lf_queue_elem_t *)ucc_malloc(sizeof(ucc_lf_queue_elem_t));
        ucc_lf_queue_init_elem(elem);
        if (!elem) {
            ucc_atomic_add32(&test->memory_err, 1);
            goto exit;
        }
        ucc_lf_queue_enqueue(&test->lf_queue, elem);
        ucc_atomic_add64((uint64_t *)&test->test_sum, (uint64_t)elem);
        ucc_atomic_add32(&test->elems_num, 1);
    }
exit:
    ucc_atomic_sub32(&test->active_producers_threads,1);
    return 0;
}

void *consumer_thread(void *arg)
{
    ucc_test_queue_t *test = (ucc_test_queue_t *)arg;
    while(test->active_producers_threads || test->elems_num){
        ucc_lf_queue_elem_t *elem = ucc_lf_queue_dequeue(&test->lf_queue, 1);
        if (elem) {
            ucc_atomic_sub64((uint64_t *)&test->test_sum, (uint64_t)elem);
            ucc_atomic_sub32(&test->elems_num, 1);
            ucc_free(elem);
        }
    }
    return 0;
}

class test_lf_queue : public ucc::test
{
  public:
    ucc_test_queue_t       test;
    int                    i;
    std::vector<pthread_t> producers_threads;
    std::vector<pthread_t> consumers_threads;
    int                    lf_test(int num_of_producers, int num_of_consumers);
};

int test_lf_queue::lf_test(int num_of_producers, int num_of_consumers){
    producers_threads.resize(num_of_producers);
    consumers_threads.resize(num_of_consumers);
    memset(&test, 0, sizeof(ucc_test_queue_t));
    ucc_lf_queue_init(&test.lf_queue);
    for (i = 0; i < num_of_producers; i++) {
        ucc_atomic_add32(&test.active_producers_threads, 1);
        pthread_create(&producers_threads[i], NULL, &producer_thread,
                       (void *)&test);
    }
    for (i = 0; i < num_of_consumers; i++) {
        pthread_create(&consumers_threads[i], NULL, &consumer_thread,
                       (void *)&test);
    }
    for (i = 0; i < num_of_producers; i++) {
        pthread_join(producers_threads[i], NULL);
    }
    for (i = 0; i < num_of_consumers; i++) {
        pthread_join(consumers_threads[i], NULL);
    }
    ucc_lf_queue_destroy(&test.lf_queue);
    if (test.memory_err) {
        return 1;
    }
    if (test.test_sum) {
        return 1;
    }
    return 0;
}


UCC_TEST_F(test_lf_queue, oneProducerOneConsumer)
{
    EXPECT_EQ(lf_test(1, 1), 0);
}

UCC_TEST_F(test_lf_queue, oneProducerManyConsumers)
{
    EXPECT_EQ(lf_test(1, 7), 0);
}

UCC_TEST_F(test_lf_queue, manyProducersManyConsumers)
{
    EXPECT_EQ(lf_test(7, 7), 0);
}
