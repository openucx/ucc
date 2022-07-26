/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_H_
#define UCC_EC_H_

#include "ucc/api/ucc.h"
#include "components/ec/base/ucc_ec_base.h"

ucc_status_t ucc_ec_init(const ucc_ec_params_t *ec_params);

ucc_status_t ucc_ec_available(ucc_ee_type_t ee_type);

ucc_status_t ucc_ec_get_attr(ucc_ec_attr_t *attr);

ucc_status_t ucc_ec_finalize();

ucc_status_t ucc_ec_task_post(void *ee_context, ucc_ee_type_t ee_type,
                              void **ee_task);

ucc_status_t ucc_ec_task_query(void *ee_task, ucc_ee_type_t ee_type);

ucc_status_t ucc_ec_task_end(void *ee_task, ucc_ee_type_t ee_type);

ucc_status_t ucc_ec_create_event(void **event, ucc_ee_type_t ee_type);

ucc_status_t ucc_ec_destroy_event(void *event, ucc_ee_type_t ee_type);

ucc_status_t ucc_ec_event_post(void *ee_context, void *event,
                               ucc_ee_type_t ee_type);

ucc_status_t ucc_ec_event_test(void *event, ucc_ee_type_t ee_type);

ucc_status_t ucc_ee_executor_init(const ucc_ee_executor_params_t *params,
                                  ucc_ee_executor_t **executor);

ucc_status_t ucc_ee_executor_status(const ucc_ee_executor_t *executor);

ucc_status_t ucc_ee_executor_start(ucc_ee_executor_t *executor,
                                   void *ee_context);

ucc_status_t ucc_ee_executor_stop(ucc_ee_executor_t *executor);

ucc_status_t ucc_ee_executor_finalize(ucc_ee_executor_t *executor);

ucc_status_t ucc_ee_executor_task_post(ucc_ee_executor_t *executor,
                                       const ucc_ee_executor_task_args_t *task_args,
                                       ucc_ee_executor_task_t **task);

ucc_status_t ucc_ee_executor_task_test(const ucc_ee_executor_task_t *task);

ucc_status_t ucc_ee_executor_task_finalize(ucc_ee_executor_task_t *task);

#endif
