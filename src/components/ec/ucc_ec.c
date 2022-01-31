/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "base/ucc_ec_base.h"
#include "ucc_ec.h"
#include "core/ucc_global_opts.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"

static const ucc_ec_ops_t *ec_ops[UCC_EE_LAST];

#define UCC_CHECK_EC_AVAILABLE(ee)                                             \
    do {                                                                       \
        if (NULL == ec_ops[ee]) {                                              \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

ucc_status_t ucc_ec_init(const ucc_ec_params_t *ec_params)
{
    int            i, n_ecs;
    ucc_ec_base_t *ec;
    ucc_status_t   status;
    ucc_ec_attr_t  attr;

    memset(ec_ops, 0, UCC_EE_LAST * sizeof(ucc_ec_ops_t *));
    n_ecs = ucc_global_config.ec_framework.n_components;
    for (i = 0; i < n_ecs; i++) {
        ec = ucc_derived_of(ucc_global_config.ec_framework.components[i],
                            ucc_ec_base_t);
        if (ec->ref_cnt == 0) {
            ec->config = ucc_malloc(ec->config_table.size);
            if (!ec->config) {
                ucc_error("failed to allocate %zd bytes for ec config",
                          ec->config_table.size);
                continue;
            }
            status = ucc_config_parser_fill_opts(
                ec->config, ec->config_table.table, "UCC_",
                ec->config_table.prefix, 1);
            if (UCC_OK != status) {
                ucc_info("failed to parse config for EC component: %s (%d)",
                         ec->super.name, status);
                ucc_free(ec->config);
                continue;
            }
            status = ec->init(ec_params);
            if (UCC_OK != status) {
                ucc_info("ec_init failed for component: %s, skipping (%d)",
                         ec->super.name, status);
                ucc_config_parser_release_opts(ec->config,
                                               ec->config_table.table);
                ucc_free(ec->config);
                continue;
            }
            ucc_debug("ec %s initialized", ec->super.name);
        } else {
            attr.field_mask = UCC_EC_ATTR_FIELD_THREAD_MODE;
            status = ec->get_attr(&attr);
            if (status != UCC_OK) {
                return status;
            }
            if (attr.thread_mode < ec_params->thread_mode) {
                ucc_warn("ec %s was allready initilized with "
                         "different thread mode: current tm %d, provided tm %d",
                         ec->super.name, attr.thread_mode,
                         ec_params->thread_mode);
            }
        }
        ec->ref_cnt++;
        ec_ops[ec->type] = &ec->ops;
    }

    return UCC_OK;
}

ucc_status_t ucc_ec_available(ucc_ee_type_t ee_type)
{
    if (NULL == ec_ops[ee_type]) {
        return UCC_ERR_NOT_FOUND;
    }

    return UCC_OK;
}

ucc_status_t ucc_ec_finalize()
{
    ucc_ee_type_t  et;
    ucc_ec_base_t *ec;

    for (et = UCC_EE_CPU_THREAD; et < UCC_EE_LAST; et++) {
        if (NULL != ec_ops[et]) {
            ec = ucc_container_of(ec_ops[et], ucc_ec_base_t, ops);
            ec->ref_cnt--;
            if (ec->ref_cnt == 0) {
                ec->finalize();
                ucc_config_parser_release_opts(ec->config,
                                               ec->config_table.table);
                ucc_free(ec->config);
                ec_ops[et] = NULL;
            }
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_ec_task_post(void *ee_context, ucc_ee_type_t ee_type,
                              void **ee_task)
{
    UCC_CHECK_EC_AVAILABLE(ee_type);
    return ec_ops[ee_type]->task_post(ee_context, ee_task);
}

ucc_status_t ucc_ec_task_query(void *ee_task, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EC_AVAILABLE(ee_type);
    return ec_ops[ee_type]->task_query(ee_task);
}

ucc_status_t ucc_ec_task_end(void *ee_task, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EC_AVAILABLE(ee_type);
    return ec_ops[ee_type]->task_end(ee_task);
}

ucc_status_t ucc_ec_create_event(void **event, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EC_AVAILABLE(ee_type);
    return ec_ops[ee_type]->create_event(event);
}

ucc_status_t ucc_ec_destroy_event(void *event, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EC_AVAILABLE(ee_type);
    return ec_ops[ee_type]->destroy_event(event);
}

ucc_status_t ucc_ec_event_post(void *ee_context, void *event,
                               ucc_ee_type_t ee_type)
{
    UCC_CHECK_EC_AVAILABLE(ee_type);
    return ec_ops[ee_type]->event_post(ee_context, event);
}

ucc_status_t ucc_ec_event_test(void *event, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EC_AVAILABLE(ee_type);
    return ec_ops[ee_type]->event_test(event);
}
