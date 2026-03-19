/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_mc_user_component.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_mem_type.h"
#include "utils/ucc_log.h"
#include <stdlib.h>
#include <string.h>

/* User component registry - only modified during initialization (single-threaded),
 * read-only afterwards. No locking needed. */
static ucc_mc_user_component_entry_t *mc_user_component_list = NULL;
static ucc_memory_type_t              next_memory_type       = UCC_MEMORY_TYPE_USER_FIRST;

ucc_status_t ucc_mc_user_component_register(
    ucc_mc_base_t *mc, ucc_memory_type_t *memory_type)
{
    ucc_mc_user_component_entry_t *entry;
    ucc_memory_type_t              mt;

    if (!mc || !memory_type) {
        ucc_error("invalid parameters for user component registration");
        return UCC_ERR_INVALID_PARAM;
    }

    if (!mc->ops.mem_alloc || !mc->ops.mem_free ||
        !mc->ops.memcpy     || !mc->ops.memset) {
        ucc_error("user component '%s' must implement mem_alloc, mem_free, "
                  "memcpy, and memset", mc->super.name);
        return UCC_ERR_INVALID_PARAM;
    }

    /* Check if user component with same name already exists */
    entry = mc_user_component_list;
    while (entry) {
        if (strcmp(entry->mc->super.name, mc->super.name) == 0) {
            ucc_warn("MC user component '%s' already registered", mc->super.name);
            return UCC_ERR_NO_RESOURCE;
        }
        entry = entry->next;
    }

    entry = (ucc_mc_user_component_entry_t *)ucc_malloc(
        sizeof(ucc_mc_user_component_entry_t), "mc_user_component_entry");
    if (!entry) {
        ucc_error("failed to allocate memory for MC user component entry");
        return UCC_ERR_NO_MEMORY;
    }

    mt = next_memory_type++;
    mc->ref_cnt++;

    entry->mc          = mc;
    entry->memory_type = mt;
    entry->next        = mc_user_component_list;
    mc_user_component_list = entry;
    *memory_type           = mt;

    ucc_info("MC user component '%s' registered with memory_type=%d",
             mc->super.name, mt);
    return UCC_OK;
}

ucc_status_t ucc_mc_user_component_unregister(ucc_memory_type_t memory_type)
{
    ucc_mc_user_component_entry_t *entry, *prev;

    if (memory_type < UCC_MEMORY_TYPE_USER_FIRST) {
        ucc_error("cannot unregister builtin memory type %d", memory_type);
        return UCC_ERR_INVALID_PARAM;
    }

    prev  = NULL;
    entry = mc_user_component_list;

    while (entry) {
        if (entry->memory_type == memory_type) {
            if (prev) {
                prev->next = entry->next;
            } else {
                mc_user_component_list = entry->next;
            }
            entry->mc->ref_cnt--;
            if (entry->mc->ref_cnt == 0) {
                entry->mc->finalize();
                ucc_config_parser_release_opts(entry->mc->config,
                                               entry->mc->config_table.table);
                ucc_free(entry->mc->config);
            }
            ucc_free(entry);
            ucc_info("MC user component memory_type=%d unregistered", memory_type);
            return UCC_OK;
        }
        prev  = entry;
        entry = entry->next;
    }

    ucc_error("MC user component memory_type=%d not found", memory_type);
    return UCC_ERR_NOT_FOUND;
}

const char *ucc_mc_user_component_get_name(ucc_memory_type_t memory_type)
{
    ucc_mc_user_component_entry_t *entry = ucc_mc_user_component_get_entry(memory_type);

    return entry ? entry->mc->super.name : NULL;
}

int ucc_mc_is_user_component(ucc_memory_type_t memory_type)
{
    return (memory_type >= UCC_MEMORY_TYPE_USER_FIRST) ? 1 : 0;
}

ucc_mc_user_component_entry_t *ucc_mc_user_component_get_entry(ucc_memory_type_t memory_type)
{
    ucc_mc_user_component_entry_t *entry;

    if (memory_type < UCC_MEMORY_TYPE_USER_FIRST) {
        return NULL;
    }

    entry = mc_user_component_list;
    while (entry) {
        if (entry->memory_type == memory_type) {
            return entry;
        }
        entry = entry->next;
    }

    return NULL;
}

void ucc_mc_user_component_finalize_all(void)
{
    ucc_mc_user_component_entry_t *entry, *next;

    entry = mc_user_component_list;
    while (entry) {
        next = entry->next;
        entry->mc->ref_cnt--;
        if (entry->mc->ref_cnt == 0) {
            entry->mc->finalize();
            ucc_config_parser_release_opts(entry->mc->config,
                                           entry->mc->config_table.table);
            ucc_free(entry->mc->config);
        }
        ucc_free(entry);
        entry = next;
    }

    mc_user_component_list = NULL;
    next_memory_type       = UCC_MEMORY_TYPE_USER_FIRST;
}

int ucc_mc_total_mem_types(void)
{
    return (int)next_memory_type;
}

ucc_status_t ucc_mc_user_component_iterate(ucc_mc_user_component_iter_cb_t callback, void *ctx)
{
    ucc_mc_user_component_entry_t *entry;
    ucc_status_t                   status;

    if (!callback) {
        return UCC_ERR_INVALID_PARAM;
    }

    entry = mc_user_component_list;
    while (entry) {
        status = callback(entry, ctx);
        if (status != UCC_OK) {
            return status;
        }
        entry = entry->next;
    }

    return UCC_OK;
}
