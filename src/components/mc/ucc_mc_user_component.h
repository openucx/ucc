/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_USER_COMPONENT_H_
#define UCC_MC_USER_COMPONENT_H_

#include "base/ucc_mc_base.h"

/**
 * Internal user component registry entry
 */
typedef struct ucc_mc_user_component_entry {
    ucc_mc_base_t                      *mc;          /* Loaded component */
    ucc_memory_type_t                   memory_type; /* Dynamically assigned */
    struct ucc_mc_user_component_entry *next;
} ucc_mc_user_component_entry_t;

/**
 * Register a user component (internal use)
 * Called during ucc_mc_init() for components loaded from .so files.
 * Assigns a unique memory type value beyond UCC_MEMORY_TYPE_LAST.
 *
 * @param [in]  mc          Loaded component
 * @param [out] memory_type Assigned memory type
 *
 * @return UCC_OK on success
 */
ucc_status_t ucc_mc_user_component_register(
    ucc_mc_base_t *mc, ucc_memory_type_t *memory_type);

/**
 * Unregister a user component (internal use)
 *
 * @param [in] memory_type Memory type to unregister
 *
 * @return UCC_OK on success
 */
ucc_status_t ucc_mc_user_component_unregister(ucc_memory_type_t memory_type);

/**
 * Get user component name by memory type
 *
 * @param [in] memory_type  Memory type
 *
 * @return User component name or NULL if not found
 */
const char *ucc_mc_user_component_get_name(ucc_memory_type_t memory_type);

/**
 * Check if memory type belongs to a user component
 *
 * @param [in] memory_type  Memory type to check
 *
 * @return 1 if user component (>= UCC_MEMORY_TYPE_LAST), 0 if builtin
 */
int ucc_mc_is_user_component(ucc_memory_type_t memory_type);

/**
 * Get user component entry by memory type (internal use)
 *
 * @param [in] memory_type  Memory type
 *
 * @return User component entry or NULL if not found
 */
ucc_mc_user_component_entry_t *ucc_mc_user_component_get_entry(ucc_memory_type_t memory_type);

/**
 * Finalize all registered user components and reset the registry.
 * Called from ucc_mc_finalize().
 */
void ucc_mc_user_component_finalize_all(void);

/**
 * Iterate through all registered user components with a callback function
 *
 * @param [in] callback  Callback function called for each user component;
 *                       return UCC_OK to continue, any other value to stop
 * @param [in] ctx       User context passed to callback
 *
 * @return UCC_OK on success
 */
typedef ucc_status_t (*ucc_mc_user_component_iter_cb_t)(
    ucc_mc_user_component_entry_t *user_component, void *ctx);
ucc_status_t ucc_mc_user_component_iterate(ucc_mc_user_component_iter_cb_t callback, void *ctx);

/**
 * Returns the total number of memory type slots needed to cover all built-in
 * and currently registered user memory types. Use this as the n_mem_types
 * argument when allocating score arrays.
 */
int ucc_mc_total_mem_types(void);

#endif /* UCC_MC_USER_COMPONENT_H_ */
