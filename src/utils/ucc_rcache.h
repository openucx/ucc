/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_RCACHE_H_
#define UCC_RCACHE_H_

#include <ucs/memory/rcache.h>
#include <ucm/api/ucm.h>
#include <utils/ucc_sys.h>

//TODO: handle external events
#define ucc_rcache_t                 ucs_rcache_t
#define ucc_rcache_ops_t             ucs_rcache_ops_t
#define ucc_rcache_params_t          ucs_rcache_params_t
#define ucc_rcache_region_t          ucs_rcache_region_t

#define ucc_rcache_destroy           ucs_rcache_destroy
#define ucc_rcache_region_hold       ucs_rcache_region_hold
#define ucc_rcache_region_put        ucs_rcache_region_put
#define ucc_rcache_region_invalidate ucs_rcache_region_invalidate

/* Wrapper functions for status conversion */
static inline ucc_status_t
ucc_rcache_create(const ucc_rcache_params_t *params,
                  const char *name, ucc_rcache_t **rcache_p)
{
#ifndef UCS_HAVE_RCACHE_REGION_ALIGNMENT
    ucc_rcache_params_t params_dup = *params;
    params_dup.alignment     = UCS_PGT_ADDR_ALIGN;
    params_dup.max_alignment = ucc_get_page_size();

    return ucs_status_to_ucc_status(ucs_rcache_create(
                                    &params_dup, name, NULL, rcache_p));
#else
    return ucs_status_to_ucc_status(ucs_rcache_create(
                                    params, name, NULL, rcache_p));
#endif
}

/* [arg] parameter allows passing additional information from mem_reg callabck.
   For example, it can be used to indicate whether the entry was found in the
   cache or new registration happened. */
static inline ucc_status_t
ucc_rcache_get(ucc_rcache_t *rcache, void *address, size_t length,
               void *arg, ucc_rcache_region_t **region_p)
{
#ifdef UCS_HAVE_RCACHE_REGION_ALIGNMENT
    return ucs_status_to_ucc_status(ucs_rcache_get(
                                       rcache, address, length,
                                       ucc_get_page_size(),
                                       PROT_READ | PROT_WRITE, arg, region_p));
#else
    return ucs_status_to_ucc_status(ucs_rcache_get(
                                       rcache, address, length,
                                       PROT_READ | PROT_WRITE, arg, region_p));
#endif
}

#endif
