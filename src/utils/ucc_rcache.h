/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_RCACHE_H_
#define UCC_RCACHE_H_

#include <ucs/memory/rcache.h>
#include <ucm/api/ucm.h>

//TODO: handle external events
#define ucc_rcache_t                 ucs_rcache_t
#define ucc_rcache_ops_t             ucs_rcache_ops_t
#define ucc_rcache_params_t          ucs_rcache_params_t
#define ucc_rcache_region_t          ucs_rcache_region_t

#define ucc_rcache_create            ucs_rcache_create
#define ucc_rcache_destroy           ucs_rcache_destroy
#define ucc_rcache_get               ucs_rcache_get
#define ucc_rcache_region_hold       ucs_rcache_region_hold
#define ucc_rcache_region_put        ucs_rcache_region_put
#define ucc_rcache_region_invalidate ucs_rcache_region_invalidate

#endif
