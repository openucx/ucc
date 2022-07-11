/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_EP_HASH_H_
#define UCC_TL_UCP_EP_HASH_H_
#include "config.h"
#include "core/ucc_context.h"
#include "utils/khash.h"
#include <stdint.h>

static inline uint32_t tl_ucp_ctx_id_hash_fn_impl(uint32_t h, uint32_t k)
{
    uint32_t H = h & 0xf8000000;
    h = h << 5;
    h = h ^ ( H >> 27 );
    h = h ^ k;
    return h;
}

/* Collisions are handled in khash implementation */
static inline khint32_t tl_ucp_ctx_id_hash_fn(ucc_context_id_t k)
{
    uint32_t h = 0;

    ucc_assert(sizeof(k.pi.host_hash) == 8);
    h = tl_ucp_ctx_id_hash_fn_impl(h,
                                   kh_int64_hash_func(k.pi.host_hash));
    h = tl_ucp_ctx_id_hash_fn_impl(h, k.pi.pid);
    h = tl_ucp_ctx_id_hash_fn_impl(h, k.seq_num);
    return (khint32_t)h;
}

#define tl_ucp_ctx_id_equal_fn(_a, _b)                                         \
    (((_a).pi.host_hash == (_b).pi.host_hash) &&                               \
     ((_a).pi.pid == (_b).pi.pid) && ((_a).seq_num == (_b).seq_num))

KHASH_INIT(tl_ucp_ep_hash, ucc_context_id_t, void*, 1, \
           tl_ucp_ctx_id_hash_fn, tl_ucp_ctx_id_equal_fn);

#define tl_ucp_ep_hash_t khash_t(tl_ucp_ep_hash)

static inline void* tl_ucp_hash_get(tl_ucp_ep_hash_t *h, ucc_context_id_t key)
{
    khiter_t k;
    void    *value;
    k = kh_get(tl_ucp_ep_hash, h , key);
    if (k == kh_end(h)) {
        return NULL;
    }
    value = kh_value(h, k);
    return value;
}

static inline void tl_ucp_hash_put(tl_ucp_ep_hash_t *h, ucc_context_id_t key,
                                   void *value)
{
    int ret;
    khiter_t k;
    k = kh_put(tl_ucp_ep_hash, h, key, &ret);
    kh_value(h, k) = value;
}

static inline void* tl_ucp_hash_pop(tl_ucp_ep_hash_t *h)
{
    void    *ep = NULL;
    khiter_t k;
    k = kh_begin(h);
    while (k != kh_end(h)) {
        if (kh_exist(h, k)) {
            ep = kh_value(h, k);
            break;
        }
        k++;
    }
    if (ep) {
        kh_del(tl_ucp_ep_hash, h, k);
    }
    return ep;
}
#endif
