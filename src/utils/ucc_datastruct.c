/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_datastruct.h"
#include "ucc_malloc.h"
#include "ucc_compiler_def.h"
#include "ucc_log.h"

ucc_status_t ucc_mrange_uint_copy(ucc_mrange_uint_t       *dst,
                                  const ucc_mrange_uint_t *src)
{
    ucc_mrange_t *r, *r_dup;

    dst->default_value = src->default_value;
    ucc_list_head_init(&dst->ranges);
    ucc_list_for_each(r, &src->ranges, list_elem) {
        r_dup = ucc_malloc(sizeof(*r_dup), "range_dup");
        if (ucc_unlikely(!r_dup)) {
            ucc_error("failed to allocate %zd bytes for mrange",
                      sizeof(*r_dup));
            goto err;
        }
        r_dup->start  = r->start;
        r_dup->end    = r->end;
        r_dup->value  = r->value;
        r_dup->mtypes = r->mtypes;
        ucc_list_add_tail(&dst->ranges, &r_dup->list_elem);
    }

    return UCC_OK;
err:
    ucc_mrange_uint_destroy(dst);
    return UCC_ERR_NO_MEMORY;
}

void ucc_mrange_uint_destroy(ucc_mrange_uint_t *param)
{
    ucc_mrange_t *r, *r_tmp;

    ucc_list_for_each_safe(r, r_tmp, &param->ranges, list_elem) {
        ucc_list_del(&r->list_elem);
        ucc_free(r);
    }
}
