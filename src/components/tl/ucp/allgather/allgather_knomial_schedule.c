/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "allgather.h"

#include <stdlib.h>

ucc_status_t ucc_tl_ucp_allgather_knomial_parse_radices(
    const char *value, ucc_rank_t team_size, ucc_kn_radix_t *radices,
    uint8_t *nradices)
{
    const char   *p = value;
    char         *end;
    unsigned long parsed;
    ucc_rank_t    product = 1;
    uint8_t       n       = 0;

    *nradices             = 0;
    if (value == NULL || value[0] == '\0') {
        return UCC_ERR_NOT_FOUND;
    }

    while (*p != '\0') {
        if (n == UCC_KN_MAX_RADIX_PHASES) {
            return UCC_ERR_INVALID_PARAM;
        }
        parsed = strtoul(p, &end, 10);
        if (end == p || parsed < 2 || parsed > UINT16_MAX ||
            product > UCC_RANK_MAX / parsed) {
            return UCC_ERR_INVALID_PARAM;
        }
        radices[n++] = (ucc_kn_radix_t)parsed;
        product *= (ucc_rank_t)parsed;
        if (*end == '\0') {
            break;
        }
        if (*end != ',' || end[1] == '\0') {
            return UCC_ERR_INVALID_PARAM;
        }
        p = end + 1;
    }

    if (product != team_size) {
        return UCC_ERR_INVALID_PARAM;
    }
    *nradices = n;
    return UCC_OK;
}
