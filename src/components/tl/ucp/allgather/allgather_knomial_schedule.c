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

#define UCC_TL_UCP_ALLGATHER_KN_LARGE_MSG_SIZE (1ull << 30)
#define UCC_TL_UCP_ALLGATHER_KN_MAX_AUTO_RADIX 9

/* Powers of two first, then remaining radices descending. */
static const ucc_kn_radix_t ucc_tl_ucp_allgather_knomial_small_msg_radices[] = {
    8, 4, 2, 9, 7, 6, 5, 3};

static void ucc_tl_ucp_allgather_knomial_search_small_msg(
    ucc_rank_t remaining, uint8_t first, ucc_kn_radix_t *current,
    uint8_t ncurrent, uint8_t nphases, unsigned fanout, ucc_kn_radix_t *best,
    unsigned *best_fanout)
{
    ucc_kn_radix_t radix;
    uint8_t        i;

    if (ncurrent == nphases) {
        if (remaining == 1 && fanout < *best_fanout) {
            for (i = 0; i < ncurrent; i++) {
                best[i] = current[i];
            }
            *best_fanout = fanout;
        }
        return;
    }
    if (remaining == 1 || fanout >= *best_fanout) {
        return;
    }
    for (i = first;
         i < sizeof(ucc_tl_ucp_allgather_knomial_small_msg_radices) /
                 sizeof(ucc_tl_ucp_allgather_knomial_small_msg_radices[0]);
         i++) {
        radix = ucc_tl_ucp_allgather_knomial_small_msg_radices[i];
        if (remaining % radix != 0) {
            continue;
        }
        current[ncurrent] = radix;
        ucc_tl_ucp_allgather_knomial_search_small_msg(
            remaining / radix,
            i,
            current,
            ncurrent + 1,
            nphases,
            fanout + radix - 1,
            best,
            best_fanout);
    }
}

static uint8_t ucc_tl_ucp_allgather_knomial_min_phases(ucc_rank_t size)
{
    uint8_t nphases = 0;

    while (size > 1) {
        size = size / UCC_TL_UCP_ALLGATHER_KN_MAX_AUTO_RADIX +
               (size % UCC_TL_UCP_ALLGATHER_KN_MAX_AUTO_RADIX != 0);
        nphases++;
    }
    return nphases;
}

static int ucc_tl_ucp_allgather_knomial_factor_large_msg(
    ucc_rank_t remaining, ucc_kn_radix_t *radices, uint8_t *nradices)
{
    ucc_kn_radix_t radix;

    for (radix = 2; radix <= UCC_TL_UCP_ALLGATHER_KN_MAX_AUTO_RADIX; radix++) {
        while (remaining % radix == 0) {
            radices[(*nradices)++] = radix;
            remaining /= radix;
        }
    }
    return remaining == 1;
}

int ucc_tl_ucp_allgather_knomial_select_radices(
    ucc_rank_t team_size, size_t msg_size, ucc_kn_radix_t *radix,
    ucc_kn_radix_t *radices, uint8_t *nradices)
{
    ucc_kn_radix_t current[UCC_KN_MAX_RADIX_PHASES];
    uint8_t        nphases;
    uint8_t        i;

    *nradices = 0;
    if (team_size < 2) {
        return 0;
    }

    if (msg_size >= UCC_TL_UCP_ALLGATHER_KN_LARGE_MSG_SIZE) {
        if (!ucc_tl_ucp_allgather_knomial_factor_large_msg(
                team_size, radices, nradices)) {
            *nradices = 0;
            return 0;
        }
    } else {
        for (nphases = ucc_tl_ucp_allgather_knomial_min_phases(team_size);
             nphases <= UCC_KN_MAX_RADIX_PHASES;
             nphases++) {
            unsigned best_fanout = UINT_MAX;

            ucc_tl_ucp_allgather_knomial_search_small_msg(
                team_size, 0, current, 0, nphases, 0, radices, &best_fanout);
            if (best_fanout != UINT_MAX) {
                *nradices = nphases;
                break;
            }
        }
        if (*nradices == 0) {
            return 0;
        }
    }

    for (i = 1; i < *nradices; i++) {
        if (radices[i] != radices[0]) {
            return 1;
        }
    }
    *radix    = radices[0];
    *nradices = 0;
    return 1;
}
