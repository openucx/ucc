/**
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEAM_CACHE_H_
#define UCC_TEAM_CACHE_H_

#include "config.h"
#include "ucc/api/ucc.h"
#include "ucc/api/ucc_status.h"
#include "utils/ucc_datastruct.h"
#include <stdint.h>

typedef struct ucc_team    ucc_team_t;
typedef struct ucc_context ucc_context_t;

/* Normalized team identity; @hash covers membership only, not @ext_id */
typedef struct ucc_team_cache_identity {
    uint64_t    hash;
    ucc_rank_t  size;
    ucc_rank_t  self_ep;
    uint16_t    ext_id; /* 0 for pool-id teams */
    /* Per-adoption stamp that identifies WHICH cached team was picked.
       Normally the vote's key lane carries @ext_id, which is enough to prove
       every rank selected the same entry. RESEAT breaks that: its lookup
       deliberately ignores @ext_id so a drifted id can still match on
       membership, and two ranks holding different dormant teams of identical
       membership would then agree on a lane that no longer distinguishes them.
       The cookie is stamped when a team is adopted and voted on instead, so the
       agreement still proves a common choice. 0 until the vote stamps it. */
    uint64_t    instance_cookie;
    ucc_rank_t *members; /* heap owned, length == size */
} ucc_team_cache_identity_t;

/* Materialize membership from @params into @identity, which owns its members */
ucc_status_t ucc_team_cache_identity_build(
    const ucc_team_params_t *params, ucc_team_cache_identity_t *identity);

/* Compare membership and ext_id */
int ucc_team_cache_identity_equal(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b);

/* Compare membership only */
int ucc_team_cache_identity_equal_membership(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b);

/* Free @identity->members and zero @identity; idempotent */
void ucc_team_cache_identity_free(ucc_team_cache_identity_t *identity);

/* Non-zero if no optional behavioral team param is set in params->mask */
int  ucc_team_cache_is_cacheable(const ucc_team_params_t *params);

#endif /* UCC_TEAM_CACHE_H_ */
