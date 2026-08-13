/**
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team_cache.h"
#include "ucc_team.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_compiler_def.h"
#include <string.h>

/* FNV-1a over {size, self_ep, members}, used only as a bucket key */
#define UCC_TEAM_CACHE_FNV1A_OFFSET 0xcbf29ce484222325ULL
#define UCC_TEAM_CACHE_FNV1A_PRIME  0x00000100000001b3ULL

static void ucc_team_cache_fnv1a_accumulate(
    uint64_t *h, const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t         b;

    for (b = 0; b < len; b++) {
        *h ^= (uint64_t)p[b];
        *h *= UCC_TEAM_CACHE_FNV1A_PRIME;
    }
}

ucc_status_t ucc_team_cache_identity_build(
    const ucc_team_params_t *params, ucc_team_cache_identity_t *identity)
{
    ucc_rank_t  size;
    ucc_rank_t  self_ep;
    ucc_rank_t *members;
    ucc_rank_t  i;
    uint64_t    h;

    if (!(params->mask & UCC_TEAM_PARAM_FIELD_EP_MAP)) {
        ucc_debug("team cache identity: no EP_MAP in params, not cacheable");
        return UCC_ERR_INVALID_PARAM;
    }

    size    = (ucc_rank_t)params->ep_map.ep_num;
    self_ep = (params->mask & UCC_TEAM_PARAM_FIELD_EP) ? (ucc_rank_t)params->ep
                                                       : UCC_RANK_INVALID;

    /* An external id is its own id/tag domain, so it is part of the identity */
    identity->ext_id = ((params->mask & UCC_TEAM_PARAM_FIELD_ID) &&
                        (params->id <= UCC_TEAM_ID_MAX))
                           ? (uint16_t)(((uint16_t)params->id) |
                                        UCC_TEAM_ID_EXTERNAL_BIT)
                           : 0;

    if (size < 1) {
        ucc_debug("team cache identity: empty ep_map");
        return UCC_ERR_INVALID_PARAM;
    }

    members = ucc_malloc(size * sizeof(*members), "team_cache_members");
    if (ucc_unlikely(!members)) {
        ucc_error(
            "failed to allocate %zu bytes for team cache members",
            size * sizeof(*members));
        return UCC_ERR_NO_MEMORY;
    }

    /* Copy the map so the identity never aliases caller-owned storage */
    for (i = 0; i < size; i++) {
        members[i] = ucc_ep_map_eval(params->ep_map, i);
    }

    identity->size            = size;
    identity->self_ep         = self_ep;
    identity->members         = members;
    identity->instance_cookie = 0; /* stamped later by the agreement vote */

    /* ext_id is deliberately not hashed, so the bucket is membership only */
    h                         = UCC_TEAM_CACHE_FNV1A_OFFSET;
    ucc_team_cache_fnv1a_accumulate(&h, &size, sizeof(size));
    ucc_team_cache_fnv1a_accumulate(&h, &self_ep, sizeof(self_ep));
    ucc_team_cache_fnv1a_accumulate(
        &h, members, (size_t)size * sizeof(*members));
    identity->hash = h;

    return UCC_OK;
}

int ucc_team_cache_identity_equal(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b)
{
    return ucc_team_cache_identity_equal_membership(a, b) &&
           a->ext_id == b->ext_id;
}

int ucc_team_cache_identity_equal_membership(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b)
{
    if (a->hash != b->hash || a->size != b->size || a->self_ep != b->self_ep) {
        return 0;
    }
    return memcmp(
               a->members, b->members, (size_t)a->size * sizeof(*a->members)) ==
           0;
}

void ucc_team_cache_identity_free(ucc_team_cache_identity_t *identity)
{
    if (!identity) {
        return;
    }
    ucc_free(identity->members);
    memset(identity, 0, sizeof(*identity));
}

int ucc_team_cache_is_cacheable(const ucc_team_params_t *params)
{
    /* These are not part of the identity, so a reuse could change semantics */
    uint64_t optional = UCC_TEAM_PARAM_FIELD_ORDERING |
                        UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS |
                        UCC_TEAM_PARAM_FIELD_SYNC_TYPE |
                        UCC_TEAM_PARAM_FIELD_P2P_CONN |
                        UCC_TEAM_PARAM_FIELD_MEM_PARAMS;

    return (params->mask & optional) == 0;
}
