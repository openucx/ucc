/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team.h"
#include "ucc_lib.h"
#include "components/cl/ucc_cl.h"
#include "components/tl/ucc_tl.h"
#include "ucc_service_coll.h"
#include <inttypes.h>

static ucc_status_t ucc_team_alloc_id(ucc_team_t *team);
static void ucc_team_release_id(ucc_team_t *team);
static ucc_status_t ucc_team_destroy_single(ucc_team_h team);
static ucc_status_t ucc_team_teardown_for_rebuild(ucc_team_t *team);
static ucc_status_t ucc_team_reset_for_rebuild(
    ucc_context_t *context, ucc_team_t *team);

void ucc_copy_team_params(ucc_team_params_t *dst, const ucc_team_params_t *src)
{
    dst->mask = src->mask;
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_ORDERING, ordering);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS,
                            outstanding_colls);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_EP, ep);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_EP_RANGE, ep_range);
    //TODO do we need to copy ep_list ?
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_TEAM_SIZE,
                            team_size);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_SYNC_TYPE,
                            sync_type);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_OOB, oob);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_P2P_CONN, p2p_conn);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_MEM_PARAMS,
                            mem_params);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_EP_MAP, ep_map);
}

ucc_team_artifacts_t *ucc_team_artifacts_alloc(void)
{
    ucc_team_artifacts_t *a;

    a = ucc_calloc(1, sizeof(*a), "team_artifacts");
    if (!a) {
        ucc_error(
            "failed to allocate %zd bytes for team artifacts", sizeof(*a));
        return NULL;
    }
    a->refcount = 1;
    a->heap     = 1;
    ucc_spinlock_init(&a->lock, 0);
    return a;
}

/* Initialize an embedded (non-heap) holder in place: used for teams that will
   never be cached or shared, so no separate allocation is made. Same refcount/
   lock lifecycle as a heap holder, but ucc_team_artifacts_put cleans its
   contents without freeing the struct (it lives inside the ucc_team_t). */
void ucc_team_artifacts_init_inline(ucc_team_artifacts_t *a)
{
    memset(a, 0, sizeof(*a));
    a->refcount = 1;
    a->heap     = 0;
    ucc_spinlock_init(&a->lock, 0);
}

int ucc_team_artifacts_refcount(ucc_team_artifacts_t *artifacts)
{
    int refcount;

    ucc_assert(artifacts != NULL);
    ucc_spin_lock(&artifacts->lock);
    refcount = artifacts->refcount;
    ucc_spin_unlock(&artifacts->lock);
    return refcount;
}

ucc_team_artifacts_t *ucc_team_artifacts_get(ucc_team_artifacts_t *artifacts)
{
    ucc_assert(artifacts != NULL);
    ucc_spin_lock(&artifacts->lock);
    ucc_assert(artifacts->refcount > 0);
    artifacts->refcount++;
    ucc_spin_unlock(&artifacts->lock);
    return artifacts;
}

void ucc_team_artifacts_put(ucc_team_artifacts_t *artifacts)
{
    int refcount;

    if (!artifacts) {
        return;
    }
    ucc_spin_lock(&artifacts->lock);
    ucc_assert(artifacts->refcount > 0);
    refcount = --artifacts->refcount;
    ucc_spin_unlock(&artifacts->lock);

    if (refcount > 0) {
        return;
    }

    /* Last reference: free what the holder owns. topo teardown moved here from
       terminal teardown; ctx_ranks is the UCC-owned backing array (NULL when
       ctx_map aliases params->ep_map). ctx_map is POD, nothing to free. */
    ucc_topo_cleanup(artifacts->topo);
    ucc_free(artifacts->ctx_ranks);
    ucc_spinlock_destroy(&artifacts->lock);
    /* Embedded holders live inside their ucc_team_t and are freed with it; only
       heap holders are freed here. */
    if (artifacts->heap) {
        ucc_free(artifacts);
    }
}

int ucc_team_can_derive_from(const ucc_team_t *parent)
{
    /* Derivable iff fully built with a shareable holder: ctx_map built
       (ADDR_EXCHANGE ran) and, for size > 1, the topo materialized (build-once
       in ucc_team_create_cls). A size-1 team has no topo, so require topo only
       for size > 1. These are reused as-is, never recomputed. */
    if (parent == NULL || parent->artifacts == NULL) {
        return 0;
    }
    if (parent->state != UCC_TEAM_ACTIVE) {
        return 0;
    }
    if (parent->size > 1 && parent->artifacts->ctx_map.ep_num == 0) {
        /* ctx_map never filled (zero-initialized holder, e.g. a per-team
           addr_storage team that took the ucc_core_addr_exchange branch). A
           built multi-rank ctx_map always has ep_num == size >= 2. Not
           shareable - fall back to a full create. */
        return 0;
    }
    if (parent->size > 1 && parent->artifacts->topo == NULL) {
        return 0;
    }
    return 1;
}

/* Re-adopt guard for a DORMANT derived team: its borrowed holder is valid only
   while the parent stays live. Under the held cache->lock, checks a live
   membership-matching parent still exists AND the holder is still shared
   (refcount > 1). Either failing means full-create. */
static int ucc_team_derived_reuse_valid(
    ucc_team_cache_t *cache, const ucc_team_t *derived,
    ucc_team_cache_identity_t *id)
{
    ucc_team_t *parent;

    ucc_assert(derived->is_derived);
    if (derived->artifacts == NULL) {
        return 0;
    }
    parent = ucc_team_cache_lookup_live(cache, id);
    if (parent == NULL || parent == derived) {
        /* No live sibling parent pins the borrowed holder anymore. */
        return 0;
    }
    /* refcount > 1: at least the live parent + this dormant child reference the
       shared holder, so the borrowed pointer is not stale.  Read under the
       artifacts lock that guards refcount writes (ucc_team_artifacts_get/put);
       cache->lock does not synchronize those. */
    return ucc_team_artifacts_refcount(derived->artifacts) > 1;
}

/* Re-seat a DORMANT derived team's external id to @new_ext_id and fan it out to
   every CL/TL team (via update_id) plus the service team, so the team can be
   re-adopted for a same-membership create whose cid drifted (experimental RESEAT
   path). The new cid is unique among live comms, so the re-seated tags cannot
   alias. */
static void ucc_team_reseat_id(ucc_team_t *team, uint16_t new_ext_id)
{
    int i;

    ucc_assert(team->is_derived);

    team->id                    = new_ext_id;
    team->bp.id                 = new_ext_id;
    team->cache_identity.ext_id = new_ext_id;

    if (team->service_team) {
        UCC_TL_TEAM_IFACE(team->service_team)
            ->scoll.update_id(&team->service_team->super, new_ext_id);
    }
    for (i = 0; i < team->n_cl_teams; i++) {
        ucc_base_team_iface_t *cl_iface;

        if (!team->cl_teams[i]) {
            continue;
        }
        cl_iface = &UCC_CL_TEAM_IFACE(team->cl_teams[i])->team;
        if (cl_iface->update_id) {
            cl_iface->update_id(&team->cl_teams[i]->super, new_ext_id);
        }
    }
}

ucc_status_t ucc_team_init_derived(
    ucc_team_t *team, ucc_team_artifacts_t *held_artifacts, uint16_t parent_id)
{
    /* @held_artifacts is the parent's holder, already refcount-bumped by the
       caller under cache->lock; we CONSUME that reference here (the parent's
       lifetime is now irrelevant). Free the fresh unused holder create_post
       allocated for this team, since a derived team never rebuilds its
       artifacts. The last put (in ucc_team_destroy_single) frees the shared
       holder once every borrowing team is gone. */
    ucc_team_artifacts_put(team->artifacts);
    team->artifacts  = held_artifacts;
    team->is_derived = 1;
    team->parent_id  = parent_id;

    /* Derived teams are cached: the pending-insert marker set by create_post
       makes the generic ACTIVE-insert path chain this team onto the parent's
       bucket keyed by its OWN cid. It then takes the LIVE->DORMANT retain path
       and is re-adopted on the next identical create while the parent stays live
       and pins the holder; its eventual destroy puts the ref. */

    ucc_debug(
        "team %p: derived-create (shared artifacts %p, parent id=%u) - "
        "skipping ADDR_EXCHANGE + topo build",
        (void *)team,
        (void *)team->artifacts,
        parent_id);
    return UCC_OK;
}

ucc_status_t ucc_team_get_attr(ucc_team_h team, ucc_team_attr_t *team_attr)
{
    uint64_t supported_fields =
        UCC_TEAM_ATTR_FIELD_SIZE | UCC_TEAM_ATTR_FIELD_EP;

    if (team_attr->mask & ~supported_fields) {
        ucc_error("ucc_team_get_attr() is not implemented for specified field");
        return UCC_ERR_NOT_IMPLEMENTED;
    }

    if (team_attr->mask & UCC_TEAM_ATTR_FIELD_SIZE) {
        team_attr->size = team->size;
    }

    if (team_attr->mask & UCC_TEAM_ATTR_FIELD_EP) {
        team_attr->ep = team->rank;
    }

    return UCC_OK;
}

static ucc_status_t ucc_team_create_post_single(ucc_context_t *context,
                                                ucc_team_t *team)
{
    ucc_status_t status;

    if (context->service_team && team->size > 1) {
        /* Use internal service team for OOB, skip OOB if team size is 1 */
        ucc_subset_t subset = {.myrank     = team->rank,
                               .map.ep_num = team->size,
                               .map.type   = UCC_EP_MAP_FULL};
        status = ucc_internal_oob_init(team, subset, &team->bp.params.oob);
        if (UCC_OK != status) {
            return status;
        }
        team->bp.params.mask |= UCC_TEAM_PARAM_FIELD_OOB;
    }

    /* A team of size > 1 needs an OOB to bootstrap its TL teams (address
       exchange, etc.). It comes either from the
       context service team (above) or from a user-provided team OOB. If
       neither is available - e.g. the service team could not be created
       because tl/ucp failed on some ranks - the team cannot be created. */
    if (team->size > 1 && !(team->bp.params.mask & UCC_TEAM_PARAM_FIELD_OOB)) {
        ucc_error("cannot create team of size %d: no OOB available (context "
                  "service team unavailable and no team OOB provided)",
                  team->size);
        return UCC_ERR_NO_RESOURCE;
    }

    team->cl_teams = ucc_malloc(sizeof(ucc_cl_team_t *) * context->n_cl_ctx);
    if (!team->cl_teams) {
        ucc_error("failed to allocate %zd bytes for cl teams array",
                  sizeof(ucc_cl_team_t *) * context->n_cl_ctx);
        return UCC_ERR_NO_MEMORY;
    }
    team->bp.rank                 = team->rank;
    team->bp.size                 = team->size;
    team->bp.team                 = team;
    team->bp.map.type             = UCC_EP_MAP_FULL;
    team->bp.map.ep_num           = team->size;
    if (team->is_derived) {
        /* Derived team: artifacts were borrowed, so ctx_map/topo are already
           materialized. Skip UCC_TEAM_ADDR_EXCHANGE (which builds ctx_map) and
           start at UCC_TEAM_SERVICE_TEAM: it STILL runs ALLOC_ID (own team-id ->
           own tag/seq domain), CL/TL team creation, and score_map build. A
           size-1 team has no exchange regardless, so its start state is
           unchanged. */
        team->state = (team->size > 1) ? UCC_TEAM_SERVICE_TEAM
                                       : UCC_TEAM_CL_CREATE;
    } else {
        team->state = (team->size > 1) ? UCC_TEAM_ADDR_EXCHANGE
                                       : UCC_TEAM_CL_CREATE;
    }
    team->last_team_create_posted = -1;
    return UCC_OK;
}

/* Allocate and initialize a fresh ucc_team shell (no CL/TL build yet): artifacts
   holder (heap when @id_built - shareable/cacheable; else embedded inline),
   contexts array, base params, external id. On @id_built the built identity is
   MOVED onto the team (caller's @id is zeroed) and cache_pending_insert is set.
   Returns the team, or NULL with *status_out set (all its own allocations freed).
   The caller still owns any pinned derive_artifacts. Shared by the direct-reuse
   miss path and the agreement path. */
static ucc_team_t *ucc_team_alloc_shell(
    ucc_context_h *contexts, uint32_t num_contexts,
    const ucc_team_params_t *params, uint64_t team_size, uint64_t team_rank,
    int id_built, ucc_team_cache_identity_t *id, ucc_status_t *status_out)
{
    ucc_team_t *team;

    team = ucc_calloc(1, sizeof(ucc_team_t), "ucc_team");
    if (!team) {
        ucc_error(
            "failed to allocate %zd bytes for ucc team", sizeof(ucc_team_t));
        *status_out = UCC_ERR_NO_MEMORY;
        return NULL;
    }
    if (id_built) {
        team->artifacts = ucc_team_artifacts_alloc();
        if (!team->artifacts) {
            ucc_free(team);
            *status_out = UCC_ERR_NO_MEMORY;
            return NULL;
        }
    } else {
        team->artifacts = &team->artifacts_inline;
        ucc_team_artifacts_init_inline(team->artifacts);
    }
    team->runtime_oob  = params->oob;
    team->num_contexts = num_contexts;
    team->size         = (ucc_rank_t)team_size;
    team->rank         = (ucc_rank_t)team_rank;
    team->seq_num      = 0;
    team->refcount     = 1;
    if (id_built) {
        team->cache_identity = *id; /* move: caller's id owns nothing now */
        team->cache_pending_insert = 1;
        memset(id, 0, sizeof(*id));
    } else {
        memset(&team->cache_identity, 0, sizeof(team->cache_identity));
        team->cache_pending_insert = 0;
    }
    ucc_list_head_init(&team->cache_link);
    ucc_list_head_init(&team->bucket_link);
    team->cache_state = UCC_TEAM_CACHE_STATE_NONE;
    team->contexts    = ucc_malloc(
        sizeof(ucc_context_t *) * num_contexts, "ucc_team_ctx");
    if (!team->contexts) {
        ucc_error(
            "failed to allocate %zd bytes for ucc team contexts array",
            sizeof(ucc_context_t *) * num_contexts);
        ucc_team_cache_identity_free(&team->cache_identity);
        ucc_team_artifacts_put(team->artifacts);
        ucc_free(team);
        *status_out = UCC_ERR_NO_MEMORY;
        return NULL;
    }
    memcpy(team->contexts, contexts, sizeof(ucc_context_t *) * num_contexts);
    ucc_copy_team_params(&team->bp.params, params);
    if ((params->mask & UCC_TEAM_PARAM_FIELD_ID) &&
        (params->id <= UCC_TEAM_ID_MAX)) {
        team->id = ((uint16_t)params->id) | UCC_TEAM_ID_EXTERNAL_BIT;
    }
    *status_out = UCC_OK;
    return team;
}

/* Evict a stale dormant-derived candidate (its live parent is gone): detach from
   the table and registry, free its identity, park it for teardown, and rebook the
   lookup's provisional hit as a miss. Caller holds cache->lock. */
static void ucc_team_cache_evict_stale_derived(
    ucc_team_cache_t *cache, ucc_team_t *cached)
{
    ucc_team_cache_table_erase(cache, cached);
    ucc_team_cache_registry_remove(cache, cached);
    ucc_team_cache_identity_free(&cached->cache_identity);
    cached->cache_state = UCC_TEAM_CACHE_STATE_NONE;
    ucc_list_add_tail(&cache->pending_destroy, &cached->cache_link);
    cache->stats.evictions++;
    ucc_assert(cache->stats.hits > 0);
    cache->stats.hits--;
    cache->stats.misses++;
}

/* Rebook a lookup miss as a hit: a dormant-derived reseat reused a cached team the
   exact lookup had missed (cid drift). Caller holds cache->lock. */
static void ucc_team_cache_rebook_miss_as_hit(ucc_team_cache_t *cache)
{
    ucc_assert(cache->stats.misses > 0);
    cache->stats.misses--;
    cache->stats.hits++;
}

/* Undo a posted-vote setup on a fatal post failure. EXACT: return the reserved
   candidate to DORMANT (refcount was never bumped). DERIVED/MISS: release any
   pinned parent artifacts and free the fresh shell. */
static void ucc_team_agreement_rollback(
    ucc_team_cache_t *cache, ucc_team_t *handle, ucc_team_cache_action_t action)
{
    if (action == UCC_TEAM_CACHE_ACTION_EXACT_REUSE ||
        action == UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED) {
        /* Both reserved a DORMANT candidate; return it to DORMANT (refcount was
           never bumped). A RESEAT_DERIVED candidate is NOT re-seated until the
           vote agrees, so its id/tag domain is unchanged and needs no undo. */
        ucc_spin_lock(&cache->lock);
        handle->cache_state = UCC_TEAM_CACHE_STATE_DORMANT;
        ucc_team_cache_registry_make_dormant(cache, handle);
        ucc_spin_unlock(&cache->lock);
        return;
    }
    if (action == UCC_TEAM_CACHE_ACTION_DERIVED_FROM_LIVE &&
        handle->cache_derive_artifacts) {
        ucc_team_artifacts_put(handle->cache_derive_artifacts);
        handle->cache_derive_artifacts = NULL;
    }
    ucc_team_destroy_single(handle);
}

/* Agreement create path (cacheable size>1 team). Classify locally, RESERVE a
   dormant EXACT candidate or allocate a fresh shell (DERIVED/MISS), then POST a
   member-scoped vote (progressed in ucc_team_create_test). Returns @new_team set
   to the handle, or an error on fatal vote-post failure. Identity-build failure is
   non-fatal: this rank still posts a MISS vote so peers do not hang. */
static ucc_status_t ucc_team_agreement_create_post(
    ucc_context_h *contexts, uint32_t num_contexts,
    const ucc_team_params_t *params, uint64_t team_size, uint64_t team_rank,
    ucc_team_cache_t *cache, ucc_team_h *new_team)
{
    ucc_team_cache_identity_t id;
    int                       id_built = 0;
    ucc_team_t               *cached   = NULL;
    ucc_team_t               *handle;
    ucc_team_artifacts_t     *derive_artifacts = NULL;
    uint16_t                  derive_parent_id = 0;
    ucc_team_cache_action_t   action           = UCC_TEAM_CACHE_ACTION_MISS;
    uint64_t                  key              = 0;
    /* Per-instance cookie vote payload. @cookie proves cross-rank agreement on the
       reuse/reseat candidate instance; @parent_cookie the derive/reseat parent's;
       @proposed_cookie carries rank-0's fresh-instance proposal. */
    uint64_t                  cookie           = 0;
    uint64_t                  parent_cookie    = 0;
    uint16_t                  reseat_new_id    = 0;
    int                       is_rank0         = (team_rank == 0);
    uint64_t                  proposed_cookie  = 0;
    ucc_subset_t              subset;
    ucc_status_t              status;

    if (ucc_team_cache_identity_build(params, &id) == UCC_OK) {
        id_built = 1;
        ucc_spin_lock(&cache->lock);
        /* Rank 0 always reserves a fresh cookie so a MISS outcome has a value to
           adopt; it is distributed via the vote's lane [9]. Non-fresh outcomes
           (EXACT/DERIVED/RESEAT) simply keep the candidate/instance cookie, so
           the reserved value is harmlessly skipped (monotonic, never recycled). */
        if (is_rank0) {
            proposed_cookie = ucc_team_cache_next_cookie(cache);
        }
        cached = ucc_team_cache_lookup(cache, &id);
        if (cached != NULL && cached->is_derived &&
            !ucc_team_derived_reuse_valid(cache, cached, &id)) {
            /* Stale dormant-derived (parent gone): evict, classify as miss. */
            ucc_team_cache_evict_stale_derived(cache, cached);
            cached = NULL;
        }
        if (cached != NULL) {
            /* EXACT reuse: full membership+ext_id match. RESERVE for the vote
               (off dormant, refcount unchanged). The instance cookie is NOT
               needed here - the exact identity (incl. ext_id) already pins the
               instance - so vote lane [5] stays 0/all-ones. */
            action = UCC_TEAM_CACHE_ACTION_EXACT_REUSE;
            key    = cached->id;
            ucc_team_cache_registry_make_reserved(cache, cached);
            cached->cache_state = UCC_TEAM_CACHE_STATE_RESERVED;
        } else if (
            cache->reseat &&
            (cached = ucc_team_cache_lookup_dormant_derived(cache, &id)) !=
                NULL &&
            cached->cache_identity.instance_cookie != 0 &&
            ucc_team_derived_reuse_valid(cache, cached, &id)) {
            /* RESEAT_DERIVED (UCC_TEAM_CACHE_RESEAT, experimental, default off):
               the exact lookup missed (cid drifted) but a DORMANT derived team of
               the same membership exists with a still-valid holder. Agreement-safe
               only because the vote reconciles the candidate's per-instance cookie:
               a membership-only lookup can return different physical instances per
               rank (cids recycle), so the cookie proves all ranks picked the same
               one. Decline candidates with cookie==0: those are from the direct/
               size==1 path and lack a per-instance stamp, so the cookie vote would
               accept any zero from any candidate - defeating the proof. The reseat
               is committed under the lock in create_test only on unanimous agreement. */
            action        = UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED;
            key           = cached->id;
            cookie        = cached->cache_identity.instance_cookie;
            parent_cookie = cached->cache_parent_instance_cookie;
            reseat_new_id = id.ext_id;
            ucc_team_cache_registry_make_reserved(cache, cached);
            cached->cache_state = UCC_TEAM_CACHE_STATE_RESERVED;
            /* A global-MISS rollback in create_test re-corrects this to a miss. */
            ucc_team_cache_rebook_miss_as_hit(cache);
        } else {
            ucc_team_t *live;

            cached = NULL; /* dormant-derived probe above may have set it */
            live   = cache->derived ? ucc_team_cache_lookup_live(cache, &id)
                                    : NULL;
            if (live != NULL && ucc_team_can_derive_from(live)) {
                action           = UCC_TEAM_CACHE_ACTION_DERIVED_FROM_LIVE;
                key              = live->id;
                parent_cookie    = live->cache_identity.instance_cookie;
                derive_artifacts = ucc_team_artifacts_get(live->artifacts);
                derive_parent_id = live->id;
            }
        }
        ucc_spin_unlock(&cache->lock);
    }

    if (action == UCC_TEAM_CACHE_ACTION_EXACT_REUSE ||
        action == UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED) {
        handle = cached;
        ucc_team_cache_identity_free(
            &id); /* candidate carries its own identity */
        handle->cache_reseat_new_id = reseat_new_id;
        /* Refresh the membership map so a global-miss rebuild (ucc_team_exchange)
           reads a valid ep_map, not the candidate's stale one from a since-freed
           communicator. On reuse this is not read operationally (owned ctx_map). */
        handle->bp.params.ep_map    = params->ep_map;
        handle->bp.params.mask |= UCC_TEAM_PARAM_FIELD_EP_MAP;
    } else {
        handle = ucc_team_alloc_shell(
            contexts,
            num_contexts,
            params,
            team_size,
            team_rank,
            id_built,
            &id,
            &status);
        if (handle == NULL) {
            if (id_built) {
                ucc_team_cache_identity_free(&id);
            }
            if (derive_artifacts) {
                ucc_team_artifacts_put(derive_artifacts);
            }
            return status;
        }
        status = ucc_team_create_post_single(contexts[0], handle);
        if (status < 0) {
            if (derive_artifacts) {
                ucc_team_artifacts_put(derive_artifacts);
            }
            ucc_team_destroy_single(handle);
            return status;
        }
        if (action == UCC_TEAM_CACHE_ACTION_DERIVED_FROM_LIVE) {
            handle->cache_derive_artifacts       = derive_artifacts;
            handle->cache_derive_parent_id       = derive_parent_id;
            handle->cache_parent_instance_cookie = parent_cookie;
        }
    }

    handle->cache_local_action = action;
    /* Fill the vote: cookie/parent_cookie lanes prove same-instance selection for
       RESEAT/DERIVED; rank-0 proposes the fresh-instance cookie on lane [9]. */
    ucc_team_cache_vote_fill(
        handle->cache_vote_in,
        action != UCC_TEAM_CACHE_ACTION_MISS,
        action,
        key,
        cookie,
        parent_cookie,
        is_rank0,
        proposed_cookie);
    subset.myrank = handle->rank;
    subset
        .map = params
                   ->ep_map; /* member index -> ctx rank (valid this create) */
    /* This vote and the later UCC_TEAM_ALLOC_ID allreduce both ride
       ctx->service_team on the shared UCC_TL_UCP_SERVICE_TAG but never alias: per
       team the vote is finalized before the state machine reaches ALLOC_ID (strictly
       sequential), and the ucc.h contract permits only one outstanding create/destroy
       per context, so no concurrent create adds another service op. */
    status = ucc_service_allreduce_ctx(
        handle,
        &handle->cache_vote_req,
        handle->cache_vote_in,
        handle->cache_vote_out,
        UCC_DT_UINT64,
        UCC_TEAM_CACHE_VOTE_LANES,
        UCC_OP_BAND,
        subset);
    if (status < 0) {
        ucc_team_agreement_rollback(cache, handle, action);
        return status;
    }
    handle->state = UCC_TEAM_CACHE_AGREE;
    *new_team     = handle;
    /* create_post returns UCC_OK (the vote is posted); the async work - the vote
       and the resulting build/reuse - is driven by ucc_team_create_test, which
       returns UCC_INPROGRESS until the team reaches ACTIVE. */
    return UCC_OK;
}

ucc_status_t ucc_team_create_post(ucc_context_h *contexts, uint32_t num_contexts,
                                  const ucc_team_params_t *params,
                                  ucc_team_h *new_team)
{
    uint64_t                  team_size = 0;
    uint64_t                  team_rank = UINT64_MAX;
    ucc_team_cache_t         *cache     = NULL;
    ucc_team_cache_identity_t id;
    int                       id_built         = 0;
    /* Held (refcount-bumped) reference to a coexisting LIVE parent's shared
       artifacts, captured under cache->lock so it cannot be freed across the
       unlock; consumed by ucc_team_init_derived or released on a pre-init
       failure path. No pointer to the parent team is kept. */
    ucc_team_artifacts_t     *derive_artifacts = NULL;
    uint16_t                  derive_parent_id = 0;
    int                       reseat_needed    = 0;
    uint16_t                  reseat_new_id    = 0;
    ucc_team_t               *team;
    ucc_status_t              status;

    if (num_contexts < 1) {
        return UCC_ERR_INVALID_PARAM;
    } else if (num_contexts > 1) {
        ucc_error("team creation from multiple contexts is not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (params->mask & UCC_TEAM_PARAM_FIELD_TEAM_SIZE) {
        team_size = params->team_size;
    }

    if (params->mask & UCC_TEAM_PARAM_FIELD_OOB) {
        if (team_size > 0 && params->oob.n_oob_eps != team_size) {
            ucc_error(
                "inconsistent team_sizes provided as params.team_size %llu "
                "and params.oob.n_oob_eps %llu",
                (unsigned long long)params->team_size,
                (unsigned long long)params->oob.n_oob_eps);
            return UCC_ERR_INVALID_PARAM;
        }
        team_size = params->oob.n_oob_eps;
    }

    if (params->mask & UCC_TEAM_PARAM_FIELD_EP_MAP) {
        if (team_size > 0 && params->ep_map.ep_num != team_size) {
            ucc_error(
                "inconsistent team_sizes provided as params.team_size %llu "
                "and/or params.oob.n_oob_eps %llu and/or ep_map.ep_num %llu",
                (unsigned long long)params->team_size,
                (unsigned long long)params->oob.n_oob_eps,
                (unsigned long long)params->ep_map.ep_num);
            return UCC_ERR_INVALID_PARAM;
        }
        team_size = params->ep_map.ep_num;
    }
    if (team_size < 1) {
        ucc_warn("minimal size of UCC team is 1, provided %llu",
                 (unsigned long long)team_size);
        return UCC_ERR_INVALID_PARAM;
    }

    if ((params->mask & UCC_TEAM_PARAM_FIELD_EP) &&
        (params->mask & UCC_TEAM_PARAM_FIELD_EP_RANGE) &&
        (params->ep_range == UCC_COLLECTIVE_EP_RANGE_CONTIG)) {
        if ((params->mask & UCC_TEAM_PARAM_FIELD_OOB) &&
            (params->oob.oob_ep != params->ep)) {
            ucc_error(
                "inconsistent EP value is provided as params.ep %llu "
                "and params.oob.oob_ep %llu",
                (unsigned long long)params->ep,
                (unsigned long long)params->oob.oob_ep);
            return UCC_ERR_INVALID_PARAM;
        }
        team_rank = params->ep;
    } else if (params->mask & UCC_TEAM_PARAM_FIELD_OOB) {
        team_rank = params->oob.oob_ep;
    }

    if (team_rank == UINT64_MAX) {
        /* Neither EP nor OOB_EP is provided, can't assign the rank */
        ucc_error("either UCC_TEAM_PARAM_FIELD_EP(RANGE) "
                  "or UCC_TEAM_PARAM_FIELD_OOB must be provided");
        return UCC_ERR_INVALID_PARAM;
    }

    if (team_size > (uint64_t)UCC_RANK_MAX) {
        ucc_error("team size is too large: %llu, max supported %u",
                  (unsigned long long)team_size, UCC_RANK_MAX);
        return UCC_ERR_INVALID_PARAM;
    }

    if (team_rank > (uint64_t)UCC_RANK_MAX) {
        ucc_error("team rank is too large: %llu, max supported %u",
                  (unsigned long long)team_rank, UCC_RANK_MAX);
        return UCC_ERR_INVALID_PARAM;
    }

    /* Cross-rank agreement (UCC_TEAM_CACHE_AGREEMENT, default on): for a cacheable
       size>1 team on a cache-enabled context, every member votes on the cache
       action so all reach an identical reuse-vs-fresh decision (no split hit/miss
       deadlock). EP_MAP is required (the membership source and the vote's
       member->ctx-rank map); teams without it are uncacheable and fall through. The
       vote is posted here and progressed in ucc_team_create_test. When the knob is
       off, size>1 cacheable teams take the direct-reuse path below (faster, but
       unsafe for overlapping subcommunicators - the caller opted out). */
    cache = ((ucc_context_t *)contexts[0])->team_cache;
    if (cache != NULL && cache->agreement &&
        ucc_team_cache_is_cacheable(params) && team_size > 1 &&
        (params->mask & UCC_TEAM_PARAM_FIELD_EP_MAP)) {
        return ucc_team_agreement_create_post(
            contexts,
            num_contexts,
            params,
            team_size,
            team_rank,
            cache,
            new_team);
    }

    /* Direct-reuse path (size==1 or non-cacheable, or agreement off): build the
       identity and look up a DORMANT identical-membership team. On a hit, re-adopt
       the same team object; on a miss, fall through to a normal create and remember
       the identity so it is inserted once ACTIVE. */
    if (cache != NULL && ucc_team_cache_is_cacheable(params)) {
        status = ucc_team_cache_identity_build(params, &id);
        if (status == UCC_OK) {
            ucc_team_t *cached;

            id_built = 1;

            /* Lookup + adopt MUST be atomic under a single held cache->lock so a
               concurrent create cannot adopt the same dormant team twice and a
               concurrent destroy cannot free/re-adopt it mid-window.
               ucc_team_cache_lookup / _get / registry_make_live all require the
               lock held; drop it before any drain work. */
            ucc_spin_lock(&cache->lock);
            cached = ucc_team_cache_lookup(cache, &id);
            if (cached != NULL && cached->is_derived &&
                !ucc_team_derived_reuse_valid(cache, cached, &id)) {
                /* Dormant derived hit whose parent is gone (stale holder): evict
                   and fall through to a full create. */
                ucc_debug(
                    "team cache: dormant derived team %p parent gone - "
                    "evicting stale child, falling back to full create",
                    (void *)cached);
                ucc_team_cache_evict_stale_derived(cache, cached);
                cached = NULL;
            }
            if (cache->reseat && cached == NULL) {
                /* RESEAT (UCC_TEAM_CACHE_RESEAT, experimental, default off): the
                   exact lookup missed. Re-adopt a DORMANT DERIVED team of the same
                   membership whose ext_id (cid) drifted, if its borrowed holder is
                   still valid; re-seat its id/tag domain to the new cid below. */
                ucc_team_t *reseat = ucc_team_cache_lookup_dormant_derived(
                    cache, &id);

                if (reseat != NULL &&
                    ucc_team_derived_reuse_valid(cache, reseat, &id)) {
                    cached        = reseat;
                    reseat_needed = 1;
                    reseat_new_id = id.ext_id;
                    ucc_team_cache_rebook_miss_as_hit(cache);
                }
            }
            if (cached != NULL) {
                /* DORMANT hit: claim the team (refcount++ -> LIVE) and move it
                   dormant -> live under the lock. A re-adopted derived team keeps
                   its borrowed holder and its own team id. */
                ucc_team_cache_get(cached);
                ucc_team_cache_registry_make_live(cache, cached);

                /* RESEAT: re-adopted a same-membership derived team whose cid
                   drifted. Re-seat its id and TL/service tag domains to the new cid
                   under cache->lock so no user observes a half-reseated team.
                   update_id does scalar writes only; seq_num is NOT reset. */
                if (reseat_needed &&
                    cached->cache_identity.ext_id != reseat_new_id) {
                    ucc_debug(
                        "team cache: reseat team %p id 0x%x -> 0x%x "
                        "(cid drift, same membership)",
                        (void *)cached,
                        (unsigned)cached->cache_identity.ext_id,
                        (unsigned)reseat_new_id);
                    ucc_team_reseat_id(cached, reseat_new_id);
                }
            } else {
                /* No DORMANT match: a LIVE identical-membership match means a
                   second simultaneously-live comm over the same members (e.g.
                   MPI_Comm_dup) -> derived-create. Classified under the same lock
                   as the dormant lookup. Gated by UCC_TEAM_CACHE_DERIVED (default
                   on); when off or not derivable, this is a normal full create. */
                ucc_team_t *live = cache->derived
                                       ? ucc_team_cache_lookup_live(cache, &id)
                                       : NULL;

                if (live != NULL && ucc_team_can_derive_from(live)) {
                    /* Pin the parent's artifacts under the lock (keep only the held
                       holder + parent id, never the parent pointer); consumed by
                       ucc_team_init_derived or released on a pre-init failure. */
                    derive_artifacts = ucc_team_artifacts_get(live->artifacts);
                    derive_parent_id = live->id;
                }
            }
            ucc_spin_unlock(&cache->lock);

            if (cached != NULL) {
                ucc_debug(
                    "team cache: dormant reuse / hit, team %p (hash=0x%" PRIx64
                    ")",
                    (void *)cached,
                    id.hash);
                /* The built identity only keyed the lookup; the cached team carries
                   its own (re-seated above if the cid drifted). */
                ucc_team_cache_identity_free(&id);

                /* No drain needed: the UCC API forbids collectives after
                   ucc_team_destroy, so a dormant team has no in-flight work; the
                   context progress queue is deliberately NOT drained (it is global
                   and could block on unrelated teams). seq_num is not reset, so it
                   keeps advancing and cannot alias the reused team's fresh tags.
                   The team is already ACTIVE, so ucc_team_create_test
                   short-circuits; bp.params is left as-is (identical membership). */
                *new_team = cached;
                return UCC_OK;
            }
            /* Miss: keep `id` (id_built stays set) and move it onto the new
               team below so it can be inserted at ACTIVE. */
        } else {
            /* Unbuildable identity (e.g. no EP_MAP): uncached create. */
            cache = NULL;
        }
    } else {
        cache = NULL;
    }

    team = ucc_team_alloc_shell(
        contexts,
        num_contexts,
        params,
        team_size,
        team_rank,
        id_built,
        &id,
        &status);
    if (team == NULL) {
        if (id_built) { /* freed here if the failure was before the id move */
            ucc_team_cache_identity_free(&id);
        }
        if (derive_artifacts) { /* release the pinned parent artifacts */
            ucc_team_artifacts_put(derive_artifacts);
        }
        return status;
    }
    /* Derived-create: a coexisting LIVE identical-membership parent was found and
       its artifacts pinned above. Borrow them and take the shortened create path
       (skip ADDR_EXCHANGE + topo build). Applies to EXTERNAL-id teams too (e.g.
       MPI_Comm_dup). Must run BEFORE ucc_team_create_post_single, which reads
       team->is_derived. */
    if (derive_artifacts != NULL) {
        ucc_team_init_derived(team, derive_artifacts, derive_parent_id);
    }
    status    = ucc_team_create_post_single(contexts[0], team);
    *new_team = team;
    return status;
}

static ucc_status_t ucc_team_create_service_team(ucc_context_t *context,
                                                 ucc_team_t *team)
{
    ucc_status_t status;
    if (context->service_team) {
        /* Global single service team is allocated on ucc_context.
           UCC_INTERNAL_OOB is enabled. Don't need another service team */
        return UCC_OK;
    }
    if (!team->service_team) {
        ucc_base_team_params_t b_params;
        ucc_base_team_t *      b_team;
        status = ucc_tl_context_get(context, "ucp", &context->service_ctx);
        if (UCC_OK != status) {
            ucc_warn("TL UCP context is not available, "
                     "service team can not be created");
            return status;
        }
        memcpy(&b_params, &team->bp, sizeof(ucc_base_team_params_t));
        b_params.scope =
            UCC_CL_LAST + 1; // CORE scope id - never overlaps with CL type
        b_params.scope_id = 0;
        b_params.id       = 0;
        b_params.team     = team;
        b_params.map.type = UCC_EP_MAP_FULL;
        status            = UCC_TL_CTX_IFACE(context->service_ctx)
                     ->team.create_post(&context->service_ctx->super, &b_params,
                                        &b_team);
        if (UCC_OK != status) {
            ucc_error("tl ucp service team create post failed");
            return status;
        }
        team->service_team = ucc_derived_of(b_team, ucc_tl_team_t);
    }
    status = UCC_TL_CTX_IFACE(context->service_ctx)
        ->team.create_test(&team->service_team->super);
    if (status < 0) {
        team->service_team = NULL;
        ucc_error("failed to create service tl ucp team");
    }
    return status;
}

static ucc_status_t ucc_team_create_cls(ucc_context_t *context,
                                        ucc_team_t *team)
{
    ucc_cl_iface_t  *cl_iface;
    ucc_base_team_t *b_team;
    ucc_status_t     status;
    ucc_subset_t     subset;
    int              i;

    if (context->topo && !UCC_TEAM_TOPO(team) && team->size > 1) {
        /* Context->topo is not NULL if any of the enabled CLs
           reported topo_required through the lib_attr */
        subset.map    = UCC_TEAM_CTX_MAP(team);
        subset.myrank = team->rank;
        status = ucc_topo_init(subset, context->topo, &UCC_TEAM_TOPO(team));
        if (UCC_OK != status) {
            ucc_warn("failed to init team topo");
        } else if (team->cache_pending_insert) {
            /* Build-once guard, only for teams that will actually be cached/shared
               (cache_pending_insert): materialize the lazily-filled sbgp/layout
               state now, while create_post is serialized per context, so the topo
               is effectively immutable and a derived team can share it read-only
               without racing on first-touch lazy writes under THREAD_MULTIPLE.
               Ordinary (non-cacheable / cache-disabled) teams keep the original
               lazy first-touch behavior - no eager work, no new failure modes. */
            status = ucc_topo_prepare_shared(UCC_TEAM_TOPO(team));
            if (UCC_OK != status) {
                /* Could not fully materialize the shared topo. Don't cache/share
                   a half-built topo: fall back to a normal, un-shared team whose
                   remaining sbgps fill lazily on first touch (single-user, so no
                   THREAD_MULTIPLE race). */
                ucc_warn(
                    "failed to prepare shared topo (%s); team %p will not "
                    "be cached",
                    ucc_status_string(status),
                    (void *)team);
                team->cache_pending_insert = 0;
            }
        }
    }

    if (team->last_team_create_posted >= 0) {
        cl_iface = UCC_CL_CTX_IFACE(context->cl_ctx[team->last_team_create_posted]);
        b_team   = &team->cl_teams[team->last_team_create_posted]->super;
        status   = cl_iface->team.create_test(b_team);
        if (status < 0) {
            team->n_cl_teams--;
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            cl_iface->team.destroy(b_team);
        } else if (status == UCC_INPROGRESS) {
            return status;
        }
    }

    for (i = team->last_team_create_posted + 1; i < context->n_cl_ctx; i++) {
        cl_iface = UCC_CL_CTX_IFACE(context->cl_ctx[i]);
        status   = cl_iface->team.create_post(&context->cl_ctx[i]->super,
                                              &team->bp, &b_team);
        if (status != UCC_OK) {
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            continue;
        }
        status = cl_iface->team.create_test(b_team);
        if (status < 0) {
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            cl_iface->team.destroy(b_team);
            continue;
        }
        team->cl_teams[team->n_cl_teams++] =
            ucc_derived_of(b_team, ucc_cl_team_t);
        if (status == UCC_INPROGRESS) {
            team->last_team_create_posted = i;
            /* workaround to fix oob allgather issue if multiple teams use it
               simultaneously*/
            return UCC_INPROGRESS;
        }
    }
    if (0 == team->n_cl_teams) {
        ucc_error("no CL teams were created");
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_team_exchange(ucc_context_t *context,
                                             ucc_team_t *   team)
{
    ucc_team_oob_coll_t oob = team->runtime_oob;
    ucc_status_t        status;

    if (!context->addr_storage.storage) {
        /* There is no addresses collected on the context
           (can be, e.g., if user did not pass OOB for ctx
           creation). Need to exchange addresses here */
        return ucc_core_addr_exchange(context, &oob, &team->addr_storage);
    }
    /* We only need to exchange ctx_ranks and build map to ctx array */
    ucc_assert(context->addr_storage.storage != NULL);
    if (team->bp.params.mask & UCC_TEAM_PARAM_FIELD_EP_MAP) {
        if (team->cache_pending_insert) {
            /* A cacheable team can outlive the caller's communicator (it is
               retained dormant and re-adopted after MPI_Comm_free). The caller's
               params.ep_map may be a UCC_EP_MAP_CB closure whose cb_ctx IS that
               communicator (OMPI coll/ucc passes cb_ctx = comm), so aliasing it
               would leave the operational map dangling once the comm is freed.
               Materialize the membership into a UCC-owned ctx_ranks array NOW
               (while the caller map is still valid) and build a self-contained map
               over it; ownership then lives in the artifacts holder, freed by
               ucc_team_artifacts_put. ucc_ep_map_from_array collapses a
               contiguous/full pattern to a STRIDED/FULL map (freeing ctx_ranks) or
               keeps the owned array in an ARRAY map - either way no caller pointer
               is retained. */
            ucc_rank_t i;

            if (!UCC_TEAM_CTX_RANKS(team)) {
                UCC_TEAM_CTX_RANKS(team) = ucc_malloc(
                    team->size * sizeof(ucc_rank_t), "ctx_ranks");
                if (!UCC_TEAM_CTX_RANKS(team)) {
                    ucc_error(
                        "failed to allocate %zd bytes for ctx ranks array",
                        team->size * sizeof(ucc_rank_t));
                    return UCC_ERR_NO_MEMORY;
                }
                for (i = 0; i < team->size; i++) {
                    UCC_TEAM_CTX_RANKS(team)
                    [i] = (ucc_rank_t)ucc_ep_map_eval(
                        team->bp.params.ep_map, i);
                }
            }
            UCC_TEAM_CTX_MAP(team) = ucc_ep_map_from_array(
                &UCC_TEAM_CTX_RANKS(team),
                team->size,
                context->addr_storage.size,
                1);
        } else {
            /* Non-cacheable team: the caller guarantees params.ep_map outlives the
               team, so aliasing is safe and skips materialization. */
            UCC_TEAM_CTX_MAP(team) = team->bp.params.ep_map;
        }
    } else {
        if (!UCC_TEAM_CTX_RANKS(team)) {
            UCC_TEAM_CTX_RANKS(team) = ucc_malloc(
                team->size * sizeof(ucc_rank_t), "ctx_ranks");
            if (!UCC_TEAM_CTX_RANKS(team)) {
                ucc_error("failed to allocate %zd bytes for ctx ranks array",
                          team->size * sizeof(ucc_rank_t));
                return UCC_ERR_NO_MEMORY;
            }
            status = oob.allgather(
                &context->rank,
                UCC_TEAM_CTX_RANKS(team),
                sizeof(ucc_rank_t),
                oob.coll_info,
                &team->oob_req);
            if (UCC_OK != status) {
                ucc_error("failed to start oob allgather for proc info exchange");
                ucc_free(UCC_TEAM_CTX_RANKS(team));
                UCC_TEAM_CTX_RANKS(team) = NULL;
                return status;
            }
        }
        status = oob.req_test(team->oob_req);
        if (status < 0) {
            oob.req_free(team->oob_req);
            ucc_error("oob req test failed during team proc info exchange");
            return status;
        } else if (UCC_INPROGRESS == status) {
            return status;
        }
        oob.req_free(team->oob_req);
        ucc_assert(team->size >= 2);
        UCC_TEAM_CTX_MAP(team) = ucc_ep_map_from_array(
            &UCC_TEAM_CTX_RANKS(team),
            team->size,
            context->addr_storage.size,
            1);
    }
    ucc_debug(
        "team %p rank %d, ctx_rank %d, map_type %d",
        team,
        team->rank,
        context->rank,
        UCC_TEAM_CTX_MAP(team).type);
    return UCC_OK;
}

static ucc_status_t ucc_team_build_score_map(ucc_team_t *team)
{
    ucc_coll_score_t *score, *score_merge, *score_next;
    ucc_status_t      status;
    int               i;

    ucc_assert(team->n_cl_teams > 0);
    status = UCC_CL_TEAM_IFACE(team->cl_teams[0])
                 ->team.get_scores(&team->cl_teams[0]->super, &score);
    if (UCC_OK != status) {
        ucc_error("failed to get cl %s scores",
                  UCC_CL_TEAM_IFACE(team->cl_teams[0])->super.name);
        return status;
    }
    for (i = 1; i < team->n_cl_teams; i++) {
        status = UCC_CL_TEAM_IFACE(team->cl_teams[i])
                     ->team.get_scores(&team->cl_teams[i]->super, &score_next);
        if (UCC_OK != status) {
            ucc_error("failed to get cl %s scores",
                      UCC_CL_TEAM_IFACE(team->cl_teams[i])->super.name);
            ucc_coll_score_free(score);
            return status;
        }
        status = ucc_coll_score_merge(score, score_next, &score_merge, 1);
        if (UCC_OK != status) {
            ucc_error("failed to merge scores");
            ucc_coll_score_free(score);
            ucc_coll_score_free(score_next);
            return status;
        }
        score = score_merge;
    }
    status = ucc_coll_score_build_map(score, &team->score_map);
    if (UCC_OK != status) {
        ucc_error("failed to build score map");
    }
    return status;
}

/* Insert a freshly-built (miss-path), now-ACTIVE team into its context's cache:
   stamp its per-instance cookie, reclaim completed evictions, evict a victim if at
   capacity, then insert and transition DORMANT -> LIVE (directly, refcount is
   already 1). A full/collision insert leaves the team uncached but functional. */
static void ucc_team_cache_admit(ucc_team_t *team)
{
    ucc_context_t    *ctx   = team->contexts[0];
    ucc_team_cache_t *cache = ctx->team_cache;
    uint64_t          new_cookie;

    team->cache_pending_insert = 0;

    /* Adopt the instance cookie every member agreed on: the vote's distribution
       lane carried team-rank 0's proposal (a real value is neither 0 nor all-ones;
       the direct/size==1 path leaves the vote zeroed, which reuse never consults). */
    new_cookie = ucc_team_cache_vote_new_cookie(team->cache_vote_out);
    if (new_cookie != 0 && new_cookie != ~(uint64_t)0) {
        team->cache_identity.instance_cookie = new_cookie;
    }

    if (cache == NULL) {
        return;
    }

    /* Reclaim completed evictions, then make room at capacity by evicting the
       victim. evict_one / progress_pending take cache->lock themselves and drive
       teardown outside it, so they run before the insert critical section.
       cache->max_size is clamped against the team-ID pool at context create
       (UCC_TEAM_CACHE_ID_HEADROOM), so a free pool ID is guaranteed. If nothing is
       evictable (all cached teams live), the new team is admitted un-cached. */
    ucc_team_cache_progress_pending(cache);

    if (cache->eviction != UCC_TEAM_CACHE_EVICTION_NONE &&
        cache->size >= cache->max_size) {
        if (UCC_ERR_NO_RESOURCE == ucc_team_cache_evict_one(cache)) {
            ucc_debug(
                "team cache at pool-safe capacity (size=%u/%u), all entries "
                "live; admitting team %p (hash=0x%" PRIx64 ") un-cached",
                cache->size,
                cache->max_size,
                (void *)team,
                team->cache_identity.hash);
        }
    }

    ucc_spin_lock(&cache->lock);
    if (UCC_OK == ucc_team_cache_insert(cache, team) &&
        team->cache_state == UCC_TEAM_CACHE_STATE_DORMANT) {
        ucc_list_del(&team->cache_link);
        team->cache_state = UCC_TEAM_CACHE_STATE_LIVE;
        ucc_team_cache_registry_add_live(cache, team);
        ucc_debug(
            "team cache: insert (hash=0x%" PRIx64 ") team %p -> LIVE "
            "refcount=%d",
            team->cache_identity.hash,
            (void *)team,
            team->refcount);
    } else if (team->cache_state == UCC_TEAM_CACHE_STATE_NONE) {
        /* Cache full or hash collision: team stays uncached and functional. */
        ucc_debug(
            "team cache: insert skipped (hash=0x%" PRIx64 ") team %p stays "
            "uncached",
            team->cache_identity.hash,
            (void *)team);
    }
    ucc_spin_unlock(&cache->lock);
}

/* Promote a RESERVED reuse candidate to LIVE under cache->lock after a unanimous
   vote. When @reseat, also re-seat the team's id/tag domain to the new cid
   (experimental RESEAT); the instance cookie is unchanged (same physical instance). */
static void ucc_team_agreement_promote_reserved(
    ucc_context_t *context, ucc_team_t *team, int reseat)
{
    ucc_team_cache_t *cache = context->team_cache;

    ucc_spin_lock(&cache->lock);
    ucc_team_cache_get(team); /* refcount 0 -> 1, RESERVED -> LIVE */
    ucc_team_cache_registry_make_live(cache, team);
    if (reseat && team->cache_identity.ext_id != team->cache_reseat_new_id) {
        ucc_debug(
            "team cache: agreed RESEAT team %p id 0x%x -> 0x%x",
            (void *)team,
            (unsigned)team->cache_identity.ext_id,
            (unsigned)team->cache_reseat_new_id);
        ucc_team_reseat_id(team, team->cache_reseat_new_id);
    }
    ucc_spin_unlock(&cache->lock);
}

ucc_status_t ucc_team_create_test_single(ucc_context_t *context,
                                         ucc_team_t    *team)
{
    ucc_status_t            status = UCC_OK;
    ucc_team_cache_action_t agreed;

    switch (team->state) {
    case UCC_TEAM_CACHE_AGREE:
        /* Progress the member vote posted in create_post. On completion, commit
           the unanimously-agreed action or fall to a fresh build. */
        status = ucc_service_coll_test(&team->cache_vote_req);
        if (status == UCC_INPROGRESS) {
            return UCC_INPROGRESS;
        }
        if (status < 0) {
            ucc_service_coll_finalize(&team->cache_vote_req);
            ucc_error(
                "team cache: agreement vote failed: %s",
                ucc_status_string(status));
            goto out;
        }
        ucc_service_coll_finalize(&team->cache_vote_req);
        agreed = ucc_team_cache_vote_result(team->cache_vote_out);

        if (agreed == UCC_TEAM_CACHE_ACTION_EXACT_REUSE &&
            team->cache_local_action == UCC_TEAM_CACHE_ACTION_EXACT_REUSE) {
            /* Unanimous reuse: promote the RESERVED candidate to LIVE. */
            ucc_team_agreement_promote_reserved(context, team, 0);
            ucc_debug("team cache: agreed EXACT reuse, team %p", (void *)team);
            team->state = UCC_TEAM_ACTIVE;
            return UCC_OK;
        }
        if (agreed == UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED &&
            team->cache_local_action == UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED) {
            /* Unanimous RESEAT (cid drift): every member proved (via the cookie
               lanes) it selected the same physical dormant-derived instance.
               Promote to LIVE and re-seat to the new cid. */
            ucc_team_agreement_promote_reserved(context, team, 1);
            team->state = UCC_TEAM_ACTIVE;
            return UCC_OK;
        }
        if (agreed == UCC_TEAM_CACHE_ACTION_DERIVED_FROM_LIVE &&
            team->cache_local_action ==
                UCC_TEAM_CACHE_ACTION_DERIVED_FROM_LIVE) {
            /* Unanimous derive: borrow the parent's artifacts and build fresh. */
            ucc_team_init_derived(
                team,
                team->cache_derive_artifacts,
                team->cache_derive_parent_id);
            team->cache_derive_artifacts = NULL;
            team->state = (team->size > 1) ? UCC_TEAM_SERVICE_TEAM
                                           : UCC_TEAM_CL_CREATE;
            return UCC_INPROGRESS;
        }
        /* Global MISS (or a defensive action mismatch): every member fresh-builds. */
        if (team->cache_local_action == UCC_TEAM_CACHE_ACTION_EXACT_REUSE ||
            team->cache_local_action == UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED) {
            /* This rank reserved a DORMANT candidate; detach it from the cache and
               rebuild the same handle in place after draining its old CL/TL tag
               domain. The candidate may be a derived team (an EXACT_REUSE or
               RESEAT_DERIVED hit on a dormant derived team), so always clear
               is_derived: the rebuild is a FRESH FULL build and must run
               ADDR_EXCHANGE to materialize its own ctx_map/topo, not skip it as a
               derived team would. For a RESEAT candidate the ext_id is also stale
               (drifted cid): re-key it to the new create's cid. Under the lock the
               candidate is off every list, so no peer sees the transient. */
            ucc_team_cache_t *vote_cache = context->team_cache;
            ucc_spin_lock(&vote_cache->lock);
            ucc_team_cache_table_erase(vote_cache, team);
            ucc_team_cache_registry_remove(vote_cache, team);
            if (team->cache_local_action ==
                UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED) {
                if (UCC_TEAM_ID_IS_EXTERNAL(team)) {
                    team->id = team->cache_reseat_new_id;
                }
                team->cache_identity.ext_id = team->cache_reseat_new_id;
            }
            team->is_derived = 0;
            team->parent_id  = 0;
            ucc_spin_unlock(&vote_cache->lock);
            team->cache_state          = UCC_TEAM_CACHE_STATE_NONE;
            team->cache_pending_insert = 1;
            team->state                = UCC_TEAM_CACHE_MISS_TEARDOWN;
            ucc_debug(
                "team cache: agreement lost, rebuilding team %p in place",
                (void *)team);
            /* fall through to CACHE_MISS_TEARDOWN */
        } else {
            if (team->cache_local_action ==
                UCC_TEAM_CACHE_ACTION_DERIVED_FROM_LIVE) {
                ucc_team_artifacts_put(team->cache_derive_artifacts);
                team->cache_derive_artifacts = NULL;
            }
            team->state = (team->size > 1) ? UCC_TEAM_ADDR_EXCHANGE
                                           : UCC_TEAM_CL_CREATE;
            return UCC_INPROGRESS;
        }
        /* fall through */
    case UCC_TEAM_CACHE_MISS_TEARDOWN:
        /* Poll the rejected candidate's teardown to terminal UCC_OK (the CL/TL
           destroys are the (team_id, seq_num) wire-tag alias barrier), then reset
           and re-enter the normal build. */
        status = ucc_team_teardown_for_rebuild(team);
        if (status == UCC_INPROGRESS) {
            ucc_context_progress(context);
            return UCC_INPROGRESS;
        }
        if (status < 0) {
            goto out;
        }
        status = ucc_team_reset_for_rebuild(context, team);
        if (status < 0) {
            goto out;
        }
        return UCC_INPROGRESS; /* re-enter at the reset start state */
    case UCC_TEAM_ADDR_EXCHANGE:
        status = ucc_team_exchange(context, team);
        if (UCC_OK != status) {
            goto out;
        }
        team->state = UCC_TEAM_SERVICE_TEAM;
        /* fall through */
    case UCC_TEAM_SERVICE_TEAM:
        if ((context->cl_flags & UCC_BASE_LIB_FLAG_SERVICE_TEAM_REQUIRED) ||
            ((context->cl_flags & UCC_BASE_LIB_FLAG_TEAM_ID_REQUIRED) &&
             (team->id == 0))) {
            /* We need service team either when it is explicitly required
             * by any CL/TL (e.g. CL/HIER) or if TEAM_ID is required but
             * not provided by the user
             */
            status = ucc_team_create_service_team(context, team);
            if (UCC_OK != status) {
                goto out;
            }
        }
        team->state = UCC_TEAM_ALLOC_ID;
        /* fall through */
    case UCC_TEAM_ALLOC_ID:
        /* ucc_team_alloc_id runs a service allreduce over the shared
           UCC_TL_UCP_SERVICE_TAG; it cannot alias the agreement vote - see the note
           at the vote-post site in ucc_team_agreement_create_post. */
        if (context->cl_flags & UCC_BASE_LIB_FLAG_TEAM_ID_REQUIRED) {
            status = ucc_team_alloc_id(team);
            if (UCC_OK != status) {
                goto out;
            }
        }
        /* A derived team must get its own team-id, distinct from the parent whose
           artifacts it borrows, so it owns an independent TL/UCP tag/seq domain. */
        if (team->is_derived && team->id != 0) {
            ucc_assert(team->id != team->parent_id);
        }
        team->bp.id = team->id;
        team->state = UCC_TEAM_CL_CREATE;
        if (team->service_team) {
            /* update service team id */
            UCC_TL_TEAM_IFACE(team->service_team)->scoll.update_id
                (&team->service_team->super, team->id);
        }
        /* fall through */
    case UCC_TEAM_CL_CREATE:
        status = ucc_team_create_cls(context, team);
        break;
    case UCC_TEAM_ACTIVE:
        return UCC_OK;
    }
out:
    if (UCC_OK == status) {
        team->state = UCC_TEAM_ACTIVE;
        status = ucc_team_build_score_map(team);
    }

    if (UCC_OK == status &&
        ucc_global_config.log_component.log_level >= UCC_LOG_LEVEL_INFO &&
        team->rank == 0) {
        ucc_info("===== COLL_SCORE_MAP (team_id %d, size %u) =====",
                 team->id, team->size);
        ucc_coll_score_map_print_info(team->score_map,
                                      ucc_global_config.log_component.log_level);
        ucc_info("================================================");
    }
    /* Insert the freshly-built (miss-path) team into the cache once it is ACTIVE
       and the score map is built. Re-adopted (hit-path) teams short-circuit
       earlier and never reach here. */
    if (UCC_OK == status && team->cache_pending_insert) {
        ucc_team_cache_admit(team);
    }
    return status;
}

ucc_status_t ucc_team_create_test(ucc_team_h team)
{
    if (NULL == team) {
        ucc_error("ucc_team_create_test: invalid team handle: NULL");
        return UCC_ERR_INVALID_PARAM;
    }
    /* we don't support multiple contexts per team yet */
    ucc_assert(team->num_contexts == 1);
    if (team->state == UCC_TEAM_ACTIVE) {
        return UCC_OK;
    }
    return ucc_team_create_test_single(team->contexts[0], team);
}

/* Tear down a team's components (service/CL/TL teams, artifacts, score_map,
   addr_storage, team-id). When @for_rebuild is set (global-miss rebuild of a
   rejected EXACT candidate), drives every component destroy to terminal UCC_OK
   - including the CL/TL destroys that are the (team_id, seq_num) wire-tag alias
   barrier - but KEEPS the struct-lifetime allocations (cl_teams array, contexts,
   cache_identity, the team struct itself) so ucc_team_reset_for_rebuild can
   rebuild the same handle in place. Otherwise it is the terminal full teardown. */
static ucc_status_t ucc_team_destroy_single_ex(ucc_team_h team, int for_rebuild)
{
    ucc_cl_iface_t *cl_iface;
    int             i;
    ucc_status_t    status;

    if (team->service_team) {
        if (UCC_OK != (status = UCC_TL_CTX_IFACE(team->contexts[0]->service_ctx)
                       ->team.destroy(&team->service_team->super))) {
            return status;
        }
        team->service_team = NULL;
        ucc_tl_context_put(team->contexts[0]->service_ctx);
    }
    for (i = 0; i < team->n_cl_teams; i++) {
        if (!team->cl_teams[i])
            continue;
        cl_iface = UCC_CL_TEAM_IFACE(team->cl_teams[i]);
        if (UCC_OK !=
            (status = cl_iface->team.destroy(&team->cl_teams[i]->super))) {
            return status;
        }
        team->cl_teams[i] = NULL;
    }

    /* Drop this team's reference to the shared artifacts holder. A solo team is
       the sole reference (1 -> 0), so topo and ctx_ranks are torn down here; a
       derived team only decrements and the last team out frees them. Done before
       ctx teardown because the TL nested maps aliasing &artifacts->ctx_map are
       already destroyed by the CL/TL team destroys above. */
    ucc_team_artifacts_put(team->artifacts);
    team->artifacts = NULL;

    if (team->contexts[0]->service_team && team->size > 1) {
        ucc_internal_oob_finalize(&team->bp.params.oob);
    }

    if ((ucc_global_config.log_component.log_level >= UCC_LOG_LEVEL_INFO) &&
        (team->rank == 0)) {
        ucc_info("team destroyed, team_id %d", team->id);
    }

    ucc_coll_score_free_map(team->score_map);
    team->score_map = NULL;
    ucc_free(team->addr_storage.storage);
    /* team->ctx_ranks (now the holder's ctx_ranks) was freed by
       ucc_team_artifacts_put above at the last reference. */
    ucc_team_release_id(team);
    if (for_rebuild) {
        /* Keep contexts, cache_identity, and the struct itself: the handle is
           reused and rebuilt in place. Free the cl_teams array (create_post_single
           re-allocates it) and clear addr_storage. All transport is now drained
           and the team-id released, so ucc_team_reset_for_rebuild + the normal
           build can safely redraw an id without tag aliasing. */
        ucc_free(team->cl_teams);
        team->cl_teams = NULL;
        memset(&team->addr_storage, 0, sizeof(team->addr_storage));
        return UCC_OK;
    }
    ucc_free(team->cl_teams);
    ucc_free(team->contexts);
    /* Free the cache identity at terminal teardown. Cached paths free it before
     * driving the destroy (zeroing members makes this an idempotent no-op), and
     * never-cached teams have a zeroed identity. This is the ONLY free for a team
     * that built an identity on the miss path but was admitted un-cached (cache
     * full with no evictable victim, or a hash collision). */
    ucc_team_cache_identity_free(&team->cache_identity);
    ucc_free(team);
    return UCC_OK;
}

static ucc_status_t ucc_team_destroy_single(ucc_team_h team)
{
    return ucc_team_destroy_single_ex(team, 0);
}

/* Global-miss recovery. Drive the rejected EXACT candidate's components to
   terminal UCC_OK (polled - returns UCC_INPROGRESS until every CL/TL/service
   destroy completes, the wire-tag alias barrier), keeping the struct. */
static ucc_status_t ucc_team_teardown_for_rebuild(ucc_team_t *team)
{
    return ucc_team_destroy_single_ex(team, 1);
}

/* Restore a torn-down (for_rebuild) team to the pristine pre-build state so the
   normal state machine rebuilds it in place, re-inserting under its retained
   cache_identity. Membership (contexts/size/rank) and cache_identity are kept. */
static ucc_status_t ucc_team_reset_for_rebuild(
    ucc_context_t *context, ucc_team_t *team)
{
    ucc_assert(team->service_team == NULL);
    ucc_assert(team->sreq == NULL);

    team->n_cl_teams = 0; /* cl_teams array was freed by teardown */
    team->seq_num    = 0;
    /* External-id teams keep their id (ucc_team_alloc_id preserves it); internal
       ids were released to the pool and must be redrawn, so zero them. */
    if (!UCC_TEAM_ID_IS_EXTERNAL(team)) {
        team->id = 0;
    }
    team->bp.id       = 0;
    team->oob_req     = NULL;
    team->refcount    = 1;
    team->cache_state = UCC_TEAM_CACHE_STATE_NONE;
    team->cache_pending_insert =
        1; /* rebuilt team re-inserts under its identity */
    team->cache_derive_artifacts = NULL;
    ucc_list_head_init(&team->cache_link);
    ucc_list_head_init(&team->bucket_link);

    /* A fresh heap holder (the old one was released by teardown). Cacheable teams
       always use a heap (shareable) holder. */
    team->artifacts = ucc_team_artifacts_alloc();
    if (!team->artifacts) {
        return UCC_ERR_NO_MEMORY;
    }

    /* Re-run the per-create setup: re-allocs cl_teams, re-inits internal OOB,
       repopulates bp, sets last_team_create_posted and the start state. The
       global-MISS branch cleared team->is_derived, so it starts at ADDR_EXCHANGE
       (size>1) and rebuilds its own ctx_map/topo. */
    return ucc_team_create_post_single(context, team);
}

ucc_status_t ucc_team_destroy(ucc_team_h team)
{
    if (NULL == team) {
        ucc_error("ucc_team_destroy: invalid team handle: NULL");
        return UCC_ERR_INVALID_PARAM;
    }

    if (team->state != UCC_TEAM_ACTIVE) {
        ucc_error("team %p is used before team_create is completed", team);
        return UCC_ERR_INVALID_PARAM;
    }

    /* we don't support multiple contexts per team yet */
    ucc_assert(team->num_contexts == 1);

    /* Cache-aware destroy, on team->cache_state:
     *   - NONE : never cached; fall straight through to ucc_team_destroy_single.
     *   - LIVE : cached and backing this communicator. Put the refcount (1 -> 0)
     *            to DORMANT, move it live -> dormant on the registry, and return
     *            UCC_OK IMMEDIATELY without tearing it down; the dormant team
     *            keeps its team-id and CL/TL teams for a later re-adopt. Teardown
     *            is deferred to eviction / the context-finalize drain.
     *   - DORMANT : unreachable (the state guard rejects a non-ACTIVE handle);
     *            fall through to _single defensively.
     *
     * The put + make_dormant happen under cache->lock, the SAME lock the re-adopt
     * takes, so a concurrent create cannot re-adopt a team mid-transition and a
     * dormant team is atomically eligible for the next lookup. */
    if (team->cache_state == UCC_TEAM_CACHE_STATE_LIVE) {
        ucc_context_t    *ctx   = team->contexts[0];
        ucc_team_cache_t *cache = ctx->team_cache;
        int               n;

        ucc_assert(cache != NULL);
        ucc_spin_lock(&cache->lock);
        n = ucc_team_cache_put(team); /* 1 -> 0, sets cache_state = DORMANT */
        ucc_team_cache_registry_make_dormant(cache, team);
        ucc_spin_unlock(&cache->lock);

        ucc_debug(
            "team cache: team %p now dormant, retained for reuse "
            "(hash=0x%" PRIx64 ", live_users=%d)",
            (void *)team,
            team->cache_identity.hash,
            n);
        /* Return UCC_OK immediately: callers spin on UCC_INPROGRESS, so a
         * retained dormant team must NOT stay INPROGRESS. */
        return UCC_OK;
    }

    return ucc_team_destroy_single(team);
}

/* Terminal teardown-failure handling: ucc_team_destroy_single returned a hard
 * error, so the team is only partially destroyed and some CL/TL/service state may
 * still alias the shared artifacts holder. Freeing the struct or putting the
 * artifacts could dangle/double-free that state, so we do not. We do reclaim the
 * scarce team-id pool bit (a local bit-clear, safe regardless of component state)
 * so a failed teardown cannot starve id allocation, and log loudly. The team is
 * already detached from every cache list, so it can never be re-observed; its
 * component memory is an accepted, bounded leak. */
static void ucc_team_cache_abandon_failed(ucc_team_t *team, ucc_status_t status)
{
    ucc_error(
        "cached team %p teardown failed terminally (%s); reclaiming "
        "team-id %u and abandoning partially destroyed component state",
        (void *)team,
        ucc_status_string(status),
        (unsigned)team->id);
    ucc_team_release_id(team);
}

void ucc_team_cache_drain(ucc_context_t *context)
{
    ucc_team_cache_t *cache = context->team_cache;
    ucc_team_t       *team, *tmp;
    ucc_status_t      status;

    if (cache == NULL) {
        return;
    }

    /* Reap every remaining DORMANT team, single-threaded at context teardown
     * (no concurrent create/destroy, so no cache lock needed to walk). Runs
     * BEFORE CL/TL context destroy and the shared service-team teardown, because
     * a dormant team still holds CL/TL/service refs and a team-id. For each team:
     * detach from both registries and the khash bucket, and free the cache
     * identity HERE while `team` is valid (a cache-only field _single never
     * reads), then drive the (possibly async) teardown to completion - _single
     * releases the team-id and frees the struct only on terminal UCC_OK. */
    ucc_list_for_each_safe (team, tmp, &cache->dormant, cache_link) {
        ucc_assert(team->cache_state == UCC_TEAM_CACHE_STATE_DORMANT);

        ucc_team_cache_registry_remove(cache, team);
        ucc_team_cache_table_erase(cache, team);
        ucc_team_cache_identity_free(&team->cache_identity);

        while (UCC_INPROGRESS == (status = ucc_team_destroy_single(team))) {
            ucc_context_progress(context);
        }
        if (status < 0) {
            ucc_team_cache_abandon_failed(team, status);
        }
    }

    ucc_assert(ucc_list_is_empty(&cache->dormant));

    /* An evicted victim may still be mid-teardown on the pending-destroy list;
     * flush it to UCC_OK now so nothing lingers before CL/TL ctx destroy runs. */
    while (!ucc_list_is_empty(&cache->pending_destroy)) {
        ucc_team_cache_progress_pending(cache);
        if (!ucc_list_is_empty(&cache->pending_destroy)) {
            ucc_context_progress(context);
        }
    }
}

/*
 * Pending-destroy state machine.
 *
 * ucc_team_destroy_single early-returns UCC_INPROGRESS while a CL/TL destroy is
 * in flight and releases the team-id + frees the struct only on terminal UCC_OK,
 * so an evicted victim cannot be torn down in one shot; it is parked on
 * cache->pending_destroy and progressed to UCC_OK by this driver.
 *
 * LOCK DISCIPLINE: ucc_team_destroy_single may call ucc_context_progress and MUST
 * NOT run under cache->lock. This detaches the whole pending list under the lock,
 * drops it, drives one destroy attempt per team outside the lock, then re-parks
 * the still-UCC_INPROGRESS teams under the lock. A team on pending_destroy is off
 * the table and both registries, so no other thread can observe or re-adopt it
 * mid-destroy.
 */
ucc_status_t ucc_team_cache_progress_pending(ucc_team_cache_t *cache)
{
    ucc_list_link_t work;
    ucc_team_t     *team, *tmp;
    ucc_status_t    status;

    if (cache == NULL) {
        return UCC_OK;
    }

    /* Detach the whole pending list under the lock so the destroys below run
     * OUTSIDE cache->lock. */
    ucc_list_head_init(&work);
    ucc_spin_lock(&cache->lock);
    ucc_list_for_each_safe (team, tmp, &cache->pending_destroy, cache_link) {
        ucc_list_del(&team->cache_link);
        ucc_list_add_tail(&work, &team->cache_link);
    }
    ucc_spin_unlock(&cache->lock);

    ucc_list_for_each_safe (team, tmp, &work, cache_link) {
        /* Capture id and pointer BEFORE destroy: on UCC_OK the struct is freed
         * and must not be accessed afterwards; on UCC_INPROGRESS it is re-parked
         * below. */
        void    *team_ptr = (void *)team;
        uint16_t team_id  = team->id;

        ucc_list_del(&team->cache_link);
        status = ucc_team_destroy_single(team);
        if (status == UCC_INPROGRESS) {
            ucc_spin_lock(&cache->lock);
            ucc_list_add_tail(&cache->pending_destroy, &team->cache_link);
            ucc_spin_unlock(&cache->lock);
        } else if (status < 0) {
            ucc_team_cache_abandon_failed(team, status);
        } else {
            /* UCC_OK: ucc_team_release_id returned any pool bit, restoring the
             * clamp headroom (external-ID teams hold no pool bit). */
            ucc_debug(
                "team cache: evicted team %p destroy complete "
                "(UCC_OK, id=%u); team-id pool headroom restored",
                team_ptr,
                (unsigned)team_id);
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_team_cache_evict_one(ucc_team_cache_t *cache)
{
    ucc_team_t *victim;
    uint64_t    hash;

    ucc_spin_lock(&cache->lock);

    victim = ucc_team_cache_pick_lru_victim(cache);
    if (victim == NULL) {
        /* All cached teams are live (pinned by a live communicator); nothing is
         * evictable.  Sentinel so the admission path leaves the new team
         * uncached but fully functional. */
        ucc_spin_unlock(&cache->lock);
        return UCC_ERR_NO_RESOURCE;
    }
    /* Only DORMANT teams are ever evicted. The victim may differ per rank (under
     * LFU/LRU, or under FIFO with overlapping scopes); no consensus is needed here
     * because the cache-action agreement (UCC_TEAM_CACHE_AGREEMENT) reconciles any
     * resulting hit/miss split into a consistent fresh build. */
    ucc_assert(victim->cache_state == UCC_TEAM_CACHE_STATE_DORMANT);
    hash = victim->cache_identity.hash;

    /* Detach the victim entirely (erase the khash bucket, unlink from the dormant
     * registry) and free cache_identity here under the lock, BEFORE parking it on
     * pending_destroy: once parked, a concurrent progress_pending may claim and
     * destroy it, so touching victim after the unlock would be a use-after-free.
     * A team on pending_destroy is on neither the table nor any registry, so
     * lookup can never find it. */
    ucc_team_cache_table_erase(cache, victim);
    ucc_team_cache_registry_remove(cache, victim);
    ucc_team_cache_identity_free(&victim->cache_identity);
    victim->cache_state = UCC_TEAM_CACHE_STATE_NONE;
    ucc_list_add_tail(&cache->pending_destroy, &victim->cache_link);
    cache->stats.evictions++;

    ucc_debug(
        "team cache %p: evicting dormant team %p (hash=0x%" PRIx64
        ", size now %u, evictions=%" PRIu64 ")",
        (void *)cache,
        (void *)victim,
        hash,
        cache->size,
        cache->stats.evictions);

    ucc_spin_unlock(&cache->lock);

    /* Drive the (nonblocking) teardown OUTSIDE the lock. */
    ucc_team_cache_progress_pending(cache);
    return UCC_OK;
}

/* Non-static (declared in ucc_team.h) so the id-pool boundary math can be unit
   tested directly; see test/gtest/core/test_team_cache.cc. */
int ucc_team_id_pool_ffs_clear(uint64_t *value)
{
    int i;
    for (i=0; i<64; i++) {
        if (*value & ((uint64_t)1 << i)) {
            *value &= ~((uint64_t)1 << i);
            return i+1;
        }
    }
    return 0;
}

void ucc_team_id_pool_set_bit(uint64_t *local, int id)
{
    /* Inverse of the allocation layout in ucc_team_alloc_id: a bit at index
       (pos-1) of word i encodes id = i*64 + pos (pos in 1..64), so id maps back
       to word (id-1)/64, bit (id-1)%64.  Using id/64 for the word mis-indexes
       every multiple of 64 (e.g. id 64 -> word 1 instead of word 0). */
    int map_pos = (id - 1) / 64;
    int pos     = (id - 1) % 64;
    ucc_assert(id >= 1);
    local[map_pos] |= ((uint64_t)1 << pos);
}

static ucc_status_t ucc_team_alloc_id(ucc_team_t *team)
{
    /* at least 1 ctx is always available */
    ucc_context_t   *ctx      = team->contexts[0];
    uint64_t        *local, *global;
    ucc_status_t     status;
    int              pos, i;

    if (team->id > 0) {
        ucc_assert(UCC_TEAM_ID_IS_EXTERNAL(team));
        return UCC_OK;
    }

    if (!ctx->ids.pool) {
        ctx->ids.pool = ucc_malloc(ctx->ids.pool_size*2*sizeof(uint64_t), "ids_pool");
        if (!ctx->ids.pool) {
            ucc_error("failed to allocate %zd bytes for team_ids_pool",
                      ctx->ids.pool_size*2*sizeof(uint64_t));
            return UCC_ERR_NO_MEMORY;
        }
        /* init all bits to 1 - all available */
        memset(ctx->ids.pool, 255, ctx->ids.pool_size*2*sizeof(uint64_t));
    }
    local  = ctx->ids.pool;
    global = ctx->ids.pool + ctx->ids.pool_size;

    if (!team->sreq) {
        ucc_subset_t subset = {.map.type   = UCC_EP_MAP_FULL,
                               .map.ep_num = team->size,
                               .myrank     = team->rank};
        status = ucc_service_allreduce(team, local, global, UCC_DT_UINT64,
                                       ctx->ids.pool_size,
                                       UCC_OP_BAND, subset,
                                       &team->sreq);
        if (status < 0) {
            return status;
        }
    }
    ucc_context_progress(ctx);
    status = ucc_service_coll_test(team->sreq);
    if (status < 0) {
        ucc_error("service allreduce test failure: %s",
                  ucc_status_string(status));
        return status;
    } else if (status != UCC_OK) {
        return status;
    }
    ucc_service_coll_finalize(team->sreq);
    team->sreq = NULL;
    memcpy(local, global, ctx->ids.pool_size*sizeof(uint64_t));
    pos = 0;
    for (i=0; i<ctx->ids.pool_size; i++) {
        if ((pos = ucc_team_id_pool_ffs_clear(&local[i])) > 0) {
            break;
        }
    }
    if (pos > 0) {
        ucc_assert(pos <= 64);
        team->id = (uint16_t)(i*64+pos);
        ucc_debug("allocated ID %d for team %p", team->id, team);
    } else {
        ucc_warn("could not allocate team id, whole id space is occupied, "
                 "try increasing UCC_TEAM_IDS_POOL_SIZE");
        return UCC_ERR_NO_RESOURCE;
    }
    ucc_assert(team->id > 0);
    return UCC_OK;
}

static void ucc_team_release_id(ucc_team_t *team)
{
    ucc_context_t *ctx = team->contexts[0];
    /* release the id pool bit if it was not provided by user */
    if (0 != team->id && !UCC_TEAM_ID_IS_EXTERNAL(team)) {
        ucc_team_id_pool_set_bit(ctx->ids.pool, team->id);
    }
}
