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
static ucc_status_t ucc_team_destroy_single_ex(
    ucc_team_h team, int for_rebuild);
static ucc_status_t ucc_team_teardown_for_rebuild(ucc_team_t *team);
static ucc_status_t ucc_team_reset_for_rebuild(
    ucc_context_t *context, ucc_team_t *team);

void ucc_team_artifacts_init_inline(ucc_team_artifacts_t *a)
{
    memset(a, 0, sizeof(*a));
    a->refcount = 1;
    a->heap     = 0;
    ucc_spinlock_init(&a->lock, 0);
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

    /* ctx_ranks is NULL when ctx_map aliases the caller's ep_map */
    ucc_topo_cleanup(artifacts->topo);
    ucc_free(artifacts->ctx_ranks);
    ucc_spinlock_destroy(&artifacts->lock);
    if (artifacts->heap) {
        ucc_free(artifacts);
    }
}

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
    team->state                   = (team->size > 1) ? UCC_TEAM_ADDR_EXCHANGE
                                                     : UCC_TEAM_CL_CREATE;
    team->last_team_create_posted = -1;
    return UCC_OK;
}

/* Allocate an unbuilt team; on @id_built the identity MOVES from @id to it.
   The move happens only on success: when this returns NULL the caller still
   owns @id and must free it. Both call sites rely on that split. */
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
    team->artifacts = &team->artifacts_inline;
    ucc_team_artifacts_init_inline(team->artifacts);
    team->runtime_oob  = params->oob;
    team->num_contexts = num_contexts;
    team->size         = (ucc_rank_t)team_size;
    team->rank         = (ucc_rank_t)team_rank;
    team->seq_num      = 0;
    team->refcount     = 1;
    if (id_built) {
        team->cache_identity       = *id; /* the caller's id now owns nothing */
        team->cache_pending_insert = 1;
        memset(id, 0, sizeof(*id));
    } else {
        memset(&team->cache_identity, 0, sizeof(team->cache_identity));
        team->cache_pending_insert = 0;
    }
    ucc_list_head_init(&team->cache_link);
    team->cache_state        = UCC_TEAM_CACHE_STATE_NONE;
    team->cache_local_action = UCC_TEAM_CACHE_ACTION_MISS;
    team->contexts           = ucc_malloc(
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
    /* check if user provides team id and if it is not too large */
    if ((params->mask & UCC_TEAM_PARAM_FIELD_ID) &&
        (params->id <= UCC_TEAM_ID_MAX)) {
        team->id = ((uint16_t)params->id) | UCC_TEAM_ID_EXTERNAL_BIT;
    }
    *status_out = UCC_OK;
    return team;
}

/* Classify this rank's cache action and post the vote that reconciles it */
static ucc_status_t ucc_team_agreement_create_post(
    ucc_context_h *contexts, uint32_t num_contexts,
    const ucc_team_params_t *params, uint64_t team_size, uint64_t team_rank,
    ucc_team_cache_t *cache, ucc_team_h *new_team)
{
    ucc_team_cache_identity_t id;
    int                       id_built = 0;
    ucc_team_t               *cached   = NULL;
    ucc_team_t               *handle;
    ucc_team_cache_action_t   action          = UCC_TEAM_CACHE_ACTION_MISS;
    uint64_t                  key             = 0;
    int                       is_rank0        = (team_rank == 0);
    uint64_t                  proposed_cookie = 0;
    ucc_subset_t              subset;
    ucc_status_t              status;

    if (ucc_team_cache_identity_build(params, &id) == UCC_OK) {
        id_built = 1;
        ucc_spin_lock(&cache->lock);
        /* Drawn unconditionally, so a MISS outcome has a cookie to adopt */
        if (is_rank0) {
            proposed_cookie = ucc_team_cache_next_cookie(cache);
        }
        cached = ucc_team_cache_lookup(cache, &id);
        if (cached != NULL) {
            action = UCC_TEAM_CACHE_ACTION_EXACT_REUSE;
            key    = cached->id;
            ucc_team_cache_registry_make_reserved(cache, cached);
            cached->cache_state = UCC_TEAM_CACHE_STATE_RESERVED;
        }
        ucc_spin_unlock(&cache->lock);
    }

    if (action == UCC_TEAM_CACHE_ACTION_EXACT_REUSE) {
        handle = cached;
        /* The candidate already carries its own identity */
        ucc_team_cache_identity_free(&id);
        /* Refresh the map, since the candidate's may name a freed team */
        handle->bp.params.ep_map = params->ep_map;
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
            return status;
        }
        status = ucc_team_create_post_single(contexts[0], handle);
        if (status < 0) {
            ucc_team_destroy_single(handle);
            return status;
        }
    }

    handle->cache_local_action = action;
    ucc_team_cache_vote_fill(
        handle->cache_vote_in,
        action != UCC_TEAM_CACHE_ACTION_MISS,
        action,
        key,
        /*cookie=*/0,
        /*parent_cookie=*/0,
        is_rank0,
        proposed_cookie);
    /* ep_map is valid for this create and maps member index to ctx rank */
    subset.myrank = handle->rank;
    subset.map    = params->ep_map;
    status        = ucc_service_allreduce_ctx(
        handle,
        handle->cache_vote_in,
        handle->cache_vote_out,
        UCC_DT_UINT64,
        UCC_TEAM_CACHE_VOTE_LANES,
        UCC_OP_BAND,
        subset,
        &handle->cache_vote_req);
    if (status < 0) {
        if (action == UCC_TEAM_CACHE_ACTION_EXACT_REUSE) {
            ucc_spin_lock(&cache->lock);
            handle->cache_state = UCC_TEAM_CACHE_STATE_DORMANT;
            ucc_team_cache_registry_make_dormant(cache, handle);
            ucc_spin_unlock(&cache->lock);
        } else {
            ucc_team_destroy_single(handle);
        }
        return status;
    }
    handle->state = UCC_TEAM_CACHE_AGREE;
    *new_team     = handle;
    return UCC_OK;
}

ucc_status_t ucc_team_create_post(ucc_context_h *contexts, uint32_t num_contexts,
                                  const ucc_team_params_t *params,
                                  ucc_team_h *new_team)
{
    uint64_t     team_size = 0;
    uint64_t     team_rank = UINT64_MAX;
    ucc_team_t  *team;
    ucc_status_t status;
    ucc_team_cache_t         *cache    = NULL;
    ucc_team_cache_identity_t id;
    int                       id_built = 0;

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

    /* Agreement keeps every member on the same reuse decision.

       Every condition below must evaluate identically on every member. A rank
       that skips the vote (caching or agreement off, a differing params->mask,
       no EP_MAP) builds its team directly while its peers wait on a
       member-scoped allreduce that never completes, so a non-uniform
       configuration hangs the create rather than degrading it. This is
       documented for users under "Team-cache settings must be identical on
       every rank" in docs/user_guide.md. */
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

    /* Direct reuse, for a single-rank team or when agreement is disabled */
    if (cache != NULL && ucc_team_cache_is_cacheable(params)) {
        status = ucc_team_cache_identity_build(params, &id);
        if (status == UCC_OK) {
            ucc_team_t *cached;

            id_built = 1;

            /* One lock spans lookup and adopt, so no team is adopted twice */
            ucc_spin_lock(&cache->lock);
            cached = ucc_team_cache_lookup(cache, &id);
            if (cached != NULL) {
                cached->cache_local_action = UCC_TEAM_CACHE_ACTION_EXACT_REUSE;
                ucc_team_cache_get(cached);
                ucc_team_cache_registry_make_live(cache, cached);
            }
            ucc_spin_unlock(&cache->lock);

            if (cached != NULL) {
                ucc_debug(
                    "team cache: dormant reuse / hit, team %p (hash=0x%" PRIx64
                    ")",
                    (void *)cached,
                    id.hash);
                ucc_team_cache_identity_free(&id);
                *new_team = cached;
                return UCC_OK;
            }
            /* On a miss @id moves onto the new team, to insert once ACTIVE */
        }
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
        if (id_built) {
            ucc_team_cache_identity_free(&id);
        }
        return status;
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
            /* A cached team outlives the caller's map, so copy it now */
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
                    UCC_TEAM_CTX_RANKS(team)[i] =
                        (ucc_rank_t)ucc_ep_map_eval(team->bp.params.ep_map, i);
                }
            }
            UCC_TEAM_CTX_MAP(team) = ucc_ep_map_from_array(
                &UCC_TEAM_CTX_RANKS(team), team->size,
                context->addr_storage.size, 1);
        } else {
            /* The caller's ep_map outlives the team, so aliasing it is safe */
            UCC_TEAM_CTX_MAP(team) = team->bp.params.ep_map;
        }
    } else {
        if (!UCC_TEAM_CTX_RANKS(team)) {
            UCC_TEAM_CTX_RANKS(team) =
                ucc_malloc(team->size * sizeof(ucc_rank_t), "ctx_ranks");
            if (!UCC_TEAM_CTX_RANKS(team)) {
                ucc_error("failed to allocate %zd bytes for ctx ranks array",
                          team->size * sizeof(ucc_rank_t));
                return UCC_ERR_NO_MEMORY;
            }
            status = oob.allgather(&context->rank, UCC_TEAM_CTX_RANKS(team),
                                   sizeof(ucc_rank_t), oob.coll_info,
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
        UCC_TEAM_CTX_MAP(team) =
            ucc_ep_map_from_array(&UCC_TEAM_CTX_RANKS(team), team->size,
                                  context->addr_storage.size, 1);
    }
    ucc_debug("team %p rank %d, ctx_rank %d, map_type %d", team, team->rank,
              context->rank, UCC_TEAM_CTX_MAP(team).type);
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

/* Detach @team from the cache table and from whichever list it is on, and mark
   it uncached. Callers hold @cache->lock, so the state change happens under the
   lock as ucc_team_cache_state_t requires. The identity is left intact: the
   teardown paths still log from it, and the callers that must release it do so
   themselves. */
static void ucc_team_cache_detach(ucc_team_cache_t *cache, ucc_team_t *team)
{
    ucc_team_cache_table_erase(cache, team);
    ucc_team_cache_registry_remove(team);
    team->cache_state = UCC_TEAM_CACHE_STATE_NONE;
}

/* Add a freshly built, now ACTIVE team to its context cache */
static void ucc_team_cache_admit(ucc_team_t *team)
{
    ucc_context_t    *ctx   = team->contexts[0];
    ucc_team_cache_t *cache = ctx->team_cache;
    uint64_t          new_cookie;

    team->cache_pending_insert = 0;

    /* The direct path leaves the vote zeroed, which reuse never consults */
    new_cookie = ucc_team_cache_vote_new_cookie(team->cache_vote_out);
    if (new_cookie != 0 && new_cookie != ~(uint64_t)0) {
        team->cache_identity.instance_cookie = new_cookie;
    }

    if (cache == NULL) {
        return;
    }

    /* Both take cache->lock, so they must run before the insert below */
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
        ucc_debug(
            "team cache: insert skipped (hash=0x%" PRIx64 ") team %p stays "
            "uncached",
            team->cache_identity.hash,
            (void *)team);
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
            ucc_team_cache_t *vote_cache = context->team_cache;

            ucc_spin_lock(&vote_cache->lock);
            ucc_team_cache_get(team); /* RESERVED -> LIVE */
            ucc_team_cache_registry_make_live(vote_cache, team);
            ucc_spin_unlock(&vote_cache->lock);
            ucc_debug("team cache: agreed EXACT reuse, team %p", (void *)team);
            team->state = UCC_TEAM_ACTIVE;
            return UCC_OK;
        }
        /* Not unanimous, so every member builds a fresh team */
        if (team->cache_local_action == UCC_TEAM_CACHE_ACTION_EXACT_REUSE) {
            /* Detach the rejected candidate and rebuild the same handle */
            ucc_team_cache_t *vote_cache = context->team_cache;

            ucc_spin_lock(&vote_cache->lock);
            ucc_team_cache_table_erase(vote_cache, team);
            ucc_team_cache_registry_remove(team);
            ucc_spin_unlock(&vote_cache->lock);
            team->cache_pending_insert = 1;
            team->state                = UCC_TEAM_CACHE_MISS_TEARDOWN;
            ucc_debug(
                "team cache: agreement lost, rebuilding team %p in place",
                (void *)team);
            /* fall through to CACHE_MISS_TEARDOWN */
        } else {
            team->state = (team->size > 1) ? UCC_TEAM_ADDR_EXCHANGE
                                           : UCC_TEAM_CL_CREATE;
            return UCC_INPROGRESS;
        }
        /* fall through */
    case UCC_TEAM_CACHE_MISS_TEARDOWN:
        /* A candidate that lost the vote is torn down rather than just marked
           uncached: peers are about to build a fresh team reusing this team's
           id, and its CL/TL teams still hold the matching wire tags. Destroying
           them is the alias barrier that keeps a late in-flight message from
           the retired team out of the rebuilt one's tag space.

           This state is re-entered on every progress call until the teardown
           completes, so ucc_team_teardown_for_rebuild must be restartable: it
           advances per component and returns UCC_INPROGRESS without repeating
           the destroys it already finished. */
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
        if (context->cl_flags & UCC_BASE_LIB_FLAG_TEAM_ID_REQUIRED) {
            status = ucc_team_alloc_id(team);
            if (UCC_OK != status) {
                goto out;
            }
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
    /* TODO: add team/coll selection and check if some teams are never
             used after selection and clean them up */
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

/* Tear down a team; @for_rebuild keeps what reset_for_rebuild reuses */
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

    /* Safe here: the TL nested maps aliasing ctx_map are destroyed above */
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
    ucc_team_release_id(team);

    if (for_rebuild) {
        /* create_post_single reallocates cl_teams */
        ucc_free(team->cl_teams);
        team->cl_teams = NULL;
        memset(&team->addr_storage, 0, sizeof(team->addr_storage));
        return UCC_OK;
    }
    ucc_free(team->cl_teams);
    ucc_free(team->contexts);
    ucc_team_cache_identity_free(&team->cache_identity);
    ucc_free(team);
    return UCC_OK;
}

static ucc_status_t ucc_team_destroy_single(ucc_team_h team)
{
    return ucc_team_destroy_single_ex(team, 0);
}

/* Polls, returning UCC_INPROGRESS until every component has been destroyed */
static ucc_status_t ucc_team_teardown_for_rebuild(ucc_team_t *team)
{
    return ucc_team_destroy_single_ex(team, 1);
}

/* Return a torn-down team to its pre-build state, keeping its membership */
static ucc_status_t ucc_team_reset_for_rebuild(
    ucc_context_t *context, ucc_team_t *team)
{
    ucc_assert(team->service_team == NULL);
    ucc_assert(team->sreq == NULL);

    team->n_cl_teams = 0;
    team->seq_num    = 0;
    /* A pool id was released by the teardown and has to be redrawn */
    if (!UCC_TEAM_ID_IS_EXTERNAL(team)) {
        team->id = 0;
    }
    team->bp.id                = 0;
    team->oob_req              = NULL;
    team->refcount             = 1;
    team->cache_state          = UCC_TEAM_CACHE_STATE_NONE;
    team->cache_pending_insert = 1;
    ucc_list_head_init(&team->cache_link);
    /* Teardown released the old holder, so re-init the inline one */
    team->artifacts = &team->artifacts_inline;
    ucc_team_artifacts_init_inline(team->artifacts);

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

    /* A cached team is retained, keeping its id and its CL/TL teams */
    if (team->cache_state == UCC_TEAM_CACHE_STATE_LIVE) {
        ucc_context_t    *ctx   = team->contexts[0];
        ucc_team_cache_t *cache = ctx->team_cache;
        int               n;

        ucc_assert(cache != NULL);
        ucc_spin_lock(&cache->lock);
        n = ucc_team_cache_put(team); /* LIVE -> DORMANT */
        ucc_team_cache_registry_make_dormant(cache, team);
        ucc_spin_unlock(&cache->lock);

        ucc_debug(
            "team cache: team %p now dormant, retained for reuse "
            "(hash=0x%" PRIx64 ", live_users=%d)",
            (void *)team,
            team->cache_identity.hash,
            n);
        return UCC_OK; /* callers spin on UCC_INPROGRESS, so never return it */
    }

    return ucc_team_destroy_single(team);
}

/* Reclaim the team id after a failed teardown; the rest is a bounded leak */
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

    /* Context teardown is single threaded, so the walk needs no lock */
    ucc_list_for_each_safe (team, tmp, &cache->dormant, cache_link) {
        ucc_assert(team->cache_state == UCC_TEAM_CACHE_STATE_DORMANT);

        ucc_team_cache_detach(cache, team);

        while (UCC_INPROGRESS == (status = ucc_team_destroy_single(team))) {
            ucc_context_progress(context);
        }
        if (status < 0) {
            ucc_team_cache_abandon_failed(team, status);
        }
    }

    ucc_assert(ucc_list_is_empty(&cache->dormant));

    while (!ucc_list_is_empty(&cache->pending_destroy)) {
        ucc_team_cache_progress_pending(cache);
        if (!ucc_list_is_empty(&cache->pending_destroy)) {
            ucc_context_progress(context);
        }
    }
}

/* Teams are popped under the lock, since a destroy must not run holding it */
void ucc_team_cache_progress_pending(ucc_team_cache_t *cache)
{
    ucc_team_t  *team;
    ucc_status_t status;
    unsigned     pending, i;
    uint16_t     team_id;

    if (cache == NULL) {
        return;
    }

    ucc_spin_lock(&cache->lock);
    pending = (unsigned)ucc_list_length(&cache->pending_destroy);
    ucc_spin_unlock(&cache->lock);

    /* Bounded by the initial count so a re-queued team is not retried here */
    for (i = 0; i < pending; i++) {
        ucc_spin_lock(&cache->lock);
        if (ucc_list_is_empty(&cache->pending_destroy)) {
            ucc_spin_unlock(&cache->lock);
            break;
        }
        team = ucc_list_extract_head(&cache->pending_destroy, ucc_team_t,
                                     cache_link);
        ucc_spin_unlock(&cache->lock);

        team_id = team->id;
        status  = ucc_team_destroy_single(team);
        if (status == UCC_INPROGRESS) {
            ucc_spin_lock(&cache->lock);
            ucc_list_add_tail(&cache->pending_destroy, &team->cache_link);
            ucc_spin_unlock(&cache->lock);
        } else if (status < 0) {
            ucc_team_cache_abandon_failed(team, status);
        } else {
            ucc_debug("team cache: evicted team destroy complete "
                      "(UCC_OK, id=%u); team-id pool headroom restored",
                      (unsigned)team_id);
        }
    }
}

ucc_status_t ucc_team_cache_evict_one(ucc_team_cache_t *cache)
{
    ucc_team_t *victim;
    uint64_t    hash;

    ucc_spin_lock(&cache->lock);

    victim = ucc_team_cache_pick_victim(cache);
    if (victim == NULL) {
        ucc_spin_unlock(&cache->lock);
        return UCC_ERR_NO_RESOURCE;
    }
    ucc_assert(victim->cache_state == UCC_TEAM_CACHE_STATE_DORMANT);
    hash = victim->cache_identity.hash;

    ucc_team_cache_detach(cache, victim);
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

    ucc_team_cache_progress_pending(cache);
    return UCC_OK;
}

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
    int map_pos;
    int pos;

    ucc_assert(id >= 1);
    map_pos = (id-1) / 64;
    pos     = (id-1) % 64;
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
