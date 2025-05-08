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

static ucc_status_t ucc_team_alloc_id(ucc_team_t *team);
static void ucc_team_release_id(ucc_team_t *team);

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

ucc_status_t ucc_team_create_post(ucc_context_h *contexts, uint32_t num_contexts,
                                  const ucc_team_params_t *params,
                                  ucc_team_h *new_team)
{
    uint64_t     team_size = 0;
    uint64_t     team_rank = UINT64_MAX;
    ucc_team_t  *team;
    ucc_status_t status;

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

    team = ucc_calloc(1, sizeof(ucc_team_t), "ucc_team");
    if (!team) {
        ucc_error("failed to allocate %zd bytes for ucc team",
                  sizeof(ucc_team_t));
        return UCC_ERR_NO_MEMORY;
    }
    team->runtime_oob  = params->oob;
    team->num_contexts = num_contexts;
    team->size         = (ucc_rank_t)team_size;
    team->rank         = (ucc_rank_t)team_rank;
    team->seq_num      = 0;
    team->contexts     = ucc_malloc(sizeof(ucc_context_t *) * num_contexts,
                                    "ucc_team_ctx");
    if (!team->contexts) {
        ucc_error("failed to allocate %zd bytes for ucc team contexts array",
                  sizeof(ucc_context_t) * num_contexts);
        status = UCC_ERR_NO_MEMORY;
        goto err_ctx_alloc;
    }

    memcpy(team->contexts, contexts, sizeof(ucc_context_t *) * num_contexts);
    ucc_copy_team_params(&team->bp.params, params);
    /* check if user provides team id and if it is not too large */
    if ((params->mask & UCC_TEAM_PARAM_FIELD_ID) &&
        (params->id <= UCC_TEAM_ID_MAX)) {
        team->id = ((uint16_t)params->id) | UCC_TEAM_ID_EXTERNAL_BIT;
    }
    status    = ucc_team_create_post_single(contexts[0], team);
    *new_team = team;
    return status;

err_ctx_alloc:
    *new_team = NULL;
    ucc_free(team);
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

    if (context->topo && !team->topo && team->size > 1) {
        /* Context->topo is not NULL if any of the enabled CLs
           reported topo_required through the lib_attr */
        subset.map    = team->ctx_map;
        subset.myrank = team->rank;
        status        = ucc_topo_init(subset, context->topo, &team->topo);
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
        team->ctx_map = team->bp.params.ep_map;
    } else {
        if (!team->ctx_ranks) {
            team->ctx_ranks =
                ucc_malloc(team->size * sizeof(ucc_rank_t), "ctx_ranks");
            if (!team->ctx_ranks) {
                ucc_error("failed to allocate %zd bytes for ctx ranks array",
                          team->size * sizeof(ucc_rank_t));
                return UCC_ERR_NO_MEMORY;
            }
            status = oob.allgather(&context->rank, team->ctx_ranks,
                                   sizeof(ucc_rank_t), oob.coll_info,
                                   &team->oob_req);
            if (UCC_OK != status) {
                ucc_error("failed to start oob allgather for proc info exchange");
                ucc_free(team->ctx_ranks);
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
        team->ctx_map = ucc_ep_map_from_array(&team->ctx_ranks, team->size,
                                              context->addr_storage.size, 1);
    }
    ucc_debug("team %p rank %d, ctx_rank %d, map_type %d", team, team->rank,
              context->rank, team->ctx_map.type);
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

ucc_status_t ucc_team_create_test_single(ucc_context_t *context,
                                         ucc_team_t    *team)
{
    ucc_status_t status = UCC_OK;

    switch (team->state) {
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

static ucc_status_t ucc_team_destroy_single(ucc_team_h team)
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

    ucc_topo_cleanup(team->topo);

    if (team->contexts[0]->service_team && team->size > 1) {
        ucc_internal_oob_finalize(&team->bp.params.oob);
    }

    if ((ucc_global_config.log_component.log_level >= UCC_LOG_LEVEL_INFO) &&
        (team->rank == 0)) {
        ucc_info("team destroyed, team_id %d", team->id);
    }

    ucc_coll_score_free_map(team->score_map);
    ucc_free(team->addr_storage.storage);
    ucc_free(team->ctx_ranks);
    ucc_team_release_id(team);
    ucc_free(team->cl_teams);
    ucc_free(team->contexts);
    ucc_free(team);
    return UCC_OK;
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
    return ucc_team_destroy_single(team);
}

static inline int
find_first_set_and_zero(uint64_t *value) {
    int i;
    for (i=0; i<64; i++) {
        if (*value & ((uint64_t)1 << i)) {
            *value &= ~((uint64_t)1 << i);
            return i+1;
        }
    }
    return 0;
}

static inline void
set_id_bit(uint64_t *local, int id) {
    int map_pos = id / 64;
    int pos = (id-1) % 64;
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
        if ((pos = find_first_set_and_zero(&local[i])) > 0) {
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
        set_id_bit(ctx->ids.pool, team->id);
    }
}
