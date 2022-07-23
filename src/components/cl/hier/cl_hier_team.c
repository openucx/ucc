/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_hier.h"
#include "utils/ucc_malloc.h"
#include "core/ucc_team.h"
#include "core/ucc_service_coll.h"
#include "cl_hier_coll.h"

#define SBGP_SET(_team, _sbgp, _enable)                                        \
    _team->sbgps[UCC_HIER_SBGP_##_sbgp].sbgp_type = UCC_SBGP_##_sbgp;          \
    _team->sbgps[UCC_HIER_SBGP_##_sbgp].state     = UCC_HIER_SBGP_##_enable;

/* The function below must enable/disable those hier sbgps that will be
   used to construct hierarchical schedules.
   Currently just enable two sbgps as example and for testing purposes.
   Next step is to enable sbgps based on the requested hierarchical algs. */
static void ucc_cl_hier_enable_sbgps(ucc_cl_hier_team_t *team)
{
    SBGP_SET(team, NET, ENABLED);
    SBGP_SET(team, NODE, ENABLED);
    SBGP_SET(team, NODE_LEADERS, ENABLED);
    SBGP_SET(team, FULL, ENABLED); //todo parse score if a2av is enabled
}

UCC_CLASS_INIT_FUNC(ucc_cl_hier_team_t, ucc_base_context_t *cl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_cl_hier_context_t *ctx =
        ucc_derived_of(cl_context, ucc_cl_hier_context_t);
    ucc_cl_hier_lib_t *lib = ucc_derived_of(cl_context->lib, ucc_cl_hier_lib_t);
    int                i, j, t, n_sbgp_teams;
    ucc_status_t       status;
    ucc_hier_sbgp_t   *hs;
    ucc_config_names_array_t  *tls;
    ucc_subset_t               subset;
    struct ucc_team_team_desc *d;

    if (!params->team->topo) {
        cl_info(cl_context->lib,
                "can't create hier team without topology data");
        return UCC_ERR_INVALID_PARAM;
    }

    if (ucc_topo_is_single_node(params->team->topo)) {
        cl_debug(cl_context->lib, "skipping single node team");
        return UCC_ERR_INVALID_PARAM;
    }

    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_team_t, &ctx->super, params);

    memset(self->sbgps, 0, sizeof(self->sbgps));
    ucc_cl_hier_enable_sbgps(self);
    n_sbgp_teams = 0;
    for (i = 0; i < UCC_HIER_SBGP_LAST; i++) {
        hs            = &self->sbgps[i];
        if (hs->state == UCC_HIER_SBGP_ENABLED) {
            hs->sbgp = ucc_topo_get_sbgp(params->team->topo, hs->sbgp_type);
            if (hs->sbgp->status != UCC_SBGP_ENABLED) {
                /* SBGP of that type either not exists or the calling process
                   is not part of subgroup */
                cl_debug(cl_context->lib, "sbgp %s is not enabled",
                         ucc_sbgp_str(hs->sbgp_type));
                hs->state = UCC_HIER_SBGP_DISABLED;
                continue;
            }
            hs->n_tls = 0;
            tls       = &lib->cfg.sbgp_tls[i].array;
            for (j = 0; j < tls->count; j++) {
                status =
                    ucc_tl_context_get(ctx->super.super.ucc_context,
                                       tls->names[j], &hs->tl_ctxs[hs->n_tls]);
                if (UCC_OK != status) {
                    cl_debug(cl_context->lib,
                             "tl context %s is not available for sbgp %s",
                             tls->names[j], ucc_sbgp_str(hs->sbgp_type));
                } else {
                    hs->n_tls++;
                    n_sbgp_teams++;
                    ucc_assert(hs->n_tls <= CL_HIER_MAX_SBGP_TLS);
                }
            }
        }
    }
    status = ucc_team_multiple_req_alloc(&self->team_create_req, n_sbgp_teams);
    if (UCC_OK != status) {
        cl_error(cl_context->lib, "failed to allocate team req multiple");
        goto err;
    }

    j                              = 0;
    self->team_create_req->n_teams = n_sbgp_teams;
    /* Initialize base params for ALL tl teams we need to create:
       for each hier sbgp we have n_tls potentially requested tl teams */
    for (i = 0; i < UCC_HIER_SBGP_LAST; i++) {
        hs = &self->sbgps[i];
        if (hs->state == UCC_HIER_SBGP_ENABLED) {
            for (t = 0; t < hs->n_tls; t++) {
                d                         = &self->team_create_req->descs[j];
                d->param.team             = params->team;
                d->param.rank             = hs->sbgp->group_rank;
                d->param.size             = hs->sbgp->group_size;
                d->param.params.mask =
                    UCC_TEAM_PARAM_FIELD_EP_RANGE | UCC_TEAM_PARAM_FIELD_EP |
                    UCC_TEAM_PARAM_FIELD_TEAM_SIZE | UCC_TEAM_PARAM_FIELD_OOB;
                d->param.params.ep       = (uint64_t)hs->sbgp->group_rank;
                d->param.params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
                d->ctx                   = hs->tl_ctxs[t];
                d->param.scope           = UCC_CL_HIER;
                d->param.id              = params->id;
                d->param.scope_id        = i;
                d->param.map             = hs->sbgp->map;
                subset.myrank            = hs->sbgp->group_rank;
                subset.map               = hs->sbgp->map;
                /* Internal oob will execute allgather over subset */
                status = ucc_internal_oob_init(params->team, subset,
                                               &d->param.params.oob);
                if (UCC_OK != status) {
                    cl_error(cl_context->lib, "failed to init oob for sbgp %s",
                             ucc_sbgp_str(hs->sbgp->type));
                    goto err;
                }
                d->args[0] = i;
                d->args[1] = t;
                j++;
            }
        }
    }

    status = ucc_tl_team_create_multiple(self->team_create_req);
    if (status < 0) {
        cl_error(cl_context->lib, "failed to post tl team create (%d)", status);
        goto err;
    }
    cl_info(cl_context->lib, "posted cl team: %p", self);
    return UCC_OK;
err:
    ucc_team_multiple_req_free(self->team_create_req);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_hier_team_t)
{
    cl_info(self->super.super.context->lib, "finalizing cl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_hier_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_cl_hier_team_t, ucc_cl_team_t);

ucc_status_t ucc_cl_hier_team_destroy(ucc_base_team_t *cl_team)
{
    ucc_cl_hier_team_t    *team   = ucc_derived_of(cl_team, ucc_cl_hier_team_t);
    ucc_cl_hier_context_t *ctx    = UCC_CL_HIER_TEAM_CTX(team);
    ucc_status_t           status = UCC_OK;
    int                    i, j;
    ucc_hier_sbgp_t       *hs;

    if (NULL == team->team_create_req) {
        status = ucc_team_multiple_req_alloc(&team->team_create_req,
                                             team->n_tl_teams);
        if (UCC_OK != status) {
            cl_error(ctx->super.super.lib,
                     "failed to allocate team req multiple");
            return status;
        }
        team->team_create_req->n_teams = 0;
        for (i = 0; i < UCC_HIER_SBGP_LAST; i++) {
            hs = &team->sbgps[i];
            if (hs->state == UCC_HIER_SBGP_ENABLED) {
                if (hs->score_map) {
                    ucc_coll_score_free_map(hs->score_map);
                }
                for (j = 0; j < hs->n_tls; j++) {
                    if (hs->tl_teams[j]) {
                        ucc_tl_context_put(hs->tl_ctxs[j]);
                        team->team_create_req
                            ->descs[team->team_create_req->n_teams++]
                            .team = hs->tl_teams[j];
                    }
                }
            }
        }
    }
    status = ucc_tl_team_destroy_multiple(team->team_create_req);
    if (UCC_INPROGRESS == status) {
        return status;
    }
    for (i = 0; i < team->team_create_req->n_teams; i++) {
        if (team->team_create_req->descs[i].status != UCC_OK) {
            cl_error(ctx->super.super.lib, "tl team destroy failed (%d)",
                     status);
            status = team->team_create_req->descs[i].status;
        }
    }
    ucc_team_multiple_req_free(team->team_create_req);
    UCC_CLASS_DELETE_FUNC_NAME(ucc_cl_hier_team_t)(cl_team);
    return status;
}

ucc_status_t ucc_cl_hier_team_create_test(ucc_base_team_t *cl_team)
{
    ucc_cl_hier_team_t    *team = ucc_derived_of(cl_team, ucc_cl_hier_team_t);
    ucc_cl_hier_context_t *ctx  = UCC_CL_HIER_TEAM_CTX(team);
    ucc_status_t           status;
    int                    i;
    ucc_coll_score_t      *score, *score_merge;
    struct ucc_team_team_desc *d;
    ucc_hier_sbgp_t           *hs;

    status = ucc_tl_team_create_multiple(team->team_create_req);

    if (status != UCC_OK) {
        return status;
    }

    team->n_tl_teams = 0;

    /* TL teams are created: get scores and merge them to produce
       score map for each sbgp */
    for (i = 0; i < team->team_create_req->n_teams; i++) {
        d                       = &team->team_create_req->descs[i];
        ucc_hier_sbgp_type_t st = (ucc_hier_sbgp_type_t)d->args[0];
        int                  tl = (int)d->args[1];

        hs = &team->sbgps[st];
        if (d->status == UCC_OK) {
            hs->tl_teams[tl] = d->team;
            team->n_tl_teams++;
            status = UCC_TL_TEAM_IFACE(d->team)->team.get_scores(
                &d->team->super, &score);
            if (UCC_OK != status) {
                cl_warn(ctx->super.super.lib, "failed to get tl %s scores",
                        UCC_TL_TEAM_IFACE(d->team)->super.name);
                continue;
                //goto cleanup ?
            }
            if (hs->score == NULL) {
                hs->score = score;
            } else {
                status =
                    ucc_coll_score_merge(hs->score, score, &score_merge, 1);
                if (UCC_OK != status) {
                    cl_warn(ctx->super.super.lib, "failed to merge scores");
                } else {
                    hs->score = score_merge;
                }
            }
            cl_info(ctx->super.super.lib, "initialized tl %s team for sbgp %s",
                    UCC_TL_CTX_IFACE(d->ctx)->super.name,
                    ucc_sbgp_str(hs->sbgp_type));
        } else {
            cl_debug(ctx->super.super.lib, "failed to create tl %s team",
                     UCC_TL_CTX_IFACE(d->ctx)->super.name);
            hs->tl_teams[tl] = NULL;
            hs->tl_ctxs[tl]  = NULL;
            ucc_tl_context_put(d->ctx);
        }
    }

    for (i = 0; i < UCC_HIER_SBGP_LAST; i++) {
        hs = &team->sbgps[i];
        if (hs->score == NULL) {
            if (hs->state == UCC_HIER_SBGP_ENABLED) {
                /* we tried to enable that sbgp, which means the subgroup
                   exists. however failed to create a single team there */
                cl_error(ctx->super.super.lib,
                         "no tl teams were created for sbgp %s",
                         ucc_sbgp_str(hs->sbgp_type));
                status = UCC_ERR_NO_RESOURCE;
                break;
            }
            hs->state = UCC_HIER_SBGP_DISABLED;
        } else {
            status = ucc_coll_score_build_map(hs->score, &hs->score_map);
            if (UCC_OK != status) {
                cl_error(ctx->super.super.lib, "failed to build score map");
                hs->state = UCC_HIER_SBGP_DISABLED;
                status = UCC_ERR_NO_RESOURCE;
                break;
            }
        }
    }
    ucc_team_multiple_req_free(team->team_create_req);
    team->team_create_req = NULL;

    if (SBGP_EXISTS(team, NODE_LEADERS)) {
        team->top_sbgp = UCC_HIER_SBGP_NODE_LEADERS;
    } else {
        ucc_assert(SBGP_EXISTS(team, NODE));
        team->top_sbgp = UCC_HIER_SBGP_NODE;
    }

    return status;
}

ucc_status_t ucc_cl_hier_team_get_scores(ucc_base_team_t   *cl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_cl_hier_team_t *team  = ucc_derived_of(cl_team, ucc_cl_hier_team_t);
    ucc_base_lib_t     *lib   = UCC_CL_TEAM_LIB(team);
    ucc_base_context_t *ctx   = UCC_CL_TEAM_CTX(team);
    ucc_memory_type_t   mt[2] = {UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA};
    ucc_coll_score_t   *score;
    ucc_status_t        status;
    int                 i;

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        cl_error(lib, "faild to alloc score_t");
        return status;
    }

    for (i = 0; i < 2; i++) {
        status = ucc_coll_score_add_range(
            score, UCC_COLL_TYPE_ALLTOALLV, mt[i], 0, UCC_MSG_MAX,
            /* low priority 1: to be enabled manually */
            1, ucc_cl_hier_alltoallv_init, cl_team);
        if (UCC_OK != status) {
            cl_error(lib, "failed to add range to score_t");
            return status;
        }

        status = ucc_coll_score_add_range(
            score, UCC_COLL_TYPE_ALLTOALL, mt[i], 0, UCC_MSG_MAX,
            /* low priority 1: to be enabled manually */
            1, ucc_cl_hier_alltoall_init, cl_team);
        if (UCC_OK != status) {
            cl_error(lib, "failed to add range to score_t");
            return status;
        }
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
        0, UCC_MSG_MAX, UCC_CL_HIER_DEFAULT_SCORE,
        ucc_cl_hier_barrier_init, cl_team);
    if (UCC_OK != status) {
        cl_error(lib, "faild to add range to score_t");
        return status;

    }

    for (i = 0; i < UCC_CL_HIER_N_DEFAULT_ALG_SELECT_STR; i++) {
        status = ucc_coll_score_update_from_str(
            ucc_cl_hier_default_alg_select_str[i], score,
            UCC_TL_TEAM_SIZE(team), ucc_cl_hier_coll_init, &team->super.super,
            UCC_CL_HIER_DEFAULT_SCORE, ucc_cl_hier_alg_id_to_init);
        if (UCC_OK != status) {
            cl_error(lib, "failed to apply default coll select setting: %s",
                     ucc_cl_hier_default_alg_select_str[i]);
            goto err;
        }
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, score, UCC_CL_TEAM_SIZE(team), NULL, cl_team,
            UCC_CL_HIER_DEFAULT_SCORE, ucc_cl_hier_alg_id_to_init);

        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
    }
    *score_p = score;
    return UCC_OK;
err:
    ucc_coll_score_free(score);
    *score_p = NULL;
    return status;
}
