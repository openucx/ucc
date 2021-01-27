/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team.h"
#include "ucc_lib.h"
#include "components/cl/ucc_cl.h"

void ucc_copy_team_params(ucc_team_params_t *dst, const ucc_team_params_t *src)
{
    dst->mask = src->mask;
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_ORDERING, ordering);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS,
                            outstanding_colls);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_EP, ep);
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

static ucc_status_t ucc_team_create_post_single(ucc_context_t *context,
                                                ucc_team_t *team)
{
    int                    i;
    ucc_status_t           status;
    ucc_base_team_t       *b_team;
    ucc_cl_iface_t        *cl_iface;
    ucc_base_team_params_t b_params;
    memcpy(&b_params.params, &team->params, sizeof(ucc_team_params_t));
    team->cl_teams = ucc_malloc(sizeof(ucc_cl_team_t *) * context->n_cl_ctx);
    if (!team) {
        ucc_error("failed to allocate %zd bytes for cl teams array",
                  sizeof(ucc_cl_team_t *) * context->n_cl_ctx);
        return UCC_ERR_NO_MEMORY;
    }
    for (i = 0; i < context->n_cl_ctx; i++) {
        cl_iface = UCC_CL_CTX_IFACE(context->cl_ctx[i]);
        status   = cl_iface->team.create_post(&context->cl_ctx[i]->super,
                                            &b_params, &b_team);

        if (status != UCC_OK) {
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            /* TODO: what is the logic here? continue or report error?
               maybe need to recheck supported matrix after the whole
               ucc_team is created and report error if not all requested
               capabilities are provided */
            continue;
        }
        status = cl_iface->team.create_test(b_team);
        if (status < 0) {
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            /* TODO: see comment above */
            continue;
        }
        team->cl_teams[team->n_cl_teams++] =
            ucc_derived_of(b_team, ucc_cl_team_t);
        if (status == UCC_INPROGRESS) {
            /* workaround to fix oob allgather issue if multiple teams use it
               simultaneously*/
            break;
        }
    }
    if (team->n_cl_teams == 0) {
        ucc_error("no cl teams have been created for ucc_team");
        return UCC_ERR_NO_MESSAGE;
    }
    team->last_team_create_posted = i;
    team->status = (i < context->n_cl_ctx) ? UCC_INPROGRESS : UCC_OK;
    return UCC_OK;
}

ucc_status_t ucc_team_create_post(ucc_context_h *contexts, uint32_t num_contexts,
                                  const ucc_team_params_t *params,
                                  ucc_team_h *new_team)
{
    ucc_team_t  *team;
    ucc_status_t status;
    if (num_contexts < 1) {
        return UCC_ERR_INVALID_PARAM;
    } else if (num_contexts > 1) {
        ucc_error("team creation from multiple contexts is not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }
    team = ucc_malloc(sizeof(ucc_team_t), "ucc_team");
    if (!team) {
        ucc_error("failed to allocate %zd bytes for ucc team",
                  sizeof(ucc_team_t));
        return UCC_ERR_NO_MEMORY;
    }
    team->n_cl_teams   = 0;
    team->num_contexts = num_contexts;
    team->contexts =
        ucc_malloc(sizeof(ucc_context_t *) * num_contexts, "ucc_team_ctx");
    if (!team->contexts) {
        ucc_error("failed to allocate %zd bytes for ucc team contexts array",
                  sizeof(ucc_context_t) * num_contexts);
        status = UCC_ERR_NO_MEMORY;
        goto err_ctx_alloc;
    }
    memcpy(team->contexts, contexts, sizeof(ucc_context_t *) * num_contexts);
    ucc_copy_team_params(&team->params, params);
    status    = ucc_team_create_post_single(contexts[0], team);
    *new_team = team;
    return status;

err_ctx_alloc:
    free(team);
    return status;
}

ucc_status_t ucc_team_create_test_single(ucc_context_t *context,
                                         ucc_team_t    *team)
{
    int                    i;
    ucc_cl_iface_t        *cl_iface;
    ucc_base_team_t       *b_team;
    ucc_status_t           status;
    ucc_base_team_params_t b_params;

    cl_iface = UCC_CL_CTX_IFACE(context->cl_ctx[team->last_team_create_posted]);
    b_team   = &team->cl_teams[team->last_team_create_posted]->super;
    status   = cl_iface->team.create_test(b_team);
    if (status < 0) {
        ucc_debug("failed to create CL %s team", cl_iface->super.name);
        /* TODO: see comment above */
    } else if (status == UCC_INPROGRESS) {
        return status;
    }

    memcpy(&b_params.params, &team->params, sizeof(ucc_team_params_t));
    for (i = team->last_team_create_posted + 1; i < context->n_cl_ctx; i++) {
        cl_iface = UCC_CL_CTX_IFACE(context->cl_ctx[i]);
        status   = cl_iface->team.create_post(&context->cl_ctx[i]->super,
                                            &b_params, &b_team);
        if (status != UCC_OK) {
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            /* TODO: see comment above*/
            continue;
        }
        status = cl_iface->team.create_test(b_team);
        if (status < 0) {
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            /* TODO: see comment above */
            continue;
        }
        team->cl_teams[team->n_cl_teams++] =
            ucc_derived_of(b_team, ucc_cl_team_t);
        if (status == UCC_INPROGRESS) {
            /* workaround to fix oob allgather issue if multiple teams use it
               simultaneously*/
            return UCC_INPROGRESS;
        }
    }
    team->status = UCC_OK;
    /* TODO: add team/coll selection and check if some teams are never
             used after selection and clean them up */
    return UCC_OK;
}

ucc_status_t ucc_team_create_test(ucc_team_h team)
{
    /* we don't support multiple contexts per team yet */
    ucc_assert(team->num_contexts == 1);
    if (team->status == UCC_OK) {
        return UCC_OK;
    }
    return ucc_team_create_test_single(team->contexts[0], team);
}

static ucc_status_t ucc_team_destroy_single(ucc_team_h team)
{
    ucc_cl_iface_t *cl_iface;
    int             i;
    ucc_status_t    status;
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
    ucc_free(team);
    return UCC_OK;
}

ucc_status_t ucc_team_destroy_nb(ucc_team_h team)
{
    if (team->status != UCC_OK) {
        ucc_error("team %p is used before team_create is completed", team);
        return UCC_ERR_INVALID_PARAM;
    }

    /* we don't support multiple contexts per team yet */
    ucc_assert(team->num_contexts == 1);
    return ucc_team_destroy_single(team);
}
ucc_status_t ucc_team_destroy(ucc_team_h team)
{
    ucc_status_t status;
    while (UCC_INPROGRESS == (status = ucc_team_destroy_single(team))) {
        ; //TODO call ucc progress here
    }
    return status;
}
