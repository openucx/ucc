/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include "ucc_team.h"
#include "utils/ucc_log.h"
static int compare_teams_by_priority(const void* t1, const void* t2)
{
    const ucc_tl_team_t** team1 = (const ucc_tl_team_t**)t1;
    const ucc_tl_team_t** team2 = (const ucc_tl_team_t**)t2;
    return (*team2)->iface->priority - (*team1)->iface->priority;
}

static inline void fill_tl_contexts_array(ucc_context_t **contexts,
                                          uint32_t n_ctx, ucc_team_lib_t *tl_lib,
                                          ucc_tl_context_t **tl_ctxs,
                                          uint32_t *n_tl_ctx)
{
    ucc_tl_iface_t *iface = tl_lib->iface;
    int i, j;
    *n_tl_ctx = 0;
    for (i=0; i<n_ctx; i++) {
        for (j=0; j<contexts[i]->n_tl_ctx; j++) {
            if (contexts[i]->tl_ctx[j]->tl_lib == tl_lib) {
                tl_ctxs[*n_tl_ctx] = contexts[i]->tl_ctx[j];
                (*n_tl_ctx)++;
                break;
            }
        }
    }
}

ucc_status_t ucc_team_create_post(ucc_context_t **contexts,
                                  uint32_t n_ctx,
                                  const ucc_team_params_t *params,
                                  ucc_team_t **ucc_team)
{
    int i;
    ucc_coll_type_t c;
    ucc_team_t *team;
    ucc_tl_context_t *tl_ctx;
    ucc_status_t status;
    ucc_tl_context_t **tl_ctxs;
    ucc_lib_info_t *lib;
    uint32_t n_tl_ctx;

    *ucc_team = NULL;
    if (n_ctx < 1) {
        ucc_error("No library contexts available");
        status = UCC_ERR_INVALID_PARAM;
        goto error;
    }
    lib = contexts[0]->lib;
    for (i=1; i<n_ctx; i++) {
        if (lib != contexts[i]->lib) {
            ucc_error("contexts used for team creation constructed "
                      "from different lib objects");
            status = UCC_ERR_INVALID_PARAM;
            goto error;
        }
    }
    team = (ucc_team_t*)malloc(sizeof(*team) +
                                sizeof(ucc_tl_team_t*)*(n_ctx-1));
    if (!team) {
        ucc_error("failed to allocate team");
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    team->contexts = (ucc_context_t**)malloc(n_ctx*sizeof(ucc_context_t*));
    if (!team->contexts) {
        ucc_error("failed to allocate team contexts array");
        status = UCC_ERR_NO_MEMORY;
        goto error_ctx;
    }
    memcpy(team->contexts, contexts, n_ctx*sizeof(ucc_context_t*));

    tl_ctxs = (ucc_tl_context_t**)malloc(n_ctx*sizeof(ucc_tl_context_t*));
    if (!tl_ctxs) {
        ucc_error("failed to allocate tl_ctxs array");
        status = UCC_ERR_NO_MEMORY;
        goto error_tl_ctxs;
    }

    team->n_tl_teams = 0;
    for (i=0; i<lib->n_libs_opened; i++) {
        fill_tl_contexts_array(contexts, n_ctx, lib->libs[i],
                               tl_ctxs, &n_tl_ctx);
        if (n_tl_ctx > 0) {
            status = lib->libs[i]->iface->
                team_create_post(tl_ctxs, n_tl_ctx, params,
                                 &team->tl_teams[team->n_tl_teams]);
            if (UCC_OK == status) {
                team->tl_teams[team->n_tl_teams]->iface = lib->libs[i]->iface;
                team->tl_teams[team->n_tl_teams]->tl_lib = lib->libs[i];
                team->n_tl_teams++;
            }
            ucc_info("tl team %s create_post status: %d", lib->libs[i]->iface->name, status);
        }
    }
    free(tl_ctxs);

    if (team->n_tl_teams == 0) {
        ucc_error("No teams created");
        status  = UCC_ERR_NO_MESSAGE;
        goto error_tl_ctxs;
    }

    team->status = UCC_INPROGRESS;
    *ucc_team = team;
    return UCC_OK;

error_tl_ctxs:
    free(team->contexts);
error_ctx:
    free(team);
error:
    return status;
}

ucc_status_t ucc_team_create_test(ucc_team_t *team)
{
    int i, c;
    for (i=0; i<team->n_tl_teams; i++) {
        if (UCC_INPROGRESS ==
            team->tl_teams[i]->iface->team_create_test(team->tl_teams[i])) {
            return UCC_INPROGRESS;
        }
    }
    qsort(team->tl_teams, team->n_tl_teams, sizeof(ucc_tl_team_t*),
          compare_teams_by_priority);
    for (c = 0; c < UCC_COLL_LAST; c++) {
        for (i=0; i<team->n_tl_teams; i++) {
            if (team->tl_teams[i]->iface->params.coll_types & UCS_BIT(c)) {
                team->coll_team_id[c] = i;
                break;
            }
        }
    }
    team->status = UCC_OK;
    /* TODO: check if some teams are never used after selection and clean them up */
    return UCC_OK;
}

void ucc_team_destroy(ucc_team_t *team)
{
    int               i;
    if (team->status != UCC_OK) {
        ucc_error("team %p is used before team_create is completed", team);
        return;
    }
    for (i=0; i<team->n_tl_teams; i++) {
        team->tl_teams[i]->iface->team_destroy(team->tl_teams[i]);
    }
    free(team);
}
