/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgather.h"
#include "../allgatherv/allgatherv.h"
#include "../cl_hier_coll.h"
#include "core/ucc_team.h"

ucc_base_coll_alg_info_t
    ucc_cl_hier_allgather_algs[UCC_CL_HIER_ALLGATHER_ALG_LAST + 1] = {
        [UCC_CL_HIER_ALLGATHER_ALG_GAB] =
            {.id   = UCC_CL_HIER_ALLGATHER_ALG_GAB,
             .name = "gab",
             .desc = "gatherv + allgatherv + bcast"},
        [UCC_CL_HIER_ALLGATHER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

/* Allgather is allgatherv with uniform per rank counts, so rather than
   duplicating the gatherv/allgatherv/bcast schedule building logic here,
   just validate the Allgather-specific preconditions and hand off to
   allgatherv init unchanged (still tagged UCC_COLL_TYPE_ALLGATHER):
   ucc_cl_hier_allgatherv_init() detects that tag itself, synthesizes the
   uniform counts/displacements pair it needs as part of its own scratch
   buffer, and re-validates the generic mem_type/persistent/active-set
   preconditions against the translated dst.info_v.* view, so there's
   nothing left to build or re-check here. */
UCC_CL_HIER_PROFILE_FUNC(
    ucc_status_t, ucc_cl_hier_allgather_init, (coll_args, team, task),
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_topo_t          *topo    = team->params.team->topo;

    if (topo->node_leader_rank_id != 0) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!SBGP_ENABLED(cl_team, NODE) && !SBGP_ENABLED(cl_team, NODE_LEADERS)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    return ucc_cl_hier_allgatherv_init(coll_args, team, task);
}
