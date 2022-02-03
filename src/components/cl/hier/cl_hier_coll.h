/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_CL_HIER_COLL_H_
#define UCC_CL_HIER_COLL_H_

#include "cl_hier.h"
#include "schedule/ucc_schedule_pipelined.h"
#include "components/mc/ucc_mc.h"

typedef struct ucc_cl_hier_schedule_t {
    ucc_schedule_pipelined_t super;
    ucc_mc_buffer_header_t  *scratch;
} ucc_cl_hier_schedule_t;

static inline ucc_cl_hier_schedule_t *
ucc_cl_hier_get_schedule(ucc_cl_hier_team_t *team)
{
    ucc_cl_hier_context_t  *ctx      = UCC_CL_HIER_TEAM_CTX(team);
    ucc_cl_hier_schedule_t *schedule = ucc_mpool_get(&ctx->sched_mp);

    UCC_CL_HIER_PROFILE_REQUEST_NEW(schedule, "cl_hier_sched_p", 0);
    return schedule;
}

static inline void ucc_cl_hier_put_schedule(ucc_schedule_t *schedule)
{
    UCC_CL_HIER_PROFILE_REQUEST_FREE(schedule);
    ucc_mpool_put(schedule);
}

#endif
