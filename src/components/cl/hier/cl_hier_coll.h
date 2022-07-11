/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_CL_HIER_COLL_H_
#define UCC_CL_HIER_COLL_H_

#include "cl_hier.h"
#include "schedule/ucc_schedule_pipelined.h"
#include "components/mc/ucc_mc.h"
#include "allreduce/allreduce.h"
#include "alltoallv/alltoallv.h"
#include "alltoall/alltoall.h"
#include "barrier/barrier.h"

#define UCC_CL_HIER_N_DEFAULT_ALG_SELECT_STR 1

extern const char
    *ucc_cl_hier_default_alg_select_str[UCC_CL_HIER_N_DEFAULT_ALG_SELECT_STR];

typedef struct ucc_cl_hier_schedule_t {
    ucc_schedule_pipelined_t super;
    ucc_mc_buffer_header_t  *scratch;
    union {
        struct {
            uint64_t *counts;
        } allreduce_split_rail;
    };
} ucc_cl_hier_schedule_t;

static inline ucc_cl_hier_schedule_t *
ucc_cl_hier_get_schedule(ucc_cl_hier_team_t *team)
{
    ucc_cl_hier_context_t  *ctx      = UCC_CL_HIER_TEAM_CTX(team);
    ucc_cl_hier_schedule_t *schedule = ucc_mpool_get(&ctx->sched_mp);

    schedule->scratch = NULL;
    UCC_CL_HIER_PROFILE_REQUEST_NEW(schedule, "cl_hier_sched_p", 0);
    return schedule;
}

static inline void ucc_cl_hier_put_schedule(ucc_schedule_t *schedule)
{
    UCC_CL_HIER_PROFILE_REQUEST_FREE(schedule);
    ucc_mpool_put(schedule);
}

ucc_status_t ucc_cl_hier_alg_id_to_init(int alg_id, const char *alg_id_str,
                                        ucc_coll_type_t   coll_type,
                                        ucc_memory_type_t mem_type, //NOLINT
                                        ucc_base_coll_init_fn_t *init);
#endif
