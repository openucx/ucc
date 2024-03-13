/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_CL_DOCA_UROM_COLL_H_
#define UCC_CL_DOCA_UROM_COLL_H_

#include "cl_doca_urom.h"
#include "schedule/ucc_schedule_pipelined.h"
#include "components/mc/ucc_mc.h"

#include "../../tl/ucp/tl_ucp.h"

#define UCC_CL_DOCA_UROM_N_DEFAULT_ALG_SELECT_STR 2

extern const char
    *ucc_cl_doca_urom_default_alg_select_str[UCC_CL_DOCA_UROM_N_DEFAULT_ALG_SELECT_STR];

typedef struct ucc_cl_doca_urom_schedule_t {
    ucc_schedule_pipelined_t       super;
    struct ucc_cl_doca_urom_result res;
    struct export_buf              src_ebuf;
    struct export_buf              dst_ebuf;
} ucc_cl_doca_urom_schedule_t;

static inline ucc_cl_doca_urom_schedule_t *
ucc_cl_doca_urom_get_schedule(ucc_cl_doca_urom_team_t *team)
{
    ucc_cl_doca_urom_context_t  *ctx      = UCC_CL_DOCA_UROM_TEAM_CTX(team);
    ucc_cl_doca_urom_schedule_t *schedule = ucc_mpool_get(&ctx->sched_mp);

    return schedule;
}

static inline void ucc_cl_doca_urom_put_schedule(ucc_schedule_t *schedule)
{
    ucc_mpool_put(schedule);
}

#endif
