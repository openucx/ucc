/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "utils/ucc_coll_utils.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_allreduce_algs[UCC_TL_UCP_ALLREDUCE_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL,
             .name = "knomial",
             .desc =
                 "recursive k-ing with arbitrary radix (latency oriented alg)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL,
             .name = "sra_knomial",
             .desc = "recursive k-nomial scatter-reduce followed by k-nomial "
                     "allgather (bw oriented alg)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

#define CHECK_USERDEFINED_OP(_args, _team)                                     \
    do {                                                                       \
        if (_args.mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) {         \
            tl_error(UCC_TL_TEAM_LIB(_team),                                   \
                     "userdefined reductions are not supported yet");          \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define CHECK_INPLACE(_args, _team)                                            \
    do {                                                                       \
        if (!UCC_IS_INPLACE(_args) &&                                          \
            (_args.src.info.mem_type != _args.dst.info.mem_type)) {            \
            tl_error(UCC_TL_TEAM_LIB(_team),                                   \
                     "assymetric src/dst memory types are not supported yet"); \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define ALLREDUCE_TASK_CHECK(_args, _team)                                     \
    CHECK_USERDEFINED_OP((_args), (_team));                                    \
    CHECK_INPLACE((_args), (_team));

ucc_status_t ucc_tl_ucp_allreduce_init(ucc_tl_ucp_task_t *task)
{
    ALLREDUCE_TASK_CHECK(task->args, task->team);
    task->super.post     = ucc_tl_ucp_allreduce_knomial_start;
    task->super.progress = ucc_tl_ucp_allreduce_knomial_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_allreduce_knomial_start;
    task->super.progress = ucc_tl_ucp_allreduce_knomial_progress;
    *task_h              = &task->super;
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allreduce_sra_knomial_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t *     team,
                                      ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_allreduce_sra_knomial_start;
    task->super.progress = ucc_tl_ucp_allreduce_sra_knomial_progress;
    *task_h              = &task->super;
    return UCC_OK;
}
