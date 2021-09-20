/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team.h"
#include "ucc_context.h"
#include "ucc_mc.h"
#include "components/cl/ucc_cl.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_coll_utils.h"
#include "utils/profile/ucc_profile_core.h"
#include "schedule/ucc_schedule.h"

/* NOLINTNEXTLINE  */
static ucc_cl_team_t *ucc_select_cl_team(ucc_coll_args_t *coll_args,
                                         ucc_team_t *team)
{
    /* TODO1: collective CL selection logic will be there.
       for now just return 1st CL on a list
       TODO2: remove NOLINT once TODO1 is done */
    return team->cl_teams[0];
}

#define UCC_BUFFER_INFO_CHECK_MEM_TYPE(_info) do {                             \
    if ((_info).mem_type == UCC_MEMORY_TYPE_UNKNOWN) {                         \
        ucc_mem_attr_t mem_attr;                                               \
        ucc_status_t st;                                                       \
        mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;                     \
        st = ucc_mc_get_mem_attr((_info).buffer, &mem_attr);                   \
        if (ucc_unlikely(st != UCC_OK)) {                                      \
            return st;                                                         \
        }                                                                      \
        (_info).mem_type = mem_attr.mem_type;                                  \
    }                                                                          \
} while(0)

#define UCC_BUFFER_INFO_CHECK_DATATYPE(_info1, _info2) do {                    \
    if ((_info1).datatype != (_info2).datatype) {                              \
        ucc_error("datatype missmatch");                                       \
        return UCC_ERR_INVALID_PARAM;                                          \
    }                                                                          \
} while(0)

#define UCC_IS_ROOT(_args, _myrank) ((_args).root == (_myrank))

#if ENABLE_DEBUG == 1
static ucc_status_t ucc_check_coll_args(const ucc_coll_args_t *coll_args,
                                        ucc_rank_t rank)
{
    switch (coll_args->coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_DATATYPE(coll_args->src.info,
                                           coll_args->dst.info);
        }
        break;
    case UCC_COLL_TYPE_REDUCE:
        if (!UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_DATATYPE(coll_args->src.info,
                                           coll_args->dst.info);
        }
        break;
    default:
        return UCC_OK;
    }
    return UCC_OK;
}
#else
#define ucc_check_coll_args(...) UCC_OK
#endif

static ucc_status_t ucc_coll_args_check_mem_type(ucc_coll_args_t *coll_args,
                                                 ucc_rank_t rank)
{
    switch (coll_args->coll_type) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return UCC_OK;
    case UCC_COLL_TYPE_BCAST:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        return UCC_OK;
    case UCC_COLL_TYPE_ALLREDUCE:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        } else {
            coll_args->src.info.mem_type = coll_args->dst.info.mem_type;
        }
        return UCC_OK;
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info_v);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_ALLTOALLV:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info_v);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info_v);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_GATHER:
    	// TODO: Check logic for Gather once implemented
    case UCC_COLL_TYPE_REDUCE:
        if (!UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        } else {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
            if (!UCC_IS_INPLACE(*coll_args)) {
                UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        	}
        }
        return UCC_OK;
    case UCC_COLL_TYPE_GATHERV:
        if (UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info_v);
        }
        if (!(UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank))) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_SCATTER:
        if (UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        if (!(UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank))) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_SCATTERV:
        if (UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info_v);
        }
        if (!(UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank))) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        }
        return UCC_OK;
    default:
        ucc_error("unknown collective type");
        return UCC_ERR_INVALID_PARAM;
    };
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_init,
                      (coll_args, request, team), ucc_coll_args_t *coll_args,
                      ucc_coll_req_h *request, ucc_team_h team)
{
    ucc_cl_team_t         *cl_team;
    ucc_coll_task_t       *task;
    ucc_base_coll_args_t   op_args;
    ucc_status_t           status;

    status = ucc_coll_args_check_mem_type(coll_args, team->rank);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("memory type detection failed");
        return status;
    }
    status = ucc_check_coll_args(coll_args, team->rank);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("collective arguments check failed");
        return status;
    }

    /* TO discuss: maybe we want to pass around user pointer ? */
    op_args.mask = 0;
    memcpy(&op_args.args, coll_args, sizeof(ucc_coll_args_t));
    op_args.team = team;
    cl_team      = ucc_select_cl_team(coll_args, team);
    status =
        UCC_CL_TEAM_IFACE(cl_team)->coll.init(&op_args, &cl_team->super, &task);
    if (UCC_ERR_NOT_SUPPORTED == status) {
        ucc_debug("failed to init collective: not supported");
        return status;
    } else if (ucc_unlikely(status < 0)) {
        ucc_error("failed to init collective: %s", ucc_status_string(status));
        return status;
    }
    if (coll_args->mask & UCC_COLL_ARGS_FIELD_CB) {
        task->cb = coll_args->cb;
        task->flags |= UCC_COLL_TASK_FLAG_CB;
    }
    if (ucc_global_config.log_component.log_level >= UCC_LOG_LEVEL_DEBUG) {
        char coll_debug_str[256];
        ucc_coll_str(&op_args, coll_debug_str, sizeof(coll_debug_str));
        ucc_debug("coll_init: %s, req %p", coll_debug_str, task);
    }
    *request = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_collective_post(ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);

    ucc_debug("coll_post: req %p", task);
    return task->post(task);
}

ucc_status_t ucc_collective_triggered_post(ucc_ee_h ee, ucc_ev_t *ev)
{
    ucc_coll_task_t *task = ucc_derived_of(ev->req, ucc_coll_task_t);

    ucc_debug("coll_triggered_post: req %p", task);
    task->ee = ee;
    return task->triggered_post(ee, ev, task);
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_finalize, (request),
                      ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);

    ucc_debug("coll_finalize: req %p", task);
    return task->finalize(task);
}
