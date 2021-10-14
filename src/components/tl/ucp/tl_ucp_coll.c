/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_team.h"
#include "barrier/barrier.h"
#include "alltoall/alltoall.h"
#include "alltoallv/alltoallv.h"
#include "allreduce/allreduce.h"
#include "allgather/allgather.h"
#include "allgatherv/allgatherv.h"
#include "bcast/bcast.h"
#include "reduce/reduce.h"
const char
    *ucc_tl_ucp_default_alg_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR] = {
        UCC_TL_UCP_ALLREDUCE_DEFAULT_ALG_SELECT_STR,
        UCC_TL_UCP_BCAST_DEFAULT_ALG_SELECT_STR,
        UCC_TL_UCP_ALLTOALL_DEFAULT_ALG_SELECT_STR};

void ucc_tl_ucp_send_completion_cb(void *request, ucs_status_t status,
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in send completion %s",
                 ucs_status_string(status));
        task->super.super.status = ucs_status_to_ucc_status(status);
    }
    task->send_completed++;
    ucp_request_free(request);
}

void ucc_tl_ucp_recv_completion_cb(void *request, ucs_status_t status,
                                   const ucp_tag_recv_info_t *info, /* NOLINT */
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in send completion %s",
                 ucs_status_string(status));
        task->super.super.status = ucs_status_to_ucc_status(status);
    }
    task->recv_completed++;
    ucp_request_free(request);
}

ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_ucp_put_task(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *team,
                                  ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_task_t    *task = ucc_tl_ucp_init_task(coll_args, team);
    ucc_status_t          status;

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_BARRIER:
        status = ucc_tl_ucp_barrier_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        status = ucc_tl_ucp_alltoall_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        status = ucc_tl_ucp_alltoallv_init(task);
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        status = ucc_tl_ucp_allreduce_init(task);
        break;
    case UCC_COLL_TYPE_ALLGATHER:
        status = ucc_tl_ucp_allgather_init(task);
        break;
    case UCC_COLL_TYPE_ALLGATHERV:
        status = ucc_tl_ucp_allgatherv_init(task);
        break;
    case UCC_COLL_TYPE_BCAST:
        status = ucc_tl_ucp_bcast_init(task);
        break;
    case UCC_COLL_TYPE_REDUCE:
        status = ucc_tl_ucp_reduce_init(task);
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    tl_trace(team->context->lib, "init coll req %p", task);
    *task_h = &task->super;
    return status;
}

static inline int alg_id_from_str(ucc_coll_type_t coll_type, const char *str)
{
    switch (coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_tl_ucp_allreduce_alg_from_str(str);
    case UCC_COLL_TYPE_BCAST:
        return ucc_tl_ucp_bcast_alg_from_str(str);
    case UCC_COLL_TYPE_ALLTOALL:
        return ucc_tl_ucp_alltoall_alg_from_str(str);
    default:
        break;
    }
    return -1;
}

ucc_status_t ucc_tl_ucp_alg_id_to_init(int alg_id, const char *alg_id_str,
                                       ucc_coll_type_t   coll_type,
                                       ucc_memory_type_t mem_type, //NOLINT
                                       ucc_base_coll_init_fn_t *init)
{
    ucc_status_t status = UCC_OK;
    if (alg_id_str) {
        alg_id = alg_id_from_str(coll_type, alg_id_str);
    }

    switch (coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        switch (alg_id) {
        case UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL:
            *init = ucc_tl_ucp_allreduce_knomial_init;
            break;
        case UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL:
            *init = ucc_tl_ucp_allreduce_sra_knomial_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_BCAST:
        switch (alg_id) {
        case UCC_TL_UCP_BCAST_ALG_KNOMIAL:
            *init = ucc_tl_ucp_bcast_knomial_init;
            break;
        case UCC_TL_UCP_BCAST_ALG_SAG_KNOMIAL:
            *init = ucc_tl_ucp_bcast_sag_knomial_init;
            break;
        default:
           status = UCC_ERR_INVALID_PARAM;
           break;
        };
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        switch (alg_id) {
        case UCC_TL_UCP_ALLTOALL_ALG_PAIRWISE:
            *init = ucc_tl_ucp_alltoall_pairwise_init;
            break;
        case UCC_TL_UCP_ALLTOALL_ALG_ONESIDED:
            *init = ucc_tl_ucp_alltoall_onesided_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
        break;
    }
    return status;
}
