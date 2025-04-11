/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_team.h"
#include "barrier/barrier.h"
#include "alltoall/alltoall.h"
#include "alltoallv/alltoallv.h"
#include "allreduce/allreduce.h"
#include "allgather/allgather.h"
#include "allgatherv/allgatherv.h"
#include "reduce_scatter/reduce_scatter.h"
#include "reduce_scatterv/reduce_scatterv.h"
#include "bcast/bcast.h"
#include "reduce/reduce.h"
#include "gather/gather.h"
#include "gatherv/gatherv.h"
#include "fanin/fanin.h"
#include "fanout/fanout.h"
#include "scatterv/scatterv.h"

const ucc_tl_ucp_default_alg_desc_t
    ucc_tl_ucp_default_alg_descs[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR] = {
        {
            .select_str = NULL,
            .str_get_fn = ucc_tl_ucp_allgather_score_str_get
        },
        {
            .select_str = UCC_TL_UCP_ALLGATHERV_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = NULL,
            .str_get_fn = ucc_tl_ucp_alltoall_score_str_get
        },
        {
            .select_str = UCC_TL_UCP_ALLREDUCE_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_BCAST_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_REDUCE_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_REDUCE_SCATTERV_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_ALLTOALLV_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        }
};

ucc_status_t ucc_tl_ucp_team_default_score_str_alloc(ucc_tl_ucp_team_t *team,
    char *default_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR])
{
    ucc_status_t st = UCC_OK;
    int i;

    for (i = 0; i < UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR; i++) {
        if (ucc_tl_ucp_default_alg_descs[i].select_str) {
            default_select_str[i] = strdup(ucc_tl_ucp_default_alg_descs[i].select_str);
        } else {
            default_select_str[i] = ucc_tl_ucp_default_alg_descs[i].str_get_fn(team);
        }
        if (!default_select_str[i]) {
            st = UCC_ERR_NO_MEMORY;
            goto exit;
        }

    }

exit:
    return st;
}

void ucc_tl_ucp_team_default_score_str_free(
    char *default_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR])
{
    int i;

    for (i = 0; i < UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR; i++) {
        ucc_free(default_select_str[i]);
    }
}

void ucc_tl_ucp_send_completion_cb_st(void *request, ucs_status_t status,
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in send completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ++task->tagged.send_completed;
    ucp_request_free(request);
}

void ucc_tl_ucp_send_completion_cb_mt(void *request, ucs_status_t status,
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in send completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ucc_atomic_add32(&task->tagged.send_completed, 1);
    ucp_request_free(request);
}

void ucc_tl_ucp_put_completion_cb(void *request, ucs_status_t status,
                                  void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in put completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    task->onesided.put_completed++;
    ucp_request_free(request);
}

void ucc_tl_ucp_get_completion_cb(void *request, ucs_status_t status,
                                  void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in get completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    task->onesided.get_completed++;
    ucp_request_free(request);
}

void ucc_tl_ucp_recv_completion_cb_mt(void *request, ucs_status_t status,
                                      const ucp_tag_recv_info_t *info, /* NOLINT */
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ucc_atomic_add32(&task->tagged.recv_completed, 1);
    ucp_request_free(request);
}

void ucc_tl_ucp_recv_completion_cb_st(void *request, ucs_status_t status,
                                      const ucp_tag_recv_info_t *info, /* NOLINT */
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ++task->tagged.recv_completed;
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
    case UCC_COLL_TYPE_GATHER:
        status = ucc_tl_ucp_gather_init(task);
        break;
    case UCC_COLL_TYPE_FANIN:
        status = ucc_tl_ucp_fanin_init(task);
        break;
    case UCC_COLL_TYPE_FANOUT:
        status = ucc_tl_ucp_fanout_init(task);
        break;
    case UCC_COLL_TYPE_SCATTERV:
        status = ucc_tl_ucp_scatterv_init(task);
        break;
    case UCC_COLL_TYPE_GATHERV:
        status = ucc_tl_ucp_gatherv_init(task);
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
    case UCC_COLL_TYPE_ALLGATHER:
        return ucc_tl_ucp_allgather_alg_from_str(str);
    case UCC_COLL_TYPE_ALLGATHERV:
        return ucc_tl_ucp_allgatherv_alg_from_str(str);
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_tl_ucp_allreduce_alg_from_str(str);
    case UCC_COLL_TYPE_ALLTOALL:
        return ucc_tl_ucp_alltoall_alg_from_str(str);
    case UCC_COLL_TYPE_ALLTOALLV:
        return ucc_tl_ucp_alltoallv_alg_from_str(str);
    case UCC_COLL_TYPE_BCAST:
        return ucc_tl_ucp_bcast_alg_from_str(str);
    case UCC_COLL_TYPE_REDUCE:
        return ucc_tl_ucp_reduce_alg_from_str(str);
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return ucc_tl_ucp_reduce_scatter_alg_from_str(str);
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return ucc_tl_ucp_reduce_scatterv_alg_from_str(str);
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
    case UCC_COLL_TYPE_ALLGATHER:
        switch (alg_id) {
        case UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL:
            *init = ucc_tl_ucp_allgather_knomial_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_RING:
            *init = ucc_tl_ucp_allgather_ring_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_NEIGHBOR:
            *init = ucc_tl_ucp_allgather_neighbor_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_BRUCK:
            *init = ucc_tl_ucp_allgather_bruck_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_SPARBIT:
            *init = ucc_tl_ucp_allgather_sparbit_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLGATHERV:
        switch (alg_id) {
        case UCC_TL_UCP_ALLGATHERV_ALG_KNOMIAL:
            *init = ucc_tl_ucp_allgatherv_knomial_init;
            break;
        case UCC_TL_UCP_ALLGATHERV_ALG_RING:
            *init = ucc_tl_ucp_allgatherv_ring_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        switch (alg_id) {
        case UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL:
            *init = ucc_tl_ucp_allreduce_knomial_init;
            break;
        case UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL:
            *init = ucc_tl_ucp_allreduce_sra_knomial_init;
            break;
        case UCC_TL_UCP_ALLREDUCE_ALG_DBT:
            *init = ucc_tl_ucp_allreduce_dbt_init;
            break;
        case UCC_TL_UCP_ALLREDUCE_ALG_SLIDING_WINDOW:
            *init = ucc_tl_ucp_allreduce_sliding_window_init;
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
        case UCC_TL_UCP_BCAST_ALG_DBT:
            *init = ucc_tl_ucp_bcast_dbt_init;
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
        case UCC_TL_UCP_ALLTOALL_ALG_BRUCK:
            *init = ucc_tl_ucp_alltoall_bruck_init;
            break;
        case UCC_TL_UCP_ALLTOALL_ALG_ONESIDED:
            *init = ucc_tl_ucp_alltoall_onesided_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        switch (alg_id) {
        case UCC_TL_UCP_ALLTOALLV_ALG_PAIRWISE:
            *init = ucc_tl_ucp_alltoallv_pairwise_init;
            break;
        case UCC_TL_UCP_ALLTOALLV_ALG_HYBRID:
            *init = ucc_tl_ucp_alltoallv_hybrid_init;
            break;
        case UCC_TL_UCP_ALLTOALLV_ALG_ONESIDED:
            *init = ucc_tl_ucp_alltoallv_onesided_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_REDUCE:
        switch (alg_id) {
        case UCC_TL_UCP_REDUCE_ALG_KNOMIAL:
            *init = ucc_tl_ucp_reduce_knomial_init;
            break;
        case UCC_TL_UCP_REDUCE_ALG_DBT:
            *init = ucc_tl_ucp_reduce_dbt_init;
            break;
        case UCC_TL_UCP_REDUCE_ALG_SRG:
            *init = ucc_tl_ucp_reduce_srg_knomial_init;
            break;
        default:
           status = UCC_ERR_INVALID_PARAM;
           break;
        };
        break;
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        switch (alg_id) {
        case UCC_TL_UCP_REDUCE_SCATTER_ALG_RING:
            *init = ucc_tl_ucp_reduce_scatter_ring_init;
            break;
        case UCC_TL_UCP_REDUCE_SCATTER_ALG_KNOMIAL:
            *init = ucc_tl_ucp_reduce_scatter_knomial_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        switch (alg_id) {
        case UCC_TL_UCP_REDUCE_SCATTERV_ALG_RING:
            *init = ucc_tl_ucp_reduce_scatterv_ring_init;
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
