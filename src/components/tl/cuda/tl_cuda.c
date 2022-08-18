/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"
#include "core/ucc_team.h"
#include "components/mc/base/ucc_mc_base.h"
#include "allgather/allgather.h"
#include "allgatherv/allgatherv.h"

static ucc_config_field_t ucc_tl_cuda_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_cuda_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"MAX_CONCURRENT", "8",
     "Maximum number of outstanding colls",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, max_concurrent),
     UCC_CONFIG_TYPE_UINT},

    {"SCRATCH_SIZE", "2Mb",
     "Size of the internal scratch buffer",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, scratch_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLGATHER_RING_MAX_RINGS", "2",
     "Max number of rings used in allgather and allgatherv ring algorithms",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, allgather_ring_max_rings),
     UCC_CONFIG_TYPE_UINT},

    {"ALLGATHER_RING_NUM_CHUNKS", "4",
     "Number of chunks each ring message will be split into",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, allgather_ring_num_chunks),
     UCC_CONFIG_TYPE_UINT},

    {"REDUCE_SCATTER_RING_MAX_RINGS", "2",
     "Max number of rings used in reduce_scatter and "
     "reduce_scatterv ring algorithms",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, reduce_scatter_ring_max_rings),
     UCC_CONFIG_TYPE_UINT},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_lib_t, ucc_base_lib_t);

ucc_status_t ucc_tl_cuda_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t *base_attr);

static ucs_config_field_t ucc_tl_cuda_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_cuda_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {NULL}};

ucc_status_t ucc_tl_cuda_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t *base_attr);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_cuda_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_cuda_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

ucc_status_t ucc_tl_cuda_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p);

UCC_TL_IFACE_DECLARE(cuda, CUDA);

__attribute__((constructor)) static void tl_cuda_iface_init(void)
{

    ucc_tl_cuda.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLGATHER)] =
        ucc_tl_cuda_allgather_algs;
    ucc_tl_cuda.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLGATHERV)] =
        ucc_tl_cuda_allgatherv_algs;
}
