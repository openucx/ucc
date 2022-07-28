/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp_coll_perf_params.h"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_1_128                       \
    "allreduce:0-1k:@0#allreduce:1k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_2_1                         \
    "allreduce:0-256k:@0#allreduce:256k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_4_1                         \
    "allreduce:0-16k:@0#allreduce:16k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_8_1                         \
    "allreduce:0-8k:@0#allreduce:8k-inf:@1"

static void
ucc_tl_ucp_amd_rome_single_node_128(ucc_tl_ucp_perf_params_t *p,
                                    ucc_tl_ucp_lib_config_t  *cfg,
                                    size_t                    data_size)
{
    p->allreduce_kn_radix           = 2;
    p->allreduce_sra_radix          = data_size < 131072 ? 4 :
                                          (data_size == 4194304 ? 2 : 8);
    p->allreduce_sra_n_frags        = cfg->allreduce_sra_kn_n_frags;
    p->allreduce_sra_pipeline_depth = cfg->allreduce_sra_kn_pipeline_depth;
    p->allreduce_sra_frag_thresh    = cfg->allreduce_sra_kn_frag_thresh;
}

static void
ucc_tl_ucp_amd_rome_2nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                ucc_tl_ucp_lib_config_t  *cfg,
                                size_t                    data_size)
{
    p->allreduce_kn_radix           = 2;
    p->allreduce_sra_radix          = 2;
    p->allreduce_sra_n_frags        = 3;
    p->allreduce_sra_pipeline_depth = 3;
    p->allreduce_sra_frag_thresh    = data_size >= 262144 ? 0 :
                                          cfg->allreduce_sra_kn_frag_thresh;
}

static void
ucc_tl_ucp_amd_rome_4nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                ucc_tl_ucp_lib_config_t  *cfg,
                                size_t                    data_size)
{
    p->allreduce_kn_radix           = 4;
    p->allreduce_sra_radix          = 4;
    p->allreduce_sra_n_frags        = 3;
    p->allreduce_sra_pipeline_depth = 3;
    p->allreduce_sra_frag_thresh    = data_size >= 2097152 ? 0 :
                                          cfg->allreduce_sra_kn_frag_thresh;
}

static void
ucc_tl_ucp_amd_rome_8nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                ucc_tl_ucp_lib_config_t  *cfg,
                                size_t                    data_size)
{
    p->allreduce_kn_radix           = data_size <= 8192 ? 8 : 2;
    p->allreduce_sra_radix          = 8;
    p->allreduce_sra_n_frags        = 3;
    p->allreduce_sra_pipeline_depth = 3;
    p->allreduce_sra_frag_thresh    = data_size >= 4194304 ? 0 :
                                          cfg->allreduce_sra_kn_frag_thresh;
}

TL_UCP_PERF_KEY_DECLARE_BASE(amd_rome_single_node_128ppn, AMD, ROME, 1, 128,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_1_128,
                             ucc_tl_ucp_amd_rome_single_node_128,
                             UCC_MEMORY_TYPE_HOST);

TL_UCP_PERF_KEY_DECLARE_BASE(amd_rome_2nodes_1ppn, AMD, ROME, 2, 1,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_2_1,
                             ucc_tl_ucp_amd_rome_2nodes_1ppn,
                             UCC_MEMORY_TYPE_HOST);

TL_UCP_PERF_KEY_DECLARE_BASE(amd_rome_4nodes_1ppn, AMD, ROME, 4, 1,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_4_1,
                             ucc_tl_ucp_amd_rome_4nodes_1ppn,
                             UCC_MEMORY_TYPE_HOST);

TL_UCP_PERF_KEY_DECLARE_BASE(amd_rome_8nodes_1ppn, AMD, ROME, 8, 1,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_8_1,
                             ucc_tl_ucp_amd_rome_8nodes_1ppn,
                             UCC_MEMORY_TYPE_HOST);
