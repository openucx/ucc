/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp_coll_perf_params.h"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_1_40                        \
    "allreduce:0-2k:@0#allreduce:2k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_2_1                         \
    "allreduce:0-16k:@0#allreduce:16k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_4_1                         \
    "allreduce:0-8k:@0#allreduce:8k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_8_1                         \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_32_1                        \
    "allreduce:0-2k:@0#allreduce:2k-inf:@1"

static void
ucc_tl_ucp_intel_skylake_single_node_40(ucc_tl_ucp_perf_params_t *p,
                                        ucc_tl_ucp_lib_config_t  *cfg,
                                        ucc_memory_type_t         mtype,
                                        size_t                    data_size)
{
    if (mtype == UCC_MEMORY_TYPE_HOST) {
        p->allreduce_kn_radix           = 2;
        p->allreduce_sra_radix          = data_size < 262144 ? 2 : 8;
        p->allreduce_sra_n_frags        = cfg->allreduce_sra_kn_n_frags;
        p->allreduce_sra_pipeline_depth = cfg->allreduce_sra_kn_pipeline_depth;
        p->allreduce_sra_frag_thresh    = cfg->allreduce_sra_kn_frag_thresh;
    } else {
        ucc_tl_ucp_perf_params_generic_allreduce(p, cfg, mtype, data_size);
    }
}

static void
ucc_tl_ucp_intel_skylake_2nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                     ucc_tl_ucp_lib_config_t  *cfg,
                                     ucc_memory_type_t         mtype,
                                     size_t                    data_size)
{
    if (mtype == UCC_MEMORY_TYPE_HOST) {
        p->allreduce_kn_radix           = 2;
        p->allreduce_sra_radix          = 2;
        p->allreduce_sra_n_frags        = data_size <= 262144 ? 2 : 4;
        p->allreduce_sra_pipeline_depth = data_size <= 262144 ? 2 : 4;
        p->allreduce_sra_frag_thresh    = 0;
    } else {
        ucc_tl_ucp_perf_params_generic_allreduce(p, cfg, mtype, data_size);
    }
}

static void
ucc_tl_ucp_intel_skylake_4nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                     ucc_tl_ucp_lib_config_t  *cfg,
                                     ucc_memory_type_t         mtype,
                                     size_t                    data_size)
{
    if (mtype == UCC_MEMORY_TYPE_HOST) {
        p->allreduce_kn_radix           = 4;
        p->allreduce_sra_radix          = 4;
        p->allreduce_sra_n_frags        = 3;
        p->allreduce_sra_pipeline_depth = 3;
        p->allreduce_sra_frag_thresh    = data_size >= 131072 ? 0 :
            cfg->allreduce_sra_kn_frag_thresh;
    } else {
        ucc_tl_ucp_perf_params_generic_allreduce(p, cfg, mtype, data_size);
    }
}

static void
ucc_tl_ucp_intel_skylake_8nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                     ucc_tl_ucp_lib_config_t  *cfg,
                                     ucc_memory_type_t         mtype,
                                     size_t                    data_size)
{
    if (mtype == UCC_MEMORY_TYPE_HOST) {
        p->allreduce_kn_radix           = data_size < 16384 ? 8 : 2;
        p->allreduce_sra_radix          = 8;
        p->allreduce_sra_n_frags        = 3;
        p->allreduce_sra_pipeline_depth = 3;
        p->allreduce_sra_frag_thresh    = data_size >= 1048576 ? 0 :
            cfg->allreduce_sra_kn_frag_thresh;
    } else {
        ucc_tl_ucp_perf_params_generic_allreduce(p, cfg, mtype, data_size);
    }
}

static void
ucc_tl_ucp_intel_skylake_32nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                      ucc_tl_ucp_lib_config_t  *cfg,
                                      ucc_memory_type_t         mtype,
                                      size_t                    data_size)
{
    if (mtype == UCC_MEMORY_TYPE_HOST) {
        p->allreduce_kn_radix           = data_size < 16384 ? 8 : 2;
        p->allreduce_sra_radix          = 8;
        p->allreduce_sra_n_frags        = 3;
        p->allreduce_sra_pipeline_depth = 3;
        p->allreduce_sra_frag_thresh    = data_size >= 1048576 ? 0 :
            cfg->allreduce_sra_kn_frag_thresh;
    } else {
        ucc_tl_ucp_perf_params_generic_allreduce(p, cfg, mtype, data_size);
    }
}

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_single_node_40ppn, INTEL, SKYLAKE,
                             1, 40, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_1_40,
                             ucc_tl_ucp_intel_skylake_single_node_40);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_2nodes_1ppn, INTEL, SKYLAKE,
                             2, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_2_1,
                             ucc_tl_ucp_intel_skylake_2nodes_1ppn);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_4nodes_1ppn, INTEL, SKYLAKE,
                             4, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_4_1,
                             ucc_tl_ucp_intel_skylake_4nodes_1ppn);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_8nodes_1ppn, INTEL, SKYLAKE,
                             8, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_8_1,
                             ucc_tl_ucp_intel_skylake_8nodes_1ppn);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_32nodes_1ppn, INTEL, SKYLAKE,
                             32, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_32_1,
                             ucc_tl_ucp_intel_skylake_32nodes_1ppn);
