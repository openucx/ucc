/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp_coll_perf_params.h"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_1_28                    \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_2_1                     \
    "allreduce:0-128k:@0#allreduce:128k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_4_1                     \
    "allreduce:0-16k:@0#allreduce:16k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_8_1                     \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1"

static void
ucc_tl_ucp_intel_broadwell_single_node_28(ucc_tl_ucp_perf_params_t *p,
                                          ucc_tl_ucp_lib_config_t  *cfg,
                                          ucc_memory_type_t         mtype,
                                          size_t data_size /* NOLINT */)
{
    if (mtype == UCC_MEMORY_TYPE_HOST) {
        p->allreduce_kn_radix  = 2;
        p->allreduce_sra_radix = 7;
        p->allreduce_sra_n_frags        = cfg->allreduce_sra_kn_n_frags;
        p->allreduce_sra_pipeline_depth = cfg->allreduce_sra_kn_pipeline_depth;
        p->allreduce_sra_frag_thresh    = cfg->allreduce_sra_kn_frag_thresh;
    } else {
        ucc_tl_ucp_perf_params_generic_allreduce(p, cfg, mtype, data_size);
    }
}

static void
ucc_tl_ucp_intel_broadwell_2nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                       ucc_tl_ucp_lib_config_t  *cfg,
                                       ucc_memory_type_t         mtype,
                                       size_t data_size /* NOLINT */)
{
    if (mtype == UCC_MEMORY_TYPE_HOST) {
        p->allreduce_kn_radix  = 2;
        p->allreduce_sra_radix = 2;
        p->allreduce_sra_n_frags        = cfg->allreduce_sra_kn_n_frags;
        p->allreduce_sra_pipeline_depth = cfg->allreduce_sra_kn_pipeline_depth;
        p->allreduce_sra_frag_thresh    = cfg->allreduce_sra_kn_frag_thresh;
    } else {
        ucc_tl_ucp_perf_params_generic_allreduce(p, cfg, mtype, data_size);
    }
}

static void
ucc_tl_ucp_intel_broadwell_4nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                       ucc_tl_ucp_lib_config_t  *cfg,
                                       ucc_memory_type_t         mtype,
                                       size_t data_size /* NOLINT */)
{
    if (mtype == UCC_MEMORY_TYPE_HOST) {
        p->allreduce_kn_radix  = 4;
        p->allreduce_sra_radix = 4;
        p->allreduce_sra_n_frags        = cfg->allreduce_sra_kn_n_frags;
        p->allreduce_sra_pipeline_depth = cfg->allreduce_sra_kn_pipeline_depth;
        p->allreduce_sra_frag_thresh    = cfg->allreduce_sra_kn_frag_thresh;
    } else {
        ucc_tl_ucp_perf_params_generic_allreduce(p, cfg, mtype, data_size);
    }
}

static void
ucc_tl_ucp_intel_broadwell_8nodes_1ppn(ucc_tl_ucp_perf_params_t *p,
                                       ucc_tl_ucp_lib_config_t  *cfg,
                                       ucc_memory_type_t         mtype,
                                       size_t data_size /* NOLINT */)
{
    if (mtype == UCC_MEMORY_TYPE_HOST) {
        p->allreduce_kn_radix  = 8;
        p->allreduce_sra_radix = 8;
        p->allreduce_sra_n_frags        = cfg->allreduce_sra_kn_n_frags;
        p->allreduce_sra_pipeline_depth = cfg->allreduce_sra_kn_pipeline_depth;
        p->allreduce_sra_frag_thresh    = cfg->allreduce_sra_kn_frag_thresh;
    } else {
        ucc_tl_ucp_perf_params_generic_allreduce(p, cfg, mtype, data_size);
    }
}

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_single_node_28ppn, INTEL,
                             BROADWELL, 1, 28,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_1_28,
                             ucc_tl_ucp_intel_broadwell_single_node_28);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_2nodes_1ppn, INTEL, BROADWELL,
                             2, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_2_1,
                             ucc_tl_ucp_intel_broadwell_2nodes_1ppn);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_4nodes_1ppn, INTEL, BROADWELL,
                             4, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_4_1,
                             ucc_tl_ucp_intel_broadwell_4nodes_1ppn);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_8nodes_1ppn, INTEL, BROADWELL,
                             8, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_8_1,
                             ucc_tl_ucp_intel_broadwell_8nodes_1ppn);

