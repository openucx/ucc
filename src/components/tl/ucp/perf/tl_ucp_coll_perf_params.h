/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

extern ucc_tl_ucp_perf_key_t intel_broadwell_single_node_28ppn;
extern ucc_tl_ucp_perf_key_t intel_broadwell_2nodes_1ppn;
extern ucc_tl_ucp_perf_key_t intel_broadwell_4nodes_1ppn;
extern ucc_tl_ucp_perf_key_t intel_broadwell_8nodes_1ppn;
extern ucc_tl_ucp_perf_key_t intel_skylake_single_node_40ppn;
extern ucc_tl_ucp_perf_key_t intel_skylake_2nodes_1ppn;
extern ucc_tl_ucp_perf_key_t intel_skylake_4nodes_1ppn;
extern ucc_tl_ucp_perf_key_t intel_skylake_8nodes_1ppn;
extern ucc_tl_ucp_perf_key_t intel_skylake_32nodes_1ppn;
extern ucc_tl_ucp_perf_key_t amd_rome_single_node_128ppn;
extern ucc_tl_ucp_perf_key_t amd_rome_2nodes_1ppn;
extern ucc_tl_ucp_perf_key_t amd_rome_4nodes_1ppn;
extern ucc_tl_ucp_perf_key_t amd_rome_8nodes_1ppn;

#define UCC_TL_UCP_ALLREDUCE_DEFAULT_ALG_SELECT_STR                           \
    "allreduce:0-8k:@0#allreduce:8k-inf:@1"

static inline void
ucc_tl_ucp_perf_params_generic_allreduce(ucc_tl_ucp_perf_params_t *p,
                                         ucc_tl_ucp_lib_config_t  *cfg,
                                         size_t data_size /* NOLINT */)
{
    p->allreduce_kn_radix           = cfg->allreduce_kn_radix;
    p->allreduce_sra_radix          = cfg->allreduce_sra_kn_radix;
    p->allreduce_sra_n_frags        = cfg->allreduce_sra_kn_n_frags;
    p->allreduce_sra_pipeline_depth = cfg->allreduce_sra_kn_pipeline_depth;
    p->allreduce_sra_frag_thresh    = cfg->allreduce_sra_kn_frag_thresh;
}

#define TL_UCP_PERF_KEY_DECLARE_BASE(_name, _vendor, _model, _nnodes, _ppn,   \
                                     _ar_alg_thresh, _allreduce_fn, _mt...)   \
    ucc_tl_ucp_perf_key_t _name = {                                           \
        .cpu_vendor           = UCC_CPU_VENDOR_ ## _vendor,                   \
        .cpu_model            = UCC_CPU_MODEL_ ## _vendor ## _ ## _model,     \
        .label                = UCC_PP_MAKE_STRING(_name),                    \
        .allreduce_alg_thresh = _ar_alg_thresh,                               \
        .nnodes               = _nnodes,                                      \
        .ppn                  = _ppn,                                         \
        .perf_allreduce_func  = (_mt == UCC_MEMORY_TYPE_HOST) ? _allreduce_fn \
            : ucc_tl_ucp_perf_params_generic_allreduce} /* TODO: add support for gpu */
