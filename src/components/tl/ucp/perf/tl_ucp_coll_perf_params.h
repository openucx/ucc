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
extern ucc_tl_ucp_perf_key_t intel_broadwell_32nodes_1ppn;
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

#define TL_UCP_PERF_KEY_DECLARE_BASE(_name, _vendor, _model, _nnodes,         \
                                     _ppn, _ar_alg_thresh,                    \
                                     /*_allreduce_fn,*/...)                   \
    ucc_tl_ucp_perf_key_t _name = {                                           \
        .cpu_vendor           = UCC_CPU_VENDOR_ ## _vendor,                   \
        .cpu_model            = UCC_CPU_MODEL_ ## _vendor ## _ ## _model,     \
        .label                = UCC_PP_MAKE_STRING(_name),                    \
        .allreduce_alg_thresh = _ar_alg_thresh,                               \
        .nnodes               = _nnodes,                                      \
        .ppn                  = _ppn} /*,                            \
        .allreduce_func       = _allreduce_fn} */
