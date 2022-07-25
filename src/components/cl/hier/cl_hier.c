/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_hier.h"
#include "utils/ucc_malloc.h"
#include "allreduce/allreduce.h"
#include "alltoall/alltoall.h"
#include "alltoallv/alltoallv.h"

ucc_status_t ucc_cl_hier_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t  *base_attr);
ucc_status_t ucc_cl_hier_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t      *base_attr);

static ucc_config_field_t ucc_cl_hier_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_hier_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_lib_config_table)},

    {"NODE_SBGP_TLS", "ucp",
     "TLS to be used for NODE subgroup.\n"
     "NODE subgroup contains processes of a team located on the same node",
     ucc_offsetof(ucc_cl_hier_lib_config_t, sbgp_tls[UCC_HIER_SBGP_NODE]),
     UCC_CONFIG_TYPE_ALLOW_LIST},

    {"NODE_LEADERS_SBGP_TLS", "ucp",
     "TLS to be used for NODE_LEADERS subgroup.\n"
     "NODE_LEADERS subgroup contains processes of a team with local node rank "
     "equal 0",
     ucc_offsetof(ucc_cl_hier_lib_config_t,
                  sbgp_tls[UCC_HIER_SBGP_NODE_LEADERS]),
     UCC_CONFIG_TYPE_ALLOW_LIST},

    {"NET_SBGP_TLS", "ucp",
     "TLS to be used for NET subgroup.\n"
     "NET subgroup contains processes of a team with identical local node "
     "rank.\n"
     "This subgroup only exists for teams with equal number of processes "
     "across the nodes",
     ucc_offsetof(ucc_cl_hier_lib_config_t, sbgp_tls[UCC_HIER_SBGP_NET]),
     UCC_CONFIG_TYPE_ALLOW_LIST},

    {"FULL_SBGP_TLS", "ucp",
     "TLS to be used for FULL subgroup.\n"
     "FULL subgroup contains all processes of the team",
     ucc_offsetof(ucc_cl_hier_lib_config_t, sbgp_tls[UCC_HIER_SBGP_FULL]),
     UCC_CONFIG_TYPE_ALLOW_LIST},

    {"ALLTOALLV_SPLIT_NODE_THRESH", "0",
     "Messages larger than that threshold will be sent via node sbgp tl",
     ucc_offsetof(ucc_cl_hier_lib_config_t, a2av_node_thresh),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLREDUCE_SPLIT_RAIL_FRAG_THRESH", "inf",
     "Threshold to enable fragmentation and pipelining of Split_Rail "
     "allreduce alg",
     ucc_offsetof(ucc_cl_hier_lib_config_t, allreduce_split_rail_frag_thresh),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLREDUCE_SPLIT_RAIL_FRAG_SIZE", "inf",
     "Maximum allowed fragment size of Split_Rail alg",
     ucc_offsetof(ucc_cl_hier_lib_config_t, allreduce_split_rail_frag_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLREDUCE_SPLIT_RAIL_N_FRAGS", "2",
     "Number of fragments each allreduce is split into when Split_Rail alg is "
     "used\n"
     "The actual number of fragments can be larger if fragment size exceeds\n"
     "ALLREDUCE_SPLIT_RAIL_FRAG_SIZE",
     ucc_offsetof(ucc_cl_hier_lib_config_t, allreduce_split_rail_n_frags),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_SPLIT_RAIL_PIPELINE_DEPTH", "2",
     "Number of fragments simultaneously progressed by the Split_Rail alg",
     ucc_offsetof(ucc_cl_hier_lib_config_t,
                  allreduce_split_rail_pipeline_depth),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_SPLIT_RAIL_SEQUENTIAL", "n",
     "Type of pipelined schedule for Split_Rail alg (sequential/parallel)",
     ucc_offsetof(ucc_cl_hier_lib_config_t, allreduce_split_rail_seq),
     UCC_CONFIG_TYPE_BOOL},

    {NULL}};

static ucs_config_field_t ucc_cl_hier_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_hier_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_context_config_table)},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_hier_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_hier_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_hier_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_hier_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_hier_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_cl_hier_team_create_test(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_hier_team_destroy(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_hier_team_get_scores(ucc_base_team_t   *cl_team,
                                         ucc_coll_score_t **score);
UCC_CL_IFACE_DECLARE(hier, HIER);

__attribute__((constructor)) static void cl_hier_iface_init(void)
{
    ucc_cl_hier.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLREDUCE)] =
        ucc_cl_hier_allreduce_algs;
    ucc_cl_hier.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLTOALL)] =
        ucc_cl_hier_alltoall_algs;
    ucc_cl_hier.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLTOALLV)] =
        ucc_cl_hier_alltoallv_algs;
}
