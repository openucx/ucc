/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "utils/ucc_malloc.h"
#include "components/mc/ucc_mc.h"
#include "components/mc/base/ucc_mc_base.h"
#include "allreduce/allreduce.h"
#include "bcast/bcast.h"
#include "barrier/barrier.h"
#include "alltoall/alltoall.h"
#include "alltoallv/alltoallv.h"
#include "allgather/allgather.h"
#include "allgatherv/allgatherv.h"
#include "reduce_scatter/reduce_scatter.h"
#include "reduce_scatterv/reduce_scatterv.h"
#include "reduce/reduce.h"
#include "gather/gather.h"
#include "fanout/fanout.h"
#include "fanin/fanin.h"

ucc_status_t ucc_tl_ucp_get_lib_attr(const ucc_base_lib_t *lib,
                                     ucc_base_lib_attr_t  *base_attr);
ucc_status_t ucc_tl_ucp_get_context_attr(const ucc_base_context_t *context,
                                         ucc_base_ctx_attr_t      *base_attr);

static ucc_config_field_t ucc_tl_ucp_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_ucp_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"ALLTOALL_PAIRWISE_NUM_POSTS", "1",
     "Maximum number of outstanding send and receive messages in alltoall "
     "pairwise algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoall_pairwise_num_posts),
     UCC_CONFIG_TYPE_UINT},

    {"ALLTOALLV_PAIRWISE_NUM_POSTS", "1",
     "Maximum number of outstanding send and receive messages in alltoallv "
     "pairwise algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoallv_pairwise_num_posts),
     UCC_CONFIG_TYPE_UINT},

    {"KN_RADIX", "0",
     "Radix of all algorithms based on knomial pattern. When set to a "
     "positive value it is used as a convinience parameter to set all "
     "other KN_RADIX values",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, kn_radix), UCC_CONFIG_TYPE_UINT},

    {"BARRIER_KN_RADIX", "4",
     "Radix of the recursive-knomial barrier algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, barrier_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"FANIN_KN_RADIX", "4", "Radix of the knomial tree fanin algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, fanin_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"FANOUT_KN_RADIX", "4", "Radix of the knomial tree fanout algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, fanout_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_KN_RADIX", "4",
     "Radix of the recursive-knomial allreduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_SRA_KN_RADIX", "4",
     "Radix of the scatter-reduce-allgather (SRA) knomial allreduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_sra_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_SRA_KN_FRAG_THRESH", "inf",
     "Threshold to enable fragmentation and pipelining of SRA Knomial "
     "allreduce alg",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_sra_kn_frag_thresh),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLREDUCE_SRA_KN_FRAG_SIZE", "inf",
     "Maximum allowed fragment size of SRA knomial alg",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_sra_kn_frag_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLREDUCE_SRA_KN_N_FRAGS", "2",
     "Number of fragments each allreduce is split into when SRA knomial alg is "
     "used\n"
     "The actual number of fragments can be larger if fragment size exceeds\n"
     "ALLREDUCE_SRA_KN_FRAG_SIZE",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_sra_kn_n_frags),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_SRA_KN_PIPELINE_DEPTH", "2",
     "Number of fragments simultaneously progressed by the SRA knomial alg",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_sra_kn_pipeline_depth),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_SRA_KN_SEQUENTIAL", "n",
     "Type of pipelined schedule for SRA knomial alg (sequential/parallel)",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_sra_kn_seq),
     UCC_CONFIG_TYPE_BOOL},

    {"REDUCE_SCATTER_KN_RADIX", "4",
     "Radix of the knomial reduce-scatter algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, reduce_scatter_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"ALLGATHER_KN_RADIX", "4", "Radix of the knomial allgather algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allgather_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"BCAST_KN_RADIX", "4", "Radix of the recursive-knomial bcast algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, bcast_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"BCAST_SAG_KN_RADIX", "4",
     "Radix of the scatter-allgather (SAG) knomial bcast algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, bcast_sag_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"REDUCE_KN_RADIX", "4", "Radix of the knomial tree reduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, reduce_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"GATHER_KN_RADIX", "4", "Radix of the knomial tree reduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, gather_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"SCATTER_KN_RADIX", "4", "Radix of the knomial scatter algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, scatter_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"REDUCE_AVG_PRE_OP", "1",
     "Reduce will perform division by team_size in early stages of the "
     "algorithm,\n"
     "else - in result",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, reduce_avg_pre_op),
     UCC_CONFIG_TYPE_BOOL},

    {"REDUCE_SCATTER_RING_BIDIRECTIONAL", "y",
     "Launch 2 inverted rings concurrently during ReduceScatter Ring algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, reduce_scatter_ring_bidirectional),
     UCC_CONFIG_TYPE_BOOL},

    {"REDUCE_SCATTERV_RING_BIDIRECTIONAL", "y",
     "Launch 2 inverted rings concurrently  during ReduceScatterV Ring "
     "algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, reduce_scatterv_ring_bidirectional),
     UCC_CONFIG_TYPE_BOOL},

    {NULL}};

static ucs_config_field_t ucc_tl_ucp_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_ucp_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {"PRECONNECT", "0",
     "Threshold that defines the number of ranks in the UCC team/context "
     "below which the team/context enpoints will be preconnected during "
     "corresponding team/context create call",
     ucc_offsetof(ucc_tl_ucp_context_config_t, preconnect),
     UCC_CONFIG_TYPE_UINT},

    {"NPOLLS", "10",
     "Number of ucp progress polling cycles for p2p requests testing",
     ucc_offsetof(ucc_tl_ucp_context_config_t, n_polls), UCC_CONFIG_TYPE_UINT},

    {"OOB_NPOLLS", "20",
     "Number of polling cycles for oob allgather and service coll request",
     ucc_offsetof(ucc_tl_ucp_context_config_t, oob_npolls),
     UCC_CONFIG_TYPE_UINT},

    {"PRE_REG_MEM", "0", "Pre Register collective memory region with UCX",
     ucc_offsetof(ucc_tl_ucp_context_config_t, pre_reg_mem),
     UCC_CONFIG_TYPE_UINT},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_ucp_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_ucp_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_ucp_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_ucp_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_ucp_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_ucp_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_ucp_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *team,
                                  ucc_coll_task_t **task);

ucc_status_t ucc_tl_ucp_populate_rcache(void *addr, size_t length,
                                        ucs_memory_type_t mem_type,
                                        ucc_tl_ucp_context_t *ctx);

ucc_status_t ucc_tl_ucp_service_allreduce(ucc_base_team_t *team, void *sbuf,
                                          void *rbuf, ucc_datatype_t dt,
                                          size_t count, ucc_reduction_op_t op,
                                          ucc_subset_t      subset,
                                          ucc_coll_task_t **task);

ucc_status_t ucc_tl_ucp_service_allgather(ucc_base_team_t *team, void *sbuf,
                                          void *rbuf, size_t msgsize,
                                          ucc_subset_t      subset,
                                          ucc_coll_task_t **task_p);

ucc_status_t ucc_tl_ucp_service_bcast(ucc_base_team_t *team, void *buf,
                                      size_t msgsize, ucc_rank_t root,
                                      ucc_subset_t      subset,
                                      ucc_coll_task_t **task_p);

ucc_status_t ucc_tl_ucp_service_test(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_service_cleanup(ucc_coll_task_t *task);

void         ucc_tl_ucp_service_update_id(ucc_base_team_t *team, uint16_t id);

ucc_status_t ucc_tl_ucp_team_get_scores(ucc_base_team_t   *tl_team,
                                        ucc_coll_score_t **score);

UCC_TL_IFACE_DECLARE(ucp, UCP);

ucs_memory_type_t ucc_memtype_to_ucs[UCC_MEMORY_TYPE_LAST+1] = {
    [UCC_MEMORY_TYPE_HOST]         = UCS_MEMORY_TYPE_HOST,
    [UCC_MEMORY_TYPE_CUDA]         = UCS_MEMORY_TYPE_CUDA,
    [UCC_MEMORY_TYPE_CUDA_MANAGED] = UCS_MEMORY_TYPE_CUDA_MANAGED,
    [UCC_MEMORY_TYPE_ROCM]         = UCS_MEMORY_TYPE_ROCM,
    [UCC_MEMORY_TYPE_ROCM_MANAGED] = UCS_MEMORY_TYPE_ROCM_MANAGED,
    [UCC_MEMORY_TYPE_UNKNOWN]      = UCS_MEMORY_TYPE_UNKNOWN
};

UCC_TL_UCP_PROFILE_FUNC_VOID(ucc_tl_ucp_pre_register_mem, (team, addr, length,
                             mem_type), ucc_tl_ucp_team_t *team, void *addr,
                             size_t length, ucc_memory_type_t mem_type)
{
    void *base_address  = addr;
    size_t alloc_length = length;
    ucc_mem_attr_t mem_attr;
    ucc_status_t status;

    if ((addr == NULL) || (length == 0)) {
        return;
    }

    mem_attr.field_mask   = UCC_MEM_ATTR_FIELD_BASE_ADDRESS |
                            UCC_MEM_ATTR_FIELD_ALLOC_LENGTH;
    mem_attr.alloc_length = length;
    status = ucc_mc_get_mem_attr(addr, &mem_attr);
    if (ucc_likely(status == UCC_OK)) {
        base_address = mem_attr.base_address;
        alloc_length = mem_attr.alloc_length;
    } else {
        tl_warn(UCC_TL_TEAM_LIB(team), "failed to query base addr and len");
    }

    status = ucc_tl_ucp_populate_rcache(base_address, alloc_length,
                                        ucc_memtype_to_ucs[mem_type],
                                        UCC_TL_UCP_TEAM_CTX(team));
    if (ucc_unlikely(status != UCC_OK)) {
        tl_warn(UCC_TL_TEAM_LIB(team), "ucc_tl_ucp_mem_map failed");
    }
}

__attribute__((constructor)) static void tl_ucp_iface_init(void)
{
    ucc_tl_ucp.super.scoll.allreduce = ucc_tl_ucp_service_allreduce;
    ucc_tl_ucp.super.scoll.allgather = ucc_tl_ucp_service_allgather;
    ucc_tl_ucp.super.scoll.bcast     = ucc_tl_ucp_service_bcast;
    ucc_tl_ucp.super.scoll.update_id = ucc_tl_ucp_service_update_id;

    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLREDUCE)] =
        ucc_tl_ucp_allreduce_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_BCAST)] =
        ucc_tl_ucp_bcast_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_BARRIER)] =
        ucc_tl_ucp_barrier_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLTOALL)] =
        ucc_tl_ucp_alltoall_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLTOALLV)] =
        ucc_tl_ucp_alltoallv_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_REDUCE_SCATTER)] =
        ucc_tl_ucp_reduce_scatter_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_REDUCE_SCATTERV)] =
        ucc_tl_ucp_reduce_scatterv_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_REDUCE)] =
        ucc_tl_ucp_reduce_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_GATHER)] =
        ucc_tl_ucp_gather_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_FANIN)] =
        ucc_tl_ucp_fanin_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_FANOUT)] =
        ucc_tl_ucp_fanout_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLGATHER)] =
        ucc_tl_ucp_allgather_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLGATHERV)] =
        ucc_tl_ucp_allgatherv_algs;

    ucc_components_load("tlcp_ucp", &ucc_tl_ucp.super.coll_plugins);
}
