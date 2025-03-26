/**
 * Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gatherv/gatherv.h"
#include "fanout/fanout.h"
#include "fanin/fanin.h"
#include "scatterv/scatterv.h"

ucc_status_t ucc_tl_ucp_get_lib_attr(const ucc_base_lib_t *lib,
                                     ucc_base_lib_attr_t  *base_attr);

ucc_status_t ucc_tl_ucp_get_lib_properties(ucc_base_lib_properties_t *prop);

ucc_status_t ucc_tl_ucp_get_context_attr(const ucc_base_context_t *context,
                                         ucc_base_ctx_attr_t      *base_attr);

ucc_config_field_t ucc_tl_ucp_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_ucp_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"ALLTOALL_PAIRWISE_NUM_POSTS", "auto",
     "Maximum number of outstanding send and receive messages in alltoall "
     "pairwise algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoall_pairwise_num_posts),
     UCC_CONFIG_TYPE_ULUNITS},

    {"ALLTOALLV_PAIRWISE_NUM_POSTS", "auto",
     "Maximum number of outstanding send and receive messages in alltoallv "
     "pairwise algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoallv_pairwise_num_posts),
     UCC_CONFIG_TYPE_ULUNITS},

/* TODO: add radix to config once it's fully supported by the algorithm
    {"ALLTOALLV_HYBRID_RADIX", "2",
     "Radix of the Hybrid Alltoallv algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoallv_hybrid_radix),
     UCC_CONFIG_TYPE_UINT},
*/
    {"ALLTOALLV_HYBRID_NUM_SCRATCH_SENDS", "1",
     "Number of send operations issued from scratch buffer per radix step",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoallv_hybrid_num_scratch_sends),
     UCC_CONFIG_TYPE_UINT},

    {"ALLTOALLV_HYBRID_NUM_SCRATCH_RECVS", "3",
     "Number of recv operations issued from scratch buffer per radix step",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoallv_hybrid_num_scratch_recvs),
     UCC_CONFIG_TYPE_UINT},

    {"ALLTOALLV_HYBRID_PAIRWISE_NUM_POSTS", "3",
     "The maximum number of pairwise messages to send before waiting for "
     "completion",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoallv_hybrid_pairwise_num_posts),
     UCC_CONFIG_TYPE_UINT},

    {"ALLTOALLV_HYBRID_BUFF_SIZE", "256k",
     "Total size of scratch buffer, used for sends and receives",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoallv_hybrid_buff_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLTOALLV_HYBRID_CHUNK_BYTE_LIMIT", "12k",
     "Max size of data send in pairwise step of hybrid alltoallv algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, alltoallv_hybrid_chunk_byte_limit),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"KN_RADIX", "0",
     "Radix of all algorithms based on knomial pattern. When set to a "
     "positive value it is used as a convenience parameter to set all "
     "other KN_RADIX values",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, kn_radix), UCC_CONFIG_TYPE_UINT},

    {"BARRIER_KN_RADIX", "8",
     "Radix of the recursive-knomial barrier algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, barrier_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"FANIN_KN_RADIX", "4", "Radix of the knomial tree fanin algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, fanin_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"FANOUT_KN_RADIX", "4", "Radix of the knomial tree fanout algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, fanout_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_KN_RADIX", "auto",
     "Radix of the recursive-knomial allreduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_kn_radix),
     UCC_CONFIG_TYPE_UINT_RANGED},

    {"ALLREDUCE_SLIDING_WIN_BUF_SIZE", "65536",
     "Buffer size of the sliding window allreduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_sliding_window_buf_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLREDUCE_SLIDING_WIN_PUT_WINDOW_SIZE", "0",
     "Max concurrent puts in SW Allreduce. 0 means set to team size",
     ucc_offsetof(ucc_tl_ucp_lib_config_t,
                  allreduce_sliding_window_put_window_size),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_SLIDING_WIN_NUM_GET_BUFS", "0",
     "Number of get buffers for sliding window AR. 0 means set to team size",
     ucc_offsetof(ucc_tl_ucp_lib_config_t,
                  allreduce_sliding_window_num_get_bufs),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_SRA_KN_RADIX", "auto",
     "Radix of the scatter-reduce-allgather (SRA) knomial allreduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_sra_kn_radix),
     UCC_CONFIG_TYPE_UINT_RANGED},

    {"ALLREDUCE_SRA_KN_PIPELINE", "auto",
     "Pipelining settings for SRA Knomial allreduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allreduce_sra_kn_pipeline),
     UCC_CONFIG_TYPE_PIPELINE_PARAMS},

    {"REDUCE_SCATTER_KN_RADIX", "4",
     "Radix of the knomial reduce-scatter algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, reduce_scatter_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"ALLGATHER_KN_RADIX", "auto", "Radix of the knomial allgather algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, allgather_kn_radix),
     UCC_CONFIG_TYPE_UINT_RANGED},

    {"BCAST_KN_RADIX", "4", "Radix of the recursive-knomial bcast algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, bcast_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"BCAST_SAG_KN_RADIX", "auto",
     "Radix of the scatter-allgather (SAG) knomial bcast algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, bcast_sag_kn_radix),
     UCC_CONFIG_TYPE_UINT_RANGED},

    {"REDUCE_KN_RADIX", "4", "Radix of the knomial tree reduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, reduce_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"GATHER_KN_RADIX", "4", "Radix of the knomial tree reduce algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, gather_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"GATHERV_LINEAR_NUM_POSTS", "0",
     "Maximum number of outstanding send and receive messages in gatherv "
     "linear algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, gatherv_linear_num_posts),
     UCC_CONFIG_TYPE_UINT},

    {"SCATTER_KN_RADIX", "4", "Radix of the knomial scatter algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, scatter_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"SCATTER_KN_ENABLE_RECV_ZCOPY", "auto",
     "Receive scatter data to user buffer with correct offset using zcopy",
     ucs_offsetof(ucc_tl_ucp_lib_config_t, scatter_kn_enable_recv_zcopy),
     UCS_CONFIG_TYPE_ON_OFF_AUTO},

    {"SCATTERV_LINEAR_NUM_POSTS", "16",
     "Maximum number of outstanding send and receive messages in scatterv "
     "linear algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, scatterv_linear_num_posts),
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
     "Launch 2 inverted rings concurrently during ReduceScatterv Ring "
     "algorithm",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, reduce_scatterv_ring_bidirectional),
     UCC_CONFIG_TYPE_BOOL},

    {"USE_TOPO", "try",
     "Allow usage of tl ucp topo",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, use_topo),
     UCC_CONFIG_TYPE_TERNARY},

    {"RANKS_REORDERING", "y",
     "Use topology information in TL UCP to reorder ranks. Requires topo info",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, use_reordering),
     UCC_CONFIG_TYPE_BOOL},

    {NULL}};

const char* ucc_tl_ucp_local_copy_names[] = {
    [UCC_TL_UCP_LOCAL_COPY_TYPE_UCP]  = "ucp",
    [UCC_TL_UCP_LOCAL_COPY_TYPE_MC]   = "mc",
    [UCC_TL_UCP_LOCAL_COPY_TYPE_EC]   = "ec",
    [UCC_TL_UCP_LOCAL_COPY_TYPE_AUTO] = "auto",
    [UCC_TL_UCP_LOCAL_COPY_TYPE_LAST] = NULL
};

static ucs_config_field_t ucc_tl_ucp_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_ucp_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {"PRECONNECT", "0",
     "Threshold that defines the number of ranks in the UCC team/context "
     "below which the team/context endpoints will be preconnected during "
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

    {"SERVICE_WORKER", "n",
     "If set to 0, uses the same worker for collectives and "
     "service. If not, creates a special worker for service collectives "
     "for which UCX_TL and UCX_NET_DEVICES are configured by the variables "
     "UCC_TL_UCP_SERVICE_TLS and UCC_TL_UCP_SERVICE_NET_DEVICES respectively",
     ucc_offsetof(ucc_tl_ucp_context_config_t, service_worker),
     UCC_CONFIG_TYPE_BOOL},

    {"SERVICE_THROTTLING_THRESH", "100",
     "Number of call to ucc_context_progress function between two consecutive "
     "calls to service worker progress function",
     ucc_offsetof(ucc_tl_ucp_context_config_t, service_throttling_thresh),
     UCC_CONFIG_TYPE_UINT},

    {"LOCAL_COPY_TYPE", "auto",
     "Determines what component is responsible for doing local copy "
     "during collective execution",
     ucc_offsetof(ucc_tl_ucp_context_config_t, local_copy_type),
     UCC_CONFIG_TYPE_ENUM(ucc_tl_ucp_local_copy_names)},

    {"MEMTYPE_COPY_ENABLE", "y",
     "Allows memory type copies. This option influences protocol selection in UCX. "
     "See https://github.com/openucx/ucx/pull/10490 for more details.",
     ucc_offsetof(ucc_tl_ucp_context_config_t, memtype_copy_enable),
     UCC_CONFIG_TYPE_BOOL},

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

ucs_memory_type_t ucc_memtype_to_ucs[UCC_MEMORY_TYPE_LAST + 1] = {
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
    ucc_tl_ucp.super.scoll.allgather = ucc_tl_ucp_service_allgather;
    ucc_tl_ucp.super.scoll.allreduce = ucc_tl_ucp_service_allreduce;
    ucc_tl_ucp.super.scoll.bcast     = ucc_tl_ucp_service_bcast;
    ucc_tl_ucp.super.scoll.update_id = ucc_tl_ucp_service_update_id;

    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLGATHER)] =
        ucc_tl_ucp_allgather_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLGATHERV)] =
        ucc_tl_ucp_allgatherv_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLREDUCE)] =
        ucc_tl_ucp_allreduce_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLTOALL)] =
        ucc_tl_ucp_alltoall_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLTOALLV)] =
        ucc_tl_ucp_alltoallv_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_BARRIER)] =
        ucc_tl_ucp_barrier_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_BCAST)] =
        ucc_tl_ucp_bcast_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_FANIN)] =
        ucc_tl_ucp_fanin_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_FANOUT)] =
        ucc_tl_ucp_fanout_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_GATHER)] =
        ucc_tl_ucp_gather_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_GATHERV)] =
        ucc_tl_ucp_gatherv_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_REDUCE)] =
        ucc_tl_ucp_reduce_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_REDUCE_SCATTER)] =
        ucc_tl_ucp_reduce_scatter_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_REDUCE_SCATTERV)] =
        ucc_tl_ucp_reduce_scatterv_algs;
    ucc_tl_ucp.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_SCATTERV)] =
        ucc_tl_ucp_scatterv_algs;

    /* no need to check return value, plugins can be absent */
    (void)ucc_components_load("tlcp_ucp", &ucc_tl_ucp.super.coll_plugins);
}
