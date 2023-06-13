/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_H_
#define UCC_TL_UCP_H_
#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "core/ucc_ee.h"
#include "utils/ucc_mpool.h"
#include "tl_ucp_ep_hash.h"
#include "schedule/ucc_schedule_pipelined.h"
#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>
#include "core/ucc_service_coll.h"

#ifndef UCC_TL_UCP_DEFAULT_SCORE
#define UCC_TL_UCP_DEFAULT_SCORE 10
#endif

#ifdef HAVE_PROFILING_TL_UCP
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_UCP_PROFILE_FUNC UCC_PROFILE_FUNC
#define UCC_TL_UCP_PROFILE_FUNC_VOID UCC_PROFILE_FUNC_VOID
#define UCC_TL_UCP_PROFILE_REQUEST_NEW UCC_PROFILE_REQUEST_NEW
#define UCC_TL_UCP_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_UCP_PROFILE_REQUEST_FREE UCC_PROFILE_REQUEST_FREE

#define MAX_NR_SEGMENTS 32
#define ONESIDED_SYNC_SIZE 1
#define ONESIDED_REDUCE_SIZE 4

typedef struct ucc_tl_ucp_iface {
    ucc_tl_iface_t super;
} ucc_tl_ucp_iface_t;
/* Extern iface should follow the pattern: ucc_tl_<tl_name> */
extern ucc_tl_ucp_iface_t ucc_tl_ucp;

typedef struct ucc_tl_ucp_lib_config {
    ucc_tl_lib_config_t      super;
    uint32_t                 kn_radix;
    uint32_t                 fanin_kn_radix;
    uint32_t                 fanout_kn_radix;
    uint32_t                 barrier_kn_radix;
    ucc_mrange_uint_t        allreduce_kn_radix;
    ucc_mrange_uint_t        allreduce_sra_kn_radix;
    uint32_t                 reduce_scatter_kn_radix;
    uint32_t                 allgather_kn_radix;
    uint32_t                 bcast_kn_radix;
    ucc_mrange_uint_t        bcast_sag_kn_radix;
    uint32_t                 reduce_kn_radix;
    uint32_t                 gather_kn_radix;
    uint32_t                 gatherv_linear_num_posts;
    uint32_t                 scatter_kn_radix;
    ucc_on_off_auto_value_t  scatter_kn_enable_recv_zcopy;
    uint32_t                 scatterv_linear_num_posts;
    uint32_t                 alltoall_pairwise_num_posts;
    uint32_t                 alltoallv_pairwise_num_posts;
    ucc_pipeline_params_t    allreduce_sra_kn_pipeline;
    int                      reduce_avg_pre_op;
    int                      reduce_scatter_ring_bidirectional;
    int                      reduce_scatterv_ring_bidirectional;
    uint32_t                 alltoallv_hybrid_radix;
    size_t                   alltoallv_hybrid_buff_size;
    size_t                   alltoallv_hybrid_chunk_byte_limit;
    uint32_t                 alltoallv_hybrid_num_scratch_sends;
    uint32_t                 alltoallv_hybrid_num_scratch_recvs;
    uint32_t                 alltoallv_hybrid_pairwise_num_posts;
    ucc_ternary_auto_value_t use_topo;
    int                      use_reordering;
} ucc_tl_ucp_lib_config_t;

typedef struct ucc_tl_ucp_context_config {
    ucc_tl_context_config_t super;
    uint32_t                preconnect;
    uint32_t                n_polls;
    uint32_t                oob_npolls;
    uint32_t                pre_reg_mem;
    uint32_t                service_worker;
    uint32_t                service_throttling_thresh;
} ucc_tl_ucp_context_config_t;

typedef ucc_tl_ucp_lib_config_t ucc_tl_ucp_team_config_t;

typedef struct ucc_tl_ucp_lib {
    ucc_tl_lib_t            super;
    ucc_tl_ucp_lib_config_t cfg;
    void                  **tlcp_configs;
} ucc_tl_ucp_lib_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_ucp_remote_info {
    void * va_base;
    size_t len;
    void * mem_h;
    void * packed_key;
    size_t packed_key_len;
} ucc_tl_ucp_remote_info_t;

typedef struct ucc_tl_ucp_worker {
    ucp_context_h     ucp_context;
    ucp_worker_h      ucp_worker;
    size_t            ucp_addrlen;
    ucp_address_t *   worker_address;
    tl_ucp_ep_hash_t *ep_hash;
    ucp_ep_h *        eps;
} ucc_tl_ucp_worker_t;

typedef struct ucc_tl_ucp_context {
    ucc_tl_context_t            super;
    ucc_tl_ucp_context_config_t cfg;
    ucc_tl_ucp_worker_t         worker;
    ucc_tl_ucp_worker_t         service_worker;
    uint32_t                    service_worker_throttling_count;
    ucc_mpool_t                 req_mp;
    ucc_tl_ucp_remote_info_t *  remote_info;
    ucp_rkey_h *                rkeys;
    uint64_t                    n_rinfo_segs;
    uint64_t                    ucp_memory_types;
    int                         topo_required;
} ucc_tl_ucp_context_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_ucp_task ucc_tl_ucp_task_t;
typedef struct ucc_tl_ucp_team {
    ucc_tl_team_t              super;
    ucc_status_t               status;
    uint32_t                   seq_num;
    ucc_tl_ucp_task_t         *preconnect_task;
    void *                     va_base[MAX_NR_SEGMENTS];
    size_t                     base_length[MAX_NR_SEGMENTS];
    ucc_tl_ucp_worker_t *      worker;
    ucc_tl_ucp_team_config_t   cfg;
    const char *               tuning_str;
    ucc_topo_t                *topo;
    ucc_ep_map_t               ctx_map;
    ucc_rank_t                 opt_radix;
} ucc_tl_ucp_team_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

extern ucc_config_field_t ucc_tl_ucp_lib_config_table[];

#define UCC_TL_UCP_SUPPORTED_COLLS                                             \
    (UCC_COLL_TYPE_ALLGATHER |                                                 \
     UCC_COLL_TYPE_ALLGATHERV |                                                \
     UCC_COLL_TYPE_ALLREDUCE |                                                 \
     UCC_COLL_TYPE_ALLTOALL |                                                  \
     UCC_COLL_TYPE_ALLTOALLV |                                                 \
     UCC_COLL_TYPE_BARRIER |                                                   \
     UCC_COLL_TYPE_BCAST |                                                     \
     UCC_COLL_TYPE_FANIN |                                                     \
     UCC_COLL_TYPE_FANOUT |                                                    \
     UCC_COLL_TYPE_GATHER |                                                    \
     UCC_COLL_TYPE_GATHERV |                                                   \
     UCC_COLL_TYPE_REDUCE |                                                    \
     UCC_COLL_TYPE_REDUCE_SCATTER |                                            \
     UCC_COLL_TYPE_REDUCE_SCATTERV |                                           \
     UCC_COLL_TYPE_SCATTERV)

#define UCC_TL_UCP_TEAM_LIB(_team)                                             \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_ucp_lib_t))

#define UCC_TL_UCP_TEAM_CTX(_team)                                             \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_ucp_context_t))

#define IS_SERVICE_TEAM(_team)                                                 \
    ((_team)->super.super.params.scope == UCC_CL_LAST + 1)

#define USE_SERVICE_WORKER(_team)                                              \
    (IS_SERVICE_TEAM(_team) && UCC_TL_UCP_TEAM_CTX(_team)->cfg.service_worker)

#define UCC_TL_UCP_TASK_TEAM(_task)                                            \
    (ucc_derived_of((_task)->super.team, ucc_tl_ucp_team_t))

#define UCC_TL_CTX_HAS_OOB(_ctx)                                               \
    ((_ctx)->super.super.ucc_context->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB)

#define UCC_TL_CTX_OOB(_ctx)                                                   \
    ((_ctx)->super.super.ucc_context->params.oob)

#define UCC_TL_UCP_REMOTE_RKEY(_ctx, _rank, _seg)                              \
    ((_ctx)->rkeys[_rank * _ctx->n_rinfo_segs + _seg])

extern ucs_memory_type_t ucc_memtype_to_ucs[UCC_MEMORY_TYPE_LAST+1];

void ucc_tl_ucp_pre_register_mem(ucc_tl_ucp_team_t *team, void *addr,
                                 size_t length, ucc_memory_type_t mem_type);

ucc_status_t ucc_tl_ucp_ctx_remote_populate(ucc_tl_ucp_context_t *ctx,
                                            ucc_mem_map_params_t  map,
                                            ucc_team_oob_coll_t   oob);
#endif
