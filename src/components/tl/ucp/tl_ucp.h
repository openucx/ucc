/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
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
#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>

#ifndef UCC_TL_UCP_DEFAULT_SCORE
#define UCC_TL_UCP_DEFAULT_SCORE 10
#endif

#ifdef HAVE_PROFILING_TL_UCP
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_UCP_PROFILE_FUNC UCC_PROFILE_FUNC
#define UCC_TL_UCP_PROFILE_REQUEST_NEW UCC_PROFILE_REQUEST_NEW
#define UCC_TL_UCP_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_UCP_PROFILE_REQUEST_FREE UCC_PROFILE_REQUEST_FREE

typedef struct ucc_tl_ucp_iface {
    ucc_tl_iface_t super;
} ucc_tl_ucp_iface_t;
/* Extern iface should follow the pattern: ucc_tl_<tl_name> */
extern ucc_tl_ucp_iface_t ucc_tl_ucp;

typedef struct ucc_tl_ucp_lib_config {
    ucc_tl_lib_config_t super;
    uint32_t            kn_radix;
    uint32_t            barrier_kn_radix;
    uint32_t            allreduce_kn_radix;
    uint32_t            allreduce_sra_kn_radix;
    uint32_t            reduce_scatter_kn_radix;
    uint32_t            allgather_kn_radix;
    uint32_t            bcast_kn_radix;
    uint32_t            reduce_kn_radix;
    uint32_t            alltoall_pairwise_num_posts;
    uint32_t            alltoallv_pairwise_num_posts;
    uint32_t            allreduce_sra_kn_n_frags;
    uint32_t            allreduce_sra_kn_pipeline_depth;
    int                 allreduce_sra_kn_seq;
    size_t              allreduce_sra_kn_frag_thresh;
    size_t              allreduce_sra_kn_frag_size;
} ucc_tl_ucp_lib_config_t;

typedef struct ucc_tl_ucp_context_config {
    ucc_tl_context_config_t super;
    uint32_t                preconnect;
    uint32_t                n_polls;
    uint32_t                oob_npolls;
    uint32_t                pre_reg_mem;
} ucc_tl_ucp_context_config_t;

typedef struct ucc_tl_ucp_lib {
    ucc_tl_lib_t            super;
    ucc_tl_ucp_lib_config_t cfg;
} ucc_tl_ucp_lib_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_ucp_context {
    ucc_tl_context_t            super;
    ucc_tl_ucp_context_config_t cfg;
    ucp_context_h               ucp_context;
    ucp_worker_h                ucp_worker;
    size_t                      ucp_addrlen;
    ucp_address_t              *worker_address;
    ucc_mpool_t                 req_mp;
    tl_ucp_ep_hash_t           *ep_hash;
    ucp_ep_h                   *eps;
} ucc_tl_ucp_context_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_ucp_task ucc_tl_ucp_task_t;
typedef struct ucc_tl_ucp_team {
    ucc_tl_team_t              super;
    ucc_status_t               status;
    ucc_rank_t                 size;
    ucc_rank_t                 rank;
    uint32_t                   id;
    uint32_t                   scope;
    uint32_t                   scope_id;
    uint32_t                   seq_num;
    ucc_tl_ucp_task_t         *preconnect_task;
} ucc_tl_ucp_team_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define UCC_TL_UCP_SUPPORTED_COLLS                         \
    (UCC_COLL_TYPE_ALLTOALL  | UCC_COLL_TYPE_ALLTOALLV  |  \
     UCC_COLL_TYPE_ALLGATHER | UCC_COLL_TYPE_ALLGATHERV |  \
     UCC_COLL_TYPE_ALLREDUCE | UCC_COLL_TYPE_BCAST      |  \
     UCC_COLL_TYPE_BARRIER   | UCC_COLL_TYPE_REDUCE)

#define UCC_TL_UCP_TEAM_LIB(_team)                                             \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_ucp_lib_t))

#define UCC_TL_UCP_TEAM_CTX(_team)                                             \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_ucp_context_t))

#define UCC_TL_UCP_WORKER(_team) UCC_TL_UCP_TEAM_CTX(_team)->ucp_worker

#define UCC_TL_CTX_HAS_OOB(_ctx) ((_ctx)->super.super.ucc_context->params.mask & \
                                  UCC_CONTEXT_PARAM_FIELD_OOB)

#define UCC_TL_CTX_OOB(_ctx) ((_ctx)->super.super.ucc_context->params.oob)

#define IS_SERVICE_TEAM(_team) ((_team)->scope == UCC_CL_LAST + 1)

// TODO remove once AVG is implemented
#define CHECK_AVG_OP(_args, _team)                                             \
    do {                                                                       \
        if (_args.reduce.predefined_op == UCC_OP_AVG) {                        \
            tl_error(UCC_TL_TEAM_LIB(_team),                                   \
                     "Average reduction is not supported yet");                \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while (0)

void ucc_tl_ucp_pre_register_mem(ucc_tl_ucp_team_t *team, void *addr,
                                 size_t length, ucc_memory_type_t mem_type);
#endif
