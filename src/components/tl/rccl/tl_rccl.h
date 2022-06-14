/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * Copyright (c) Facebook, Inc. and its affiliates. 2021.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_RCCL_H_
#define UCC_TL_RCCL_H_

#include "components/mc/base/ucc_mc_base.h"
#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_mpool.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifdef RCCL_OLD_HEADERS
#include <rccl.h>
#else
#include <rccl/rccl.h>
#endif

#ifndef UCC_TL_RCCL_DEFAULT_SCORE
#define UCC_TL_RCCL_DEFAULT_SCORE 20
#endif

#ifdef HAVE_PROFILING_TL_RCCL
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_RCCL_PROFILE_FUNC UCC_PROFILE_FUNC
#define UCC_TL_RCCL_PROFILE_FUNC_VOID UCC_PROFILE_FUNC_VOID
#define UCC_TL_RCCL_PROFILE_REQUEST_NEW UCC_PROFILE_REQUEST_NEW
#define UCC_TL_RCCL_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_RCCL_PROFILE_REQUEST_FREE UCC_PROFILE_REQUEST_FREE

typedef struct ucc_tl_rccl_iface {
    ucc_tl_iface_t super;
} ucc_tl_rccl_iface_t;

extern ucc_tl_rccl_iface_t ucc_tl_rccl;

typedef struct ucc_tl_rccl_lib_config {
    ucc_tl_lib_config_t super;
} ucc_tl_rccl_lib_config_t;

typedef enum ucc_tl_rccl_completion_sync_type {
    UCC_TL_RCCL_COMPLETION_SYNC_TYPE_EVENT,
    UCC_TL_RCCL_COMPLETION_SYNC_TYPE_MEMOPS,
    UCC_TL_RCCL_COMPLETION_SYNC_TYPE_AUTO,
    UCC_TL_RCCL_COMPLETION_SYNC_TYPE_LAST
} ucc_tl_rccl_completion_sync_type_t;

typedef struct ucc_tl_rccl_context_config {
    ucc_tl_context_config_t            super;
    ucc_tl_rccl_completion_sync_type_t sync_type;
} ucc_tl_rccl_context_config_t;

typedef struct ucc_tl_rccl_lib {
    ucc_tl_lib_t super;
} ucc_tl_rccl_lib_t;
UCC_CLASS_DECLARE(ucc_tl_rccl_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_rccl_context {
    ucc_tl_context_t             super;
    ucc_tl_rccl_context_config_t cfg;
    ucc_mpool_t                  req_mp;
    void                        *scratch_buf;
} ucc_tl_rccl_context_t;
UCC_CLASS_DECLARE(ucc_tl_rccl_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_rccl_team {
    ucc_tl_team_t        super;
    ncclUniqueId        *unique_id;
    void                *oob_req;
    ncclComm_t           rccl_comm;
    hipStream_t          stream;
} ucc_tl_rccl_team_t;

typedef struct ucc_tl_rccl_task {
    ucc_coll_task_t         super;
    ucc_status_t            host_status;
    ucc_status_t           *dev_status;
    void                   *completed;
    union {
        struct {
            ucc_mc_buffer_header_t *scratch;
            size_t                  max_count;
        } allgatherv_bcopy;
    };
} ucc_tl_rccl_task_t;

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_rccl_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_rccl_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_rccl_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

#define UCC_TL_RCCL_SUPPORTED_COLLS                                            \
    (UCC_COLL_TYPE_ALLTOALL       | UCC_COLL_TYPE_ALLTOALLV  |                 \
     UCC_COLL_TYPE_ALLGATHER      | UCC_COLL_TYPE_ALLGATHERV |                 \
     UCC_COLL_TYPE_ALLREDUCE      | UCC_COLL_TYPE_BCAST      |                 \
     UCC_COLL_TYPE_REDUCE_SCATTER | UCC_COLL_TYPE_REDUCE     |                 \
     UCC_COLL_TYPE_BARRIER)

UCC_CLASS_DECLARE(ucc_tl_rccl_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define RCCLCHECK_GOTO(_cmd, _label, _status, _lib)                            \
    do {                                                                       \
        ncclResult_t e = _cmd;                                                 \
        if (ncclSuccess != e) {                                                \
            tl_error(_lib, "RCCL error %d %s", e, ncclGetErrorString(e));      \
            _status = UCC_ERR_NO_MESSAGE;                                      \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#define HIPCHECK_GOTO(_cmd, _label, _status, _lib)                             \
    do {                                                                       \
        hipError_t e = _cmd;                                                   \
        if (hipSuccess != e) {                                                 \
            tl_error(_lib, "HIP error %d %s", e, hipGetErrorName(e));          \
            _status = UCC_ERR_NO_MESSAGE;                                      \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#define UCC_TL_RCCL_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_rccl_lib_t))

#endif
