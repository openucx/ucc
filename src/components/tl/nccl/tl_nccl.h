/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_NCCL_H_
#define UCC_TL_NCCL_H_

#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_mpool.h"

#include <nccl.h>
#include <cuda.h>

typedef struct ucc_tl_nccl_iface {
    ucc_tl_iface_t super;
} ucc_tl_nccl_iface_t;

extern ucc_tl_nccl_iface_t ucc_tl_nccl;

typedef struct ucc_tl_nccl_lib_config {
    ucc_tl_lib_config_t super;
} ucc_tl_nccl_lib_config_t;

typedef struct ucc_tl_nccl_context_config {
    ucc_tl_context_config_t super;
} ucc_tl_nccl_context_config_t;

typedef struct ucc_tl_nccl_lib {
    ucc_tl_lib_t super;
} ucc_tl_nccl_lib_t;
UCC_CLASS_DECLARE(ucc_tl_nccl_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_nccl_context {
    ucc_tl_context_t             super;
    ucc_tl_nccl_context_config_t cfg;
    ucc_mpool_t                  req_mp;
} ucc_tl_nccl_context_t;
UCC_CLASS_DECLARE(ucc_tl_nccl_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_nccl_team {
    ucc_tl_team_t        super;
    ncclUniqueId        *unique_id;
    void                *oob_req;
    ucc_team_oob_coll_t  oob;
    ncclComm_t           nccl_comm;
    uint32_t             rank;
    uint32_t             size;
    cudaStream_t         stream;
} ucc_tl_nccl_team_t;

typedef struct ucc_tl_nccl_task {
    ucc_coll_task_t     super;
    ucc_tl_nccl_team_t *team;
    ucc_coll_args_t     args;
    cudaEvent_t         completed;
} ucc_tl_nccl_task_t;

UCC_CLASS_DECLARE(ucc_tl_nccl_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define NCCLCHECK_GOTO(_cmd, _label, _status, _lib) do {                       \
  ncclResult_t e = _cmd;                                                       \
  if(ncclSuccess != e) {                                                       \
    tl_error(_lib, "NCCL error %d %s", e, ncclGetErrorString(e));              \
    _status = UCC_ERR_NO_MESSAGE;                                              \
    goto _label;                                                               \
  }                                                                            \
} while(0)

#define CUDACHECK_GOTO(_cmd, _label, _status, _lib) do {                       \
  cudaError_t e = _cmd;                                                        \
  if(cudaSuccess != e) {                                                       \
    tl_error(_lib, "CUDA error %d %s", e, cudaGetErrorName(e));                \
    _status = UCC_ERR_NO_MESSAGE;                                              \
    goto _label;                                                               \
  }                                                                            \
} while(0)

#endif
