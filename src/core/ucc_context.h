/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_CONTEXT_H_
#define UCC_CONTEXT_H_

#include "ucc/api/ucc.h"
#include "ucc_progress_queue.h"
#include "utils/ucc_list.h"
#include "utils/ucc_proc_info.h"

typedef struct ucc_lib_info          ucc_lib_info_t;
typedef struct ucc_cl_context        ucc_cl_context_t;
typedef struct ucc_tl_context        ucc_tl_context_t;
typedef struct ucc_cl_context_config ucc_cl_context_config_t;

typedef unsigned (*ucc_context_progress_fn_t)(void *progress_arg);
typedef struct ucc_context_progress {
    ucc_context_progress_fn_t progress_fn;
    void                     *progress_arg;
} ucc_context_progress_t;

typedef struct ucc_team_id_pool {
    uint64_t *pool;
    uint32_t  pool_size;
} ucc_team_id_pool_t;

typedef struct ucc_context_id {
    ucc_host_id_t host_id;
    pid_t         pid;
    uint32_t      seq_num;
} ucc_context_id_t;

typedef struct ucc_context {
    ucc_lib_info_t         *lib;
    ucc_context_params_t    params;
    ucc_context_attr_t      attr;
    ucc_thread_mode_t       thread_mode;
    ucc_cl_context_t      **cl_ctx;
    ucc_tl_context_t      **tl_ctx;
    ucc_tl_context_t       *service_ctx;
    int                     n_cl_ctx;
    int                     n_tl_ctx;
    ucc_list_link_t         progress_list;
    ucc_progress_queue_t   *pq;
    ucc_team_id_pool_t      ids;
    ucc_context_id_t        id;
} ucc_context_t;

typedef struct ucc_context_config {
    ucc_lib_info_t           *lib;
    ucc_cl_context_config_t **configs;
    int                       n_cl_cfg;
    uint32_t                  team_ids_pool_size;
    uint32_t                  estimated_num_eps;
    uint32_t                  estimated_num_ppn;
} ucc_context_config_t;

/* Any internal UCC component (TL, CL, etc) may register its own
   progress callback fn (and argument for the callback) into core
   ucc context. Those callbacks will be triggered as part of
   ucc_context_progress.
   Any progress callback fn inserted is required to be thread safe.
   If not, we need to add to this engine a thread safe mechanism. */

ucc_status_t ucc_context_progress_register(ucc_context_t *ctx,
                                           ucc_context_progress_fn_t fn,
                                           void *progress_arg);
void         ucc_context_progress_deregister(ucc_context_t *ctx,
                                             ucc_context_progress_fn_t fn,
                                             void *progress_arg);
#endif
