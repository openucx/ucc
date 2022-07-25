/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_CONTEXT_H_
#define UCC_CONTEXT_H_

#include "ucc/api/ucc.h"
#include "ucc_progress_queue.h"
#include "utils/ucc_list.h"
#include "utils/ucc_proc_info.h"
#include "components/topo/ucc_topo.h"

typedef struct ucc_lib_info          ucc_lib_info_t;
typedef struct ucc_cl_context        ucc_cl_context_t;
typedef struct ucc_tl_context        ucc_tl_context_t;
typedef struct ucc_cl_context_config ucc_cl_context_config_t;
typedef struct ucc_tl_context_config ucc_tl_context_config_t;
typedef struct ucc_tl_team           ucc_tl_team_t;

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
    ucc_proc_info_t pi;
    uint32_t        seq_num;
} ucc_context_id_t;

#define UCC_CTX_ID_EQUAL(_id1, _id2) (UCC_PROC_INFO_EQUAL((_id1).pi, (_id2).pi) \
                                      && (_id1).seq_num == (_id2).seq_num)
typedef struct ucc_addr_storage {
    void      *storage;
    void      *oob_req;
    size_t     addr_len;
    ucc_rank_t size;
    ucc_rank_t rank;
} ucc_addr_storage_t;

typedef struct ucc_context {
    ucc_lib_info_t          *lib;
    ucc_context_params_t     params;
    ucc_context_attr_t       attr;
    ucc_thread_mode_t        thread_mode;
    ucc_cl_context_t       **cl_ctx;
    ucc_tl_context_t       **tl_ctx;
    ucc_tl_context_t        *service_ctx;
    int                      n_cl_ctx;
    int                      n_tl_ctx;
    int                      n_addr_packed; /*< Number of LT/CL components whose addresses are packed
                                              into ucc_context->attr.addr */
    ucc_config_names_array_t all_tls;
    ucc_list_link_t          progress_list;
    ucc_progress_queue_t    *pq;
    ucc_team_id_pool_t       ids;
    ucc_context_id_t         id;
    ucc_addr_storage_t       addr_storage;
    ucc_rank_t               rank; /*< rank of a process in the "global" (with
                                     OOB) context */
    ucc_context_topo_t      *topo;
    uint64_t                 cl_flags;
    ucc_tl_team_t           *service_team;
} ucc_context_t;

typedef struct ucc_context_config {
    ucc_lib_info_t           *lib;
    ucc_cl_context_config_t **cl_cfgs;
    ucc_tl_context_config_t **tl_cfgs;
    int                       n_cl_cfg;
    int                       n_tl_cfg;
    uint32_t                  team_ids_pool_size;
    uint32_t                  estimated_num_eps;
    uint32_t                  estimated_num_ppn;
    uint32_t                  lock_free_progress_q;
    uint32_t                  internal_oob;
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
/* Performs address exchange between the processes group defined by OOB.
   This function can be used either at context creation time
   (if ctx is global) or at team creation time. The corresponding oob
   arguments must be provided (c_oob for context and t_oob for team).
   The function is non-blocking and can return UCC_INPROGRESS.
   If caller needs a blocking behavior then the function
   must be called until UCC_OK is returned.

   The addresses are stored in the addr_storage data structure.

   The addressing data of rank "i" (according to OOB) can be accessed
   with UCC_ADDR_STORAGE_RANK_HEADER macro defined below.
*/
ucc_status_t ucc_core_addr_exchange(ucc_context_t          *context,
                                    ucc_context_oob_coll_t *c_oob,
                                    ucc_team_oob_coll_t    *t_oob,
                                    ucc_addr_storage_t     *addr_storage);

/* UCC context packed address layout:
   --------------------------------------------------------------------------
   |n_components|id0|offset0|id1|offset1|..|idN|offsetN|data0|data1|..|dataN|
   --------------------------------------------------------------------------
   each component can extract its own addressing using offset into data.
   Offset is found by id. */
typedef struct ucc_context_addr_header {
    ucc_context_id_t ctx_id;
    int n_components; // Number of CL/TL components whose address is packed
    struct {
        unsigned long id;     // id of component computed during framework load
        ptrdiff_t     offset; // offset of the address of the component in the
            // packed data array. Component from the start of the header
    } components[1];
} ucc_context_addr_header_t;

#define UCC_CONTEXT_ADDR_HEADER_SIZE(_n_components)                            \
    ({                                                                         \
        ucc_context_addr_header_t _h;                                          \
        size_t                    _size;                                       \
        _size = sizeof(_h) + sizeof(_h.components[0]) * (_n_components - 1);   \
        _size;                                                                 \
    })

#define UCC_CONTEXT_ADDR_DATA(_header)                                         \
    PTR_OFFSET(_header, UCC_CONTEXT_ADDR_HEADER_SIZE(_header->n_components))

#define UCC_ADDR_STORAGE_RANK_HEADER(_storage, _rank)                          \
    (ucc_context_addr_header_t *)PTR_OFFSET((_storage)->storage,               \
                                            (_storage)->addr_len *(_rank))
#endif
