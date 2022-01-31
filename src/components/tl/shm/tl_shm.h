/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHM_H_
#define UCC_TL_SHM_H_

#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_mpool.h"
#include "utils/ucc_math.h"
#include "tl_shm_knomial_pattern.h"

#include <assert.h> //needed?
#include <errno.h>
#include <sys/shm.h>
#include <sys/types.h>

#ifndef UCC_TL_SHM_DEFAULT_SCORE
#define UCC_TL_SHM_DEFAULT_SCORE 20
#endif

#ifdef HAVE_PROFILING_TL_SHM
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_SHM_PROFILE_FUNC UCC_PROFILE_FUNC
#define UCC_TL_SHM_PROFILE_FUNC_VOID UCC_PROFILE_FUNC_VOID
#define UCC_TL_SHM_PROFILE_REQUEST_NEW UCC_PROFILE_REQUEST_NEW
#define UCC_TL_SHM_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_SHM_PROFILE_REQUEST_FREE UCC_PROFILE_REQUEST_FREE

#define BCOL_SHMSEG_PROBE_COUNT 100
//TODO take arch code from ucs ??
#define SHMSEG_WMB()  __asm__ __volatile__("": : :"memory")
#define SHMSEG_ISYNC() __asm__ __volatile__("": : :"memory")

#define UCC_TL_SHM_SUPPORTED_COLLS                                            \
    (UCC_COLL_TYPE_BCAST | UCC_COLL_TYPE_REDUCE | UCC_COLL_TYPE_BARRIER)

typedef struct ucc_kn_tree ucc_kn_tree_t;

typedef struct ucc_tl_shm_iface {
    ucc_tl_iface_t super;
} ucc_tl_shm_iface_t;

extern ucc_tl_shm_iface_t ucc_tl_shm;

typedef struct ucc_tl_shm_lib_config {
    ucc_tl_lib_config_t  super;
    uint32_t             n_concurrent;
    uint32_t             data_size;
    uint32_t             ctrl_size;
    uint32_t             page_size;
    uint32_t             bcast_alg;
    uint32_t             bcast_base_radix;
    uint32_t             bcast_top_radix;
    uint32_t             reduce_alg;
    uint32_t             reduce_base_radix;
    uint32_t             reduce_top_radix;
    uint32_t             fanin_base_radix;
    uint32_t             fanin_top_radix;
    uint32_t             fanout_base_radix;
    uint32_t             fanout_top_radix;
    uint32_t             barrier_base_radix;
    uint32_t             barrier_top_radix;
    uint32_t             max_trees_cached;
    uint32_t             n_polls;
    uint32_t             base_tree_only;
    uint32_t             set_perf_params;
    char                *group_mode;
} ucc_tl_shm_lib_config_t;

typedef struct ucc_tl_shm_context_config {
    ucc_tl_context_config_t            super;
} ucc_tl_shm_context_config_t;

typedef struct ucc_tl_shm_lib {
    ucc_tl_lib_t            super;
    ucc_tl_shm_lib_config_t cfg;
} ucc_tl_shm_lib_t;
UCC_CLASS_DECLARE(ucc_tl_shm_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_shm_context {
    ucc_tl_context_t            super;
    ucc_tl_shm_context_config_t cfg;
    ucc_mpool_t                  req_mp;
} ucc_tl_shm_context_t;
UCC_CLASS_DECLARE(ucc_tl_shm_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_shm_ctrl {
    volatile uint32_t pi;      /* producer index */
    volatile uint32_t ci;      /* consumer index */
    char              data[1]; /* start of inline data */
} ucc_tl_shm_ctrl_t;

typedef struct ucc_tl_shm_seg {
    volatile void *ctrl; /* control array = start of seg */
    volatile void *data; /* start of the data segments */
} ucc_tl_shm_seg_t;

typedef struct ucc_tl_shm_tree {
    ucc_kn_tree_t *base_tree; /* tree for base group, always != NULL */
    ucc_kn_tree_t *top_tree;  /* tree for leaders group, can be NULL if the group
                                 does not exists or the process is not part of it */
} ucc_tl_shm_tree_t;

typedef struct ucc_tl_shm_tree_cache_keys {
    ucc_rank_t      base_radix;
    ucc_rank_t      top_radix;
    ucc_rank_t      root;
    ucc_coll_type_t coll_type;
    int             base_tree_only;
} ucc_tl_shm_tree_cache_keys_t;

typedef struct ucc_tl_shm_tree_cache_elems {
	ucc_tl_shm_tree_cache_keys_t  keys;
	ucc_tl_shm_tree_t            *tree;
} ucc_tl_shm_tree_cache_elems_t;

typedef struct ucc_tl_shm_tree_cache {
	size_t                         size;
	ucc_tl_shm_tree_cache_elems_t *elems;
} ucc_tl_shm_tree_cache_t;

typedef struct ucc_tl_shm_team {
    ucc_tl_team_t            super;
    void                    *oob_req;
    ucc_tl_shm_seg_t        *segs;
    uint32_t                 seq_num;
    uint32_t                 n_base_groups;
    uint32_t                 my_group_id;
    int                     *allgather_dst;
    int                      n_concurrent;
    ucc_sbgp_t              *base_groups;
    ucc_sbgp_t              *leaders_group;
    ucc_topo_t              *topo;
    void                    *shm_buffer;
    ucc_ep_map_t             group_rank_map;
    ucc_ep_map_t             rank_group_id_map;
    ucc_ep_map_t             ctrl_map;
    size_t                   ctrl_size;
    size_t                   data_size;
    size_t                   max_inline;
    ucc_tl_shm_tree_cache_t *tree_cache;
    ucc_status_t             status;
} ucc_tl_shm_team_t;

//typedef enum {
//    UCC_TL_SHM_BASE_GROUP,
//    UCC_TL_SHM_LEADERS_GROUP,
//} ucc_tl_shm_group_t; // needed?


UCC_CLASS_DECLARE(ucc_tl_shm_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define TASK_TEAM(_task)                                                      \
    (ucc_derived_of((_task)->super.team, ucc_tl_shm_team_t))
#define TASK_CTX(_task)                                                       \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_shm_context_t))
#define TASK_LIB(_task)                                                       \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_shm_lib_t))
#define UCC_TL_SHM_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_shm_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

#endif
