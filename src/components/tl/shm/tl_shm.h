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

#include <assert.h> //needed?
#include <errno.h>
#include <sys/shm.h>
#include <sys/types.h>

#ifndef UCC_TL_SHM_DEFAULT_SCORE
#define UCC_TL_SHM_DEFAULT_SCORE 20
#endif
//
//#define SHMEM_128b 128
//#define SHMEM_2K   4096
///* Barrier uses offset 2 for progress */
//#define SHMEM_AT_ITERATION 2
//
///* Allreduce up to 64b, uses 2-9 for data and 10 for progress aka. level */
//#define SHMEM_DATA 2
//#define SHMEM_AT_LEVEL 10
//
//#define SHMSEG_128B_RADIX 4
//#define SHMSEG_2K_RADIX 2
//
//#define SHMEM_STATE(_var, _rank, _rw) (((_var) + (_rank))->state[(_rw)])
//#define aSHMEM_STATE(_var, _rank, _rw) (void*)(&(((_var) + (_rank))->state[(_rw)]))
//
//#define SHMEM_LEAF 0
//#define SHMEM_ROOT 1
//
///* Barrier uses offset 2 for progress */
//#define SHMEM_AT_ITERATION 2
//#define SHMSEG_BARRIER_RADIX 8
//
///* Allreduce for 33-2K, needs root/leaf for signaling,
// * radix (iteration)
// * offset into 2K seg
// * and its on_node rank
// */
//#define SHMEM_2K_AT_RADIX_I  3
//#define SHMEM_2K_OFFSET      4
////#define SHMEM_2K_ONNODE_RANK 5 // not used in xccl, needed?
//
///* Shared memory segment sizes */
//#define SHMEM_4K 4096
//#define SHMEM_8K 8192
///* Convenience macro's for setup */
//#define ROOT_AT_LEVEL(_rank, _radix, _level) \
//    !((_rank) & ((int) pow((_radix), (_level)+1)-1))
//
//#define NEXT_PARTNER(radix, level) \
//    (pow(radix, level))
//
///* Used in allreduce */
//#define IS_ROOT_AT_LEVEL(_rank, _level) !(((_rank) >> (_level)) & 1)
//#define R_PARTNER_AT_LEVEL(rank, level) ((_rank) + (1 << (_level)))
//
#define BCOL_SHMSEG_PROBE_COUNT 100
//TODO take arch code from ucs ??
#define SHMSEG_WMB()  __asm__ __volatile__("": : :"memory") //why do we need bpth wmb and isync?
#define SHMSEG_ISYNC() __asm__ __volatile__("": : :"memory")

#define UCC_TL_SHM_SUPPORTED_COLLS                                             \
    (UCC_COLL_TYPE_BCAST | UCC_COLL_TYPE_REDUCE)

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
    uint32_t             bcast_kn_radix;
    uint32_t             max_trees_cached;
    uint32_t             n_polls;
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
//    ucc_mpool_t                  req_mp;
} ucc_tl_shm_context_t;
UCC_CLASS_DECLARE(ucc_tl_shm_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_shm_ctrl {
    volatile int32_t pi;      /* producer index */
    volatile int32_t ci;      /* consumer index */
    char             data[1]; /* start of inline data */
} ucc_tl_shm_ctrl_t;

typedef struct ucc_tl_shm_seg {
    volatile void *ctrl; /* control array = start of seg */
    volatile void *data; /* start of the data segments */
} ucc_tl_shm_seg_t;

typedef struct ucc_tl_shm_tree {ucc_kn_tree_t
    ucc_kn_tree_t *base_tree; /* tree for base group, always != NULL */
    ucc_kn_tree_t *top_tree;  /* tree for leaders group, can be NULL if the group
                                 does not exists or the process is not part of it */
} ucc_tl_shm_tree_t

typedef struct ucc_tl_shm_tree_cache_keys {
    ucc_rank_t radix;
    ucc_rank_t root;
    ucc_rank_t team_size;
//    ucc_coll_type_t coll_type;
} ucc_tl_shm_tree_cache_keys_t;

typedef struct ucc_tl_shm_tree_cache {
	ucc_tl_shm_tree_cache_keys_t **keys;
	ucc_tl_shm_tree_t            **trees;
	size_t                         size;
} ucc_tl_shm_tree_cache_t;

typedef struct ucc_tl_shm_team {
    ucc_tl_team_t           super;
    void                   *oob_req;
    ucc_tl_shm_seg_t       *segs;
    uint32_t                seq_num;
    uint32_t                n_base_groups;
    uint32_t                my_group_id;
    int                    *allgather_dst;
    int                     n_concurrent;
    ucc_sbgp_t             *base_groups;
    ucc_sbgp_t             *leaders_group;
    ucc_topo_t             *topo; //?
    void                   *shm_buffer;
    ucc_ep_map_t            group_rank_map;
    ucc_ep_map_t            rank_group_id_map;
    ucc_ep_map_t            ctrl_map;
    size_t                  ctrl_size;
    size_t                  data_size;
    ucc_tl_shm_tree_cache_t tree_cache;
    ucc_status_t            status;
} ucc_tl_shm_team_t;

typedef enum {
    UCC_TL_SHM_BASE_GROUP,
    UCC_TL_SHM_LEADERS_GROUP,
} ucc_tl_shm_group_t; // needed?


UCC_CLASS_DECLARE(ucc_tl_shm_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_shm_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_shm_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_shm_lib_t))
#define UCC_TL_SHM_TEAM_LIB(_team)                                             \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_shm_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

#endif
