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

#define SHMEM_128b 128
#define SHMEM_2K   4096
/* Barrier uses offset 2 for progress */
#define SHMEM_AT_ITERATION 2

/* Allreduce up to 64b, uses 2-9 for data and 10 for progress aka. level */
#define SHMEM_DATA 2
#define SHMEM_AT_LEVEL 10

#define SHMSEG_128B_RADIX 4
#define SHMSEG_2K_RADIX 2

#define SHMEM_STATE(_var, _rank, _rw) (((_var) + (_rank))->state[(_rw)])
#define aSHMEM_STATE(_var, _rank, _rw) (void*)(&(((_var) + (_rank))->state[(_rw)]))

#define SHMEM_LEAF 0
#define SHMEM_ROOT 1

/* Barrier uses offset 2 for progress */
#define SHMEM_AT_ITERATION 2
#define SHMSEG_BARRIER_RADIX 8

/* Allreduce for 33-2K, needs root/leaf for signaling,
 * radix (iteration)
 * offset into 2K seg
 * and its on_node rank
 */
#define SHMEM_2K_AT_RADIX_I  3
#define SHMEM_2K_OFFSET      4
//#define SHMEM_2K_ONNODE_RANK 5 // not used in xccl, needed?

/* Shared memory segment sizes */
#define SHMEM_4K 4096
#define SHMEM_8K 8192
/* Convenience macro's for setup */
#define ROOT_AT_LEVEL(_rank, _radix, _level) \
    !((_rank) & ((int) pow((_radix), (_level)+1)-1))

#define NEXT_PARTNER(radix, level) \
    (pow(radix, level))

/* Used in allreduce */
#define IS_ROOT_AT_LEVEL(_rank, _level) !(((_rank) >> (_level)) & 1)
#define R_PARTNER_AT_LEVEL(rank, level) ((_rank) + (1 << (_level)))

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
    ucc_tl_lib_config_t super;
} ucc_tl_shm_lib_config_t;

typedef struct ucc_tl_shm_context_config {
    ucc_tl_context_config_t            super;
} ucc_tl_shm_context_config_t;

typedef struct ucc_tl_shm_lib {
    ucc_tl_lib_t super;
} ucc_tl_shm_lib_t;
UCC_CLASS_DECLARE(ucc_tl_shm_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_shm_context {
    ucc_tl_context_t             super;
    ucc_tl_shm_context_config_t  cfg;
//    ucc_mpool_t                  req_mp;
} ucc_tl_shm_context_t;
UCC_CLASS_DECLARE(ucc_tl_shm_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct shmem_sync {
    volatile int64_t state[16];
} shmem_sync_t;

typedef struct _ar2k_data_shmseg_t {
    void* _base[2];
} ar2k_data_shmseg_t;

typedef struct {
    int isRoot;
    int my_offset;
    int partners_at_level;
    int partner_offset;
} barrier_radix_info_t;

typedef struct ucc_shm_seg_data {
    int  ar2k_sync_shmid; // shmid for sync to connect at uma level
    int  ar2k_data_shmid; // shmid for specific socket (and uma) shm seg
    void *ar128b_shmseg[2];
    void *ar2k_sync_shmseg; // array of shm pointers (shm seg per socket) for uma level
    void *ar2k_data_shmseg[2]; // uma shmseg
    void *ar2k_data_shmseg_mine[2]; // socket shmseg
    int  seq_num; // why in both team and shm_seg_data?
} ucc_shm_seg_data_t;

typedef struct ucc_shm_seg {
    /* shmseg ar8 */
    int ar64_logx_group_size;
    int *ar64_radix_array;
    int *ar64_bcol_to_node_group_list;
    int ar64_radix_array_length;
    int my_ar64_node_root_rank;

    /* shmseg ar2k */
    shmem_sync_t*       ar2k_sync_shmseg; /* used for signaling */
    shmem_sync_t**      ar2k_sync_sockets_shmseg;
    shmem_sync_t*       barrier_shmseg;
    ar2k_data_shmseg_t* ar2k_data_sockets_shmseg;

    int  ar2k_logx_group_size;
    int* ar2k_radix_array;
    int  ar2k_radix_array_length;
    int  barrier_logx_group_size;
    barrier_radix_info_t* barrier_radix_info;
    int  my_ar2k_root_rank;
    int  on_node_rank;
} ucc_shm_seg_t;

typedef struct ucc_tl_shm_team {
    ucc_tl_team_t        super;
    void                *oob_req;
    ucc_team_oob_coll_t  oob;
    ucc_rank_t           rank;
    ucc_rank_t           size;
    ucc_shm_seg_t        shm_seg;
    ucc_shm_seg_data_t  *shm_seg_data;
    shmem_sync_t        *seg;
    uint16_t             seq_num;
    int                 allgather_src[2];
    int                 *allgather_dst;
    ucc_status_t         status;
} ucc_tl_shm_team_t;

UCC_CLASS_DECLARE(ucc_tl_shm_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#endif
