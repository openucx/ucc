/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_H_
#define UCC_TL_CUDA_H_

#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_mpool.h"
#include "utils/ucc_datastruct.h"
#include "tl_cuda_ep_hash.h"
#include "tl_cuda_topo.h"
#include "tl_cuda_team_topo.h"
#include <cuda_runtime.h>

#ifndef UCC_TL_CUDA_DEFAULT_SCORE
#define UCC_TL_CUDA_DEFAULT_SCORE 40
#endif

#define UCC_TL_CUDA_MAX_PEERS 8
#define UCC_TL_CUDA_MAX_RING_CHUNKS 8

#define UCC_TL_CUDA_SUPPORTED_COLLS                                            \
    (UCC_COLL_TYPE_ALLTOALL | UCC_COLL_TYPE_ALLTOALLV |                        \
     UCC_COLL_TYPE_ALLGATHER | UCC_COLL_TYPE_ALLGATHERV |                      \
     UCC_COLL_TYPE_BCAST |                                                     \
     UCC_COLL_TYPE_REDUCE_SCATTER | UCC_COLL_TYPE_REDUCE_SCATTERV)

#define UCC_TL_CUDA_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_cuda_lib_t))

#define UCC_TL_CUDA_TEAM_CTX(_team)                                            \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_cuda_context_t))

#define UCC_TL_CUDA_TEAM_SYNC(_team, _rank, _id)                               \
    ({                                                                         \
        size_t _ctrl_size_rank =                                               \
            (sizeof(ucc_tl_cuda_sync_t) +                                      \
             sizeof(ucc_tl_cuda_sync_data_t) * (UCC_TL_TEAM_SIZE(_team) - 1)); \
        size_t _ctrl_size = _ctrl_size_rank * UCC_TL_TEAM_SIZE(_team);         \
        void  *_sync      = PTR_OFFSET(_team->sync, _ctrl_size * (_id) +       \
                                       _ctrl_size_rank * (_rank));             \
        (ucc_tl_cuda_sync_t *)_sync;                                           \
    })

#define UCC_TL_CUDA_TEAM_BARRIER(_team, _id)                                   \
    ({                                                                         \
        size_t _bar_size = sizeof(ucc_tl_cuda_shm_barrier_t);                  \
        void  *_bar      = PTR_OFFSET(_team->bar, _bar_size * (_id));          \
        (ucc_tl_cuda_shm_barrier_t *)_bar;                                     \
    })

#ifdef HAVE_PROFILING_TL_CUDA
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_CUDA_PROFILE_FUNC          UCC_PROFILE_FUNC
#define UCC_TL_CUDA_PROFILE_REQUEST_NEW   UCC_PROFILE_REQUEST_NEW
#define UCC_TL_CUDA_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_CUDA_PROFILE_REQUEST_FREE  UCC_PROFILE_REQUEST_FREE

typedef struct ucc_tl_cuda_iface {
    ucc_tl_iface_t super;
} ucc_tl_cuda_iface_t;

extern ucc_tl_cuda_iface_t ucc_tl_cuda;

typedef struct ucc_tl_cuda_lib_config {
    ucc_tl_lib_config_t super;
    uint32_t            max_concurrent; // Maximum number of tasks that can be progressed simultaneously.
    size_t              scratch_size;
    unsigned long       allgather_ring_max_rings;
    uint32_t            allgather_ring_num_chunks;
    unsigned long       reduce_scatter_ring_max_rings;
} ucc_tl_cuda_lib_config_t;

typedef struct ucc_tl_cuda_context_config {
    ucc_tl_context_config_t super;
} ucc_tl_cuda_context_config_t;

typedef struct ucc_tl_cuda_lib {
    ucc_tl_lib_t             super;
    ucc_tl_cuda_lib_config_t cfg;
} ucc_tl_cuda_lib_t;
UCC_CLASS_DECLARE(ucc_tl_cuda_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_cuda_context {
    ucc_tl_context_t             super;
    ucc_tl_cuda_context_config_t cfg;
    int                          device;
    ucc_tl_cuda_device_pci_id_t  device_id;
    ucc_tl_cuda_topo_t          *topo;
    ucc_mpool_t                  req_mp;
    tl_cuda_ep_hash_t           *ipc_cache;
} ucc_tl_cuda_context_t;
UCC_CLASS_DECLARE(ucc_tl_cuda_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef uint32_t ucc_tl_cuda_sync_state_t;

#define UCC_TL_CUDA_TAG_FREE 0xFFFFFFFFFFFFFFFF

typedef struct ucc_tl_cuda_shm_barrier {
    ucc_rank_t   size;
    ucc_rank_t   count;
    uint64_t     tag;
    int          sense;
    ucc_status_t state[UCC_TL_CUDA_MAX_PEERS];
    int          local_sense[UCC_TL_CUDA_MAX_PEERS];
} ucc_tl_cuda_shm_barrier_t;

typedef struct ucc_tl_cuda_sync_data {
    cudaEvent_t ipc_event_remote;
} ucc_tl_cuda_sync_data_t;

typedef struct ucc_tl_cuda_mem_info {
    void              *ptr;
    size_t             length;
    size_t             offset;
    cudaIpcMemHandle_t handle;
} ucc_tl_cuda_mem_info_t;

typedef struct ucc_tl_cuda_rank_id {
    ucc_tl_cuda_device_pci_id_t pci_id;
    ucc_tl_cuda_mem_info_t      scratch_info;
    int                         shm;
} ucc_tl_cuda_rank_id_t;

typedef struct ucc_tl_cuda_sync {
    int                    seq_num[UCC_TL_CUDA_MAX_RING_CHUNKS];
    ucc_tl_cuda_mem_info_t mem_info_src;
    ucc_tl_cuda_mem_info_t mem_info_dst;
    cudaEvent_t            ipc_event_local;
    cudaIpcEventHandle_t   ev_handle;
    union {
        struct {
            size_t sbytes[UCC_TL_CUDA_MAX_PEERS];
            size_t rbytes[UCC_TL_CUDA_MAX_PEERS];
            size_t sdispl_bytes[UCC_TL_CUDA_MAX_PEERS];
            size_t rdispl_bytes[UCC_TL_CUDA_MAX_PEERS];
        } alltoallv_ce;
    };
    ucc_tl_cuda_sync_data_t data[1];
} ucc_tl_cuda_sync_t;

typedef struct ucc_tl_cuda_scratch {
    void                  *loc;
    void                  *rem[UCC_TL_CUDA_MAX_PEERS];
    ucc_tl_cuda_mem_info_t rem_info[UCC_TL_CUDA_MAX_PEERS];
} ucc_tl_cuda_scratch_t;

// Team represents a communicator created within the CUDA context, typically using NVLink for inter-GPU communication
typedef struct ucc_tl_cuda_team {
    ucc_tl_team_t              super;
    uint32_t                   seq_num;            // Counter for the number of launched collective tasks for this team
    uint32_t                   seq_num_active_set; // Counter for tasks in the active set (subset of tasks requiring special handling)
    ucc_tl_cuda_team_topo_t   *topo;
    ucc_tl_cuda_sync_t        *sync;               // Pointer to shared memory segment for synchronization
    ucc_tl_cuda_sync_state_t  *sync_state;         // Tracks the task currently using the sync segment of shared memory, if free - 0
    ucc_tl_cuda_shm_barrier_t *bar;                // Pointer to the first barrier in an array of size [0; 2 * max_concurrent]. First max_concurrent barriers are for normal mode, the second one for active set mode
    ucc_tl_cuda_scratch_t      scratch;
    cudaStream_t               stream;
    ucc_tl_cuda_rank_id_t     *ids;
    ucc_team_oob_coll_t        oob;
    void                      *oob_req;
} ucc_tl_cuda_team_t;

UCC_CLASS_DECLARE(ucc_tl_cuda_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

// Task represents a collective operation that runs in the CUDA context, typically using NVLink for inter-GPU communication
typedef struct ucc_tl_cuda_task ucc_tl_cuda_task_t;
struct ucc_tl_cuda_task {
    ucc_coll_task_t            super;
    uint32_t                   seq_num; // Sequential identifier for each task started within the team
    uint32_t                   coll_id; // Index of the collective task in flight, within the range [0; max_concurrent)
    ucc_tl_cuda_shm_barrier_t *bar;     // Pointer to the reserved barrier for this task in the CUDA team
    ucc_subset_t               subset;  // Mapping information for the active set, if it is present
    union {
        struct {
            int                    stage;
            ucc_tl_cuda_mem_info_t mem_info_src;
            ucc_tl_cuda_mem_info_t mem_info_dst;
            void                  *peer_map_addr_src[UCC_TL_CUDA_MAX_PEERS];
            void                  *peer_map_addr_dst[UCC_TL_CUDA_MAX_PEERS];
            int                    num_posted;
            ucc_datatype_t         sdt;
            ucc_datatype_t         rdt;
            void                  *sbuf;
            void                  *rbuf;
            ucc_count_t           *scnts;
            ucc_count_t           *rcnts;
            ucc_aint_t            *sdispl;
            ucc_aint_t            *rdispl;
            ucc_ee_executor_task_t
                 *exec_task[UCC_TL_CUDA_MAX_PEERS * UCC_TL_CUDA_MAX_PEERS];
            size_t (*get_size)(const ucc_tl_cuda_task_t *task, size_t *bytes,
                               ucc_rank_t block);
            size_t (*get_offset)(const ucc_tl_cuda_task_t *task,
                                 size_t *displ_bytes, ucc_rank_t block);
        } alltoallv_ce;
        struct {
            int                     stage;
            int                     num_frags;
            int                     num_rings;
            int                     num_chunks;
            ucc_datatype_t          dt;
            void                   *sbuf;
            void                   *rbuf;
            ucc_ee_executor_task_t *exec_task[2 * UCC_TL_CUDA_MAX_RING_CHUNKS];
            size_t (*get_count)(const ucc_tl_cuda_task_t *task,
                                ucc_rank_t                block);
            size_t (*get_offset)(const ucc_tl_cuda_task_t *task,
                                 ucc_rank_t                block);
        } allgatherv_ring;
        struct {
            int                     stage;
            int                     num_frags;
            ucc_datatype_t          dt;
            void *                  sbuf;
            void *                  rbuf;
            ucc_ee_executor_task_t *exec_task[2];
            size_t (*get_count)(const ucc_tl_cuda_task_t *task,
                                ucc_rank_t                block);
            size_t (*get_offset)(const ucc_tl_cuda_task_t *task,
                                 ucc_rank_t                block);
        } allgatherv_linear;
        struct {
            int                     stage;
            int                     step;
            void                   *sbuf;
            ucc_datatype_t          dt;
            ucc_rank_t              root;
            size_t                  size;
            int                     num_steps;
            ucc_ee_executor_task_t *exec_task;
            uint64_t                key; // This is mix of user provided tag, root and peer to be unique for each task, algorithm uses it to mark barrier as used
        } bcast_linear;
        struct {
            int                     stage;
            int                     num_frags;
            int                     num_rings;
            ucc_datatype_t          dt;
            void                   *sbuf;
            void                   *rbuf;
            ucc_ee_executor_task_t *exec_task[UCC_TL_CUDA_MAX_RING_CHUNKS];
            size_t (*get_count)(const ucc_tl_cuda_task_t *task,
                                ucc_rank_t                block);
            size_t (*get_offset)(const ucc_tl_cuda_task_t *task,
                                 ucc_rank_t                block);
        } reduce_scatterv_ring;
        struct {
            int                     stage;
            int                     num_frags;
            ucc_datatype_t          dt;
            void *                  sbuf;
            void *                  rbuf;
            ucc_ee_executor_task_t *exec_task[2];
            size_t (*get_count)(const ucc_tl_cuda_task_t *task,
                                ucc_rank_t                block);
            size_t (*get_offset)(const ucc_tl_cuda_task_t *task,
                                 ucc_rank_t                block);
        } reduce_scatterv_linear;
    };
};

#endif
