/**
 * Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_PROC_INFO_H_
#define UCC_PROC_INFO_H_

#include "config.h"
#include <unistd.h>
#include <ucc/api/ucc_def.h>

typedef uint64_t ucc_host_id_t;
typedef uint8_t  ucc_socket_id_t;
typedef uint8_t  ucc_numa_id_t;
typedef uint8_t  ucc_device_id_t;

#define UCC_SOCKET_ID_INVALID               ((ucc_socket_id_t)-1)
#define UCC_NUMA_ID_INVALID                 ((ucc_numa_id_t)-1)
#define UCC_DEVICE_ID_INVALID               ((ucc_device_id_t)-1)
#define UCC_HOST_ID_INVALID                 ((ucc_host_id_t)-1)
#define UCC_GPU_FABRIC_CLIQUE_ID_INVALID    ((uint64_t)0)
#define UCC_GPU_FABRIC_PARTITION_ID_INVALID ((uint32_t)0)

#define UCC_MAX_SOCKET_ID (UCC_SOCKET_ID_INVALID - 1)
#define UCC_MAX_NUMA_ID   (UCC_NUMA_ID_INVALID - 1)

#define UCC_MAX_HOST_GPUS 16
#define UCC_MAX_HOST_NICS 16

/* Length of NVML clusterUuid (NVML_GPU_FABRIC_UUID_LEN). */
#define UCC_GPU_FABRIC_CLUSTER_UUID_LEN 16

typedef struct ucc_proc_info {
    ucc_host_id_t   host_hash;
    ucc_socket_id_t socket_id;
    ucc_numa_id_t   numa_id;
    ucc_host_id_t   host_id;
    pid_t           pid;
} ucc_proc_info_t;

typedef struct ucc_pci_info {
    /**< Unique host identifier to which the PCI device belongs */
    ucc_host_id_t  host_id;
    /**< PCI domain */
    uint16_t domain;
    /**< PCI bus */
    uint8_t  bus;
    /**< PCI device */
    uint8_t  device;
    /**< PCI function */
    uint8_t  function;
} ucc_pci_info_t;

typedef enum ucc_gpu_cap {
    UCC_GPU_CAP_NVLINK            = UCC_BIT(0),
    UCC_GPU_CAP_NVSWITCH          = UCC_BIT(1),
    UCC_GPU_CAP_FABRIC            = UCC_BIT(2),
} ucc_gpu_cap_t;

#define UCC_GPU_HAS_CAP(_gpu, _cap) ((_gpu)->caps & (_cap))

typedef struct ucc_gpu_info {
    ucc_pci_info_t pci;
    /**< Bitmask of ucc_gpu_cap_t flags */
    uint32_t       caps;
    /**< NVLink partition ID for GB200+ NVL sub-fabric partitions.
     *   UCC_GPU_FABRIC_PARTITION_ID_INVALID means single partition or
     *   not populated (NVML < r525). */
    uint32_t       fabric_partition_id;
    /**< NVSwitch fabric clique ID (UCC_GPU_FABRIC_CLIQUE_ID_INVALID if unknown) */
    uint64_t       fabric_clique_id;
    /**< Globally-unique NVLink fabric cluster UUID (NVML clusterUuid).
     *   All-zero means fabric info unavailable. */
    uint8_t        fabric_cluster_uuid[UCC_GPU_FABRIC_CLUSTER_UUID_LEN];
    /**< Hash of GPU UUID for unique identification */
    uint64_t       uuid;
} ucc_gpu_info_t;

/* Returns 1 if uuid is non-all-zero (i.e., real fabric info). */
static inline int ucc_gpu_fabric_cluster_uuid_is_valid(const uint8_t *uuid)
{
    int i;
    for (i = 0; i < UCC_GPU_FABRIC_CLUSTER_UUID_LEN; i++) {
        if (uuid[i] != 0) {
            return 1;
        }
    }
    return 0;
}

 typedef struct ucc_nic_info {
    ucc_pci_info_t pci;
    /**< IB port number */
    uint8_t  port;
    /**< Unique NIC identifier (GUID) */
    uint64_t guid;
    /**< Device name, e.g., "mlx5_0" */
    char     name[16];
} ucc_nic_info_t;

typedef struct ucc_host_info {
    /**< Unique host identifier */
    ucc_host_id_t  host_id;
    /**< Number of GPUs on host */
    uint8_t        n_gpus;
    /**<Mask of visible GPUs on this host */
    uint32_t       visible_gpus;
    /**< All GPUs on this host */
    ucc_gpu_info_t gpus[UCC_MAX_HOST_GPUS];
    /**< Number of NICs on host */
    uint8_t        n_nics;
    /**<Mask of visible NICs on this host */
    uint32_t       visible_nics;
    /**< All NICs on this host */
    ucc_nic_info_t nics[UCC_MAX_HOST_NICS];
    /**< nvlink_matrix[i][j] = number of NVLink connections between GPU i and j */
    uint8_t        nvlink_matrix[UCC_MAX_HOST_GPUS][UCC_MAX_HOST_GPUS];
} ucc_host_info_t;

extern ucc_proc_info_t ucc_local_proc;
extern ucc_host_info_t ucc_local_host;

static inline int ucc_pci_distance(const void *a, const void *b)
{
    const ucc_pci_info_t *pci1 = (const ucc_pci_info_t *)a;
    const ucc_pci_info_t *pci2 = (const ucc_pci_info_t *)b;
    int                   dist = 0;
    int                   diff;

    if (pci1->host_id != pci2->host_id) {
        dist += 10000;
    }

    if (pci1->domain != pci2->domain) {
        dist += 1000;
    }

    diff = (pci1->bus > pci2->bus) ? (pci1->bus - pci2->bus)
                                   : (pci2->bus - pci1->bus);
    dist += diff * 10;
    diff = (pci1->device > pci2->device) ? (pci1->device - pci2->device)
                                         : (pci2->device - pci1->device);
    dist += diff * 2;
    diff = (pci1->function > pci2->function)
               ? (pci1->function - pci2->function)
               : (pci2->function - pci1->function);
    dist += diff;

    return dist;
}

static inline int ucc_compare_pci_info(const void *a, const void *b)
{
    return ucc_pci_distance(a, b) == 0;
}

#define UCC_PROC_INFO_EQUAL(_pi1, _pi2)                                        \
    (((_pi1).host_hash == (_pi2).host_hash) &&                                 \
     ((_pi1).pid == (_pi2).pid)) //TODO maybe need tid ?

void ucc_proc_info_print(const ucc_proc_info_t *info);

void ucc_host_info_print(const ucc_host_info_t *info);

ucc_status_t ucc_local_proc_info_init();

uint64_t ucc_get_system_id();

#endif
