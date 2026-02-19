/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define UCC_SOCKET_ID_INVALID ((ucc_socket_id_t)-1)
#define UCC_NUMA_ID_INVALID   ((ucc_numa_id_t)-1)
#define UCC_DEVICE_ID_INVALID ((ucc_device_id_t)-1)
#define UCC_HOST_ID_INVALID   ((ucc_host_id_t)-1)

#define UCC_MAX_SOCKET_ID (UCC_SOCKET_ID_INVALID - 1)
#define UCC_MAX_NUMA_ID   (UCC_NUMA_ID_INVALID - 1)

#define UCC_MAX_HOST_GPUS 16
#define UCC_MAX_HOST_NICS 16

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

typedef struct ucc_gpu_info {
    ucc_pci_info_t pci;
    /**< 1 if GPU has NVLink hardware */
    uint8_t  nvlink_capable;
    /**< 1 if multi-node NVLink supported */
    uint8_t  fabric_capable;
    /**< NVSwitch fabric clique ID (0 if unknown) */
    uint8_t  nvswitch_connected;
    /**< NVSwitch fabric clique ID (0 if unknown) */
    uint64_t fabric_clique_id;
    /**< Hash of GPU UUID for unique identification */
    uint64_t uuid;
} ucc_gpu_info_t;

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
