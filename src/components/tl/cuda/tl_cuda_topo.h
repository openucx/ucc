/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_TOPO_H_
#define UCC_TL_CUDA_TOPO_H_

#include "components/tl/ucc_tl_log.h"
#include "utils/khash.h"

typedef struct ucc_tl_cuda_device_id {
    uint16_t domain;   /* range: 0 to ffff */
    uint8_t  bus;      /* range: 0 to ff */
    uint8_t  device;   /* range: 0 to 1f */
    uint8_t  function; /* range: 0 to 7 */
} ucc_tl_cuda_device_pci_id_t;

typedef enum ucc_tl_cuda_topo_dev_type {
    UCC_TL_CUDA_TOPO_DEV_TYPE_GPU,
    UCC_TL_CUDA_TOPO_DEV_TYPE_SWITCH,
    UCC_TL_CUDA_TOPO_DEV_TYPE_LAST
} ucc_tl_cuda_topo_dev_type_t;

static inline int
ucc_tl_cuda_topo_device_id_equal(const ucc_tl_cuda_device_pci_id_t *id1,
                                 const ucc_tl_cuda_device_pci_id_t *id2)
{
    return ((id1->domain   == id2->domain) &&
            (id1->bus      == id2->bus)    &&
            (id1->device   == id2->device) &&
            (id1->function == id2->function));
}

typedef struct ucc_tl_cuda_topo_link {
    ucc_list_link_t             list_link;
    ucc_tl_cuda_device_pci_id_t pci_id;
    int                         width;
} ucc_tl_cuda_topo_link_t;

typedef struct ucc_tl_cuda_topo_node {
    ucc_tl_cuda_device_pci_id_t pci_id;
    ucc_tl_cuda_topo_dev_type_t type;
    ucc_tl_cuda_topo_link_t     link;
} ucc_tl_cuda_topo_node_t;

KHASH_MAP_INIT_INT64(bus_to_node, int);

typedef struct ucc_tl_cuda_topo {
    const ucc_base_lib_t    *lib;
    khash_t(bus_to_node)     bus_to_node_hash;
    int                      num_nodes;
    ucc_tl_cuda_topo_node_t *graph;
} ucc_tl_cuda_topo_t;

ucc_status_t ucc_tl_cuda_topo_get_pci_id(const ucc_base_lib_t *lib,
                                         int device,
                                         ucc_tl_cuda_device_pci_id_t *pci_id);

ucc_status_t ucc_tl_cuda_topo_create(const ucc_base_lib_t *lib,
                                     ucc_tl_cuda_topo_t **cuda_topo);

ucc_status_t ucc_tl_cuda_topo_destroy(ucc_tl_cuda_topo_t *cuda_topo);

ucc_status_t ucc_tl_cuda_topo_num_links(const ucc_tl_cuda_topo_t *topo,
                                        const ucc_tl_cuda_device_pci_id_t *dev1,
                                        const ucc_tl_cuda_device_pci_id_t *dev2,
                                        int *num_links);

#endif
