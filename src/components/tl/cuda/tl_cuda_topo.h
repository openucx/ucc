/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
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
} ucc_tl_cuda_device_id_t;

typedef struct ucc_tl_cuda_topo_link {
    ucc_list_link_t         list_link;
    ucc_tl_cuda_device_id_t pci_id;
    int                     width;
} ucc_tl_cuda_topo_link_t;

typedef struct ucc_tl_cuda_topo_node {
    ucc_tl_cuda_device_id_t pci_id;
    ucc_tl_cuda_topo_link_t link;
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
                                         ucc_tl_cuda_device_id_t *pci_id);

ucc_status_t ucc_tl_cuda_topo_create(const ucc_base_lib_t *lib,
                                     ucc_tl_cuda_topo_t **cuda_topo);

ucc_status_t ucc_tl_cuda_topo_destroy(ucc_tl_cuda_topo_t *cuda_topo);

ucc_status_t ucc_tl_cuda_topo_num_links(const ucc_tl_cuda_topo_t *topo,
                                        const ucc_tl_cuda_device_id_t *dev1,
                                        const ucc_tl_cuda_device_id_t *dev2,
                                        int *num_links);

#endif
