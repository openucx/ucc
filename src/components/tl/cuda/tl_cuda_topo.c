/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_cuda_topo.h"
#include "utils/arch/cuda_def.h"
#include <inttypes.h>
#include <pthread.h>
#include <nvml.h>

pthread_mutex_t nvml_lock = PTHREAD_MUTEX_INITIALIZER;

#define MAX_PCI_BUS_ID_STR 16
#define MAX_PCI_DEVICES    32

#define NVMLCHECK_GOTO(_cmd, _label, _status, _lib)                            \
    do {                                                                       \
        nvmlReturn_t e = _cmd;                                                 \
        if (ucc_unlikely(NVML_SUCCESS != e)) {                                 \
            tl_error(_lib, "NVML error %d %s", e, nvmlErrorString(e));         \
            _status = UCC_ERR_NO_MESSAGE;                                      \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

static ucc_status_t
ucc_tl_cuda_topo_pci_id_from_str(const char * bus_id_str,
                                 ucc_tl_cuda_device_pci_id_t *pci_id)
{
    int n;

    n = sscanf(bus_id_str, "%hx:%hhx:%hhx.%hhx", &pci_id->domain, &pci_id->bus,
               &pci_id->device, &pci_id->function);
    if (n != 4) {
        return UCC_ERR_INVALID_PARAM;
    }
    return UCC_OK;
}

// TODO: add to topo print
static void ucc_tl_cuda_topo_pci_id_to_str(const ucc_tl_cuda_device_pci_id_t *pci_id,
                                           char *str, size_t max)
{
    ucc_snprintf_safe(str, max, "%04x:%02x:%02x.%d", pci_id->domain,
                      pci_id->bus, pci_id->device, pci_id->function);
}

ucc_status_t ucc_tl_cuda_topo_get_pci_id(const ucc_base_lib_t *lib,
                                         int device,
                                         ucc_tl_cuda_device_pci_id_t *pci_id)
{
    char pci_bus_id[MAX_PCI_BUS_ID_STR];
    ucc_status_t st;

    CUDA_CHECK_GOTO(cudaDeviceGetPCIBusId(pci_bus_id, MAX_PCI_BUS_ID_STR,
                                          device), exit, st);
    st = ucc_tl_cuda_topo_pci_id_from_str(pci_bus_id, pci_id);
exit:
    return st;
}

static uint64_t
ucc_tl_cuda_device_pci_id_to_uint64(const ucc_tl_cuda_device_pci_id_t *id)
{
    return (((uint64_t)id->domain << 24) |
            ((uint64_t)id->bus << 16)    |
            ((uint64_t)id->device << 8)  |
            ((uint64_t)id->function));
}


static ucc_status_t
ucc_tl_cuda_topo_graph_find_by_id(const ucc_tl_cuda_topo_t *topo,
                                  const ucc_tl_cuda_device_pci_id_t *dev_id,
                                  ucc_tl_cuda_topo_node_t **node)
{
    uint64_t id;
    khiter_t iter;

    id = ucc_tl_cuda_device_pci_id_to_uint64(dev_id);
    iter = kh_get(bus_to_node, &topo->bus_to_node_hash, id);
    if (iter == kh_end(&topo->bus_to_node_hash)) {
        return UCC_ERR_NOT_FOUND;
    }
    *node = &topo->graph[kh_value(&topo->bus_to_node_hash, iter)];
    return UCC_OK;
}

static ucc_status_t
ucc_tl_cuda_topo_graph_add_device(const ucc_tl_cuda_device_pci_id_t *dev_id,
                                  ucc_tl_cuda_topo_dev_type_t dev_type,
                                  ucc_tl_cuda_topo_t *topo,
                                  ucc_tl_cuda_topo_node_t **node)
{
    uint64_t key;
    khiter_t iter;
    int ret;
    int n;
    char dev_id_str[MAX_PCI_BUS_ID_STR];

    key = ucc_tl_cuda_device_pci_id_to_uint64(dev_id);
    iter = kh_put(bus_to_node, &topo->bus_to_node_hash, key, &ret);
    if (ret < 0) {
        ucc_tl_cuda_topo_pci_id_to_str(dev_id, dev_id_str, MAX_PCI_BUS_ID_STR);
        tl_error(topo->lib, "failed to add device id %s key %" PRIu64 " to hash",
                 dev_id_str, key);
        return UCC_ERR_NO_MESSAGE;
    } else if (ret == 0) {
        /* device already exists */
        *node = &topo->graph[kh_value(&topo->bus_to_node_hash, iter)];
        ucc_assert((*node)->type == dev_type);
        return UCC_OK;
    } else {
        n = topo->num_nodes;
        topo->num_nodes++;
        kh_value(&topo->bus_to_node_hash, iter) = n;
        topo->graph[n].pci_id = *dev_id;
        topo->graph[n].type   = dev_type;
        ucc_list_head_init(&topo->graph[n].link.list_link);
        *node = &topo->graph[n];
    }
    return UCC_OK;
}

static ucc_status_t
ucc_tl_cuda_topo_graph_add_link(ucc_tl_cuda_topo_t *topo,
                                ucc_tl_cuda_topo_node_t *src,
                                ucc_tl_cuda_topo_node_t *dst)
{
    ucc_tl_cuda_topo_link_t *link;
    ucc_list_for_each(link, &src->link.list_link, list_link) {
        if (ucc_tl_cuda_topo_device_id_equal(&link->pci_id, &dst->pci_id)) {
            link->width += 1;
            return UCC_OK;
        }
    }
    link = (ucc_tl_cuda_topo_link_t*)ucc_malloc(sizeof(*link), "cuda_topo_link");
    if (!link) {
        tl_error(topo->lib, "failed to allocate topo link");
        return UCC_ERR_NO_MEMORY;
    }
    link->pci_id = dst->pci_id;
    link->width  = 1;
    ucc_list_add_tail(&src->link.list_link, &link->list_link);

    return UCC_OK;
}

static void ucc_tl_cuda_topo_free_link(ucc_tl_cuda_topo_link_t *link) {
    return;
}

static void ucc_tl_cuda_topo_graph_destroy(ucc_tl_cuda_topo_t *topo)
{
    int i;

    for (i = 0; i < topo->num_nodes; i++) {
        ucc_list_destruct(&topo->graph[i].link.list_link,
                          ucc_tl_cuda_topo_link_t, ucc_tl_cuda_topo_free_link,
                          list_link);
    }
    kh_destroy_inplace(bus_to_node, &topo->bus_to_node_hash);
    free(topo->graph);
}

static ucc_status_t
ucc_tl_cuda_topo_get_remote_dev_type(ucc_tl_cuda_topo_t *topo,
                                     nvmlDevice_t dev, int link,
                                     ucc_tl_cuda_topo_dev_type_t *dev_type)
{

#if HAVE_NVML_REMOTE_DEVICE_TYPE
    ucc_status_t status = UCC_OK;
    nvmlIntNvLinkDeviceType_t nvml_dt;

    NVMLCHECK_GOTO(nvmlDeviceGetNvLinkRemoteDeviceType(dev, link, &nvml_dt),
                   exit, status, topo->lib);
    switch(nvml_dt) {
    case NVML_NVLINK_DEVICE_TYPE_GPU:
        *dev_type = UCC_TL_CUDA_TOPO_DEV_TYPE_GPU;
        break;
    case NVML_NVLINK_DEVICE_TYPE_SWITCH:
        *dev_type = UCC_TL_CUDA_TOPO_DEV_TYPE_SWITCH;
        break;
    default:
        *dev_type = UCC_TL_CUDA_TOPO_DEV_TYPE_LAST;
        break;
    }
exit:
    return status;
#else
    nvmlPciInfo_t nvml_pci;
    nvmlDevice_t nvml_dev;
    nvmlReturn_t nvml_st;

    nvml_st = nvmlDeviceGetNvLinkRemotePciInfo_v2(dev, link, &nvml_pci);
    if (nvml_st != NVML_SUCCESS) {
        *dev_type = UCC_TL_CUDA_TOPO_DEV_TYPE_LAST;
        return UCC_OK;
    }

    nvml_st = nvmlDeviceGetHandleByPciBusId_v2(nvml_pci.busId, &nvml_dev);
    if (nvml_st == NVML_SUCCESS) {
        *dev_type = UCC_TL_CUDA_TOPO_DEV_TYPE_GPU;
        return UCC_OK;
    } else if (nvml_st == NVML_ERROR_NOT_FOUND) {
        *dev_type = UCC_TL_CUDA_TOPO_DEV_TYPE_SWITCH;
        return UCC_OK;
    }
    return UCC_ERR_NOT_SUPPORTED;
#endif
}

static ucc_status_t ucc_tl_cuda_topo_graph_create(ucc_tl_cuda_topo_t *topo)
{
    ucc_status_t status = UCC_OK;
    cudaError_t cuda_st;
    char pci_bus_str[MAX_PCI_BUS_ID_STR];
    nvmlDevice_t nvml_dev;
    nvmlFieldValue_t nvml_value;
    nvmlPciInfo_t nvml_pci;
    ucc_tl_cuda_topo_dev_type_t dev_type;
    ucc_tl_cuda_device_pci_id_t pci_id;
    ucc_tl_cuda_topo_node_t *node, *peer_node;
    int num_gpus;
    int i, num_nvlinks, link;

    cuda_st = cudaGetDeviceCount(&num_gpus);
    if ((cuda_st != cudaSuccess) || (num_gpus == 0)){
        tl_info(topo->lib, "cudaGetDeviceCount failed or no GPU devices found");
        return UCC_ERR_NO_RESOURCE;
    }
    topo->num_nodes = 0;
    topo->graph = (ucc_tl_cuda_topo_node_t*)ucc_malloc(MAX_PCI_DEVICES *
                                                       sizeof(*topo->graph),
                                                       "tl cuda topo graph");
    if (!topo->graph) {
        tl_error(topo->lib, "failed to allocate tl cuda topo graph");
        return UCC_ERR_NO_MEMORY;
    }
    kh_init_inplace(bus_to_node, &topo->bus_to_node_hash);
    pthread_mutex_lock(&nvml_lock);
    NVMLCHECK_GOTO(nvmlInit_v2(), exit_free_graph, status, topo->lib);
    nvml_value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
    for (i = 0; i < num_gpus; i++) {
        CUDA_CHECK_GOTO(cudaDeviceGetPCIBusId(pci_bus_str, MAX_PCI_BUS_ID_STR, i),
                        exit_nvml_shutdown, status);
        status = ucc_tl_cuda_topo_pci_id_from_str(pci_bus_str, &pci_id);
        if (status != UCC_OK) {
            goto exit_nvml_shutdown;
        }
        status = ucc_tl_cuda_topo_graph_add_device(&pci_id,
                                                   UCC_TL_CUDA_TOPO_DEV_TYPE_GPU,
                                                   topo, &node);
        if (status != UCC_OK) {
            goto exit_nvml_shutdown;
        }
        NVMLCHECK_GOTO(nvmlDeviceGetHandleByPciBusId_v2(pci_bus_str, &nvml_dev),
                       exit_nvml_shutdown, status, topo->lib);
        NVMLCHECK_GOTO(nvmlDeviceGetFieldValues(nvml_dev, 1, &nvml_value),
                       exit_nvml_shutdown, status, topo->lib);
        num_nvlinks = ((nvml_value.nvmlReturn == NVML_SUCCESS) &&
                       (nvml_value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                      nvml_value.value.uiVal : 0;
        for (link = 0; link < num_nvlinks; link++) {
            status = ucc_tl_cuda_topo_get_remote_dev_type(topo, nvml_dev, link,
                                                          &dev_type);
            if (status != UCC_OK) {
                tl_info(topo->lib, "failed to get remote device type");
                goto exit_nvml_shutdown;
            }
            if (dev_type == UCC_TL_CUDA_TOPO_DEV_TYPE_LAST) {
               /* nvlink connected device is not supported by cuda tl */
                continue;
            }
            NVMLCHECK_GOTO(nvmlDeviceGetNvLinkRemotePciInfo_v2(nvml_dev, link,
                                                               &nvml_pci),
                           exit_nvml_shutdown, status, topo->lib);
            pci_id.domain   = nvml_pci.domain;
            pci_id.bus      = nvml_pci.bus;
            pci_id.device   = nvml_pci.device;
            pci_id.function = 0;
            status = ucc_tl_cuda_topo_graph_add_device(&pci_id, dev_type, topo,
                                                       &peer_node);
            if (status != UCC_OK) {
                goto exit_nvml_shutdown;
            }
            status = ucc_tl_cuda_topo_graph_add_link(topo, node, peer_node);
            if (status != UCC_OK) {
                goto exit_nvml_shutdown;
            }
        }
    }
    nvmlShutdown();
    pthread_mutex_unlock(&nvml_lock);
    return UCC_OK;

exit_nvml_shutdown:
    nvmlShutdown();
exit_free_graph:
    ucc_tl_cuda_topo_graph_destroy(topo);
    pthread_mutex_unlock(&nvml_lock);
    return status;
}

ucc_status_t ucc_tl_cuda_topo_create(const ucc_base_lib_t *lib,
                                     ucc_tl_cuda_topo_t **cuda_topo)
{
    ucc_tl_cuda_topo_t *topo;
    ucc_status_t status;

    topo = (ucc_tl_cuda_topo_t*)ucc_malloc(sizeof(*topo), "cuda_topo");
    topo->lib = lib;
    if (!topo) {
        tl_error(lib, "failed to alloc cuda topo");
        status = UCC_ERR_NO_MEMORY;
        goto exit_err;
    }

    status = ucc_tl_cuda_topo_graph_create(topo);
    if (status != UCC_OK) {
        goto free_topo;
    }

    *cuda_topo = topo;
    return UCC_OK;
free_topo:
    ucc_free(topo);
exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_topo_num_links(const ucc_tl_cuda_topo_t *topo,
                                        const ucc_tl_cuda_device_pci_id_t *dev1,
                                        const ucc_tl_cuda_device_pci_id_t *dev2,
                                        int *num_links)
{
    ucc_status_t status;
    ucc_tl_cuda_topo_node_t *dev1_node, *dev2_node;
    ucc_tl_cuda_topo_node_t *link_node;
    ucc_tl_cuda_topo_link_t *dev1_link, *dev2_link;

    *num_links = 0;
    status = ucc_tl_cuda_topo_graph_find_by_id(topo, dev1, &dev1_node);
    if (status != UCC_OK) {
        return status;
    }
    status = ucc_tl_cuda_topo_graph_find_by_id(topo, dev2, &dev2_node);
    if (status != UCC_OK) {
        return status;
    }

    ucc_list_for_each(dev1_link, &dev1_node->link.list_link, list_link) {
        if (ucc_tl_cuda_topo_device_id_equal(&dev1_link->pci_id, dev2)) {
            *num_links += dev1_link->width;
        } else {
            /* check if peer device is switch and it connects given devices */
            status = ucc_tl_cuda_topo_graph_find_by_id(topo, &dev1_link->pci_id,
                                                       &link_node);
            if (status != UCC_OK) {
                return status;
            }
            if (link_node->type == UCC_TL_CUDA_TOPO_DEV_TYPE_SWITCH) {
                ucc_list_for_each(dev2_link, &dev2_node->link.list_link, list_link) {
                    if (ucc_tl_cuda_topo_device_id_equal(&dev2_link->pci_id,
                                                         &dev1_link->pci_id)) {
                        *num_links += ucc_min(dev1_link->width, dev2_link->width);
                    }
                }
            }
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_topo_destroy(ucc_tl_cuda_topo_t *cuda_topo)
{
    ucc_tl_cuda_topo_graph_destroy(cuda_topo);
    ucc_free(cuda_topo);
    return UCC_OK;
}
