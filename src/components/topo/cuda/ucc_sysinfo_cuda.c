#include "config.h"
#include "ucc_sysinfo_cuda.h"
#include "components/topo/ucc_sysinfo.h"
#include "core/ucc_global_opts.h"
#include "utils/debug/log_def.h"
#include <stdio.h>
#include <cuda.h>
#ifdef HAVE_NVML_H
#include "utils/ucc_malloc.h"
#include "utils/ucc_math.h"

#include <nvml.h>
#include <pthread.h>
#include <string.h>
#endif

static ucc_config_field_t ucc_sysinfo_cuda_config_table[] = {
    {NULL}
};

#ifdef HAVE_NVML_H
static pthread_mutex_t ucc_sysinfo_cuda_nvml_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

static ucc_status_t ucc_sysinfo_cuda_init(const ucc_sysinfo_params_t *params)
{
    (void)params;
    return UCC_OK;
}

static ucc_status_t ucc_sysinfo_cuda_finalize()
{
    return UCC_OK;
}

static ucc_status_t ucc_sysinfo_cuda_set_visible_devices(
    void *info, int n_info, const ucc_config_names_array_t *devices,
    uint32_t *visible_devices)
{
    const ucc_gpu_info_t *gpus;
    CUcontext             ctx = NULL;
    CUdevice              cu_dev;
    CUresult              cu_st;
    char                  pci_bus_id[32];
    unsigned int          domain;
    unsigned int          bus;
    unsigned int          pci_device;
    unsigned int          function;
    int                   n;
    int                   i;

    if (!visible_devices || (n_info > 0 && !info)) {
        return UCC_ERR_INVALID_PARAM;
    }

    if (devices) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    *visible_devices = 0;
    if (n_info <= 0) {
        return UCC_OK;
    }

    cu_st = cuCtxGetCurrent(&ctx);
    if (cu_st != CUDA_SUCCESS || ctx == NULL) {
        if (cu_st != CUDA_SUCCESS) {
            const char *cu_err_str = NULL;

            cuGetErrorString(cu_st, &cu_err_str);
            ucc_debug("cuCtxGetCurrent failed: %d (%s)", cu_st,
                      cu_err_str ? cu_err_str : "unknown");
        }
        return UCC_OK;
    }

    cu_st = cuCtxGetDevice(&cu_dev);
    if (cu_st != CUDA_SUCCESS) {
        const char *cu_err_str = NULL;

        cuGetErrorString(cu_st, &cu_err_str);
        ucc_debug("cuCtxGetDevice failed: %d (%s)", cu_st,
                  cu_err_str ? cu_err_str : "unknown");
        return UCC_ERR_NO_MESSAGE;
    }

    cu_st = cuDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cu_dev);
    if (cu_st != CUDA_SUCCESS) {
        const char *cu_err_str = NULL;

        cuGetErrorString(cu_st, &cu_err_str);
        ucc_debug("cuDeviceGetPCIBusId failed: %d (%s)", cu_st,
                  cu_err_str ? cu_err_str : "unknown");
        return UCC_ERR_NO_MESSAGE;
    }

    n = sscanf(pci_bus_id, "%x:%x:%x.%x",
               &domain, &bus, &pci_device, &function);
    if (n < 3) {
        return UCC_ERR_INVALID_PARAM;
    }
    if (n < 4) {
        function = 0;
    }

    gpus = (const ucc_gpu_info_t *)info;
    for (i = 0; i < n_info && i < (int)(sizeof(*visible_devices) * 8); i++) {
        if (gpus[i].pci.domain == (uint16_t)domain &&
            gpus[i].pci.bus == (uint8_t)bus &&
            gpus[i].pci.device == (uint8_t)pci_device &&
            gpus[i].pci.function == (uint8_t)function) {
            *visible_devices = (1u << i);
            break;
        }
    }

    return UCC_OK;
}

#ifdef HAVE_NVML_H
static void ucc_sysinfo_cuda_fill_pci_info(const nvmlPciInfo_t *nvml_pci,
                                           ucc_gpu_info_t *gpu)
{
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int function;
    int          n;

    n = sscanf(nvml_pci->busId, "%x:%x:%x.%x",
               &domain, &bus, &device, &function);
    if (n == 4) {
        gpu->pci.domain   = (uint16_t)domain;
        gpu->pci.bus      = (uint8_t)bus;
        gpu->pci.device   = (uint8_t)device;
        gpu->pci.function = (uint8_t)function;
    } else {
        gpu->pci.domain   = (uint16_t)nvml_pci->domain;
        gpu->pci.bus      = (uint8_t)nvml_pci->bus;
        gpu->pci.device   = (uint8_t)nvml_pci->device;
        gpu->pci.function = 0;
    }
    gpu->pci.host_id = ucc_local_proc.host_id;
}

typedef struct ucc_sysinfo_pci_id {
    uint16_t domain;
    uint8_t  bus;
    uint8_t  device;
    uint8_t  function;
} ucc_sysinfo_pci_id_t;

typedef struct ucc_sysinfo_nvlink_switch {
    ucc_sysinfo_pci_id_t pci;
    int                  gpu_links[UCC_MAX_HOST_GPUS];
} ucc_sysinfo_nvlink_switch_t;

static ucc_sysinfo_pci_id_t
ucc_sysinfo_cuda_pci_id_from_nvml(const nvmlPciInfo_t *nvml_pci)
{
    ucc_sysinfo_pci_id_t pci;
    unsigned int         domain;
    unsigned int         bus;
    unsigned int         device;
    unsigned int         function;
    int                  n;

    n = sscanf(nvml_pci->busId, "%x:%x:%x.%x",
               &domain, &bus, &device, &function);
    if (n == 4) {
        pci.domain   = (uint16_t)domain;
        pci.bus      = (uint8_t)bus;
        pci.device   = (uint8_t)device;
        pci.function = (uint8_t)function;
    } else {
        pci.domain   = (uint16_t)nvml_pci->domain;
        pci.bus      = (uint8_t)nvml_pci->bus;
        pci.device   = (uint8_t)nvml_pci->device;
        pci.function = 0;
    }

    return pci;
}

static int ucc_sysinfo_cuda_pci_id_equal(const ucc_sysinfo_pci_id_t *a,
                                         const ucc_sysinfo_pci_id_t *b)
{
    return (a->domain == b->domain) && (a->bus == b->bus) &&
           (a->device == b->device) && (a->function == b->function);
}

static int ucc_sysinfo_cuda_find_gpu_by_pci(const nvmlPciInfo_t *nvml_pci,
                                            const ucc_gpu_info_t *gpus,
                                            int n_gpus)
{
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int function;
    int          n;
    int          i;

    n = sscanf(nvml_pci->busId, "%x:%x:%x.%x",
               &domain, &bus, &device, &function);
    if (n != 4) {
        domain   = nvml_pci->domain;
        bus      = nvml_pci->bus;
        device   = nvml_pci->device;
        function = 0;
    }

    for (i = 0; i < n_gpus; i++) {
        if (gpus[i].pci.domain == (uint16_t)domain &&
            gpus[i].pci.bus == (uint8_t)bus &&
            gpus[i].pci.device == (uint8_t)device &&
            gpus[i].pci.function == (uint8_t)function) {
            return i;
        }
    }

    return -1;
}

static ucc_status_t ucc_sysinfo_cuda_build_nvlink_matrix(
    ucc_gpu_info_t *gpus, int n_gpus,
    uint8_t nvlink_matrix[UCC_MAX_HOST_GPUS][UCC_MAX_HOST_GPUS])
{
    nvmlDevice_t     nvml_dev;
    nvmlFieldValue_t nvml_value;
    nvmlPciInfo_t    nvml_pci;
    nvmlReturn_t     nvml_st;
    unsigned int     num_nvlinks;
    ucc_sysinfo_nvlink_switch_t *switches;
    int             switches_cap = 8;
    int             n_switches   = 0;
    int              i, link, peer_gpu;

    memset(nvlink_matrix, 0, UCC_MAX_HOST_GPUS * UCC_MAX_HOST_GPUS);

    switches = ucc_calloc(switches_cap, sizeof(*switches),
                          "sysinfo_cuda_nvlink_switches");
    if (!switches) {
        return UCC_ERR_NO_MEMORY;
    }

    nvml_value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
    for (i = 0; i < n_gpus; i++) {
        nvml_st = nvmlDeviceGetHandleByIndex(i, &nvml_dev);
        if (nvml_st != NVML_SUCCESS) {
            continue;
        }

        nvml_st = nvmlDeviceGetFieldValues(nvml_dev, 1, &nvml_value);
        num_nvlinks = ((nvml_st == NVML_SUCCESS) &&
                       (nvml_value.nvmlReturn == NVML_SUCCESS) &&
                       (nvml_value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT))
                          ? nvml_value.value.uiVal
                          : 0;

        for (link = 0; link < (int)num_nvlinks; link++) {
            int is_switch = 0;

            nvml_st = nvmlDeviceGetNvLinkRemotePciInfo_v2(nvml_dev, link,
                                                         &nvml_pci);
            if (nvml_st != NVML_SUCCESS) {
                continue;
            }
#if HAVE_NVML_REMOTE_DEVICE_TYPE
            nvmlIntNvLinkDeviceType_t nvml_dt;

            nvml_st = nvmlDeviceGetNvLinkRemoteDeviceType(nvml_dev, link,
                                                          &nvml_dt);
            if (nvml_st == NVML_SUCCESS &&
                nvml_dt == NVML_NVLINK_DEVICE_TYPE_SWITCH) {
                is_switch = 1;
            }
#else
            nvmlDevice_t remote_dev;

            nvml_st = nvmlDeviceGetHandleByPciBusId_v2(nvml_pci.busId,
                                                       &remote_dev);
            if (nvml_st == NVML_ERROR_NOT_FOUND) {
                is_switch = 1;
            } else if (nvml_st != NVML_SUCCESS) {
                continue;
            }
#endif
            if (is_switch) {
                ucc_sysinfo_pci_id_t pci = ucc_sysinfo_cuda_pci_id_from_nvml(
                    &nvml_pci);
                int idx = -1;
                int s;

                for (s = 0; s < n_switches; s++) {
                    if (ucc_sysinfo_cuda_pci_id_equal(&switches[s].pci, &pci)) {
                        idx = s;
                        break;
                    }
                }

                if (idx == -1) {
                    if (n_switches == switches_cap) {
                        int new_cap = switches_cap * 2;
                        ucc_sysinfo_nvlink_switch_t *tmp;

                        tmp = ucc_realloc(switches,
                                          new_cap * sizeof(*switches),
                                          "sysinfo_cuda_nvlink_switches");
                        if (!tmp) {
                            ucc_free(switches);
                            return UCC_ERR_NO_MEMORY;
                        }
                        memset(tmp + switches_cap, 0,
                               (new_cap - switches_cap) * sizeof(*switches));
                        switches = tmp;
                        switches_cap = new_cap;
                    }
                    idx = n_switches++;
                    switches[idx].pci = pci;
                }

                switches[idx].gpu_links[i]++;
                continue;
            }

            peer_gpu = ucc_sysinfo_cuda_find_gpu_by_pci(&nvml_pci, gpus,
                                                        n_gpus);
            if (peer_gpu >= 0 && peer_gpu < n_gpus) {
                if (peer_gpu != i && i < peer_gpu) {
                    nvlink_matrix[i][peer_gpu]++;
                    nvlink_matrix[peer_gpu][i]++;
                }
            }
        }
    }

    for (i = 0; i < n_switches; i++) {
        int g1, g2;

        for (g1 = 0; g1 < n_gpus; g1++) {
            if (switches[i].gpu_links[g1] == 0) {
                continue;
            }
            for (g2 = 0; g2 < n_gpus; g2++) {
                if (g1 == g2 || switches[i].gpu_links[g2] == 0) {
                    continue;
                }
                nvlink_matrix[g1][g2] +=
                    ucc_min(switches[i].gpu_links[g1],
                            switches[i].gpu_links[g2]);
            }
        }
    }

    ucc_free(switches);
    return UCC_OK;
}

static unsigned int ucc_sysinfo_cuda_get_nvlink_count(nvmlDevice_t dev)
{
    nvmlFieldValue_t nvml_value;
    nvmlReturn_t     nvml_st;

    nvml_value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
    nvml_st = nvmlDeviceGetFieldValues(dev, 1, &nvml_value);
    if ((nvml_st == NVML_SUCCESS) &&
        (nvml_value.nvmlReturn == NVML_SUCCESS) &&
        (nvml_value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) {
        return nvml_value.value.uiVal;
    }
    return 0;
}

static ucc_status_t
ucc_sysinfo_cuda_get_nvswitch_connected(nvmlDevice_t dev,
                                        unsigned int num_nvlinks,
                                        uint8_t *connected)
{
    nvmlReturn_t nvml_st;
    int          link;

    *connected = 0;
    for (link = 0; link < (int)num_nvlinks; link++) {
#if HAVE_NVML_REMOTE_DEVICE_TYPE
        nvmlIntNvLinkDeviceType_t nvml_dt;

        nvml_st = nvmlDeviceGetNvLinkRemoteDeviceType(dev, link, &nvml_dt);
        if (nvml_st != NVML_SUCCESS) {
            continue;
        }
        if (nvml_dt == NVML_NVLINK_DEVICE_TYPE_SWITCH) {
            *connected = 1;
            break;
        }
#else
        nvmlPciInfo_t nvml_pci;
        nvmlDevice_t  nvml_dev;

        nvml_st = nvmlDeviceGetNvLinkRemotePciInfo_v2(dev, link, &nvml_pci);
        if (nvml_st != NVML_SUCCESS) {
            continue;
        }
        nvml_st = nvmlDeviceGetHandleByPciBusId_v2(nvml_pci.busId, &nvml_dev);
        if (nvml_st == NVML_ERROR_NOT_FOUND) {
            *connected = 1;
            break;
        }
#endif
    }

    return UCC_OK;
}

static ucc_status_t ucc_sysinfo_cuda_get_info(void **info, int *n_info)
{
    ucc_sysinfo_gpu_info_t *gpu_info;
    unsigned int            num_gpus;
    nvmlDevice_t            nvml_dev;
    nvmlPciInfo_t           nvml_pci;
    nvmlReturn_t            nvml_st;
    ucc_status_t            status;
    int                     n_gpus;
    int                     i;

    *info = NULL;
    *n_info = 0;

    pthread_mutex_lock(&ucc_sysinfo_cuda_nvml_lock);
    nvml_st = nvmlInit_v2();
    if (nvml_st != NVML_SUCCESS) {
        ucc_debug("failed to init NVML: %s", nvmlErrorString(nvml_st));
        pthread_mutex_unlock(&ucc_sysinfo_cuda_nvml_lock);
        return UCC_ERR_NO_RESOURCE;
    }

    nvml_st = nvmlDeviceGetCount(&num_gpus);
    if (nvml_st != NVML_SUCCESS) {
        ucc_debug("nvmlDeviceGetCount failed: %s",
                  nvmlErrorString(nvml_st));
        status = UCC_ERR_NO_RESOURCE;
        goto exit_nvml_shutdown;
    }

    if (num_gpus == 0) {
        status = UCC_OK;
        goto exit_nvml_shutdown;
    }

    n_gpus = ucc_min((int)num_gpus, UCC_MAX_HOST_GPUS);
    gpu_info = (ucc_sysinfo_gpu_info_t *)ucc_malloc(sizeof(*gpu_info),
                                                    "sysinfo_cuda_gpu_info");
    if (!gpu_info) {
        ucc_error("failed to allocate %zd bytes for gpu info",
                  sizeof(*gpu_info));
        status = UCC_ERR_NO_MEMORY;
        goto exit_nvml_shutdown;
    }
    memset(gpu_info, 0, sizeof(*gpu_info));
    gpu_info->n_gpus = n_gpus;

    for (i = 0; i < n_gpus; i++) {
        char uuid_str[NVML_DEVICE_UUID_BUFFER_SIZE];
        unsigned int num_nvlinks;

        nvml_st = nvmlDeviceGetHandleByIndex(i, &nvml_dev);
        if (nvml_st != NVML_SUCCESS) {
            ucc_debug("nvmlDeviceGetHandleByIndex failed: %s",
                      nvmlErrorString(nvml_st));
            status = UCC_ERR_NO_MESSAGE;
            goto free_gpu_info;
        }

        nvml_st = nvmlDeviceGetPciInfo(nvml_dev, &nvml_pci);
        if (nvml_st != NVML_SUCCESS) {
            ucc_debug("nvmlDeviceGetPciInfo failed: %s",
                      nvmlErrorString(nvml_st));
            status = UCC_ERR_NO_MESSAGE;
            goto free_gpu_info;
        }

        ucc_sysinfo_cuda_fill_pci_info(&nvml_pci, &gpu_info->gpus[i]);

        nvml_st = nvmlDeviceGetUUID(nvml_dev, uuid_str, sizeof(uuid_str));
        if (nvml_st == NVML_SUCCESS) {
            gpu_info->gpus[i].uuid = (uint64_t)ucc_str_hash_djb2(uuid_str);
        } else {
            gpu_info->gpus[i].uuid = 0;
        }

        num_nvlinks = ucc_sysinfo_cuda_get_nvlink_count(nvml_dev);
        gpu_info->gpus[i].nvlink_capable = (num_nvlinks > 0);

        status = ucc_sysinfo_cuda_get_nvswitch_connected(
            nvml_dev, num_nvlinks, &gpu_info->gpus[i].nvswitch_connected);
        if (status != UCC_OK) {
            goto free_gpu_info;
        }

        gpu_info->gpus[i].fabric_capable   = 0;
        gpu_info->gpus[i].fabric_clique_id = 0;
    }

    if (num_gpus > UCC_MAX_HOST_GPUS) {
        ucc_debug("truncating gpu info from %u to %u entries",
                  num_gpus, (unsigned int)UCC_MAX_HOST_GPUS);
    }

    status = ucc_sysinfo_cuda_build_nvlink_matrix(gpu_info->gpus, n_gpus,
                                                  gpu_info->nvlink_matrix);
    if (status != UCC_OK) {
        goto free_gpu_info;
    }

    nvmlShutdown();
    pthread_mutex_unlock(&ucc_sysinfo_cuda_nvml_lock);

    *info = gpu_info;
    *n_info = n_gpus;
    return UCC_OK;

free_gpu_info:
    ucc_free(gpu_info);
exit_nvml_shutdown:
    nvmlShutdown();
    pthread_mutex_unlock(&ucc_sysinfo_cuda_nvml_lock);
    return status;
}
#else
static ucc_status_t ucc_sysinfo_cuda_get_info(void **info, int *n_info)
{
    *info = NULL;
    *n_info = 0;
    return UCC_ERR_NOT_SUPPORTED;
}
#endif

ucc_sysinfo_cuda_t ucc_sysinfo_cuda = {
    .super.super.name             = "cuda sysinfo",
    .super.ref_cnt                = 0,
    .super.type                   = UCC_SYSINFO_TYPE_NVLINK,
    .super.init                   = ucc_sysinfo_cuda_init,
    .super.finalize               = ucc_sysinfo_cuda_finalize,
    .super.ops.get_info           = ucc_sysinfo_cuda_get_info,
    .super.ops.set_visible_devices = ucc_sysinfo_cuda_set_visible_devices,
    .super.config_table =
        {
            .name   = "CUDA sysinfo component",
            .prefix = "SYSINFO_CUDA_",
            .table  = ucc_sysinfo_cuda_config_table,
            .size   = sizeof(ucc_sysinfo_config_t),
        },
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_sysinfo_cuda.super.config_table,
                                &ucc_config_global_list);