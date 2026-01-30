/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_sysinfo.h"
#include "core/ucc_global_opts.h"
#include "components/topo/base/ucc_sysinfo_base.h"
#include "utils/debug/log_def.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_math.h"
#include <string.h>

static const ucc_sysinfo_ops_t *sysinfo_ops[UCC_SYSINFO_TYPE_LAST];
static int                      sysinfo_initialized;

ucc_status_t ucc_sysinfo_init(void)
{
    ucc_sysinfo_base_t   *sysinfo_base;
    ucc_sysinfo_params_t  sysinfo_params;
    ucc_status_t          status;
    int                   i;

    if (sysinfo_initialized) {
        return UCC_OK;
    }

    memset(&sysinfo_params, 0, sizeof(sysinfo_params));

    for (i = 0; i < ucc_global_config.sysinfo_framework.n_components; i++) {
        sysinfo_base = ucc_derived_of(
            ucc_global_config.sysinfo_framework.components[i],
            ucc_sysinfo_base_t);

        if (sysinfo_base->ref_cnt == 0) {
            sysinfo_base->config =
                ucc_malloc(sysinfo_base->config_table.size);
            if (!sysinfo_base->config) {
                ucc_error("failed to allocate %zd bytes for sysinfo config",
                          sysinfo_base->config_table.size);
                continue;
            }
            status = ucc_config_parser_fill_opts(
                sysinfo_base->config, &sysinfo_base->config_table, "UCC_", 1);
            if (UCC_OK != status) {
                ucc_error("failed to parse config for sysinfo: %s (%d)",
                          sysinfo_base->super.name, status);
                ucc_free(sysinfo_base->config);
                continue;
            }
            status = sysinfo_base->init(&sysinfo_params);
            if (UCC_OK != status) {
                ucc_error("sysinfo_init failed for component: %s, skipping (%d)",
                          sysinfo_base->super.name, status);
                ucc_config_parser_release_opts(sysinfo_base->config,
                                               sysinfo_base->config_table.table);
                ucc_free(sysinfo_base->config);
                continue;
            }
            ucc_debug("sysinfo %s initialized", sysinfo_base->super.name);
        }

        sysinfo_base->ref_cnt++;
        sysinfo_ops[sysinfo_base->type] = &sysinfo_base->ops;
    }

    sysinfo_initialized = 1;
    return UCC_OK;
}

ucc_status_t ucc_sysinfo_get_host_info(ucc_host_info_t *info)
{
    ucc_sysinfo_gpu_info_t *gpu_info;
    ucc_nic_info_t *nic_info;
    ucc_status_t    status;
    int             n_info;
    int             copy_n;

    if (!info) {
        return UCC_ERR_INVALID_PARAM;
    }

    memset(info, 0, sizeof(*info));

    info->host_id = ucc_local_proc.host_id;

    if (sysinfo_ops[UCC_SYSINFO_TYPE_NVLINK]) {
        gpu_info = NULL;
        n_info   = 0;
        status = sysinfo_ops[UCC_SYSINFO_TYPE_NVLINK]->get_info(
            (void **)&gpu_info, &n_info);
        if (status == UCC_OK && gpu_info && n_info > 0) {
            copy_n = ucc_min(n_info, UCC_MAX_HOST_GPUS);
            info->n_gpus = (uint8_t)copy_n;
            memcpy(info->gpus, gpu_info->gpus, copy_n * sizeof(*info->gpus));
            memcpy(info->nvlink_matrix, gpu_info->nvlink_matrix,
                   sizeof(info->nvlink_matrix));
            if (n_info > UCC_MAX_HOST_GPUS) {
                ucc_debug("truncating gpu info from %d to %d entries",
                          n_info, UCC_MAX_HOST_GPUS);
            }
        } else if (status == UCC_ERR_NOT_SUPPORTED ||
                   status == UCC_ERR_NO_RESOURCE ||
                   status == UCC_ERR_NO_MESSAGE) {
            status = UCC_OK;
        }
        if (gpu_info) {
            ucc_free(gpu_info);
        }
        if (status != UCC_OK) {
            return status;
        }
    }

    if (sysinfo_ops[UCC_SYSINFO_TYPE_IB]) {
        nic_info = NULL;
        n_info   = 0;
        status = sysinfo_ops[UCC_SYSINFO_TYPE_IB]->get_info(
            (void **)&nic_info, &n_info);
        if (status == UCC_OK && nic_info && n_info > 0) {
            copy_n = ucc_min(n_info, UCC_MAX_HOST_NICS);
            info->n_nics = (uint8_t)copy_n;
            memcpy(info->nics, nic_info, copy_n * sizeof(*nic_info));
            if (n_info > UCC_MAX_HOST_NICS) {
                ucc_debug("truncating nic info from %d to %d entries",
                          n_info, UCC_MAX_HOST_NICS);
            }
        } else if (status == UCC_ERR_NOT_SUPPORTED ||
                   status == UCC_ERR_NO_RESOURCE ||
                   status == UCC_ERR_NO_MESSAGE) {
            status = UCC_OK;
        }
        if (nic_info) {
            ucc_free(nic_info);
        }
        if (status != UCC_OK) {
            return status;
        }
    }

    ucc_host_info_print(info);

    return UCC_OK;
}

ucc_status_t ucc_sysinfo_set_visible_devices(ucc_host_info_t *info,
                                             ucc_sysinfo_type_t device_type,
                                             const ucc_config_names_array_t *devices)
{
    if (!info) {
        return UCC_ERR_INVALID_PARAM;
    }

    if (device_type == UCC_SYSINFO_TYPE_NVLINK) {
        if (!sysinfo_ops[UCC_SYSINFO_TYPE_NVLINK] ||
            !sysinfo_ops[UCC_SYSINFO_TYPE_NVLINK]->set_visible_devices) {
            return UCC_ERR_NOT_SUPPORTED;
        }
        return sysinfo_ops[UCC_SYSINFO_TYPE_NVLINK]->set_visible_devices(
            info->gpus, info->n_gpus, devices, &info->visible_gpus);
    } else if (device_type == UCC_SYSINFO_TYPE_IB) {
        if (!sysinfo_ops[UCC_SYSINFO_TYPE_IB] ||
            !sysinfo_ops[UCC_SYSINFO_TYPE_IB]->set_visible_devices) {
            return UCC_ERR_NOT_SUPPORTED;
        }
        return sysinfo_ops[UCC_SYSINFO_TYPE_IB]->set_visible_devices(
            info->nics, info->n_nics, devices, &info->visible_nics);
    }

    return UCC_ERR_NOT_SUPPORTED;
}

