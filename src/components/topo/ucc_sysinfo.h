/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_SYSINFO_H_
#define UCC_SYSINFO_H_

#include "components/topo/base/ucc_sysinfo_base.h"
#include "utils/ucc_proc_info.h"

typedef struct ucc_sysinfo_gpu_info {
    int            n_gpus;
    ucc_gpu_info_t gpus[UCC_MAX_HOST_GPUS];
    uint8_t        nvlink_matrix[UCC_MAX_HOST_GPUS][UCC_MAX_HOST_GPUS];
} ucc_sysinfo_gpu_info_t;

ucc_status_t ucc_sysinfo_init(void);
ucc_status_t ucc_sysinfo_get_host_info(ucc_host_info_t *info);
ucc_status_t ucc_sysinfo_set_visible_devices(
    ucc_host_info_t *info, ucc_sysinfo_type_t device_type,
    const ucc_config_names_array_t *devices);

#endif
