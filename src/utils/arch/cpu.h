/**
* Copyright (c) 2001-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
* Copyright (C) Shanghai Zhaoxin Semiconductor Co., Ltd. 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_ARCH_CPU_H
#define UCC_ARCH_CPU_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "utils/ucc_compiler_def.h"
#include "utils/ucc_log.h"
#include <stddef.h>

/* CPU models */
typedef enum ucc_cpu_model {
    UCC_CPU_MODEL_UNKNOWN,
    UCC_CPU_MODEL_INTEL_IVYBRIDGE,
    UCC_CPU_MODEL_INTEL_SANDYBRIDGE,
    UCC_CPU_MODEL_INTEL_NEHALEM,
    UCC_CPU_MODEL_INTEL_WESTMERE,
    UCC_CPU_MODEL_INTEL_HASWELL,
    UCC_CPU_MODEL_INTEL_BROADWELL,
    UCC_CPU_MODEL_INTEL_SKYLAKE,
    UCC_CPU_MODEL_ARM_AARCH64,
    UCC_CPU_MODEL_AMD_NAPLES,
    UCC_CPU_MODEL_AMD_ROME,
    UCC_CPU_MODEL_AMD_MILAN,
    UCC_CPU_MODEL_ZHAOXIN_ZHANGJIANG,
    UCC_CPU_MODEL_ZHAOXIN_WUDAOKOU,
    UCC_CPU_MODEL_ZHAOXIN_LUJIAZUI,
    UCC_CPU_MODEL_LAST
} ucc_cpu_model_t;

/* CPU vendors */
typedef enum ucc_cpu_vendor {
    UCC_CPU_VENDOR_UNKNOWN,
    UCC_CPU_VENDOR_INTEL,
    UCC_CPU_VENDOR_AMD,
    UCC_CPU_VENDOR_GENERIC_ARM,
    UCC_CPU_VENDOR_GENERIC_PPC,
    UCC_CPU_VENDOR_FUJITSU_ARM,
    UCC_CPU_VENDOR_ZHAOXIN,
    UCC_CPU_VENDOR_LAST
} ucc_cpu_vendor_t;

static inline ucc_cpu_vendor_t ucc_get_vendor_from_str(const char *v_name)
{
    if (strcmp(v_name, "intel") == 0)
        return UCC_CPU_VENDOR_INTEL;
    if (strcmp(v_name, "amd") == 0)
        return UCC_CPU_VENDOR_AMD;
    if (strcmp(v_name, "arm") == 0)
        return UCC_CPU_VENDOR_GENERIC_ARM;
    if (strcmp(v_name, "ppc") == 0)
        return UCC_CPU_VENDOR_GENERIC_PPC;
    if (strcmp(v_name, "fujitsu") == 0)
        return UCC_CPU_VENDOR_FUJITSU_ARM;
    if (strcmp(v_name, "zhaoxin") == 0)
        return UCC_CPU_VENDOR_ZHAOXIN;
    ucc_warn("vendor not found");
    return UCC_CPU_VENDOR_UNKNOWN;
}

static inline ucc_cpu_model_t ucc_get_model_from_str(const char *m_name)
{
    if (strcmp(m_name, "ivybridge") == 0)
        return UCC_CPU_MODEL_INTEL_IVYBRIDGE;
    if (strcmp(m_name, "sandybridge") == 0)
        return UCC_CPU_MODEL_INTEL_SANDYBRIDGE;
    if (strcmp(m_name, "nehalem") == 0)
        return UCC_CPU_MODEL_INTEL_NEHALEM;
    if (strcmp(m_name, "westmere") == 0)
        return UCC_CPU_MODEL_INTEL_WESTMERE;
    if (strcmp(m_name, "haswell") == 0)
        return UCC_CPU_MODEL_INTEL_HASWELL;
    if (strcmp(m_name, "broadwell") == 0)
        return UCC_CPU_MODEL_INTEL_BROADWELL;
    if (strcmp(m_name, "skylake") == 0)
        return UCC_CPU_MODEL_INTEL_SKYLAKE;
    if (strcmp(m_name, "aarch64") == 0)
        return UCC_CPU_MODEL_ARM_AARCH64;
    if (strcmp(m_name, "naples") == 0)
        return UCC_CPU_MODEL_AMD_NAPLES;
    if (strcmp(m_name, "rome") == 0)
        return UCC_CPU_MODEL_AMD_ROME;
    if (strcmp(m_name, "milan") == 0)
        return UCC_CPU_MODEL_AMD_MILAN;
    if (strcmp(m_name, "zhangjiang") == 0)
        return UCC_CPU_MODEL_ZHAOXIN_ZHANGJIANG;
    if (strcmp(m_name, "wudaokou") == 0)
        return UCC_CPU_MODEL_ZHAOXIN_WUDAOKOU;
    if (strcmp(m_name, "lujiazui") == 0)
        return UCC_CPU_MODEL_ZHAOXIN_LUJIAZUI;
    ucc_warn("model not found");
    return UCC_CPU_MODEL_UNKNOWN;
}

#if defined(__x86_64__)
#  include "x86_64/cpu.h"
#elif defined(__powerpc64__)
#  include "ppc64/cpu.h"
#elif defined(__aarch64__)
#  include "aarch64/cpu.h"
#else
#  error "Unsupported architecture"
#endif

#define UCC_CACHE_LINE_SIZE UCC_ARCH_CACHE_LINE_SIZE

#endif
