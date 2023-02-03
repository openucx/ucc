/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2023. ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
* Copyright (C) Shanghai Zhaoxin Semiconductor Co., Ltd. 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#if defined(__x86_64__)

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "utils/arch/cpu.h"

#define X86_CPUID_GENUINEINTEL    "GenuntelineI" /* GenuineIntel in magic notation */
#define X86_CPUID_AUTHENTICAMD    "AuthcAMDenti" /* AuthenticAMD in magic notation */
#define X86_CPUID_CENTAURHAULS    "CentaulsaurH" /* CentaurHauls in magic notation */
#define X86_CPUID_SHANGHAI        "  Shai  angh" /* Shanghai in magic notation */
#define X86_CPUID_GET_MODEL       0x00000001u
#define X86_CPUID_GET_BASE_VALUE  0x00000000u
#define X86_CPUID_GET_EXTD_VALUE  0x00000007u
#define X86_CPUID_GET_MAX_VALUE   0x80000000u
#define X86_CPUID_INVARIANT_TSC   0x80000007u
#define X86_CPUID_GET_CACHE_INFO  0x00000002u
#define X86_CPUID_GET_LEAF4_INFO  0x00000004u

typedef union ucc_x86_cpu_registers {
    struct {
        union {
            uint32_t     eax;
            uint8_t      max_iter; /* leaf 2 - max iterations */
        };
        union {
            struct {
                uint32_t ebx;
                uint32_t ecx;
                uint32_t edx;
            };
            char         id[sizeof(uint32_t) * 3]; /* leaf 0 - CPU ID */
        };
    };
    union {
        uint32_t         value;
        uint8_t          tag[sizeof(uint32_t)];
    }                    reg[4]; /* leaf 2 tags */
} UCC_S_PACKED ucc_x86_cpu_registers;

/* CPU version */
typedef union ucc_x86_cpu_version {
    struct {
        unsigned stepping   : 4;
        unsigned model      : 4;
        unsigned family     : 4;
        unsigned type       : 2;
        unsigned unused     : 2;
        unsigned ext_model  : 4;
        unsigned ext_family : 8;
    };
    uint32_t reg;
} UCC_S_PACKED ucc_x86_cpu_version_t;

static UCC_F_NOOPTIMIZE inline void ucc_x86_cpuid(uint32_t level,
                                                  uint32_t *a, uint32_t *b,
                                                  uint32_t *c, uint32_t *d)
{
    asm volatile ("cpuid\n\t"
                  : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
                  : "0"(level));
}

ucc_cpu_vendor_t ucc_arch_get_cpu_vendor()
{
    ucc_x86_cpu_registers reg = {}; /* Silence static checker */

    ucc_x86_cpuid(X86_CPUID_GET_BASE_VALUE,
                  ucc_unaligned_ptr(&reg.eax), ucc_unaligned_ptr(&reg.ebx),
                  ucc_unaligned_ptr(&reg.ecx), ucc_unaligned_ptr(&reg.edx));
    if (!memcmp(reg.id, X86_CPUID_GENUINEINTEL, sizeof(X86_CPUID_GENUINEINTEL) - 1)) {
        return UCC_CPU_VENDOR_INTEL;
    } else if (!memcmp(reg.id, X86_CPUID_AUTHENTICAMD, sizeof(X86_CPUID_AUTHENTICAMD) - 1)) {
        return UCC_CPU_VENDOR_AMD;
    } else if (!memcmp(reg.id, X86_CPUID_CENTAURHAULS, sizeof(X86_CPUID_CENTAURHAULS) - 1) ||
               !memcmp(reg.id, X86_CPUID_SHANGHAI, sizeof(X86_CPUID_SHANGHAI) - 1)) {
        return UCC_CPU_VENDOR_ZHAOXIN;
    }

    return UCC_CPU_VENDOR_UNKNOWN;
}

ucc_cpu_model_t ucc_arch_get_cpu_model()
{
    ucc_x86_cpu_version_t version = {}; /* Silence static checker */
    uint32_t _ebx, _ecx, _edx;
    uint32_t model, family;

    /* Get CPU model/family */
    ucc_x86_cpuid(X86_CPUID_GET_MODEL, ucc_unaligned_ptr(&version.reg),
                  &_ebx, &_ecx, &_edx);

    model  = version.model;
    family = version.family;

    /* Adjust family/model */
    if (family == 0xf) {
        family += version.ext_family;
    }
    if ((family == 0x6) || (family == 0x7) || (family == 0xf) ||
        (family == 0x17) || (family == 0x19)) {
        model = (version.ext_model << 4) | model;
    }

    if (ucc_arch_get_cpu_vendor() == UCC_CPU_VENDOR_ZHAOXIN) {
        if (family == 0x06) {
            switch (model) {
            case 0x0f:
                return UCC_CPU_MODEL_ZHAOXIN_ZHANGJIANG;
            }
        }

        if (family == 0x07) {
            switch (model) {
            case 0x1b:
                return UCC_CPU_MODEL_ZHAOXIN_WUDAOKOU;
            case 0x3b:
                return UCC_CPU_MODEL_ZHAOXIN_LUJIAZUI;
            }
        }
    } else {
        /* Check known CPUs */
        if (family == 0x06) {
            switch (model) {
            case 0x3a:
            case 0x3e:
                return UCC_CPU_MODEL_INTEL_IVYBRIDGE;
            case 0x2a:
            case 0x2d:
                return UCC_CPU_MODEL_INTEL_SANDYBRIDGE;
            case 0x1a:
            case 0x1e:
            case 0x1f:
            case 0x2e:
                return UCC_CPU_MODEL_INTEL_NEHALEM;
            case 0x25:
            case 0x2c:
            case 0x2f:
                return UCC_CPU_MODEL_INTEL_WESTMERE;
            case 0x3c:
            case 0x3f:
            case 0x45:
            case 0x46:
                return UCC_CPU_MODEL_INTEL_HASWELL;
            case 0x3d:
            case 0x47:
            case 0x4f:
            case 0x56:
                return UCC_CPU_MODEL_INTEL_BROADWELL;
            case 0x5e:
            case 0x4e:
            case 0x55:
                return UCC_CPU_MODEL_INTEL_SKYLAKE;
            }
        }

        if (family == 0x17) {
            switch (model) {
            case 0x29:
                return UCC_CPU_MODEL_AMD_NAPLES;
            case 0x31:
                return UCC_CPU_MODEL_AMD_ROME;
            }
        }

        if (family == 0x19) {
            switch (model) {
            case 0x00:
            case 0x01:
                return UCC_CPU_MODEL_AMD_MILAN;
            case 0x11:
                return UCC_CPU_MODEL_AMD_GENOA;
            }
        }
    }

    return UCC_CPU_MODEL_UNKNOWN;
}

#endif
