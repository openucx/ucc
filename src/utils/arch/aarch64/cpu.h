/**
* Copyright (c) 2001-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) ARM Ltd. 2016-2020.  ALL RIGHTS RESERVED.
* Copyright (C) Stony Brook University. 2016-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_AARCH64_CPU_H_
#define UCC_AARCH64_CPU_H_

#define UCC_ARCH_CACHE_LINE_SIZE 64

/**
 * Assume the worst - weak memory ordering.
 */

#define ucc_aarch64_dmb(_op)          asm volatile ("dmb " #_op ::: "memory")
#define ucc_aarch64_isb(_op)          asm volatile ("isb " #_op ::: "memory")
#define ucc_aarch64_dsb(_op)          asm volatile ("dsb " #_op ::: "memory")

/* The macro is used to serialize stores across Normal NC (or Device) and WB
 * memory, (see Arm Spec, B2.7.2).  Based on recent changes in Linux kernel:
 * https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=22ec71615d824f4f11d38d0e55a88d8956b7e45f
 *
 * The underlying barrier code was changed to use lighter weight DMB instead
 * of DSB. The barrier used for synchronization of access between write back
 * and device mapped memory (PCIe BAR).
 */
#define ucc_memory_bus_store_fence()  ucc_aarch64_dmb(oshst)
#define ucc_memory_bus_load_fence()   ucc_aarch64_dmb(oshld)

#define ucc_memory_cpu_fence()        ucc_aarch64_dmb(ish)
#define ucc_memory_cpu_store_fence()  ucc_aarch64_dmb(ishst)
#define ucc_memory_cpu_load_fence()   ucc_aarch64_dmb(ishld)

typedef struct ucc_aarch64_cpuid {
    int       implementer;
    int       architecture;
    int       variant;
    int       part;
    int       revision;
} ucc_aarch64_cpuid_t;

/**
 * Get ARM CPU identifier and version
 */
void ucc_aarch64_cpuid(ucc_aarch64_cpuid_t *cpuid);

static inline ucc_cpu_model_t ucc_arch_get_cpu_model()
{
    return UCC_CPU_MODEL_ARM_AARCH64;
}

static inline ucc_cpu_vendor_t ucc_arch_get_cpu_vendor()
{
    ucc_aarch64_cpuid_t cpuid;
    ucc_aarch64_cpuid(&cpuid);

    if ((cpuid.implementer == 0x46) && (cpuid.architecture == 8)) {
        return UCC_CPU_VENDOR_FUJITSU_ARM;
    }

    return UCC_CPU_VENDOR_GENERIC_ARM;
}

#endif
