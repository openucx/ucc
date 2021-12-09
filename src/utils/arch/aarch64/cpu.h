/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
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
#define ucc_memory_bus_fence()        ucc_aarch64_dmb(oshsy)
#define ucc_memory_bus_store_fence()  ucc_aarch64_dmb(oshst)
#define ucc_memory_bus_load_fence()   ucc_aarch64_dmb(oshld)

#endif
