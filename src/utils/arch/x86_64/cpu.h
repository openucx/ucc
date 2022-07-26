/**
* Copyright (c) 2001-2013, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_X86_64_H_
#define UCC_X86_64_H_

#include "utils/ucc_compiler_def.h"

#define UCC_ARCH_CACHE_LINE_SIZE 64

/**
 * In x86_64, there is strong ordering of each processor with respect to another
 * processor, but weak ordering with respect to the bus.
 */
#define ucc_memory_bus_store_fence()  asm volatile ("sfence" ::: "memory")
#define ucc_memory_bus_load_fence()   asm volatile ("lfence" ::: "memory")

#define ucc_memory_cpu_fence()        ucc_compiler_fence()
#define ucc_memory_cpu_store_fence()  ucc_compiler_fence()
#define ucc_memory_cpu_load_fence()   ucc_compiler_fence()

ucc_cpu_model_t  ucc_arch_get_cpu_model() UCC_F_NOOPTIMIZE;
ucc_cpu_vendor_t ucc_arch_get_cpu_vendor();

#endif
