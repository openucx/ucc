/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
* Copyright (C) Rivos Inc. 2023
 * SPDX-License-Identifier: BSD-3-Clause
*
* See file LICENSE for terms.
*/

#ifndef UCC_UTILS_ARCH_RISCV64_CPU_H_
#define UCC_UTILS_ARCH_RISCV64_CPU_H_

#define UCC_ARCH_CACHE_LINE_SIZE 64

/* RVWMO rules */
#define ucc_memory_bus_fence()       asm volatile("fence iorw, iorw" ::: "memory")
#define ucc_memory_bus_store_fence() asm volatile("fence ow, ow" ::: "memory")
#define ucc_memory_bus_load_fence()  asm volatile("fence ir, ir" ::: "memory")

#define ucc_memory_cpu_fence()       asm volatile("fence rw, rw" ::: "memory")
#define ucc_memory_cpu_store_fence() asm volatile("fence rw, w" ::: "memory")
#define ucc_memory_cpu_load_fence()  asm volatile("fence r, rw" ::: "memory")

static inline ucc_cpu_model_t ucc_arch_get_cpu_model()
{
    return UCC_CPU_MODEL_UNKNOWN;
}

static inline ucc_cpu_vendor_t ucc_arch_get_cpu_vendor()
{
    return UCC_CPU_VENDOR_GENERIC_RISCV;
}

#endif
