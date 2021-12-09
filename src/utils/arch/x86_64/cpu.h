/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_X86_64_H_
#define UCC_X86_64_H_

#define UCC_ARCH_CACHE_LINE_SIZE 64

/**
 * In x86_64, there is strong ordering of each processor with respect to another
 * processor, but weak ordering with respect to the bus.
 */
#define ucc_memory_bus_fence()        asm volatile ("mfence"::: "memory")
#define ucc_memory_bus_store_fence()  asm volatile ("sfence" ::: "memory")
#define ucc_memory_bus_load_fence()   asm volatile ("lfence" ::: "memory")


#endif
