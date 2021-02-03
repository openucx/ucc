/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_H_
#define UCC_MC_H_

#include "ucc/api/ucc.h"
#include "components/mc/base/ucc_mc_base.h"

ucc_status_t ucc_mc_init();

ucc_status_t ucc_mc_available(ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_type(const void *ptr, ucc_memory_type_t *mem_type);

ucc_status_t ucc_mc_alloc(void **ptr, size_t len, ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_free(void *ptr, ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_finalize();

#endif
