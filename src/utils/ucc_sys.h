/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_SYS_H_
#define UCC_SYS_H_

#include "ucc/api/ucc_status.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_log.h"
#include <stddef.h>
#include <unistd.h>
#include <assert.h>

ucc_status_t ucc_sysv_alloc(size_t *size, void **addr, int *shm_id);

ucc_status_t ucc_sysv_free(void *addr);

size_t ucc_get_page_size();

#endif
