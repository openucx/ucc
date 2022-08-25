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

const char* ucc_sys_get_lib_path();

ucc_status_t ucc_sys_dirname(const char *path, char **out);

ucc_status_t ucc_sys_path_join(const char *path1, const char *path2,
                               char **out);

size_t ucc_get_page_size();

#endif
