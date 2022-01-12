/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_SYS_H_
#define UCC_SYS_H_

#include "ucc/api/ucc_status.h"
#include <stddef.h>

ucc_status_t ucc_sysv_alloc(size_t *size, void **addr, int *shm_id);

ucc_status_t ucc_sysv_free(void *addr);

#endif
