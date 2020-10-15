/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_COMPILER_DEF_H_
#define UCC_COMPILER_DEF_H_

#include "config.h"
#include "api/ucc_status.h"
#include <ucs/type/status.h>
#include <ucs/sys/string.h>

#define ucc_offsetof ucs_offsetof
#define ucc_container_of ucs_container_of
#define ucc_strncpy_safe ucs_strncpy_safe

static inline ucc_status_t ucs_status_to_ucc_status(ucs_status_t status)
{
    switch (status) {
    case UCS_OK:
        return UCC_OK;
    case UCS_INPROGRESS:
        return UCC_INPROGRESS;
    case UCS_ERR_NO_MEMORY:
        return UCC_ERR_NO_MEMORY;
    case UCS_ERR_INVALID_PARAM:
        return UCC_ERR_INVALID_PARAM;
    case UCS_ERR_NO_RESOURCE:
        return UCC_ERR_NO_RESOURCE;
    default:
        break;
    }
    return UCC_ERR_NO_MESSAGE;
}
#endif
