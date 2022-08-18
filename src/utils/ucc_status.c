/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc/api/ucc_status.h"
#include <stdio.h>

const char *ucc_status_string(ucc_status_t status)
{
    static char error_str[128] = {0};

    switch (status) {
    case UCC_OK:
        return "Success";
    case UCC_INPROGRESS:
        return "Operation in progress";
    case UCC_OPERATION_INITIALIZED:
        return "Operation initialized";
    case UCC_ERR_NOT_SUPPORTED:
        return "Operation is not supported";
    case UCC_ERR_NOT_IMPLEMENTED:
        return "Not implemented";
    case UCC_ERR_INVALID_PARAM:
        return "Invalid parameter";
    case UCC_ERR_NO_MEMORY:
        return "Out of memory";
    case UCC_ERR_NO_RESOURCE:
        return "Resources are not available for the operation";
    case UCC_ERR_NO_MESSAGE:
        return "Unhandled error";
    case UCC_ERR_NOT_FOUND:
        return "Not found";
    case UCC_ERR_TIMED_OUT:
        return "Timeout expired";
    default:
        snprintf(error_str, sizeof(error_str) - 1, "Unknown error %d", status);
        return error_str;
    };
}
