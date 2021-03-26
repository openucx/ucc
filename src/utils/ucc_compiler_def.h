/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_COMPILER_DEF_H_
#define UCC_COMPILER_DEF_H_

#include "config.h"
#include "ucc/api/ucc_status.h"
#include <ucs/type/status.h>
#include <ucs/sys/string.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/debug/log_def.h>
#if ENABLE_DEBUG == 1
#include <assert.h>
#endif


#define ucc_offsetof      ucs_offsetof
#define ucc_container_of  ucs_container_of
#define ucc_derived_of    ucs_derived_of
#define ucc_strncpy_safe  ucs_strncpy_safe
#define ucc_snprintf_safe snprintf

typedef ucs_log_component_config_t ucc_log_component_config_t;

#define _UCC_PP_MAKE_STRING(x) #x
#define UCC_PP_MAKE_STRING(x)  _UCC_PP_MAKE_STRING(x)
#define UCC_PP_QUOTE UCS_PP_QUOTE
#define UCC_MASK     UCS_MASK

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

static inline ucs_status_t ucc_status_to_ucs_status(ucc_status_t status)
{
    switch (status) {
    case UCC_OK:
        return UCS_OK;
    case UCC_INPROGRESS:
        return UCS_INPROGRESS;
    case UCC_ERR_NO_MEMORY:
        return UCS_ERR_NO_MEMORY;
    case UCC_ERR_INVALID_PARAM:
        return UCS_ERR_INVALID_PARAM;
    case UCC_ERR_NO_RESOURCE:
        return UCS_ERR_NO_RESOURCE;
    default:
        break;
    }
    return UCS_ERR_NO_MESSAGE;
}

#if ENABLE_DEBUG == 1
#define ucc_assert(_cond) assert(_cond)
#else
#define ucc_assert(_cond)
#endif

#define UCC_CACHE_LINE_SIZE 128 //TODO detect it
#define ucc_for_each_bit ucs_for_each_bit
#endif
