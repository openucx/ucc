/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
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

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t) -1)
#endif

#define ucc_offsetof      ucs_offsetof
#define ucc_container_of  ucs_container_of
#define ucc_derived_of    ucs_derived_of
#define ucc_strncpy_safe  ucs_strncpy_safe
#define ucc_snprintf_safe snprintf
#define ucc_likely        ucs_likely
#define ucc_unlikely      ucs_unlikely
#define ucc_string_split  ucs_string_split

/*
 * Assertions which are checked in compile-time
 * In case of failure a compiler msg looks like this:
 * error: duplicate case value switch(0) {case 0:case (_cond):;}
 *
 * Usage: UCC_STATIC_ASSERT(condition)
 */
#define UCC_STATIC_ASSERT(_cond)                                               \
     switch(0) {case 0:case (_cond):;}

/* Maximal allocation size for on-stack buffers */
#define UCC_ALLOCA_MAX_SIZE  1200

/**
 * alloca which makes sure the size is small enough.
 */
 #define ucc_alloca(_size) \
 ({ \
     ucc_assert((_size) <= UCC_ALLOCA_MAX_SIZE); \
     alloca(_size); \
 })

/**
 * Prevent compiler from reordering instructions
 */
#define ucc_compiler_fence()       asm volatile(""::: "memory")

typedef int                        ucc_score_t;

#define _UCC_PP_MAKE_STRING(x) #x
#define UCC_PP_MAKE_STRING(x)  _UCC_PP_MAKE_STRING(x)
#define UCC_PP_QUOTE UCS_PP_QUOTE
#define UCC_EMPTY_STATEMENT {}

/* Packed structure */
#define UCC_S_PACKED             __attribute__((packed))

/**
 * suppress unaligned pointer warning
 */
#define ucc_unaligned_ptr(_ptr) ({void *_p = (void*)(_ptr); _p;})

/*
 * Enable compiler checks for printf-like formatting.
 *
 * @param fmtargN number of formatting argument
 * @param vargN   number of variadic argument
 */
 #define UCC_F_PRINTF(fmtargN, vargN) __attribute__((format(printf, fmtargN, vargN)))

/* A function which should not be optimized */
#if defined(HAVE_ATTRIBUTE_NOOPTIMIZE) && (HAVE_ATTRIBUTE_NOOPTIMIZE == 1)
#define UCC_F_NOOPTIMIZE __attribute__((optimize("O0")))
#else
#define UCC_F_NOOPTIMIZE
#endif

/* A function which does not return */
#define UCC_F_NORETURN __attribute__((noreturn))

#define UCC_COPY_PARAM_BY_FIELD(_dst, _src, _FIELD, _field)                    \
    do {                                                                       \
        if ((_src)->mask & (_FIELD)) {                                         \
            (_dst)->_field = (_src)->_field;                                   \
        }                                                                      \
    } while (0)

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

#define ucc_for_each_bit ucs_for_each_bit

#define UCC_CHECK_GOTO(_cmd, _label, _status)                                  \
    do {                                                                       \
        _status = (_cmd);                                                      \
        if (ucc_unlikely(_status != UCC_OK)) {                                 \
            goto _label;                                                       \
        }                                                                      \
    } while (0)


#if defined(__clang__)
    #define ucc_assume(x) __builtin_assume(x)
#elif defined(__NVCOMPILER)
    #define ucc_assume(x) __builtin_assume(x)
#elif defined(__GNUC__)
    #if (__GNUC__ >= 13)
        /* GCC 13+ has __attribute__((assume)) */
        #define ucc_assume(x) __attribute__((assume(x)))
    #else
        /* For older GCC versions, we can use __builtin_unreachable() as a fallback */
        #define ucc_assume(x) ((x) ? (void)0 : __builtin_unreachable())
    #endif
#else
    #define ucc_assume(x) do {} while (0)  /* No-op for unsupported compilers */
#endif

#endif
