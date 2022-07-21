/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Arm, Ltd. 2021. ALL RIGHTS RESERVED.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_COMPILER_DEF_H_
#define UCC_COMPILER_DEF_H_

#include "config.h"
#include "ucc/api/ucc_status.h"
#include <ucs/type/status.h> /* Delete when last use of ucs_status_t is gone */
#include <ucs/sys/string.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/debug/log_def.h>
#if ENABLE_DEBUG == 1
#include <assert.h>
#endif

/* Note: Place "@file <file name>.h" after BEGIN_C_DECS
 * to avoid bugs in a documentation */
#ifdef __cplusplus
# define BEGIN_C_DECLS  extern "C" {
# define END_C_DECLS    }
#else
# define BEGIN_C_DECLS
# define END_C_DECLS
#endif

/*
 * Assertions which are checked in compile-time
 *
 * Usage: UCC_STATIC_ASSERT(condition)
 */
#define UCC_STATIC_ASSERT(_cond) \
     switch(0) {case 0:case (_cond):;}

/* Maximal allocation size for on-stack buffers */
#define UCC_ALLOCA_MAX_SIZE  1200

/* Aliasing structure */
#define UCC_S_MAY_ALIAS __attribute__((may_alias))

/* A function without side effects */
#define UCC_F_PURE   __attribute__((pure))

/* A function which does not return */
#define UCC_F_NORETURN __attribute__((noreturn))

/* Packed structure */
#define UCC_S_PACKED             __attribute__((packed))

/* Avoid inlining the function */
#define UCC_F_NOINLINE __attribute__ ((noinline))

/* Shared library constructor and destructor */
#define UCC_F_CTOR __attribute__((constructor))
#define UCC_F_DTOR __attribute__((destructor))

/* Silence "defined but not used" error for static function */
#define UCC_F_MAYBE_UNUSED __attribute__((used))

/* Non-null return */
#define UCC_F_NON_NULL __attribute__((nonnull))

/* Always inline the function */
#ifdef __GNUC__
#define UCC_F_ALWAYS_INLINE      inline __attribute__ ((always_inline))
#else
#define UCC_F_ALWAYS_INLINE      inline
#endif

/* Silence "uninitialized variable" for stupid compilers (gcc 4.1)
 * which can't optimize properly.
 */
#if (((__GNUC__ == 4) && (__GNUC_MINOR__ == 1)) || !defined(__OPTIMIZE__))
#  define UCC_V_INITIALIZED(_v)  (_v = (ucc_typeof(_v))0)
#else
#  define UCC_V_INITIALIZED(_v)  ((void)0)
#endif

/* The i-th bit */
#define UCC_BIT(i)               (1ul << (i))

/* Mask of bits 0..i-1 */
#define UCC_MASK(i)              (UCC_BIT(i) - 1)

/*
 * Enable compiler checks for printf-like formatting.
 *
 * @param fmtargN number of formatting argument
 * @param vargN   number of variadic argument
 */
#define UCC_F_PRINTF(fmtargN, vargN) __attribute__((format(printf, fmtargN, vargN)))

/* Unused variable */
#define UCC_V_UNUSED __attribute__((unused))

/* Aligned variable */
#define UCC_V_ALIGNED(_align) __attribute__((aligned(_align)))

/* Used for labels */
#define UCC_EMPTY_STATEMENT {}

/* Helper macro for address arithmetic in bytes */
#define UCC_PTR_BYTE_OFFSET(_ptr, _offset) \
    ((void *)((intptr_t)(_ptr) + (intptr_t)(_offset)))

/* Helper macro to calculate an address with offset equal to size of _type */
#define UCC_PTR_TYPE_OFFSET(_ptr, _type) \
    ((void *)((ucc_typeof(_type) *)(_ptr) + 1))

/* Helper macro to calculate ptr difference (_end - _start) */
#define UCC_PTR_BYTE_DIFF(_start, _end) \
    ((ptrdiff_t)((uintptr_t)(_end) - (uintptr_t)(_start)))


/**
 * Size of statically-declared array
 */
#define ucc_static_array_size(_array) \
    (sizeof(_array) / sizeof((_array)[0]))


/**
 * @return Offset of _member in _type. _type is a structure type.
 */
#define ucc_offsetof(_type, _member) \
    ((unsigned long)&( ((_type*)0)->_member ))


/**
 * Get a pointer to a struct containing a member.
 *
 * @param _ptr     Pointer to the member.
 * @param _type    Container type.
 * @param _member  Element member inside the container.

 * @return Address of the container structure.
 */
#define ucc_container_of(_ptr, _type, _member) \
    ( (_type*)( (char*)(void*)(_ptr) - ucc_offsetof(_type, _member) )  )


/**
 * Get the type of a structure or variable.
 *
 * @param _type  Return the type of this argument.
 *
 * @return The type of the given argument.
 */
#define ucc_typeof(_type) \
    __typeof__(_type)


/**
 * @return Address of a derived structure. It must have a "super" member at offset 0.
 * NOTE: we use the built-in offsetof here because we can't use ucc_offsetof() in
 *       a constant expression.
 */
#define ucc_derived_of(_ptr, _type) \
    ({\
        UCC_STATIC_ASSERT(offsetof(_type, super) == 0) \
        ucc_container_of(_ptr, _type, super); \
    })

/**
 * @param _type   Structure type.
 * @param _field  Field of structure.
 *
 * @return Size of _field in _type.
 */
#define ucc_field_sizeof(_type, _field) \
    sizeof(((_type*)0)->_field)

/**
 * @param _type   Structure type.
 * @param _field  Field of structure.
 *
 * @return Type of _field in _type.
 */
#define ucc_field_type(_type, _field) \
    ucc_typeof(((_type*)0)->_field)

/**
 * Prevent compiler from reordering instructions
 */
#define ucc_compiler_fence()       asm volatile(""::: "memory")

/**
 * Prefetch cache line
 */
#define ucc_prefetch(p)            __builtin_prefetch(p)

/* Branch prediction */
#define ucc_likely(x)              __builtin_expect(x, 1)
#define ucc_unlikely(x)            __builtin_expect(x, 0)

/* Check if an expression is a compile-time constant */
#define ucc_is_constant(expr)      __builtin_constant_p(expr)

/*
 * Define code which runs at global constructor phase
 */
#define UCC_STATIC_INIT \
    static void UCC_F_CTOR UCC_PP_APPEND_UNIQUE_ID(ucc_initializer_ctor)()

/*
 * Define code which runs at global destructor phase
 */
#define UCC_STATIC_CLEANUP \
    static void UCC_F_DTOR UCC_PP_APPEND_UNIQUE_ID(ucc_initializer_dtor)()

/*
 * Check if the two types are the same
 */
#define ucs_same_type(_type1, _type2) \
    __builtin_types_compatible_p(_type1, _type2)

#define ucc_strncpy_safe  ucs_strncpy_safe /* TODO - Remove this when converted */
#define ucc_snprintf_safe snprintf

/**
 * Prevent compiler from reordering instructions
 */
#define ucc_compiler_fence()       asm volatile(""::: "memory")

typedef ucs_log_component_config_t ucc_log_component_config_t;
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

/* A function which should not be optimized */
#if defined(HAVE_ATTRIBUTE_NOOPTIMIZE) && (HAVE_ATTRIBUTE_NOOPTIMIZE == 1)
#define UCC_F_NOOPTIMIZE __attribute__((optimize("O0")))
#else
#define UCC_F_NOOPTIMIZE
#endif

#define UCC_COPY_PARAM_BY_FIELD(_dst, _src, _FIELD, _field)                    \
    do {                                                                       \
        if ((_src)->mask & (_FIELD)) {                                         \
            (_dst)->_field = (_src)->_field;                                   \
        }                                                                      \
    } while (0)

/* TODO - Delete when ucs_status_t is deleted */
static inline ucc_status_t ucs_status_to_ucc_status(ucs_status_t status)
{
    switch (status) {
    case UCS_OK:
        return UCC_OK;
    case UCS_INPROGRESS:
        return UCC_INPROGRESS;
    case UCS_ERR_NOT_IMPLEMENTED:
        return UCC_ERR_NOT_IMPLEMENTED;
    case UCS_ERR_INVALID_PARAM:
        return UCC_ERR_INVALID_PARAM;
    case UCS_ERR_NO_MEMORY:
        return UCC_ERR_NO_MEMORY;
    case UCS_ERR_NO_RESOURCE:
        return UCC_ERR_NO_RESOURCE;
    case UCS_ERR_NO_MESSAGE:
        return UCC_ERR_NO_MESSAGE;
    case UCS_ERR_TIMED_OUT:
        return UCC_ERR_TIMED_OUT;
    case UCS_ERR_IO_ERROR:
        return UCC_ERR_IO_ERROR;
    case UCS_ERR_UNREACHABLE:
        return UCC_ERR_UNREACHABLE;
    case UCS_ERR_INVALID_ADDR:
        return UCC_ERR_INVALID_ADDR;
    case UCS_ERR_MESSAGE_TRUNCATED:
        return UCC_ERR_MESSAGE_TRUNCATED;
    case UCS_ERR_NO_PROGRESS:
        return UCC_ERR_NO_PROGRESS;
    case UCS_ERR_BUFFER_TOO_SMALL:
        return UCC_ERR_BUFFER_TOO_SMALL;
    case UCS_ERR_NO_ELEM:
        return UCC_ERR_NO_ELEM;
    case UCS_ERR_SOME_CONNECTS_FAILED:
        return UCC_ERR_SOME_CONNECTS_FAILED;
    case UCS_ERR_NO_DEVICE:
        return UCC_ERR_NO_DEVICE;
    case UCS_ERR_BUSY:
        return UCC_ERR_BUSY;
    case UCS_ERR_CANCELED:
        return UCC_ERR_CANCELED;
    case UCS_ERR_SHMEM_SEGMENT:
        return UCC_ERR_SHMEM_SEGMENT;
    case UCS_ERR_ALREADY_EXISTS:
        return UCC_ERR_ALREADY_EXISTS;
    case UCS_ERR_OUT_OF_RANGE:
        return UCC_ERR_OUT_OF_RANGE;
    case UCS_ERR_EXCEEDS_LIMIT:
        return UCC_ERR_EXCEEDS_LIMIT;
    case UCS_ERR_UNSUPPORTED:
        return UCC_ERR_UNSUPPORTED;
    case UCS_ERR_REJECTED:
        return UCC_ERR_REJECTED;
    case UCS_ERR_NOT_CONNECTED:
        return UCC_ERR_NOT_CONNECTED;
    case UCS_ERR_CONNECTION_RESET:
        return UCC_ERR_CONNECTION_RESET;
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
    case UCC_ERR_NOT_IMPLEMENTED:
        return UCS_ERR_NOT_IMPLEMENTED;
    case UCC_ERR_INVALID_PARAM:
        return UCS_ERR_INVALID_PARAM;
    case UCC_ERR_NO_MEMORY:
        return UCS_ERR_NO_MEMORY;
    case UCC_ERR_NO_RESOURCE:
        return UCS_ERR_NO_RESOURCE;
    case UCC_ERR_NO_MESSAGE:
        return UCS_ERR_NO_MESSAGE;
    case UCC_ERR_TIMED_OUT:
        return UCS_ERR_TIMED_OUT;
    case UCC_ERR_IO_ERROR:
        return UCS_ERR_IO_ERROR;
    case UCC_ERR_UNREACHABLE:
        return UCS_ERR_UNREACHABLE;
    case UCC_ERR_INVALID_ADDR:
        return UCS_ERR_INVALID_ADDR;
    case UCC_ERR_MESSAGE_TRUNCATED:
        return UCS_ERR_MESSAGE_TRUNCATED;
    case UCC_ERR_NO_PROGRESS:
        return UCS_ERR_NO_PROGRESS;
    case UCC_ERR_BUFFER_TOO_SMALL:
        return UCS_ERR_BUFFER_TOO_SMALL;
    case UCC_ERR_NO_ELEM:
        return UCS_ERR_NO_ELEM;
    case UCC_ERR_SOME_CONNECTS_FAILED:
        return UCS_ERR_SOME_CONNECTS_FAILED;
    case UCC_ERR_NO_DEVICE:
        return UCS_ERR_NO_DEVICE;
    case UCC_ERR_BUSY:
        return UCS_ERR_BUSY;
    case UCC_ERR_CANCELED:
        return UCS_ERR_CANCELED;
    case UCC_ERR_SHMEM_SEGMENT:
        return UCS_ERR_SHMEM_SEGMENT;
    case UCC_ERR_ALREADY_EXISTS:
        return UCS_ERR_ALREADY_EXISTS;
    case UCC_ERR_OUT_OF_RANGE:
        return UCS_ERR_OUT_OF_RANGE;
    case UCC_ERR_EXCEEDS_LIMIT:
        return UCS_ERR_EXCEEDS_LIMIT;
    case UCC_ERR_UNSUPPORTED:
        return UCS_ERR_UNSUPPORTED;
    case UCC_ERR_REJECTED:
        return UCS_ERR_REJECTED;
    case UCC_ERR_NOT_CONNECTED:
        return UCS_ERR_NOT_CONNECTED;
    case UCC_ERR_CONNECTION_RESET:
        return UCS_ERR_CONNECTION_RESET;
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

#define ucc_for_each_bit ucs_for_each_bit
#endif
