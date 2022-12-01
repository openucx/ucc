/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_ASSERT_H
#define UCC_ASSERT_H

#include "ucc_compiler_def.h"

#if ENABLE_DEBUG == 1 || UCC_ENABLE_ASSERT == 1
#include <assert.h>
#define ucc_assert(_cond) ucc_assert_always(_cond)
#define ucc_assert_system(_cond) assert(_cond)
#else
#define ucc_assert(_cond)
#define ucc_assert_system(_cond)
#endif

BEGIN_C_DECLS

/**
 * Fail if _expression evaluates to 0
 */
#define ucc_assert_always(_expression)                                         \
    do {                                                                       \
        if (!ucc_likely(_expression)) {                                        \
            ucc_fatal_error_format(__FILE__, __LINE__, __FUNCTION__,           \
                                   "Assertion `%s' failed", #_expression);     \
        }                                                                      \
    } while (0)

/**
 * Generate a fatal error and stop the program.
 *
 * @param [in] file        Source file name
 * @param [in] line        Source line number
 * @param [in] function    Calling function name
 * @param [in] message_buf Error message buffer. Multi-line message is
 *                         supported.
 *
 * IMPORTANT NOTE: message_buf could be overridden by this function
 */
void ucc_fatal_error_message(const char *file, unsigned line,
                             const char *function, char *message_buf)
    UCS_F_NORETURN;

/**
 * Generate a fatal error and stop the program.
 *
 * @param [in] file        Source file name
 * @param [in] line        Source line number
 * @param [in] function    Calling function name
 * @param [in] format      Error message format string. Multi-line message is
 *                         supported.
 */
void ucc_fatal_error_format(const char *file, unsigned line,
                            const char *function, const char *format, ...)
    UCC_F_NORETURN;

END_C_DECLS

#endif
