/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucc_assert.h"
#include "ucc_log.h"
#include <stdio.h>

/**
 * Get pointer to file name in path, same as basename but do not
 * modify source string.
 *
 * @param path Path to parse.
 *
 * @return file name
 */
static const char* ucc_basename(const char *path)
{
    const char *name = strrchr(path, '/');

    return (name == NULL) ? path : name + 1;
}

void ucc_fatal_error_message(const char *file, unsigned line,
                             const char *function, char *message_buf) //NOLINT: function is unused
{
    char *message_line, *save_ptr = NULL;

    ucc_log_flush();

    message_line = (message_buf == NULL) ? NULL :
                   strtok_r(message_buf, "\n", &save_ptr);
    while (message_line != NULL) {
        ucc_log_fatal_error("%13s:%-4u %s", ucc_basename(file), line, message_line);
        message_line = strtok_r(NULL, "\n", &save_ptr);
    }

    abort();
}

void ucc_fatal_error_format(const char *file, unsigned line,
                            const char *function, const char *format, ...)
{
    const size_t buffer_size = ucc_log_get_buffer_size();
    char *buffer;
    va_list ap;

    buffer = alloca(buffer_size);
    va_start(ap, format);
    //NOLINTNEXTLINE
    vsnprintf(buffer, buffer_size, format, ap);
    va_end(ap);

    ucc_fatal_error_message(file, line, function, buffer);
}
