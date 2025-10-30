/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_STRING_H_
#define UCC_STRING_H_

#include "config.h"
#include "ucc/api/ucc_def.h"
#include <sys/types.h>
#include <utils/ucc_compiler_def.h>

#define      ucc_memunits_range_str ucs_memunits_range_str

char** ucc_str_split(const char *str, const char *delim);

unsigned ucc_str_split_count(char **split);

void ucc_str_split_free(char **split);

ucc_status_t ucc_str_is_number(const char *str);

ucc_status_t ucc_str_to_memunits(const char *buf, void *dest);

/* Finds last occurence of pattern in string */
const char*  ucc_strstr_last(const char* string, const char* pattern);

/* Concatenates 2 strings. The space allocated for "out" must be
   released with ucc_free. */
ucc_status_t ucc_str_concat(const char *str1, const char *str2,
                            char **out);

ucc_status_t ucc_str_concat_n(const char *strs[], int n, char **out);

ucc_status_t ucc_str_to_mtype_map(const char *str, const char* delim,
                                  uint32_t *mt_map);

void ucc_mtype_map_to_str(uint32_t mt_map, const char *delim,
                          char *buf, size_t max);

ucc_status_t ucc_str_to_memunits_range(const char *str, size_t *start,
                                       size_t *end);

/**
 * Find a string in a NULL-terminated array of strings.
 *
 * @param str          String to search for.
 * @param string_list  NULL-terminated array of strings.
 * @param case_sensitive Whether to perform case sensitive search.
 *
 * @return Index of the string in the array, or -1 if not found.
 */
ssize_t ucc_string_find_in_list(const char *str, const char **string_list,
                                int case_sensitive);

                                /**
 * Same as strncpy(), but guarantee that the last char in the buffer is '\0'.
 */
void ucc_strncpy_zero(char *dest, const char *src, size_t max);

/**
 * Format a string to a buffer of given size, and fill the rest of the buffer
 * with '\0'. Also, guarantee that the last char in the buffer is '\0'.
 *
 * @param buf  Buffer to format the string to.
 * @param size Buffer size.
 * @param fmt  Format string.
 */
void ucc_snprintf_zero(char *buf, size_t size, const char *fmt, ...)
   UCC_F_PRINTF(3, 4);

/**
 * Allocates a path buffer of size PATH_MAX.
 *
 * @param buffer_p Pointer to the buffer.
 *                 The buffer is allocated and should be released by the caller.
 * @param name     Name of the buffer for logging.
 *
 * @return UCC_OK on success, UCC_ERR_NO_MEMORY on failure.
 */
ucc_status_t ucc_string_alloc_path_buffer(char **buffer_p, const char *name);

/**
 * Fill a filename template. The following values in the string are replaced:
 *  %p - replaced by process id
 *  %h - replaced by host name
 *  %c - replaced by the first CPU we are bound to
 *  %t - replaced by local time
 *  %u - replaced by user name
 *  %e - replaced by executable basename
 *  %i - replaced by user id
 *
 * @param tmpl   File name template (possibly containing formatting sequences)
 * @param buf    Filled with resulting file name
 * @param max    Maximal size of destination buffer.
 */
void ucc_fill_filename_template(const char *tmpl, char *buf, size_t max);

/**
 * Get pointer to file name in path, same as basename but do not
 * modify source string.
 *
 * @param path Path to parse.
 *
 * @return file name
 */
 const char* ucc_basename(const char *path);

char *ucc_strdup(const char *src, const char *name);

#endif
