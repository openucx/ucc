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

#endif
