/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_STRING_H_
#define UCC_STRING_H_
#include "config.h"
#include "ucc/api/ucc_status.h"

#define      ucc_memunits_range_str ucs_memunits_range_str

char**       ucc_str_split(const char *str, const char *delim);

unsigned     ucc_str_split_count(char **split);

void         ucc_str_split_free(char **split);

ucc_status_t ucc_str_is_number(const char *str);

ucc_status_t ucc_str_to_memunits(const char *buf, void *dest);

/* Finds last occurence of pattern in string */
const char*  ucc_strstr_last(const char* string, const char* pattern);

/* Concatenates 2 strings. The space allocated for "out" must be
   released with ucc_free. */
ucc_status_t ucc_str_concat(const char *str1, const char *str2,
                            char **out);
#endif
