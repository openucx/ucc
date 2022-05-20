/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_STRING_H_
#define UCC_STRING_H_
#include "config.h"
#include "ucc/api/ucc_status.h"

char**       ucc_str_split(const char *str, const char *delim);

unsigned     ucc_str_split_count(char **split);

void         ucc_str_split_free(char **split);

ucc_status_t ucc_str_is_number(const char *str);

ucc_status_t ucc_str_to_memunits(const char *buf, void *dest);

/* Finds last occurence of pattern in string */
const char*  ucc_strstr_last(const char* string, const char* pattern);
#define      ucc_memunits_range_str ucs_memunits_range_str

#endif
