/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "ucc_string.h"
#include "ucc_malloc.h"
#include "ucc_log.h"
#include <ctype.h>
#include <ucs/sys/string.h>

char **ucc_str_split(const char *str, const char *delim)
{
    unsigned alloc_size = 8;
    unsigned size       = 0;
    char   **out;
    char    *str_copy, *token, *saveptr;
    int      i;
    out = ucc_malloc(alloc_size * sizeof(char *), "str_split");
    if (!out) {
        ucc_error("failed to allocate %zd bytes for str_split",
                  alloc_size * sizeof(char *));
        return NULL;
    }
    str_copy = strdup(str);
    if (!str_copy) {
        ucc_error("failed to duplicate string");
        goto error;
    }
    token = strtok_r(str_copy, delim, &saveptr);
    while (NULL != token) {
        out[size] = strdup(token);
        if (!out[size]) {
            ucc_error("failed to duplicate string");
            goto error;
        }
        size++;
        if (size == (alloc_size - 1)) { /* keep 1 for NULL mark */
            alloc_size *= 2;
            out = ucc_realloc(out, alloc_size * sizeof(char *), "str_split");
            if (!out) {
                ucc_error("failed to reallocate %zd bytes for str_split",
                          alloc_size * sizeof(char *));
                goto error;
            }
        }
        token = strtok_r(NULL, delim, &saveptr);
    }
    out[size] = NULL;
    ucc_free(str_copy);
    return out;
error:
    for (i = 0; i < size; i++) {
        ucc_free(out[i]);
    }
    ucc_free(out);
    ucc_free(str_copy);
    return NULL;
}

unsigned ucc_str_split_count(char **split)
{
    int count;
    if (NULL == split) {
        return 0;
    }
    count = 0;
    while (NULL != (*split)) {
        count++;
        split++;
    }
    return count;
}

void ucc_str_split_free(char **split)
{
    char **iter = split;
    if (NULL == split) {
        return;
    }
    while (NULL != (*iter)) {
        ucc_free(*iter);
        iter++;
    }
    free(split);
}

ucc_status_t ucc_str_is_number(const char *str)
{
    unsigned i, len;
    len = strlen(str);
    for (i = 0; i < len; i++) {
        if (!isdigit(str[i])) {
            return UCC_ERR_INVALID_PARAM;
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_str_to_memunits(const char *buf, void *dest)
{
    return ucs_status_to_ucc_status(ucs_str_to_memunits(buf, dest));
}

const char* ucc_strstr_last(const char* string, const char* pattern)
{
    const char *found = NULL;

    while ((string = strstr(string, pattern))) {
        found = string++;
    }
    return found;
}

ucc_status_t ucc_str_concat(const char *str1, const char *str2,
                            char **out)
{
    size_t len;
    char  *rst;

    len = strlen(str1) + strlen(str2) + 1;
    rst = ucc_malloc(len, "str_concat");
    if (!rst) {
        ucc_error("failed to allocate %zd bytes for concatenated string", len);
        return UCC_ERR_NO_MEMORY;
    }
    ucc_strncpy_safe(rst, str1, len);
    len -= strlen(str1);
    strncat(rst, str2, len);
    *out = rst;
    return UCC_OK;
}
