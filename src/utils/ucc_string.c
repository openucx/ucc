/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "ucc_string.h"
#include "ucc_malloc.h"
#include "ucc_log.h"
#include "ucc_coll_utils.h"
#include <ctype.h>
#include <ucs/sys/string.h>
#include <utils/ucc_sys.h>
#include <time.h>
#include <unistd.h>

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
    if (out) {
        for (i = 0; i < size; i++) {
            ucc_free(out[i]);
        }
        ucc_free(out);
    }
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

ucc_status_t ucc_str_concat_n(const char *strs[], int n, char **out)
{
    size_t len = 1;
    char  *rst;
    int    i;

    for (i = 0; i < n; i++) {
        len += strlen(strs[i]);
    }

    rst = ucc_malloc(len, "str_concat");
    if (!rst) {
        ucc_error("failed to allocate %zd bytes for concatenated string", len);
        return UCC_ERR_NO_MEMORY;
    }
    ucc_strncpy_safe(rst, strs[0], len);

    for (i = 1; i < n; i++) {
        len -= strlen(strs[i - 1]);
        strncat(rst, strs[i], len);
    }
    *out = rst;
    return UCC_OK;
}

ucc_status_t ucc_str_concat(const char *str1, const char *str2, char **out)
{
    const char *strs[2] = {str1, str2};

    return ucc_str_concat_n(strs, 2, out);
}

ucc_status_t ucc_str_to_memunits_range(const char *str, size_t *start,
                                       size_t *end)
{
    ucc_status_t status = UCC_OK;
    char       **munits;
    unsigned     n_munits;

    munits = ucc_str_split(str, "-");
    if (!munits) {
        return UCC_ERR_NO_MEMORY;
    }
    n_munits = ucc_str_split_count(munits);
    if (n_munits != 2 || UCC_OK != ucc_str_to_memunits(munits[0], start) ||
        UCC_OK != ucc_str_to_memunits(munits[1], end)) {
        status = UCC_ERR_INVALID_PARAM;
    }

    ucc_str_split_free(munits);
    return status;
}

ucc_status_t ucc_str_to_mtype_map(const char *str, const char *delim,
                                  uint32_t *mt_map)
{
    ucc_status_t      status = UCC_OK;
    char **           tokens;
    unsigned          i, n_tokens;
    ucc_memory_type_t t;

    *mt_map = 0;
    tokens  = ucc_str_split(str, delim);
    if (!tokens) {
        return UCC_ERR_NO_MEMORY;
    }
    n_tokens = ucc_str_split_count(tokens);

    for (i = 0; i < n_tokens; i++) {
        t = ucc_mem_type_from_str(tokens[i]);
        if (t == UCC_MEMORY_TYPE_LAST) {
            /* entry does not match any memory type name */
            status = UCC_ERR_INVALID_PARAM;
            goto out;
        }
        *mt_map |= UCC_BIT(t);
    }
out:
    ucc_str_split_free(tokens);
    return status;
}

void ucc_mtype_map_to_str(uint32_t mt_map, const char *delim,
                          char *buf, size_t max)
{
    int    i;
    size_t last;

    for (i = 0; i < UCC_MEMORY_TYPE_LAST; i++) {
        if (UCC_BIT(i) & mt_map) {
            ucc_snprintf_safe(buf, max, "%s%s",
                              ucc_mem_type_str((ucc_memory_type_t)i), delim);
            last = strlen(buf);
            if (max - last -1 <= 0) {
                /* no more space in buf for next range*/
                return;
            }

            max -= last;
            buf += last;
        }
    }
    /* remove last delimiter */
    buf -= strlen(delim);
    *buf = '\0';
}

ssize_t ucc_string_find_in_list(const char *str, const char **string_list,
                                int case_sensitive)
{
    size_t i;

    for (i = 0; string_list[i] != NULL; ++i) {
        if ((case_sensitive && (strcmp(string_list[i], str) == 0)) ||
            (!case_sensitive && (strcasecmp(string_list[i], str) == 0))) {
            return i;
        }
    }

    return -1;
}

void ucc_strncpy_zero(char *dest, const char *src, size_t max)
{
    if (max) {
        strncpy(dest, src, max - 1);
        dest[max - 1] = '\0';
    }
}

void ucc_snprintf_zero(char *buf, size_t size, const char *fmt, ...)
{
    va_list ap;

    memset(buf, 0, size);
    va_start(ap, fmt);
    // NOLINTNEXTLINE(clang-analyzer-valist.Uninitialized)
    vsnprintf(buf, size, fmt, ap);
    va_end(ap);
}

ucc_status_t ucc_string_alloc_path_buffer(char **buffer_p, const char *name)
{
    char *temp_buffer = ucc_malloc(PATH_MAX, name);

    if (temp_buffer == NULL) {
        ucc_error("failed to allocate memory for %s", name);
        return UCC_ERR_NO_MEMORY;
    }

    *buffer_p = temp_buffer;
    return UCC_OK;
}

void ucc_fill_filename_template(const char *tmpl, char *buf, size_t max)
{
    char *p, *end;
    const char *pf, *pp;
    size_t length;
    time_t t;

    p = buf;
    end = buf + max - 1;
    *end = 0;
    pf = tmpl;
    while (*pf != 0 && p < end) {
        pp = strchr(pf, '%');
        if (pp == NULL) {
            strncpy(p, pf, end - p);
            p = end;
            break;
        }

        length = ucs_min(pp - pf, end - p);
        strncpy(p, pf, length);
        p += length;
        /* default length of the modifier (e.g. %p) */
        length = 2;

        switch (*(pp + 1)) {
        case 'p':
            snprintf(p, end - p, "%d", getpid());
            break;
        case 'h':
            snprintf(p, end - p, "%s", ucc_get_host_name());
            break;
        case 'c':
            snprintf(p, end - p, "%02d", ucc_get_first_cpu());
            break;
        case 't':
            t = time(NULL);
            strftime(p, end - p, "%Y-%m-%d-%H-%M-%S", localtime(&t));
            break;
        case 'u':
            snprintf(p, end - p, "%s", ucc_basename(ucc_get_user_name()));
            break;
        case 'e':
            snprintf(p, end - p, "%s", ucc_basename(ucc_get_exe()));
            break;
        case 'i':
            snprintf(p, end - p, "%u", geteuid());
            break;
        default:
            *(p++) = *pp;
            length = 1;
            break;
        }

        pf = pp + length;
        p += strlen(p);
    }
    *p = 0;
}

/* NOLINTNEXTLINE */
char *ucc_strdup(const char *src, const char *name)
{
    char *str = strdup(src);
    return str;
}

const char* ucc_basename(const char *path)
{
    const char *name = strrchr(path, '/');

    return (name == NULL) ? path : name + 1;
}
