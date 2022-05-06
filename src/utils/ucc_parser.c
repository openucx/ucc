/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucc_parser.h"
#include "ucc_malloc.h"
#include "ucc_log.h"

ucc_status_t ucc_config_names_array_merge(ucc_config_names_array_t *dst,
                                          const ucc_config_names_array_t *src)
{
    int i, n_new;

    n_new = 0;
    if (dst->count == 0) {
        return ucc_config_names_array_dup(dst, src);
    } else {
        for (i = 0; i < src->count; i++) {
            if (ucc_config_names_search(dst, src->names[i]) < 0) {
                /* found new entry in src which is not part of dst */
                n_new++;
            }
        }

        if (n_new) {
            dst->names =
                ucc_realloc(dst->names, (dst->count + n_new) * sizeof(char *),
                            "ucc_config_names_array");
            if (ucc_unlikely(!dst->names)) {
                return UCC_ERR_NO_MEMORY;
            }
            for (i = 0; i < src->count; i++) {
                if (ucc_config_names_search(dst, src->names[i]) < 0) {
                    dst->names[dst->count++] = strdup(src->names[i]);
                    if (ucc_unlikely(!dst->names[dst->count - 1])) {
                        ucc_error("failed to dup config_names_array entry");
                        return UCC_ERR_NO_MEMORY;
                    }
                }
            }
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_config_names_array_dup(ucc_config_names_array_t *dst,
                                        const ucc_config_names_array_t *src)
{
    int i;

    dst->names = ucc_malloc(sizeof(char*) * src->count, "ucc_config_names_array");
    if (!dst->names) {
        ucc_error("failed to allocate %zd bytes for ucc_config_names_array",
                  sizeof(char *) * src->count);
        return UCC_ERR_NO_MEMORY;
    }
    dst->count = src->count;
    for (i = 0; i < src->count; i++) {
        dst->names[i] = strdup(src->names[i]);
        if (!dst->names[i]) {
            ucc_error("failed to dup config_names_array entry");
            goto err;
        }
    }
    return UCC_OK;
err:
    for (i = i - 1; i >= 0; i--) {
        free(dst->names[i]);
    }
    return UCC_ERR_NO_MEMORY;
}

void ucc_config_names_array_free(ucc_config_names_array_t *array)
{
    int i;
    for (i = 0; i < array->count; i++) {
        free(array->names[i]);
    }
    ucc_free(array->names);
}

int ucc_config_names_search(const ucc_config_names_array_t *config_names,
                            const char *                    str)
{
    unsigned i;

    for (i = 0; i < config_names->count; ++i) {
        if (!strcmp(config_names->names[i], str)) {
           return i;
        }
    }

    return -1;
}

ucc_status_t ucc_config_allow_list_process(const ucc_config_allow_list_t * list,
                                           const ucc_config_names_array_t *all,
                                           ucc_config_names_list_t *       out)
{
    ucc_status_t status = UCC_OK;
    int          i;

    out->array.names = NULL;
    out->requested   = 0;

    switch (list->mode){
    case UCC_CONFIG_ALLOW_LIST_ALLOW:
        out->requested = 1;
        status = ucc_config_names_array_dup(&out->array, &list->array);
        break;
    case UCC_CONFIG_ALLOW_LIST_ALLOW_ALL:
        status = ucc_config_names_array_dup(&out->array, all);
        break;
    case UCC_CONFIG_ALLOW_LIST_NEGATE:
        out->array.count = 0;
        out->array.names = ucc_malloc(sizeof(char *) * all->count, "names");
        if (!out->array.names) {
            ucc_error("failed to allocate %zd bytes for names array",
                      sizeof(char *) * all->count);
            status = UCC_ERR_NO_MEMORY;
            break;
        }
        for (i = 0; i < all->count; i++) {
            if (ucc_config_names_search(&list->array, all->names[i]) < 0) {
                out->array.names[out->array.count++] = strdup(all->names[i]);
            }
        }
        break;
    default:
        ucc_assert(0);
    }
    return status;
}
