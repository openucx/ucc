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
    int i, j, n_new;

    n_new = 0;
    if (dst->count == 0) {
        return ucc_config_names_array_dup(dst, src);
    } else {
        for (i = 0; i < src->count; i++) {
            for (j = 0; j < dst->count; j++) {
                if (0 != strcmp(src->names[i], dst->names[j])) {
                    break;
                }
            }
            if (j < dst->count) {
                /* found new entry in src which is not part of dst */
                n_new++;
            }
        }

        if (n_new) {
            dst->count += n_new;
            dst->names = ucc_realloc(dst->names, dst->count * sizeof(char *),
                                     "ucc_config_names_array");
            if (ucc_unlikely(!dst->names)) {
                return UCC_ERR_NO_MEMORY;
            }
            for (i = 0; i < src->count; i++) {
                for (j = 0; j < dst->count; j++) {
                    if (0 != strcmp(src->names[i], dst->names[j])) {
                        dst->names[dst->count - n_new] = strdup(src->names[i]);
                        if (ucc_unlikely(!dst->names[dst->count - n_new])) {
                            ucc_error("failed to dup config_names_array entry");
                            return UCC_ERR_NO_MEMORY;
                        }
                        n_new--;
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


int ucc_config_names_search(ucc_config_names_array_t *config_names,
                            const char *str) {
    unsigned i;

    for (i = 0; i < config_names->count; ++i) {
        if (!strcmp(config_names->names[i], str)) {
           return i;
        }
    }

    return -1;
}
