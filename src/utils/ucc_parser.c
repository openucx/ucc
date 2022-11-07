/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_parser.h"
#include "ucc_malloc.h"
#include "ucc_log.h"
#include "khash.h"
#include "schedule/ucc_schedule.h"
#include "schedule/ucc_schedule_pipelined.h"
#include "utils/ucc_string.h"

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

KHASH_MAP_INIT_STR(ucc_cfg_file, char *);

typedef struct ucc_file_config {
    khash_t(ucc_cfg_file) vars;
} ucc_file_config_t;

static int ucc_file_parse_handler(void *arg, const char *section, //NOLINT
                                  const char *name, const char *value)
{
    ucc_file_config_t *cfg      = arg;
    khash_t(ucc_cfg_file) *vars = &cfg->vars;
    khiter_t iter;
    int      result;
    char    *dup;

    if (!name) {
        return 1;
    }
    if (NULL != getenv(name)) {
        /* variable is set in env, skip it.
           Env gets precedence over file */
        ;
        return 1;
    }
    if (strlen(name) < 4 || NULL == strstr(name, "UCC_")) {
        ucc_warn("incorrect parameter name %s "
                 "(param name should start with UCC_ or <PREFIX>_UCC_)",
                 name);
        return 0;
    }

    iter = kh_get(ucc_cfg_file, vars, name);
    if (iter != kh_end(vars)) {
        ucc_warn("found duplicate '%s' in config file", name);
        return 0;
    } else {
        dup = strdup(name);
        if (!dup) {
            ucc_error("failed to dup str for kh_put");
            return 0;
        }
        iter = kh_put(ucc_cfg_file, vars, dup, &result);
        if (result == UCS_KH_PUT_FAILED) {
            ucc_free(dup);
            ucc_error("inserting '%s' to config map failed", name);
            return 0;
        }
    }
    dup = strdup(value);
    if (!dup) {
        ucc_error("failed to dup str for kh_val");
        return 0;
    }
    kh_val(vars, iter) = dup; //NOLINT
    return 1;
}

ucc_status_t ucc_parse_file_config(const char *        filename,
                                   ucc_file_config_t **cfg_p)
{
    ucc_file_config_t *cfg;
    int                ret;
    ucc_status_t       status;

    cfg = ucc_calloc(1, sizeof(*cfg), "file_cfg");
    if (!cfg) {
        ucc_error("failed to allocate %zd bytes for file config", sizeof(*cfg));
        return UCC_ERR_NO_MEMORY;
    }
    kh_init_inplace(ucc_cfg_file, &cfg->vars);
    ret = ini_parse(filename, ucc_file_parse_handler, cfg);
    if (-1 == ret) {
        /* according to ucs/ini.h -1 means error in
           file open */
        status = UCC_ERR_NOT_FOUND;
        goto out;
    } else if (ret) {
        ucc_error("failed to parse config file %s", filename);
        status = UCC_ERR_INVALID_PARAM;
        goto out;
    }

    *cfg_p = cfg;
    return UCC_OK;
out:
    ucc_free(cfg);
    return status;
}

void ucc_release_file_config(ucc_file_config_t *cfg)
{
    const char *key;
    char *      value;

    kh_foreach(&cfg->vars, key, value, {
        ucc_free((void *)key);
        ucc_free(value);
    }) kh_destroy_inplace(ucc_cfg_file, &cfg->vars);
    ucc_free(cfg);
}

static const char *ucc_file_config_get(ucc_file_config_t *cfg,
                                       const char *       var_name)
{
    khiter_t iter;
    khash_t(ucc_cfg_file) *vars = &cfg->vars;

    iter = kh_get(ucc_cfg_file, vars, var_name);
    if (iter == kh_end(vars)) {
        /* variable not found in prefix hash*/
        return NULL;
    }
    return kh_val(vars, iter);
}

static ucc_status_t ucc_apply_file_cfg_value(void *              opts,
                                             ucc_config_field_t *fields,
                                             const char *        base_prefix,
                                             const char *component_prefix,
                                             const char *name)
{
    char        var[512];
    size_t      left = sizeof(var);
    const char *base_prefix_var;
    const char *cfg_value;

    ucc_strncpy_safe(var, base_prefix, left);
    left -= strlen(base_prefix);
    strncat(var, component_prefix, left);
    left -= strlen(component_prefix);
    strncat(var, name, left);

    base_prefix_var = strstr(var, "UCC_");
    cfg_value =
        ucc_file_config_get(ucc_global_config.file_cfg, base_prefix_var);
    if (cfg_value) {
        return ucc_config_parser_set_value(opts, fields, name, cfg_value);
    };

    if (base_prefix_var != var) {
        cfg_value = ucc_file_config_get(ucc_global_config.file_cfg, var);
        if (cfg_value) {
            return ucc_config_parser_set_value(opts, fields, name, cfg_value);
        }
    }

    return UCC_ERR_NOT_FOUND;
}

static ucc_status_t ucc_apply_file_cfg(void *opts, ucc_config_field_t *fields,
                                       const char *env_prefix,
                                       const char *component_prefix)
{
    ucc_status_t status = UCC_OK;

    while (fields->name != NULL) {
        if (strlen(fields->name) == 0) {
            status = ucc_apply_file_cfg(
                opts, (ucc_config_field_t *)fields->parser.arg, env_prefix,
                component_prefix);
            if (UCC_OK != status) {
                return status;
            }
            fields++;
            continue;
        }
        status = ucc_apply_file_cfg_value(
            opts, fields, env_prefix, component_prefix ? component_prefix : "",
            fields->name);
        if (status == UCC_ERR_NOT_FOUND && component_prefix) {
            status = ucc_apply_file_cfg_value(opts, fields, env_prefix, "",
                                              fields->name);
        }
        if (status != UCC_OK && status != UCC_ERR_NOT_FOUND) {
            return status;
        }

        fields++;
    }
    return UCC_OK;
}

ucc_status_t ucc_config_parser_fill_opts(void *opts, ucs_config_global_list_entry_t *entry,
                                         const char *env_prefix, int ignore_errors)
{
    ucs_status_t ucs_status;
    ucc_status_t status;

#if UCS_HAVE_CONFIG_GLOBAL_LIST_ENTRY_FLAGS
    ucs_status = ucs_config_parser_fill_opts(opts, entry, env_prefix,
                                             ignore_errors);
#else
    ucs_status = ucs_config_parser_fill_opts(opts, entry->table, env_prefix,
                                             entry->prefix, 0);
#endif

    status     = ucs_status_to_ucc_status(ucs_status);
    if (UCC_OK == status && ucc_global_config.file_cfg) {
        status = ucc_apply_file_cfg(opts, entry->table, env_prefix,
                                    entry->prefix);
    }

    return status;
}

void ucc_config_parser_print_all_opts(FILE *stream, const char *prefix,
                                      ucc_config_print_flags_t flags,
                                      ucc_list_link_t *        config_list)
{
    ucc_config_global_list_entry_t *entry;
    ucs_config_print_flags_t        ucs_flags;
    ucc_status_t                    status;
    char                            title[64];
    void *                          opts;

    ucs_flags = ucc_print_flags_to_ucs_print_flags(flags);
    ucc_list_for_each(entry, config_list, list)
    {
        if ((entry->table == NULL) || (entry->table[0].name == NULL)) {
            /* don't print title for an empty configuration table */
            continue;
        }

        opts = ucc_malloc(entry->size, "tmp_opts");
        if (opts == NULL) {
            ucc_error("could not allocate configuration of size %zu",
                      entry->size);
            continue;
        }

        status = ucc_config_parser_fill_opts(opts, entry, prefix, 0);
        if (status != UCC_OK) {
            ucc_free(opts);
            continue;
        }

        snprintf(title, sizeof(title), "%s configuration", entry->name);
        ucs_config_parser_print_opts(stream, title, opts, entry->table,
                                     entry->prefix, prefix, ucs_flags);

        ucs_config_parser_release_opts(opts, entry->table);
        ucc_free(opts);
    }
}

#define UCC_UUNITS_AUTO    ((unsigned)-2)

static ucc_pipeline_params_t ucc_pipeline_params_auto = {
    .threshold = 0,
    .n_frags   = 0,
    .frag_size = 0,
    .pdepth    = 0,
    .order     = 0
};

static ucc_pipeline_params_t ucc_pipeline_params_no = {
    .threshold = SIZE_MAX,
    .n_frags   = 0,
    .frag_size = 0,
    .pdepth    = 1,
    .order     = 0
};

static ucc_pipeline_params_t ucc_pipeline_params_default = {
    .threshold = SIZE_MAX,
    .n_frags   = 2,
    .frag_size = SIZE_MAX,
    .pdepth    = 2,
    .order     = UCC_PIPELINE_SEQUENTIAL
};

int ucc_pipeline_params_is_auto(const ucc_pipeline_params_t *p)
{
    return 0 == memcmp(p, &ucc_pipeline_params_auto, sizeof(*p));
}

int ucc_config_sscanf_pipeline_params(const char *buf, void *dest,
                                      const void *arg) //NOLINT
{
    ucc_pipeline_params_t *p = dest;
    ucc_status_t           status;
    int                    i, n_tokens, order;
    char **                tokens, **t2;

    if (strlen(buf) == 0) {
        return 0;
    }

    /* Special value: auto */
    if (!strcasecmp(buf, UCS_VALUE_AUTO_STR)) {
        *p = ucc_pipeline_params_auto;
        return 1;
    }

    if (!strcasecmp(buf, "n")) {
        *p = ucc_pipeline_params_no;
        return 1;
    }

    *p     = ucc_pipeline_params_default;
    tokens = ucc_str_split(buf, ":");
    if (!tokens) {
        return 0;
    }
    n_tokens = ucc_str_split_count(tokens);

    for (i = 0; i < n_tokens; i++) {
        if ((order = ucs_string_find_in_list(
                 tokens[i], ucc_pipeline_order_names, 0)) >= 0) {
            p->order = (ucc_pipeline_order_t)order;
            continue;
        }
        t2 = ucc_str_split(tokens[i], "=");
        if (!t2) {
            goto out;
        }
        if (ucc_str_split_count(t2) != 2) {
            goto out;
        }
        if (0 == strcmp(t2[0], "thresh")) {
            status = ucc_str_to_memunits(t2[1], &p->threshold);
            if (UCC_OK != status) {
                goto out;
            }
        } else if (0 == strcmp(t2[0], "fragsize")) {
            status = ucc_str_to_memunits(t2[1], &p->frag_size);
            if (UCC_OK != status) {
                goto out;
            }
        } else if (0 == strcmp(t2[0], "nfrags")) {
            status = ucc_str_is_number(t2[1]);
            if (UCC_OK != status) {
                goto out;
            }
            p->n_frags = atoi(t2[1]);
        } else if (0 == strcmp(t2[0], "pdepth")) {
            status = ucc_str_is_number(t2[1]);
            if (UCC_OK != status) {
                goto out;
            }
            p->pdepth = atoi(t2[1]);
        }
        ucc_str_split_free(t2);
    }
    return 1;
out:
    if (t2) {
        ucc_str_split_free(t2);
    }
    ucc_str_split_free(tokens);
    return 0;
}

int ucc_config_sprintf_pipeline_params(char *buf, size_t max, const void *src,
                                       const void *arg) //NOLINT
{
    const ucc_pipeline_params_t *p = src;
    char                         thresh[32], frag_size[32];

    if (ucc_pipeline_params_is_auto(p)) {
        return snprintf(buf, max, "auto");
    }
    if (!memcmp(p, &ucc_pipeline_params_no, sizeof(*p))) {
        return snprintf(buf, max, "n");
    }
    return snprintf(
        buf, max, "thresh=%s:nfrags=%d:fragsize=%s:pdepth=%d:order=%s",
        ucs_memunits_to_str(p->threshold, thresh, sizeof(thresh)), p->n_frags,
        ucs_memunits_to_str(p->frag_size, frag_size, sizeof(frag_size)),
        p->pdepth, ucc_pipeline_order_names[p->order]);
}

ucs_status_t ucc_config_clone_pipeline_params(const void *src, void *dest,
                                              const void *arg) //NOLINT
{
    memcpy(dest, src, sizeof(ucc_pipeline_params_t));
    return UCS_OK;
}

void ucc_config_release_pipeline_params(void *ptr, const void *arg)
{
}
