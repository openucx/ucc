/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_parser.h"
#include "ucc_malloc.h"
#include "ucc_log.h"
#include "ucc_string.h"
#include "ini.h"
#include "schedule/ucc_schedule.h"
#include "schedule/ucc_schedule_pipelined.h"

#define UCC_ADD_KEY_VALUE_TO_HASH(_type, _h, _name, _val)                \
    do {                                                                 \
        khiter_t _iter;                                                  \
        int      _res;                                                   \
        char    *_dup;                                                   \
        _iter = kh_get(_type, _h, _name);                                \
        if (_iter != kh_end(_h)) {                                       \
            ucc_warn("found duplicate '%s' in config file", _name);      \
            return 0;                                                    \
        } else {                                                         \
            _dup = strdup(_name);                                        \
            if (!_dup) {                                                 \
                ucc_error("failed to dup str for kh_put");               \
                return 0;                                                \
            }                                                            \
            _iter = kh_put(_type, _h, _dup, &_res);                      \
            if (_res == UCS_KH_PUT_FAILED) {                             \
                ucc_free(_dup);                                          \
                ucc_error("inserting '%s' to config map failed", _name); \
                return 0;                                                \
            }                                                            \
        }                                                                \
        _dup = strdup(_val);                                             \
        if (!_dup) {                                                     \
            ucc_error("failed to dup str for kh_val");                   \
            return 0;                                                    \
        }                                                                \
        kh_val(_h, _iter) = _dup; /* NOLINT */                           \
        return 1;                                                        \
    } while (0)

static int ucc_check_section(ucc_section_desc_t sec_desc,
                             ucc_cpu_vendor_t vendor,
                             ucc_cpu_model_t model,
                             ucc_rank_t team_size,
                             ucc_rank_t ppn_min,
                             ucc_rank_t ppn_max,
                             ucc_rank_t sock_min,
                             ucc_rank_t sock_max,
                             ucc_rank_t nnodes)
{
    if (sec_desc.mask & UCC_TUNING_DESC_FIELD_VENDOR) {
        if (sec_desc.vendor != vendor) {
            return 0;
        }
    }
    if (sec_desc.mask & UCC_TUNING_DESC_FIELD_MODEL) {
        if (sec_desc.model != model) {
            return 0;
        }
    }
    if (sec_desc.mask & UCC_TUNING_DESC_FIELD_TEAM_SIZE) {
        if (team_size < sec_desc.min_team_size ||
            team_size > sec_desc.max_team_size) {
            return 0;
        }
    }
    if (sec_desc.mask & UCC_TUNING_DESC_FIELD_PPN) {
        if (ppn_min < sec_desc.min_ppn || ppn_max > sec_desc.max_ppn) {
            return 0;
        }
    }
    if (sec_desc.mask & UCC_TUNING_DESC_FIELD_SOCK) {
        if (sock_min < sec_desc.min_sock || sock_max > sec_desc.max_sock) {
            return 0;
        }
    }
    if (sec_desc.mask & UCC_TUNING_DESC_FIELD_NNODES) {
        if (nnodes < sec_desc.min_nnodes || nnodes > sec_desc.max_nnodes) {
            return 0;
        }
    }
    return 1;
}

static inline int ucc_check_range(char *range_str, ucc_rank_t *begin,
                                  ucc_rank_t *end)
{
    char   **range = ucc_str_split(range_str, "-");
    char    *str_end;
    unsigned n_range;
    long pbegin, pend;

    if (!range) {
        goto split_err;
    }

    n_range = ucc_str_split_count(range);
    pbegin  = strtol(range[0], &str_end, 10);
    pend    = pbegin;

    if (n_range > 2 || *str_end != '\0' || pbegin < 0) {
        goto val_err;
    }

    if (n_range == 2) {
        pend = strtol(range[1], &str_end, 10);
        if (*str_end != '\0' || pend < 0) {
            goto val_err;
        }
    }
    *begin = (ucc_rank_t)pbegin;
    *end = (ucc_rank_t)pend;
    ucc_str_split_free(range);
    return 1;

val_err:
    ucc_str_split_free(range);
split_err:
    ucc_warn("invalid range defined in section name\n");
    return 0;
}

static inline ucc_status_t
ucc_parse_section_name_to_desc(const char *sec_name, ucc_section_desc_t *desc)
{
    char **split, **cur_str;
    unsigned n_split, i;

    split = ucc_str_split(sec_name, " ");
    if (!split) {
        ucc_warn("invalid section name\n");
        return UCC_ERR_INVALID_PARAM;
    }

    desc->mask = 0;
    n_split = ucc_str_split_count(split);
    for (i = 0; i < n_split; i++) {
        cur_str = ucc_str_split(split[i], "=");
        if (!cur_str) {
            ucc_warn("invalid section key=value definition\n");
            goto err_cur_str;
        }
        if (strcasecmp(cur_str[0], "vendor") == 0) {
            desc->vendor = ucc_get_vendor_from_str(cur_str[1]);
            desc->mask |= UCC_TUNING_DESC_FIELD_VENDOR;
        }
        else if (strcmp(cur_str[0], "model") == 0) {
            desc->model = ucc_get_model_from_str(cur_str[1]);
            desc->mask |= UCC_TUNING_DESC_FIELD_MODEL;
        }
        else if (strcmp(cur_str[0], "team_size") == 0) {
            if (!ucc_check_range(cur_str[1], &desc->min_team_size,
                                 &desc->max_team_size)) {
                goto err_key;
            }
            desc->mask |= UCC_TUNING_DESC_FIELD_TEAM_SIZE;
        }
        else if (strcmp(cur_str[0], "ppn") == 0) {
            if (!ucc_check_range(cur_str[1], &desc->min_ppn,
                                 &desc->max_ppn)) {
                goto err_key;
            }
            desc->mask |= UCC_TUNING_DESC_FIELD_PPN;
        }
        else if (strcmp(cur_str[0], "sock") == 0) {
            if (!ucc_check_range(cur_str[1], &desc->min_sock,
                                 &desc->max_sock)) {
                goto err_key;
            }
            desc->mask |= UCC_TUNING_DESC_FIELD_SOCK;
        }
        else if (strcmp(cur_str[0], "nnodes") == 0) {
            if (!ucc_check_range(cur_str[1], &desc->min_nnodes,
                                 &desc->max_nnodes)) {
                goto err_key;
            }
            desc->mask |= UCC_TUNING_DESC_FIELD_NNODES;
        } else {
            ucc_warn("key %s not defined as part of tuning section params\n",
                     cur_str[0]);
            goto err_key;
        }
        ucc_str_split_free(cur_str);
    }
    ucc_str_split_free(split);
    return UCC_OK;
err_key:
    ucc_str_split_free(cur_str);
err_cur_str:
    ucc_str_split_free(split);
    return UCC_ERR_INVALID_PARAM;
}

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

static int ucc_file_parse_handler(void *arg, const char *section,
                                  const char *name, const char *value)
{
    ucc_file_config_t     *cfg      = arg;
    khash_t(ucc_cfg_file) *vars     = &cfg->vars;
    khash_t(ucc_sections) *sections = &cfg->sections;
    ucc_section_wrap_t    *sec_wrap;
    khiter_t               iter;
    int                    result;
    char                  *dup;
    ucc_status_t           status;

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

    if (strlen(section) > 0) {
        /* has section */
        iter = kh_get(ucc_sections, sections, section);
        if (iter == kh_end(sections)) { /* section hash table doesn't exist */
            sec_wrap = ucc_calloc(1, sizeof(*sec_wrap),
                                  strcat("section cfg: ", section));
            if (!sec_wrap) {
                ucc_error("failed to allocate %zd bytes for section config",
                          sizeof(*sec_wrap));
                return 0;
            }
            dup = strdup(section);
            if (!dup) {
                ucc_free(sec_wrap);
                ucc_error("failed to dup str for kh_put");
                return 0;
            }
            iter = kh_put(ucc_sections, sections, dup, &result);
            if (result == UCS_KH_PUT_FAILED) {
                ucc_free(sec_wrap);
                ucc_free(dup);
                ucc_error("inserting '%s' to config map failed", name);
                return 0;
            }
            status = ucc_parse_section_name_to_desc(section, &sec_wrap->desc);
            if (status != UCC_OK) {
                ucc_free(sec_wrap);
                ucc_free(dup);
                return 0;
            }
            /* new hash table for section */
            kh_init_inplace(ucc_sec, &sec_wrap->vals_h); // NOLINT
            kh_val(sections, iter) = sec_wrap; // NOLINT
        } else {
            sec_wrap = kh_val(sections, iter);
        }
        UCC_ADD_KEY_VALUE_TO_HASH(ucc_sec, &sec_wrap->vals_h, name, value);
    }
    /* param not part of a section */
    UCC_ADD_KEY_VALUE_TO_HASH(ucc_cfg_file, vars, name, value);
}

ucc_status_t ucc_parse_file_config(const char *        filename,
                                   ucc_file_config_t **cfg_p)
{
    ucc_file_config_t *cfg;
    int                ret;
    ucc_status_t       status;

    cfg = ucc_calloc(1, sizeof(*cfg), "file_cfg");
    if (!cfg) {
        ucc_error("failed to allocate %zd bytes for file config",
                  sizeof(*cfg));
        return UCC_ERR_NO_MEMORY;
    }
    kh_init_inplace(ucc_cfg_file, &cfg->vars);
    kh_init_inplace(ucc_sections, &cfg->sections);
    ret = ucc_ini_parse(filename, ucc_file_parse_handler, cfg);
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
    cfg->filename = strdup(filename);

    *cfg_p = cfg;
    return UCC_OK;
out:
    ucc_free(cfg);
    return status;
}

void ucc_release_file_config(ucc_file_config_t *cfg)
{
    const char *key, *section_key;
    char       *value, *section_val;
    khash_t(ucc_sec)   *section;
    ucc_section_wrap_t *sec_wrap;
    int j;

    ucc_free(cfg->filename);
    kh_foreach(&cfg->vars, key, value, {
        ucc_free((void *)key);
        ucc_free(value);
    }) kh_destroy_inplace(ucc_cfg_file, &cfg->vars);

    kh_foreach(&cfg->sections, key, sec_wrap, {
        section = &sec_wrap->vals_h;
        for (j = kh_begin(section); j != kh_end(section); ++j) {
        	if (!kh_exist(section, j)) continue;
        	section_key = kh_key(section, j);
        	section_val = kh_val(section, j);
        	ucc_free((void *)section_key);
        	ucc_free(section_val);
        }
        ucc_free((void *)key);
        kh_destroy_inplace(ucc_sec, section);
    }) kh_destroy_inplace(ucc_sections, &cfg->sections);

    ucc_free(cfg);
}

static const char *ucc_file_config_get_by_section(ucc_file_config_t *cfg,
                                                  const char        *var_name,
                                                  const char        *section)
{
    khash_t(ucc_sections) *sections = &cfg->sections;
    khash_t(ucc_sec) *sec;
    khiter_t iter;
    ucc_section_wrap_t *sec_wrap;

    iter = kh_get(ucc_sections, sections, section);
    if (iter == kh_end(sections)) {
        /* section not found*/
        return NULL;
    }
    sec_wrap = kh_val(sections, iter);
    sec = &sec_wrap->vals_h;
    iter = kh_get(ucc_sec, sec, var_name);
    if (iter == kh_end(sec)) {
        /* variable not found in section*/
        return NULL;
    }
    return kh_val(sec, iter);
}

static const char *ucc_file_config_get(ucc_file_config_t *cfg,
                                       const char        *var_name,
                                       const char        *section)
{
    khash_t(ucc_cfg_file) *vars = &cfg->vars;
    khiter_t iter;

    if (strlen(section) > 0) {
        return (ucc_file_config_get_by_section(cfg, var_name, section));
    }
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
                                             const char *name,
                                             const char *section)
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
        ucc_file_config_get(ucc_global_config.file_cfg, base_prefix_var,
                            section);
    if (cfg_value) {
        return ucc_config_parser_set_value(opts, fields, name, cfg_value);
    };

    if (base_prefix_var != var) {
        cfg_value = ucc_file_config_get(ucc_global_config.file_cfg, var,
                                        section);
        if (cfg_value) {
            return ucc_config_parser_set_value(opts, fields, name, cfg_value);
        }
    }

    return UCC_ERR_NOT_FOUND;
}

static ucc_status_t ucc_apply_file_cfg(void *opts, ucc_config_field_t *fields,
                                       const char *env_prefix,
                                       const char *component_prefix,
                                       const char *section)
{
    ucc_status_t status = UCC_OK;

    while (fields->name != NULL) {
        if (strlen(fields->name) == 0) {
            status = ucc_apply_file_cfg(
                opts, (ucc_config_field_t *)fields->parser.arg, env_prefix,
                component_prefix, section);
            if (UCC_OK != status) {
                return status;
            }
            fields++;
            continue;
        }
        status = ucc_apply_file_cfg_value(
            opts, fields, env_prefix, component_prefix ? component_prefix : "",
            fields->name, section);
        if (status == UCC_ERR_NOT_FOUND && component_prefix) {
            status = ucc_apply_file_cfg_value(opts, fields, env_prefix, "",
                                              fields->name, section);
        }
        if (status != UCC_OK && status != UCC_ERR_NOT_FOUND) {
            return status;
        }

        fields++;
    }
    return UCC_OK;
}

/* Team cfg table values have been previously copied from lib cfg.
 * Here, tuning values from cfg file are applied overwriting specific values
 * in team cfg table.
 * Special case needed for param tune key which is equivalent to UCC_TL_#_TUNE.
 * Returns: UCC_OK on success,
 * error status if values from cfg file cannot be applied.
*/
ucc_status_t ucc_add_team_sections(void                *team_cfg,
                                   ucc_config_field_t  *tl_fields,
                                   ucc_topo_t          *team_topo,
                                   const char         **tuning_str,
                                   const char          *tune_key,
                                   const char          *env_prefix,
                                   const char          *component_prefix)
{
    khash_t(ucc_sections) *sections  = &ucc_global_config.file_cfg->sections;
    ucc_cpu_vendor_t       vendor    = ucc_arch_get_cpu_vendor();
    ucc_cpu_model_t        model     = ucc_arch_get_cpu_model();
    ucc_rank_t             ppn_min   = ucc_topo_min_ppn(team_topo);
    ucc_rank_t             ppn_max   = ucc_topo_max_ppn(team_topo);
    ucc_rank_t             sock_min  = ucc_topo_min_socket_size(team_topo);
    ucc_rank_t             sock_max  = ucc_topo_max_socket_size(team_topo);
    ucc_rank_t             nnodes    = ucc_topo_nnodes(team_topo);
    ucc_rank_t             team_size = team_topo->set.map.ep_num;
    int                    found     = 0;
    khash_t(ucc_sec)      *sec_h;
    khiter_t               i, j;
    const char            *sec_name;
    ucc_section_wrap_t    *sec;
    ucc_status_t           status;

    for (i = kh_begin(sections); i != kh_end(sections); ++i) {
        if (!kh_exist(sections, i)) continue;
        sec_name = kh_key(sections, i);
        sec      = kh_val(sections, i);
        if (ucc_check_section(sec->desc, vendor, model, team_size,
                              ppn_min, ppn_max, sock_min, sock_max, nnodes)) {
            sec_h = &sec->vals_h;
            j = kh_get(ucc_sec, sec_h, tune_key);
            if (j != kh_end(sec_h)) {
                *tuning_str = kh_val(sec_h, j);
            }
            status = ucc_apply_file_cfg(team_cfg, tl_fields, env_prefix,
                                        component_prefix, sec_name);
            found = 1;
        }
    }
    return found ? status : UCC_ERR_NOT_FOUND;
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
                                    entry->prefix, "");
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

ucc_status_t ucc_config_clone_table(const void *src, void *dst,
                                    const void *arg)
{
    return ucs_status_to_ucc_status(ucs_config_clone_table(src, dst, arg));
}

static ucc_pipeline_params_t ucc_pipeline_params_auto = {
    .threshold = 0,
    .n_frags   = 0,
    .frag_size = 0,
    .pdepth    = 0,
    .order     = UCC_PIPELINE_PARALLEL,
};

static ucc_pipeline_params_t ucc_pipeline_params_no = {
    .threshold = SIZE_MAX,
    .n_frags   = 0,
    .frag_size = 0,
    .pdepth    = 1,
    .order     = UCC_PIPELINE_PARALLEL,
};

static ucc_pipeline_params_t ucc_pipeline_params_default = {
    .threshold = SIZE_MAX,
    .n_frags   = 2,
    .frag_size = SIZE_MAX,
    .pdepth    = 2,
    .order     = UCC_PIPELINE_SEQUENTIAL,
};

int ucc_pipeline_params_is_auto(const ucc_pipeline_params_t *p)
{
    if ((p->threshold == ucc_pipeline_params_auto.threshold) &&
        (p->n_frags == ucc_pipeline_params_auto.n_frags) &&
        (p->frag_size == ucc_pipeline_params_auto.frag_size) &&
        (p->pdepth == ucc_pipeline_params_auto.pdepth) &&
        (p->order == ucc_pipeline_params_auto.order)) {
        return 1;
    }

    return 0;
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
        if ((order = ucc_string_find_in_list(
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
    ucc_str_split_free(tokens);
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

int ucc_config_sscanf_uint_ranged(const char *buf, void *dest,
                                  const void *arg) //NOLINT
{
    ucc_mrange_uint_t *p = dest;
    char             **ranges, **tokens;
    unsigned           n_ranges, i, j, n_tokens;
    size_t             start, end;
    ucc_mrange_t      *r;
    uint32_t           mt_map;

    ucc_list_head_init(&p->ranges);
    /* Special value: auto */
    if (!strcasecmp(buf, UCS_VALUE_AUTO_STR)) {
        p->default_value = UCC_UUNITS_AUTO;
        return 1;
    }

    ranges = ucc_str_split(buf, ",");
    if (!ranges) {
        return 0;
    }
    n_ranges = ucc_str_split_count(ranges);
    for (i = 0; i < n_ranges; i++) {
        tokens = ucc_str_split(ranges[i], ":");
        if (!tokens) {
            goto err_ranges;
        }
        n_tokens = ucc_str_split_count(tokens);
        if (n_tokens > 3 ||
            (UCC_OK != ucc_str_is_number(tokens[n_tokens - 1]) &&
             strcasecmp(tokens[n_tokens - 1], UCS_VALUE_AUTO_STR) != 0)) {
            goto err_tokens;
        }
        if (n_tokens == 1) {
            if (!strcasecmp(tokens[n_tokens - 1], UCS_VALUE_AUTO_STR)) {
            /* Special value: auto */
                p->default_value = UCC_UUNITS_AUTO;
            } else {
            /* default value without range */
                p->default_value = atoi(tokens[0]);
            }
        } else {
            r = ucc_malloc(sizeof(*r), "mrange");
            if (!r) {
                goto err_tokens;
            }
            r->mtypes = UCC_MEM_TYPE_MASK_FULL;
            r->start  = 0;
            r->end    = SIZE_MAX;

            for (j = 0; j < n_tokens; j++) {
                if (UCC_OK == ucc_str_is_number(tokens[j])) {
                    /* value */
                    r->value = atoi(tokens[j]);
                    continue;
                }
                if (UCC_OK == ucc_str_to_mtype_map(tokens[j], "^", &mt_map)) {
                    r->mtypes = mt_map;
                    continue;
                }
                if (UCC_OK ==
                    ucc_str_to_memunits_range(tokens[j], &start, &end)) {
                    r->start = start;
                    r->end   = end;
                    continue;
                }
                ucc_free(r);
                goto err_tokens;
            }

            ucc_list_add_tail(&p->ranges, &r->list_elem);
        }
        ucc_str_split_free(tokens);
    }
    ucc_str_split_free(ranges);

    return 1;

err_tokens:
    ucc_str_split_free(tokens);
err_ranges:
    ucc_str_split_free(ranges);
    return 0;
}

#define MAX_TMP_BUF_LENGTH 128
int ucc_config_sprintf_uint_ranged(char *buf, size_t max, const void *src,
                                   const void *arg) // NOLINT
{
    const ucc_mrange_uint_t *s       = src;
    ucc_mrange_t            *r;
    char                     tmp_start[MAX_TMP_BUF_LENGTH];
    char                     tmp_end[MAX_TMP_BUF_LENGTH];
    char                     tmp_mtypes[MAX_TMP_BUF_LENGTH];
    size_t                   last;

    ucc_list_for_each(r, &s->ranges, list_elem) {
        ucs_memunits_to_str(r->start, tmp_start, MAX_TMP_BUF_LENGTH);
        ucs_memunits_to_str(r->end, tmp_end, MAX_TMP_BUF_LENGTH);
        if (r->mtypes == UCC_MEM_TYPE_MASK_FULL) {
            ucc_snprintf_safe(buf, max, "%s-%s:%u", tmp_start, tmp_end,
                              r->value);
        } else {
            ucc_mtype_map_to_str(r->mtypes, "^", tmp_mtypes, MAX_TMP_BUF_LENGTH);
            ucc_snprintf_safe(buf, max, "%s-%s:%s:%u", tmp_start, tmp_end,
                              tmp_mtypes, r->value);
        }
        last = strlen(buf);
        if (max - last - 1 <= 0) {
            /* no more space in buf for next range*/
            return 1;
        }

        buf[last]     = ',';
        buf[last + 1] = '\0';
        max -= last + 1;
        buf += last + 1;
    }

    if (s->default_value == UCC_UUNITS_AUTO) {
        ucc_snprintf_safe(buf, max, "%s", "auto");
    } else {
        ucc_snprintf_safe(buf, max, "%u", s->default_value);
    }

    return 1;
}

ucs_status_t ucc_config_clone_uint_ranged(const void *src, void *dest,
                                          const void *arg) //NOLINT
{
    return ucc_status_to_ucs_status(ucc_mrange_uint_copy(dest, src));
}

void ucc_config_release_uint_ranged(void *ptr, const void *arg) //NOLINT
{
    ucc_mrange_uint_destroy(ptr);
}
