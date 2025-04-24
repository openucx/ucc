/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_BASE_IFACE_H_
#define UCC_BASE_IFACE_H_
#include "ucc/api/ucc.h"
#include "core/ucc_lib.h"
#include "utils/ucc_component.h"
#include "utils/ucc_parser.h"
#include "utils/ucc_class.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_coll_utils.h"

typedef struct ucc_team ucc_team_t;
typedef struct ucc_context ucc_context_t;
typedef struct ucc_coll_score ucc_coll_score_t;
typedef struct ucc_coll_task ucc_coll_task_t;

typedef struct ucc_base_lib {
    ucc_log_component_config_t log_component;
    int                        use_tuning;
    unsigned long              min_team_size;
} ucc_base_lib_t;

typedef struct ucc_base_config {
    ucc_config_global_list_entry_t *cfg_entry;
} ucc_base_config_t;

typedef struct ucc_base_lib_config {
    ucc_base_config_t               super;
    ucc_log_component_config_t      log_component;
    int                             use_tuning;
    unsigned long                   min_team_size;
} ucc_base_lib_config_t;

typedef struct ucc_base_ctx_config {
    ucc_base_config_t               super;
    char                           *score_str;
} ucc_base_ctx_config_t;

enum {
    UCC_BASE_LIB_FLAG_TEAM_ID_REQUIRED          = UCC_BIT(1),
    UCC_BASE_LIB_FLAG_SERVICE_TEAM_REQUIRED     = UCC_BIT(2),
    UCC_BASE_LIB_FLAG_CTX_SERVICE_TEAM_REQUIRED = UCC_BIT(3)
};

enum {
    UCC_BASE_LIB_ATTR_FIELD_MIN_TEAM_SIZE = UCC_BIT(0),
    UCC_BASE_LIB_ATTR_FIELD_MAX_TEAM_SIZE = UCC_BIT(1),
    UCC_BASE_LIB_ATTR_FIELD_FLAGS         = UCC_BIT(2),
};

typedef struct ucc_base_lib_attr_t {
    uint64_t       mask;
    ucc_lib_attr_t attr;
    ucc_rank_t     min_team_size;
    ucc_rank_t     max_team_size;
    uint64_t       flags;
} ucc_base_lib_attr_t;

typedef struct ucc_base_lib_properties {
    ucc_rank_t min_team_size;
    ucc_rank_t max_team_size;
    ucc_rank_t default_team_size;
} ucc_base_lib_properties_t;

typedef struct ucc_base_lib_params {
    ucc_lib_params_t params;
    char            *full_prefix;
} ucc_base_lib_params_t;

extern ucc_config_field_t ucc_base_lib_config_table[];
extern ucc_config_field_t ucc_base_ctx_config_table[];

typedef struct ucc_base_lib_iface {
    ucc_status_t (*init)(const ucc_base_lib_params_t *params,
                         const ucc_base_config_t *config,
                         ucc_base_lib_t **lib);
    void         (*finalize)(ucc_base_lib_t *lib);
    ucc_status_t (*get_attr)(const ucc_base_lib_t *lib,
                             ucc_base_lib_attr_t  *attr);
    ucc_status_t (*get_properties)(ucc_base_lib_properties_t *prop);
} ucc_base_lib_iface_t;

typedef struct ucc_base_context_params {
    ucc_context_params_t params;
    int                  estimated_num_eps;
    int                  estimated_num_ppn;
    ucc_thread_mode_t    thread_mode;
    const char          *prefix;
    ucc_context_t       *context;
} ucc_base_context_params_t;

typedef struct ucc_base_context {
    ucc_context_t  *ucc_context;
    ucc_base_lib_t *lib;
    char           *score_str;
} ucc_base_context_t;

typedef struct ucc_base_ctx_attr_t {
    ucc_context_attr_t attr;
    uint32_t           topo_required;
} ucc_base_ctx_attr_t;

static inline void ucc_base_ctx_attr_clear(ucc_base_ctx_attr_t *attr)
{
    uint64_t mask = attr->attr.mask;
    memset(attr, 0, sizeof(ucc_base_ctx_attr_t));
    attr->attr.mask = mask;
}

typedef struct ucc_base_context_iface {
    ucc_status_t (*create)(const ucc_base_context_params_t *params,
                           const ucc_base_config_t *config,
                           ucc_base_context_t **ctx);
    ucc_status_t (*create_epilog)(ucc_base_context_t *ctx);
    void         (*destroy)(ucc_base_context_t *ctx);
    ucc_status_t (*get_attr)(const ucc_base_context_t *context,
                             ucc_base_ctx_attr_t      *attr);
} ucc_base_context_iface_t;


typedef struct ucc_base_team_params {
    ucc_team_params_t params;
    int               scope; /* Scope that allocates the team. When TL team is created
                                the scope would be a CL_TYPE. This provides a separation
                                of teams created from different CLs with the same TL_TYPE */
    int               scope_id; /* The id of the base_team in the specified scope. Use case:
                                   a single CL team (e.g. basic) creates multiple TL teams
                                   of the same type (e.g. several tl_ucp teams). In this
                                   case CL would give those teams different scope_id. */
    ucc_rank_t        rank; /* Rank of a calling process in the TL/CL team. It is a uniq
                               process identifier within a team (not job) but it has the
                               property: it is always contig and in the range [0, team_size).*/
    ucc_rank_t        size; /* Size of the TL team. size <= team->size (tl can be a subset of
                               the core team) */
    uint16_t          id;   /* core level team id */
    ucc_team_t *      team; /* core team pointer */
    ucc_ep_map_t      map;  /* ranks map to the core ucc team */
} ucc_base_team_params_t;

typedef struct ucc_base_team {
    ucc_base_context_t    *context;
    ucc_base_team_params_t params;
} ucc_base_team_t;

typedef ucc_status_t (*ucc_get_coll_scores_fn_t)(ucc_base_team_t *team,
                                                 ucc_coll_score_t **score);

typedef struct ucc_base_team_iface {
    ucc_status_t (*create_post)(ucc_base_context_t *context,
                                const ucc_base_team_params_t *params,
                                ucc_base_team_t **team);
    ucc_status_t (*create_test)(ucc_base_team_t *team);
    ucc_status_t (*destroy)(ucc_base_team_t *team);
    ucc_get_coll_scores_fn_t get_scores;
} ucc_base_team_iface_t;

enum {
    UCC_BASE_CARGS_MAX_FRAG_COUNT = UCC_BIT(0),
    /* Info is available on nonroot ranks */
    UCC_BASE_CARGS_NONROOT_INFO   = UCC_BIT(1)
};

typedef struct ucc_buffer_info_asymmetric_memtype {
    union {
        ucc_coll_buffer_info_t      info;
        ucc_coll_buffer_info_v_t    info_v;
    } old_asymmetric_buffer;
    ucc_mc_buffer_header_t *scratch;
} ucc_buffer_info_asymmetric_memtype_t;

typedef struct ucc_base_coll_args {
    uint64_t                             mask;
    ucc_coll_args_t                      args;
    ucc_team_t                          *team;
    size_t                               max_frag_count;
    /* For asymmetric mem types across ranks */
    ucc_buffer_info_asymmetric_memtype_t asymmetric_save_info;
} ucc_base_coll_args_t;

typedef ucc_status_t (*ucc_base_coll_init_fn_t)(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task);

typedef struct ucc_base_coll_iface {
    ucc_base_coll_init_fn_t init;
} ucc_base_coll_iface_t;

ucc_status_t ucc_base_config_read(const char *full_prefix,
                                  ucc_config_global_list_entry_t *cfg_entry,
                                  ucc_base_config_t **config);

static inline void ucc_base_config_release(ucc_base_config_t *config)
{
    ucc_config_parser_release_opts(config, config->cfg_entry->table);
    ucc_free(config);
}

typedef struct ucc_base_coll_alg_info {
    unsigned    id;
    const char *name;
    const char *desc;
} ucc_base_coll_alg_info_t;

#define UCC_IFACE_NAME_PREFIX(_F, _NAME, _cfg)                                 \
    .name   = UCC_PP_MAKE_STRING(_F##_NAME),                                   \
    .prefix = UCC_PP_MAKE_STRING(_F##_NAME##_)

#define UCC_IFACE_CFG(_F, _f, _cfg, _name, _NAME)                              \
    .super._f##_cfg##_config = {                                               \
        UCC_IFACE_NAME_PREFIX(_F, _NAME, _cfg),                                \
        .table = ucc_##_f##_name##_##_cfg##_config_table,                      \
        .size  = sizeof(ucc_##_f##_name##_##_cfg##_config_t)}

#define UCC_BASE_IFACE_DECLARE(_F, _f, _name, _NAME)                           \
    ucc_##_f##_name##_iface_t ucc_##_f##_name = {                              \
        UCC_IFACE_CFG(_F, _f, lib, _name, _NAME),                              \
        UCC_IFACE_CFG(_F, _f, context, _name, _NAME),                          \
        .super.super.score = UCC_##_F##_NAME##_DEFAULT_SCORE,                  \
        .super.super.name  = UCC_PP_MAKE_STRING(_name),                        \
        .super.lib.init    = UCC_CLASS_NEW_FUNC_NAME(ucc_##_f##_name##_lib_t), \
        .super.lib.finalize =                                                  \
            UCC_CLASS_DELETE_FUNC_NAME(ucc_##_f##_name##_lib_t),               \
        .super.lib.get_attr       = ucc_##_f##_name##_get_lib_attr,            \
        .super.lib.get_properties = ucc_##_f##_name##_get_lib_properties,      \
        .super.context.create =                                                \
            UCC_CLASS_NEW_FUNC_NAME(ucc_##_f##_name##_context_t),              \
        .super.context.create_epilog = NULL,                                   \
        .super.context.destroy =                                               \
            UCC_CLASS_DELETE_FUNC_NAME(ucc_##_f##_name##_context_t),           \
        .super.context.get_attr = ucc_##_f##_name##_get_context_attr,          \
        .super.team.create_post =                                              \
            UCC_CLASS_NEW_FUNC_NAME(ucc_##_f##_name##_team_t),                 \
        .super.team.create_test = ucc_##_f##_name##_team_create_test,          \
        .super.team.destroy     = ucc_##_f##_name##_team_destroy,              \
        .super.team.get_scores  = ucc_##_f##_name##_team_get_scores,           \
        .super.coll.init        = ucc_##_f##_name##_coll_init,                 \
        .super.alg_info         = {NULL}};                                     \
    UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_##_f##_name.super._f##lib_config,     \
                                    &ucc_config_global_list);                  \
    UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_##_f##_name.super._f##context_config, \
                                    &ucc_config_global_list)

#define ucc_base_log(_lib, _level, fmt, ...)                           \
    ucc_log_component(_level, (_lib)->log_component, fmt,              \
                      ##__VA_ARGS__)

#define base_error(_lib, _fmt, ...)                                 \
    ucc_base_log(_lib, UCC_LOG_LEVEL_ERROR, _fmt, ##__VA_ARGS__)
#define base_warn(_lib, _fmt, ...)                                  \
    ucc_base_log(_lib, UCC_LOG_LEVEL_WARN, _fmt, ##__VA_ARGS__)
#define base_info(_lib, _fmt, ...)                                  \
    ucc_base_log(_lib, UCC_LOG_LEVEL_INFO, _fmt, ##__VA_ARGS__)
#define base_debug(_lib, _fmt, ...)                                 \
    ucc_base_log(_lib, UCC_LOG_LEVEL_DEBUG, _fmt, ##__VA_ARGS__)
#define base_trace(_lib, _fmt, ...)                                 \
    ucc_base_log(_lib, UCC_LOG_LEVEL_TRACE, _fmt, ##__VA_ARGS__)

#endif
