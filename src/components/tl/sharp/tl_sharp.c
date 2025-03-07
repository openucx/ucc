/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_sharp.h"
#include "components/mc/base/ucc_mc_base.h"

ucc_status_t ucc_tl_sharp_get_lib_attr(const ucc_base_lib_t *lib,
                                       ucc_base_lib_attr_t *base_attr);

ucc_status_t ucc_tl_sharp_get_lib_properties(ucc_base_lib_properties_t *prop);

ucc_status_t ucc_tl_sharp_get_context_attr(const ucc_base_context_t *context,
                                           ucc_base_ctx_attr_t *base_attr);

static ucc_config_field_t ucc_tl_sharp_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_sharp_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"USE_INTERNAL_OOB", "try",
     "Use service team to create sharp context",
     ucc_offsetof(ucc_tl_sharp_lib_config_t, use_internal_oob),
     UCC_CONFIG_TYPE_TERNARY},

    {NULL}};

static ucc_config_field_t ucc_tl_sharp_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_sharp_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {"DEVICES", "",
     "SHARP device list",
     ucc_offsetof(ucc_tl_sharp_context_config_t, dev_list),
     UCC_CONFIG_TYPE_STRING},

    {"USE_RCACHE", "y",
     "Use registration cache for sharp",
     ucc_offsetof(ucc_tl_sharp_context_config_t, use_rcache),
     UCC_CONFIG_TYPE_BOOL},

    {"REG_THRESH", "0",
     "Size threshold to register buffers",
     ucc_offsetof(ucc_tl_sharp_context_config_t, reg_threshold),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"UPROGRESS_NUM_POLLS", "9999",
     "Number of polls to do before calling user progress",
     ucc_offsetof(ucc_tl_sharp_context_config_t, uprogress_num_polls),
     UCC_CONFIG_TYPE_UINT},

    {"CONTEXT_PER_TEAM", "n",
     "Create SHARP context/tree per team",
     ucc_offsetof(ucc_tl_sharp_context_config_t, context_per_team),
     UCC_CONFIG_TYPE_BOOL},

#if HAVE_DECL_SHARP_COLL_DISABLE_LAZY_GROUP_RESOURCE_ALLOC
    {"ENABLE_LAZY_GROUP_ALLOC", "n",
     "Enable lazy group resource allocation",
     ucc_offsetof(ucc_tl_sharp_context_config_t, enable_lazy_group_alloc),
     UCC_CONFIG_TYPE_BOOL},
#endif

    {"RAND_SEED", "0",
     "Seed for random sharp job ID. (0 - use default).",
     ucc_offsetof(ucc_tl_sharp_context_config_t, rand_seed),
     UCC_CONFIG_TYPE_UINT},

    {"TEAM_MAX_PPN", "1",
     "SHARP team max PPN threshold",
     ucc_offsetof(ucc_tl_sharp_context_config_t, team_max_ppn),
     UCC_CONFIG_TYPE_UINT},

    {"USE_MULTI_CHANNEL", "0",
     "Use SHARP Multi-channel feature. Options: 0-disable 1-enable",
     ucc_offsetof(ucc_tl_sharp_context_config_t, use_multi_channel),
     UCC_CONFIG_TYPE_BOOL},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_sharp_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_sharp_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_sharp_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_sharp_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_sharp_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_sharp_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_sharp_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_sharp_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task);

ucc_status_t ucc_tl_sharp_team_get_scores(ucc_base_team_t   *tl_team,
                                          ucc_coll_score_t **score_p);
UCC_TL_IFACE_DECLARE(sharp, SHARP);

ucc_status_t ucc_tl_sharp_context_create_epilog(ucc_base_context_t *context);

ucc_status_t sharp_status_to_ucc_status(int status)
{
    switch (status) {
    case SHARP_COLL_SUCCESS:
        return UCC_OK;
    case SHARP_COLL_ENOMEM:
        return UCC_ERR_NO_MEMORY;
    case SHARP_COLL_ENOT_SUPP:
        return UCC_ERR_NOT_SUPPORTED;
    case SHARP_COLL_EINVAL:
        return UCC_ERR_INVALID_PARAM;
    case SHARP_COLL_ENO_RESOURCE:
        return UCC_ERR_NO_RESOURCE;
    default:
        break;
    }
    return UCC_ERR_NO_MESSAGE;
}

__attribute__((constructor)) static void tl_sharp_iface_init(void)
{
    ucc_tl_sharp.super.context.create_epilog = ucc_tl_sharp_context_create_epilog;
}
