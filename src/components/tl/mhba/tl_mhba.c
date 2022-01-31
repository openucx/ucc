/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mhba.h"

ucc_status_t ucc_tl_mhba_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t * base_attr);
ucc_status_t ucc_tl_mhba_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t *     base_attr);

static ucc_config_field_t ucc_tl_mhba_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_mhba_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"TRANSPOSE", "0", "Boolean - with transpose or not",
     ucc_offsetof(ucc_tl_mhba_lib_config_t, transpose), UCC_CONFIG_TYPE_UINT},

    {"ASR_BARRIER", "0", "Boolean - use  service barrier or p2p sync of ASRs",
     ucc_offsetof(ucc_tl_mhba_lib_config_t, asr_barrier), UCC_CONFIG_TYPE_UINT},

    {"TRANPOSE_BUF_SIZE", "128k", "Size of the pre-allocated transpose buffer",
     ucc_offsetof(ucc_tl_mhba_lib_config_t, transpose_buf_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"DM_BUF_SIZE", "8k", "Size of the pre-allocated DeviceMemory buffer",
     ucc_offsetof(ucc_tl_mhba_lib_config_t, dm_buf_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"DM_BUF_NUM", "auto", "Number of DM buffers to alloc",
     ucc_offsetof(ucc_tl_mhba_lib_config_t, dm_buf_num),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"BLOCK_SIZE", "0",
     "Size of the blocks that are sent using blocked AlltoAll Algorithm",
     ucc_offsetof(ucc_tl_mhba_lib_config_t, block_size), UCC_CONFIG_TYPE_UINT},

    {"NUM_DCI_QPS", "16",
     "Number of parallel DCI QPs that will be used per team",
     ucc_offsetof(ucc_tl_mhba_lib_config_t, num_dci_qps), UCC_CONFIG_TYPE_UINT},

    {"RC_DC", "2", "Boolean - 1 for DC QPs, 0 for RC QPs",
     ucc_offsetof(ucc_tl_mhba_lib_config_t, rc_dc), UCC_CONFIG_TYPE_UINT},

    {NULL}};

static ucc_config_field_t ucc_tl_mhba_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_mhba_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {"NET_DEVICES", "", "Specifies which network device(s) to use",
     ucc_offsetof(ucc_tl_mhba_context_config_t, devices),
     UCC_CONFIG_TYPE_STRING_ARRAY},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_mhba_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mhba_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_mhba_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mhba_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_mhba_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_mhba_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_mhba_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_mhba_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     team,
                                   ucc_coll_task_t **    task);

ucc_status_t ucc_tl_mhba_team_get_scores(ucc_base_team_t *  tl_team,
                                         ucc_coll_score_t **score);

UCC_TL_IFACE_DECLARE(mhba, MHBA);

ucc_status_t ucc_tl_mhba_context_create_epilog(ucc_base_context_t *context);

__attribute__((constructor)) static void tl_mhba_iface_init(void)
{
    ucc_tl_mhba.super.context.create_epilog = ucc_tl_mhba_context_create_epilog;
}
