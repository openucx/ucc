/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "example_ctx.h"
#include "core/ucc_progress_queue.h"
#include "coll_patterns/recursive_knomial.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/ucc_math.h"

ucc_tl_coll_plugin_iface_t ucc_tlcp_ucp_example;

#define CONFIG(_lib) ((ucc_tlcp_ucp_example_config_t*)((_lib)->tlcp_configs[ucc_tlcp_ucp_example.id]))

static ucc_config_field_t ucc_tlcp_ucp_example_table[] = {
    {"TLCP_EXAMPLE_TUNE", "", "Collective score modifier",
     ucc_offsetof(ucc_tlcp_ucp_example_config_t, score_str), UCC_CONFIG_TYPE_STRING},

    {NULL}};

static ucs_config_global_list_entry_t ucc_tlcp_ucp_example_cfg_entry =
{
    .name   = "TLCP_EXAMPLE",
    .prefix = "TL_UCP_",
    .table  = ucc_tlcp_ucp_example_table,
    .size   = sizeof(ucc_tlcp_ucp_example_config_t)
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_tlcp_ucp_example_cfg_entry,
                                &ucc_config_global_list);

#define UCC_TLCP_UCP_EXAMPLE_SCORE 100

ucc_status_t ucc_tlcp_ucp_example_get_scores(ucc_base_team_t *tl_team,
                                              ucc_coll_score_t **score_p)
{
    ucc_tl_ucp_team_t *team = ucc_derived_of(tl_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_lib_t  *lib  = UCC_TL_UCP_TEAM_LIB(team);
    const char        *score_str;
    ucc_coll_score_t  *score;
    ucc_status_t       status;
    ucc_memory_type_t  mt   = UCC_MEMORY_TYPE_HOST;
    /* There can be a different logic for different coll_type/mem_type.
       Right now just init everything the same way. */
    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "failed to alloc score");
        return status;
    }
    status = ucc_coll_score_add_range(score, UCC_COLL_TYPE_ALLREDUCE,
                                      UCC_MEMORY_TYPE_HOST,
                                      0, 4096, UCC_TLCP_UCP_EXAMPLE_SCORE,
                                      ucc_tl_ucp_allreduce_knomial_am_init,
                                      tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "failed to add range");
        return status;
    }
    score_str = CONFIG(lib)->score_str;
    if (strlen(score_str) > 0) {

        status = ucc_coll_score_update_from_str(score_str, score,
                                                UCC_TL_TEAM_SIZE(team),
                                                ucc_tl_ucp_allreduce_knomial_am_init,
                                                &team->super.super,
                                                UCC_TLCP_UCP_EXAMPLE_SCORE,
                                                NULL, &mt, 1);
        if (status == UCC_ERR_INVALID_PARAM) {
            /* User provided incorrect input - try to proceed */
            status = UCC_OK;
        }
    }
    *score_p = score;
    return status;
}

ucs_status_t ucc_tlcp_ucp_example_am_recv_handler(void *arg, const void *header,
                                                  size_t header_length,
                                                  void *data, size_t length,
                                                  const ucp_am_recv_param_t *param)
{
    ucc_tlcp_ucp_example_context_t *ctx   = (ucc_tlcp_ucp_example_context_t *)arg;
    ucc_tlcp_ucp_example_am_msg_t  *entry;

    uint64_t *tag = (uint64_t*)header;

    entry = ucc_malloc(sizeof(ucc_tlcp_ucp_example_am_msg_t));
    ucc_assert(header_length == 8);
    if (!entry) {
        ucc_error("failed to allocate %zd bytes for am entry",
                  sizeof(*entry));
        return UCS_ERR_NO_MEMORY;
    }
    entry->tag = *tag;
    entry->msg = data;
    ucc_list_add_tail(&ctx->am_list, &entry->list_elem);
    return UCS_INPROGRESS;
}

ucc_status_t ucc_tlcp_ucp_example_context_create(const ucc_base_context_params_t *params,
                                                 const ucc_base_config_t *config,
                                                 ucc_base_context_t *tl_ctx,
                                                 void **plugin_ctx)
{
    ucc_tl_ucp_lib_t     *lib        = ucc_derived_of(tl_ctx->lib,
                                                      ucc_tl_ucp_lib_t);
    ucc_tl_ucp_context_t *tl_ucp_ctx = ucc_derived_of(tl_ctx,
                                                      ucc_tl_ucp_context_t);
    ucc_tlcp_ucp_example_context_t *ctx;
    ucc_status_t status;
    ucs_status_t ucs_status;
    ucp_am_handler_param_t am_handler_param;

    ctx = ucc_malloc(sizeof(ucc_tlcp_ucp_example_context_t),
                    "tlcp_ucp_example_context");
    if (!ctx) {
        tl_error(lib, "failed to alloc memory for plugin context");
        return UCC_ERR_NO_MEMORY;
    }

    tl_ucp_ctx = ucc_derived_of(tl_ctx, ucc_tl_ucp_context_t);
    am_handler_param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                                  UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                                  UCP_AM_HANDLER_PARAM_FIELD_CB |
                                  UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am_handler_param.id = 1;
    am_handler_param.flags = UCP_AM_FLAG_WHOLE_MSG |
                             UCP_AM_FLAG_PERSISTENT_DATA;
    am_handler_param.cb    = ucc_tlcp_ucp_example_am_recv_handler;
    am_handler_param.arg   = ctx;

    ucs_status = ucp_worker_set_am_recv_handler(tl_ucp_ctx->worker.ucp_worker,
                                                &am_handler_param);
    if (ucs_status != UCS_OK) {
        tl_error(lib, "failed to set am recv handler");
        status = ucs_status_to_ucc_status(ucs_status);
        goto free_ctx;
    }
    ucc_list_head_init(&ctx->am_list);

    *plugin_ctx = ctx;
    return UCC_OK;
free_ctx:
    ucc_free(ctx);
    return status;
}

ucc_status_t ucc_tlcp_ucp_example_context_destroy(ucc_base_context_t *tl_ctx,
                                                  void *plugin_ctx)
{
    ucc_free(plugin_ctx);
    return UCC_OK;
}

ucc_tl_coll_plugin_iface_t ucc_tlcp_ucp_example = {
    .super.name      = "tl_ucp_example",
    .super.score     = UCC_TLCP_UCP_EXAMPLE_SCORE,
    .config.table    = ucc_tlcp_ucp_example_table,
    .config.size     = sizeof(ucc_tlcp_ucp_example_config_t),
    .get_scores      = ucc_tlcp_ucp_example_get_scores,
    .context_create  = ucc_tlcp_ucp_example_context_create,
    .context_destroy = ucc_tlcp_ucp_example_context_destroy,
};
