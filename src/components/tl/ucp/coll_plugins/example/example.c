/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "components/tl/ucp/tl_ucp.h"
#include "components/tl/ucp/tl_ucp_coll.h"
#include "core/ucc_progress_queue.h"
#include "components/tl/ucp/tl_ucp_sendrecv.h"
#include "coll_patterns/recursive_knomial.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/ucc_math.h"

ucc_tl_coll_plugin_iface_t ucc_tlcp_ucp_example;

typedef struct ucc_tlcp_ucp_example_config {
    char *score_str;
} ucc_tlcp_ucp_example_config_t;

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
void ucc_tlcp_ucp_example_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    tl_info(TASK_LIB(task), "completing tl_ucp_example coll task");

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tlcp_ucp_example_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    tl_info(TASK_LIB(task), "starting tl_ucp_example coll task");

    task->super.status = UCC_INPROGRESS;
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);

    return UCC_OK;
}

ucc_status_t ucc_tlcp_ucp_example_coll_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *team,
                                             ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t    *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t    *task    = ucc_tl_ucp_get_task(tl_team);

    ucc_coll_task_init(&task->super, coll_args, team);
    task->tagged.tag     = tl_team->seq_num;
    tl_team->seq_num     = (tl_team->seq_num + 1) % UCC_TL_UCP_MAX_COLL_TAG;
    task->super.finalize = ucc_tl_ucp_coll_finalize;
    task->super.post     = ucc_tlcp_ucp_example_start;
    task->super.progress = ucc_tlcp_ucp_example_progress;
    *task_h              = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tlcp_ucp_example_get_scores(ucc_base_team_t   *tl_team,
                                              ucc_coll_score_t **score_p)
{
    ucc_tl_ucp_team_t *team = ucc_derived_of(tl_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_lib_t  *lib  = UCC_TL_UCP_TEAM_LIB(team);
    const char        *score_str;
    ucc_coll_score_t  *score;
    ucc_status_t       status;

    /* There can be a different logic for different coll_type/mem_type.
       Right now just init everything the same way. */
    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "failed to alloc score");
        return status;
    }
    status = ucc_coll_score_add_range(score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_HOST,
                                      0, 4096, UCC_TLCP_UCP_EXAMPLE_SCORE,
                                      ucc_tlcp_ucp_example_coll_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "failed to add range");
        return status;
    }
    score_str = CONFIG(lib)->score_str;
    if (strlen(score_str) > 0) {
        status = ucc_coll_score_update_from_str(score_str, score, UCC_TL_TEAM_SIZE(team),
                                                ucc_tlcp_ucp_example_coll_init,
                                                &team->super.super, UCC_TLCP_UCP_EXAMPLE_SCORE,
                                                NULL);
        if (status == UCC_ERR_INVALID_PARAM) {
            /* User provided incorrect input - try to proceed */
            status = UCC_OK;
        }
    }
    *score_p = score;
    return status;
}

ucc_tl_coll_plugin_iface_t ucc_tlcp_ucp_example = {
    .super.name   = "tl_ucp_example",
    .super.score  = UCC_TLCP_UCP_EXAMPLE_SCORE,
    .config.table = ucc_tlcp_ucp_example_table,
    .config.size  = sizeof(ucc_tlcp_ucp_example_config_t),
    .get_scores   = ucc_tlcp_ucp_example_get_scores
};
