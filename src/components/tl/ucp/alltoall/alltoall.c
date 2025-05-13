/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"

#define ALLTOALL_MAX_PATTERN_SIZE (sizeof(UCC_TL_UCP_ALLTOALL_DEFAULT_ALG_SELECT_STR_PATTERN) + 32)
#define ALLTOALL_DEFAULT_ALG_SWITCH 129
/* TODO: add as parameters */
#define MSG_MEDIUM 66000
#define NP_THRESH 32

ucc_status_t ucc_tl_ucp_alltoall_pairwise_start(ucc_coll_task_t *task);
void ucc_tl_ucp_alltoall_pairwise_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_alltoall_onesided_start(ucc_coll_task_t *task);
void ucc_tl_ucp_alltoall_onesided_progress(ucc_coll_task_t *task);

char* ucc_tl_ucp_alltoall_score_str_get(ucc_tl_ucp_team_t *team)
{
    int max_size = ALLTOALL_MAX_PATTERN_SIZE;
    char *str;

    str = ucc_malloc(max_size * sizeof(char));
    ucc_snprintf_safe(str, max_size,
                      UCC_TL_UCP_ALLTOALL_DEFAULT_ALG_SELECT_STR_PATTERN,
                      ALLTOALL_DEFAULT_ALG_SWITCH * UCC_TL_TEAM_SIZE(team));
    return str;
}

ucc_base_coll_alg_info_t
    ucc_tl_ucp_alltoall_algs[UCC_TL_UCP_ALLTOALL_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLTOALL_ALG_PAIRWISE] =
            {.id   = UCC_TL_UCP_ALLTOALL_ALG_PAIRWISE,
             .name = "pairwise",
             .desc = "pairwise two-sided implementation"},
        [UCC_TL_UCP_ALLTOALL_ALG_BRUCK] =
            {.id   = UCC_TL_UCP_ALLTOALL_ALG_BRUCK,
             .name = "bruck",
             .desc = "Bruck alltoall"},
        [UCC_TL_UCP_ALLTOALL_ALG_ONESIDED] =
            {.id   = UCC_TL_UCP_ALLTOALL_ALG_ONESIDED,
             .name = "onesided",
             .desc = "naive, linear one-sided implementation"},
        [UCC_TL_UCP_ALLTOALL_ALG_LINEAR] =
            {.id   = UCC_TL_UCP_ALLTOALL_ALG_LINEAR,
             .name = "linear",
             .desc = "linear two-sided implementation"},
        [UCC_TL_UCP_ALLTOALL_ALG_LAST] = {.id = 0, .name = NULL, .desc = NULL}};

static ucc_rank_t get_num_posts(const ucc_tl_ucp_team_t *team,
                                const ucc_coll_args_t *args)
{
    unsigned long posts = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_num_posts;
    ucc_rank_t    tsize = UCC_TL_TEAM_SIZE(team);
    size_t data_size;

    data_size = (size_t)args->src.info.count *
                ucc_dt_size(args->src.info.datatype);
    if (posts == UCC_ULUNITS_AUTO) {
        if ((data_size > MSG_MEDIUM) && (tsize > NP_THRESH)) {
            /* use pairwise algorithm */
            posts = 1;
        } else {
            /* use linear algorithm */
            posts = 0;
        }
    }

    posts = (posts > tsize || posts == 0) ? tsize: posts;
    return posts;
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init_num_posts(ucc_base_coll_args_t *coll_args,
                                                         ucc_base_team_t *team,
                                                         ucc_coll_task_t **task_h,
                                                         ucc_rank_t num_posts)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    ALLTOALL_TASK_CHECK(coll_args->args, tl_team);
    task = ucc_tl_ucp_init_task(coll_args, team);
    task->alltoall_pairwise.num_posts = num_posts;
    *task_h = &task->super;

    status = ucc_tl_ucp_alltoall_pairwise_init_common(task);
out:
    return status;
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t num_posts;

    num_posts = get_num_posts(tl_team, &coll_args->args);
    return ucc_tl_ucp_alltoall_pairwise_init_num_posts(coll_args, team, task_h,
                                                       num_posts);
}

ucc_status_t ucc_tl_ucp_alltoall_linear_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *team,
                                            ucc_coll_task_t **task_h)
{
    return ucc_tl_ucp_alltoall_pairwise_init_num_posts(coll_args, team, task_h, 0);
}

ucc_status_t ucc_tl_ucp_alltoall_init(ucc_tl_ucp_task_t *task)
{
    ucc_status_t status;

    ALLTOALL_TASK_CHECK(TASK_ARGS(task), TASK_TEAM(task));
    task->alltoall_pairwise.num_posts = get_num_posts(TASK_TEAM(task),
                                                      &TASK_ARGS(task));

    status = ucc_tl_ucp_alltoall_pairwise_init_common(task);
out:
    return status;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    ALLTOALL_TASK_CHECK(coll_args->args, tl_team);

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "global work buffer not provided nor associated with team");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "non memory mapped buffers are not supported");
            status = UCC_ERR_NOT_SUPPORTED;
            goto out;
        }
    }
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    *task_h              = &task->super;
    task->super.post     = ucc_tl_ucp_alltoall_onesided_start;
    task->super.progress = ucc_tl_ucp_alltoall_onesided_progress;
    status               = UCC_OK;
out:
    return status;
}
