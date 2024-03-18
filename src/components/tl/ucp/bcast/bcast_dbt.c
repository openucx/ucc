/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "bcast.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"

enum {
    RECV,
    SEND_T1,
    SEND_T2,
    TEST,
};

#define UCC_BCAST_DBT_CHECK_STATE(_p)                                         \
    case _p:                                                                  \
        goto _p;

#define UCC_BCAST_DBT_GOTO_STATE(_state)                                      \
    do {                                                                      \
        switch (_state) {                                                     \
            UCC_BCAST_DBT_CHECK_STATE(SEND_T1);                               \
            UCC_BCAST_DBT_CHECK_STATE(SEND_T2);                               \
            UCC_BCAST_DBT_CHECK_STATE(TEST);                                  \
        };                                                                    \
    } while (0)

static void recv_completion_common(void *request, ucs_status_t status,
                                   const ucp_tag_recv_info_t *info, /* NOLINT */
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ucc_atomic_add32(&task->tagged.recv_completed, 1);
    if (request) {
        ucp_request_free(request);
    }
}

static void recv_completion_1(void *request, ucs_status_t status,
                                   const ucp_tag_recv_info_t *info, /* NOLINT */
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    task->bcast_dbt.t1.recv++;
    recv_completion_common(request, status, info, user_data);
}

static void recv_completion_2(void *request, ucs_status_t status,
                                   const ucp_tag_recv_info_t *info, /* NOLINT */
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    task->bcast_dbt.t2.recv++;
    recv_completion_common(request, status, info, user_data);

}

void ucc_tl_ucp_bcast_dbt_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t          *task         =
        ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t          *team         = TASK_TEAM(task);
    ucc_coll_args_t            *args         = &TASK_ARGS(task);
    ucc_rank_t                  rank         = UCC_TL_TEAM_RANK(team);
    ucc_dbt_single_tree_t       t1           = task->bcast_dbt.t1;
    ucc_dbt_single_tree_t       t2           = task->bcast_dbt.t2;
    void                       *buffer       = args->src.info.buffer;
    ucc_memory_type_t           mtype        = args->src.info.mem_type;
    ucc_datatype_t              dt           = args->src.info.datatype;
    size_t                      count        = args->src.info.count;
    size_t                      count_t1     = (count % 2) ? count / 2 + 1
                                                           : count / 2;
    size_t                      data_size_t1 = count_t1 * ucc_dt_size(dt);
    size_t                      data_size_t2 = count / 2 * ucc_dt_size(dt);
    ucc_rank_t                  coll_root    = (ucc_rank_t)args->root;
    ucp_tag_recv_nbx_callback_t cb[2]        = {recv_completion_1,
                                                recv_completion_2};
    uint32_t                    i;

    UCC_BCAST_DBT_GOTO_STATE(task->bcast_dbt.state);

    if (rank != t1.root && rank != coll_root) {
        UCPCHECK_GOTO(ucc_tl_ucp_recv_cb(buffer, data_size_t1, mtype,
                                         t1.parent, team, task, cb[0],
                                         (void *)task),
                      task, out);
    }

    if (rank != t2.root && rank != coll_root) {
        UCPCHECK_GOTO(ucc_tl_ucp_recv_cb(PTR_OFFSET(buffer, data_size_t1),
                                         data_size_t2, mtype, t2.parent, team,
                                         task, cb[1], (void *)task),
                      task, out);
    }
    task->bcast_dbt.state = SEND_T1;

SEND_T1:
/* test_recv is needed to progress ucp_worker */
    ucc_tl_ucp_test_recv(task);
    if ((coll_root == rank) || (task->bcast_dbt.t1.recv > 0)) {
        for (i = 0; i < 2; i++) {
            if ((t1.children[i] != UCC_RANK_INVALID) &&
                (t1.children[i] != coll_root)) {
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(buffer, data_size_t1, mtype,
                                                 t1.children[i], team, task),
                              task, out);
            }
        }
    } else {
        goto out;
    }
    task->bcast_dbt.state = SEND_T2;

SEND_T2:
/* test_recv is needed to progress ucp_worker */
    ucc_tl_ucp_test_recv(task);
    if ((coll_root == rank) || (task->bcast_dbt.t2.recv > 0)) {
        for (i = 0; i < 2; i++) {
            if ((t2.children[i] != UCC_RANK_INVALID) &&
                (t2.children[i] != coll_root)) {
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(PTR_OFFSET(buffer,
                                                            data_size_t1),
                                                 data_size_t2, mtype,
                                                 t2.children[i], team, task),
                              task, out);
            }
        }
    } else {
        goto out;
    }

TEST:
    if (UCC_INPROGRESS == ucc_tl_ucp_test_send(task)) {
        task->bcast_dbt.state = TEST;
        return;
    }

    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_bcast_dbt_done", 0);

out:
    return;
}

ucc_status_t ucc_tl_ucp_bcast_dbt_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t          *task         =
        ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t          *team         = TASK_TEAM(task);
    ucc_coll_args_t            *args         = &TASK_ARGS(task);
    ucc_status_t                status       = UCC_OK;
    ucc_rank_t                  rank         = UCC_TL_TEAM_RANK(team);
    void                       *buffer       = args->src.info.buffer;
    ucc_memory_type_t           mtype        = args->src.info.mem_type;
    ucc_datatype_t              dt           = args->src.info.datatype;
    size_t                      count        = args->src.info.count;
    size_t                      count_t1     = (count % 2) ? count / 2 + 1
                                                           : count / 2;
    size_t                      data_size_t1 = count_t1 * ucc_dt_size(dt);
    size_t                      data_size_t2 = count / 2 * ucc_dt_size(dt);
    ucc_rank_t                  coll_root    = (ucc_rank_t)args->root;
    ucc_rank_t                  t1_root      = task->bcast_dbt.t1.root;
    ucc_rank_t                  t2_root      = task->bcast_dbt.t2.root;
    ucp_tag_recv_nbx_callback_t cb[2]        = {recv_completion_1,
                                              recv_completion_2};

    task->bcast_dbt.t1.recv = 0;
    task->bcast_dbt.t2.recv = 0;
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (rank == coll_root && coll_root != t1_root) {
        status = ucc_tl_ucp_send_nb(buffer, data_size_t1, mtype, t1_root, team,
                                    task);
        if (UCC_OK != status) {
            return status;
        }
    }

    if (rank == coll_root && coll_root != t2_root) {
        status = ucc_tl_ucp_send_nb(PTR_OFFSET(buffer, data_size_t1),
                                    data_size_t2, mtype, t2_root, team, task);
        if (UCC_OK != status) {
            return status;
        }
    }

    if (rank != coll_root && rank == t1_root) {
        status = ucc_tl_ucp_recv_cb(buffer, data_size_t1, mtype, coll_root,
                                    team, task, cb[0], (void *)task);
        if (UCC_OK != status) {
            return status;
        }
    }

    if (rank != coll_root && rank == t2_root) {
        status = ucc_tl_ucp_recv_cb(PTR_OFFSET(buffer, data_size_t1),
                                    data_size_t2, mtype, coll_root, team, task,
                                    cb[1], (void *)task);
        if (UCC_OK != status) {
            return status;
        }
    }

    task->bcast_dbt.state = RECV;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_bcast_dbt_start", 0);
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_bcast_dbt_finalize(ucc_coll_task_t *coll_task)
{
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_bcast_dbt_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t *tl_team;
    ucc_tl_ucp_task_t *task;
    ucc_rank_t         rank, size;

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_bcast_dbt_start;
    task->super.progress = ucc_tl_ucp_bcast_dbt_progress;
    task->super.finalize = ucc_tl_ucp_bcast_dbt_finalize;
    task->n_polls        = ucc_max(1, task->n_polls);
    tl_team              = TASK_TEAM(task);
    rank                 = UCC_TL_TEAM_RANK(tl_team);
    size                 = UCC_TL_TEAM_SIZE(tl_team);
    ucc_dbt_build_trees(rank, size, &task->bcast_dbt.t1,
                             &task->bcast_dbt.t2);

    *task_h = &task->super;
    return UCC_OK;
}
