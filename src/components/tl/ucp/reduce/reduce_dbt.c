/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "reduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_dt_reduce.h"

enum {
    RECV,
    REDUCE,
    TEST,
    TEST_ROOT,
};

#define UCC_REDUCE_DBT_CHECK_STATE(_p)                                        \
    case _p:                                                                  \
        goto _p;

#define UCC_REDUCE_DBT_GOTO_STATE(_state)                                     \
    do {                                                                      \
        switch (_state) {                                                     \
            UCC_REDUCE_DBT_CHECK_STATE(REDUCE);                               \
            UCC_REDUCE_DBT_CHECK_STATE(TEST);                                 \
            UCC_REDUCE_DBT_CHECK_STATE(TEST_ROOT);                            \
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
    task->tagged.recv_completed++;
    if (request) {
        ucp_request_free(request);
    }
}

static void recv_completion_1(void *request, ucs_status_t status,
                                   const ucp_tag_recv_info_t *info, /* NOLINT */
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    task->reduce_dbt.trees[0].recv++;
    recv_completion_common(request, status, info, user_data);
}

static void recv_completion_2(void *request, ucs_status_t status,
                                   const ucp_tag_recv_info_t *info, /* NOLINT */
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    task->reduce_dbt.trees[1].recv++;
    recv_completion_common(request, status, info, user_data);
}

static inline void single_tree_reduce(ucc_tl_ucp_task_t *task, void *sbuf,
                                      void *rbuf, int n_children, size_t count,
                                      size_t data_size, ucc_datatype_t dt,
                                      ucc_coll_args_t *args, int is_avg)
{
    ucc_status_t status;

    status = ucc_dt_reduce_strided(
        sbuf,rbuf, rbuf,
        n_children, count, data_size,
        dt, args,
        is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0,
        AVG_ALPHA(task), task->reduce_dbt.executor,
        &task->reduce_dbt.etask);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(task),
                    "failed to perform dt reduction");
        task->super.status = status;
        return;
    }
    EXEC_TASK_WAIT(task->reduce_dbt.etask);
}

void ucc_tl_ucp_reduce_dbt_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t          *task         = ucc_derived_of(coll_task,
                                                              ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t          *team         = TASK_TEAM(task);
    ucc_coll_args_t            *args         = &TASK_ARGS(task);
    ucc_dbt_single_tree_t      *trees        = task->reduce_dbt.trees ;
    ucc_rank_t                  rank         = UCC_TL_TEAM_RANK(team);
    ucc_rank_t                  coll_root    = (ucc_rank_t)args->root;
    int                         is_root      = rank == coll_root;
    ucp_tag_recv_nbx_callback_t cb[2]        = {recv_completion_1,
                                                recv_completion_2};
    void                       *sbuf[2], *rbuf[2];
    uint32_t                    i, j, k;
    ucc_memory_type_t           mtype;
    ucc_datatype_t              dt;
    size_t                      count, data_size, data_size_t1;
    size_t                      counts[2];
    int                         avg_pre_op, avg_post_op;

    if (is_root) {
        mtype = args->dst.info.mem_type;
        dt    = args->dst.info.datatype;
        count = args->dst.info.count;
    } else {
        mtype = args->src.info.mem_type;
        dt    = args->src.info.datatype;
        count = args->src.info.count;
    }

    counts[0]    = (count % 2) ? count / 2 + 1 : count / 2;
    counts[1]    = count / 2;
    data_size    = count * ucc_dt_size(dt);
    data_size_t1 = counts[0] * ucc_dt_size(dt);
    avg_pre_op   = ((args->op == UCC_OP_AVG) &&
                    UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op);
    avg_post_op  = ((args->op == UCC_OP_AVG) &&
                    !UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op);

    rbuf[0] = task->reduce_dbt.scratch;
    rbuf[1] = PTR_OFFSET(rbuf[0], data_size_t1 * 2);;
    sbuf[0] = avg_pre_op ? PTR_OFFSET(rbuf[0], data_size * 2)
                              : args->src.info.buffer;;
    sbuf[1] = PTR_OFFSET(sbuf[0], data_size_t1);

    UCC_REDUCE_DBT_GOTO_STATE(task->reduce_dbt.state);
    for (i = 0; i < 2; i++) {
        j = 0;
        for (k = 0; k < 2; k++) {
            if (trees[i].children[k] != UCC_RANK_INVALID) {
                UCPCHECK_GOTO(ucc_tl_ucp_recv_cb(
                                PTR_OFFSET(rbuf[i], counts[i] * ucc_dt_size(dt) * j),
                                counts[i] * ucc_dt_size(dt), mtype,
                                trees[i].children[k], team, task, cb[i],
                                (void *)task),
                            task, out);
                j++;
            }

        }
    }
    task->reduce_dbt.state = REDUCE;

REDUCE:
/* test_recv is needed to progress ucp_worker */
    ucc_tl_ucp_test_recv(task);
    for (i = 0; i < 2; i++) {
        if (trees[i].recv == trees[i].n_children &&
            !task->reduce_dbt.reduction_comp[i]) {
            if (trees[i].n_children > 0) {
                single_tree_reduce(task, sbuf[i], rbuf[i], trees[i].n_children,
                                   counts[i], counts[i] * ucc_dt_size(dt), dt,
                                   args, avg_post_op && trees[i].root == rank);
            }
            task->reduce_dbt.reduction_comp[i] = 1;
        }
    }

    for (i = 0; i < 2; i++) {
        if (rank != trees[i].root && task->reduce_dbt.reduction_comp[i] &&
            !task->reduce_dbt.send_comp[i]) {
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb((trees[i].n_children > 0) ? rbuf[i]
                                                                       : sbuf[i],
                                              counts[i] * ucc_dt_size(dt),
                                              mtype, trees[i].parent, team,
                                              task),
                        task, out);
            task->reduce_dbt.send_comp[i] = 1;
        }
    }

    if (!task->reduce_dbt.reduction_comp[0] ||
        !task->reduce_dbt.reduction_comp[1]) {
        return;
    }
TEST:
    if (UCC_INPROGRESS == ucc_tl_ucp_test_send(task)) {
        task->reduce_dbt.state = TEST;
        return;
    }

    /* tree roots send to coll root*/
    for (i = 0; i < 2; i++) {
        if (rank == trees[i].root && !is_root) {
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(rbuf[i],
                                                 counts[i] * ucc_dt_size(dt),
                                                 mtype, coll_root, team, task),
                            task, out);
        }
    }

    task->reduce_dbt.reduction_comp[0] = trees[0].recv;
    task->reduce_dbt.reduction_comp[1] = trees[1].recv;

    for (i = 0; i < 2; i++) {
        if (is_root && rank != trees[i].root) {
            UCPCHECK_GOTO(ucc_tl_ucp_recv_cb(PTR_OFFSET(args->dst.info.buffer,
                                             i * counts[0] * ucc_dt_size(dt)),
                                             counts[i] * ucc_dt_size(dt),
                                             mtype, trees[i].root, team, task,
                                             cb[i], (void *)task),
                          task, out);
            task->reduce_dbt.reduction_comp[i]++;
        }
    }

TEST_ROOT:
/* test_recv is needed to progress ucp_worker */
    ucc_tl_ucp_test_recv(task);
    if (UCC_INPROGRESS == ucc_tl_ucp_test_send(task) ||
        task->reduce_dbt.reduction_comp[0] != trees[0].recv ||
        task->reduce_dbt.reduction_comp[1] != trees[1].recv) {
        task->reduce_dbt.state = TEST_ROOT;
        return;
    }

    for (i = 0; i < 2; i++) {
        if (is_root && rank == trees[i].root) {
            UCPCHECK_GOTO(ucc_mc_memcpy(PTR_OFFSET(args->dst.info.buffer,
                                        i * counts[(i + 1) % 2] * ucc_dt_size(dt)),
                                        rbuf[i], counts[i] * ucc_dt_size(dt),
                                        mtype, mtype), task, out);
        }
    }

    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_dbt_done", 0);
out:
    return;
}

ucc_status_t ucc_tl_ucp_reduce_dbt_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task,
                                                   ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    ucc_coll_args_t   *args       = &TASK_ARGS(task);
    ucc_rank_t         rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         team_size  = UCC_TL_TEAM_SIZE(team);
    int                avg_pre_op =
        UCC_TL_UCP_TEAM_LIB(TASK_TEAM(task))->cfg.reduce_avg_pre_op;
    ucc_datatype_t     dt;
    size_t             count, data_size;
    ucc_status_t       status;

    task->reduce_dbt.trees[0].recv     = 0;
    task->reduce_dbt.trees[1].recv     = 0;
    task->reduce_dbt.reduction_comp[0] = 0;
    task->reduce_dbt.reduction_comp[1] = 0;
    task->reduce_dbt.send_comp[0]      = 0;
    task->reduce_dbt.send_comp[1]      = 0;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (args->root == rank) {
        count = args->dst.info.count;
        dt    = args->dst.info.datatype;
    } else {
        count = args->src.info.count;
        dt    = args->src.info.datatype;
    }
    data_size = count * ucc_dt_size(dt);

    status = ucc_coll_task_get_executor(&task->super,
                                        &task->reduce_dbt.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    if (UCC_IS_INPLACE(*args) && (rank == args->root)) {
        args->src.info.buffer = args->dst.info.buffer;
    }

    if (avg_pre_op && args->op == UCC_OP_AVG) {
        /* In case of avg_pre_op, each process must divide itself by team_size */
        status =
            ucc_dt_reduce(args->src.info.buffer, args->src.info.buffer,
                          PTR_OFFSET(task->reduce_dbt.scratch, data_size * 2),
                          count, dt, args, UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA,
                          1.0 / (double)(team_size * 2),
                          task->reduce_dbt.executor, &task->reduce_dbt.etask);
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to perform dt reduction");
            return status;
        }
        EXEC_TASK_WAIT(task->reduce_dbt.etask, status);
    }

    task->reduce_dbt.state = RECV;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_dbt_start", 0);
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_reduce_dbt_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    if (task->reduce_dbt.scratch_mc_header) {
        ucc_mc_free(task->reduce_dbt.scratch_mc_header);
    }

    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_reduce_dbt_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t      *team,
                                        ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team;
    ucc_tl_ucp_task_t *task;
    ucc_rank_t         rank, size;
    ucc_memory_type_t  mtype;
    ucc_datatype_t     dt;
    size_t             count;
    size_t             data_size;
    ucc_status_t       status;

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_ucp_reduce_dbt_start;
    task->super.progress = ucc_tl_ucp_reduce_dbt_progress;
    task->super.finalize = ucc_tl_ucp_reduce_dbt_finalize;
    tl_team              = TASK_TEAM(task);
    rank                 = UCC_TL_TEAM_RANK(tl_team);
    size                 = UCC_TL_TEAM_SIZE(tl_team);
    ucc_dbt_build_trees(rank, size, &task->reduce_dbt.trees[0],
                        &task->reduce_dbt.trees[1]);

    if (coll_args->args.root == rank) {
        count = coll_args->args.dst.info.count;
        dt    = coll_args->args.dst.info.datatype;
        mtype = coll_args->args.dst.info.mem_type;
    } else {
        count = coll_args->args.src.info.count;
        dt    = coll_args->args.src.info.datatype;
        mtype = coll_args->args.src.info.mem_type;
    }
    data_size                          = count * ucc_dt_size(dt);
    task->reduce_dbt.scratch_mc_header = NULL;
    status = ucc_mc_alloc(&task->reduce_dbt.scratch_mc_header, 3 * data_size,
                          mtype);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }
    task->reduce_dbt.scratch = task->reduce_dbt.scratch_mc_header->addr;
    *task_h = &task->super;
    return UCC_OK;
}
