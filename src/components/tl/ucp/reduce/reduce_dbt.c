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

    task->reduce_dbt.t1.recv++;
    recv_completion_common(request, status, info, user_data);
}

static void recv_completion_2(void *request, ucs_status_t status,
                                   const ucp_tag_recv_info_t *info, /* NOLINT */
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    task->reduce_dbt.t2.recv++;
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
    ucc_tl_ucp_task_t          *task         =
        ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t          *team         = TASK_TEAM(task);
    ucc_coll_args_t            *args         = &TASK_ARGS(task);
    ucc_rank_t                  rank         = UCC_TL_TEAM_RANK(team);
    ucc_rank_t                  coll_root    = (ucc_rank_t)args->root;
    int                         is_root      = rank == coll_root;
    ucc_dbt_single_tree_t       t1           = task->reduce_dbt.t1;
    ucc_dbt_single_tree_t       t2           = task->reduce_dbt.t2;
    ucp_tag_recv_nbx_callback_t cb[2]        = {recv_completion_1,
                                                recv_completion_2};
    void                       *t1_sbuf, *t1_rbuf, *t2_sbuf, *t2_rbuf;
    uint32_t                    i, j;
    ucc_memory_type_t           mtype;
    ucc_datatype_t              dt;
    size_t                      count, count_t1, data_size, data_size_t1,
                                data_size_t2;
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

    count_t1     = (count % 2) ? count / 2 + 1 : count / 2;
    data_size    = count * ucc_dt_size(dt);
    data_size_t1 = count_t1 * ucc_dt_size(dt);
    data_size_t2 = count / 2 * ucc_dt_size(dt);
    avg_pre_op   = ((args->op == UCC_OP_AVG) &&
                    UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op);
    avg_post_op  = ((args->op == UCC_OP_AVG) &&
                    !UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op);
    t1_rbuf      = task->reduce_dbt.scratch;
    t2_rbuf      = PTR_OFFSET(t1_rbuf, data_size_t1 * 2);
    t1_sbuf      = avg_pre_op ? PTR_OFFSET(t1_rbuf, data_size * 2)
                              : args->src.info.buffer;
    t2_sbuf      = PTR_OFFSET(t1_sbuf, data_size_t1);

    UCC_REDUCE_DBT_GOTO_STATE(task->reduce_dbt.state);
    j = 0;
    for (i = 0; i < 2; i++) {
        if (t1.children[i] != -1) {
            UCPCHECK_GOTO(ucc_tl_ucp_recv_cb(PTR_OFFSET(t1_rbuf,
                                                        data_size_t1 * j),
                                             data_size_t1, mtype,
                                             t1.children[i], team, task, cb[0],
                                             (void *)task),
                          task, out);
            j++;
        }
    }

    j = 0;
    for (i = 0; i < 2; i++) {
        if (t2.children[i] != -1) {
            UCPCHECK_GOTO(ucc_tl_ucp_recv_cb(PTR_OFFSET(t2_rbuf,
                                                        data_size_t2 * j),
                                             data_size_t2, mtype,
                                             t2.children[i], team, task, cb[1],
                                             (void *)task),
                          task, out);
            j++;
        }
    }
    task->reduce_dbt.state = REDUCE;

REDUCE:
    if (t1.recv == t1.n_children && !task->reduce_dbt.t1_reduction_comp) {
        if (t1.n_children > 0) {
            single_tree_reduce(task, t1_sbuf, t1_rbuf, t1.n_children, count_t1,
                               data_size_t1, dt, args,
                               avg_post_op && t1.root == rank);
        }
        task->reduce_dbt.t1_reduction_comp = 1;
    }
    if (t2.recv == t2.n_children && !task->reduce_dbt.t2_reduction_comp) {
        if (t2.n_children > 0) {
            single_tree_reduce(task, t2_sbuf, t2_rbuf, t2.n_children,
                               count / 2, data_size_t2, dt, args,
                               avg_post_op && t2.root == rank);
        }
        task->reduce_dbt.t2_reduction_comp = 1;
    }

    if (rank != t1.root && task->reduce_dbt.t1_reduction_comp &&
        !task->reduce_dbt.t1_send_comp) {
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb((t1.n_children > 0) ? t1_rbuf
                                                             : t1_sbuf,
                                         data_size_t1, mtype, t1.parent, team,
                                         task),
                      task, out);
        task->reduce_dbt.t1_send_comp = 1;
    }

    if (rank != t2.root && task->reduce_dbt.t2_reduction_comp &&
        !task->reduce_dbt.t2_send_comp) {
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb((t2.n_children > 0) ? t2_rbuf
                                                             : t2_sbuf,
                                         data_size_t2, mtype, t2.parent, team,
                                         task),
                      task, out);
        task->reduce_dbt.t2_send_comp = 1;
    }

    if (!task->reduce_dbt.t1_reduction_comp ||
        !task->reduce_dbt.t2_reduction_comp) {
        return;
    }
TEST:
    if (UCC_INPROGRESS == ucc_tl_ucp_test_send(task)) {
        task->reduce_dbt.state = TEST;
        return;
    }

    /* tree roots send to coll root*/
    if (rank == t1.root && !is_root) {
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(t1_rbuf, data_size_t1, mtype,
                                             coll_root, team, task),
                          task, out);
    }

    if (rank == t2.root && !is_root) {
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(t2_rbuf, data_size_t2, mtype,
                                             coll_root, team, task),
                          task, out);
    }

    task->reduce_dbt.t1_reduction_comp = t1.recv;
    task->reduce_dbt.t2_reduction_comp = t2.recv;

    if (is_root && rank != t1.root) {
        UCPCHECK_GOTO(ucc_tl_ucp_recv_cb(args->dst.info.buffer, data_size_t1,
                                         mtype, t1.root, team, task, cb[0],
                                         (void *)task),
                      task, out);
        task->reduce_dbt.t1_reduction_comp++;
    }

    if (is_root && rank != t2.root) {
        UCPCHECK_GOTO(ucc_tl_ucp_recv_cb(PTR_OFFSET(args->dst.info.buffer,
                                                    data_size_t1),
                                         data_size_t2, mtype, t2.root, team,
                                         task, cb[1], (void *)task),
                      task, out);
        task->reduce_dbt.t2_reduction_comp++;
    }
TEST_ROOT:
    if (UCC_INPROGRESS == ucc_tl_ucp_test_send(task) ||
        task->reduce_dbt.t1_reduction_comp != t1.recv ||
        task->reduce_dbt.t2_reduction_comp != t2.recv) {
        task->reduce_dbt.state = TEST_ROOT;
        return;
    }

    if (is_root && rank == t1.root) {
        UCPCHECK_GOTO(ucc_mc_memcpy(args->dst.info.buffer, t1_rbuf,
                                    data_size_t1, mtype, mtype), task, out);
    }

    if (is_root && rank == t2.root) {
        UCPCHECK_GOTO(ucc_mc_memcpy(PTR_OFFSET(args->dst.info.buffer,
                                               data_size_t1), t2_rbuf,
                                               data_size_t2, mtype, mtype),
                      task, out);
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

    task->reduce_dbt.t1.recv           = 0;
    task->reduce_dbt.t2.recv           = 0;
    task->reduce_dbt.t1_reduction_comp = 0;
    task->reduce_dbt.t2_reduction_comp = 0;
    task->reduce_dbt.t1_send_comp      = 0;
    task->reduce_dbt.t2_send_comp      = 0;
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
    ucc_dbt_build_trees(rank, size, &task->reduce_dbt.t1,
                        &task->reduce_dbt.t2);

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
