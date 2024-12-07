/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "example_ctx.h"
#include "components/tl/ucp/tl_ucp_sendrecv.h"
#include "utils/ucc_dt_reduce.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->allreduce_kn.phase = _phase;                                     \
    } while (0)

static inline ucc_status_t
ucc_tl_ucp_send_am(void *buffer, size_t msglen, ucc_memory_type_t mtype,
                   ucc_rank_t dest_group_rank, ucc_tl_ucp_team_t *team,
                   ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);
    ucc_status_t status;
    ucp_ep_h ep;
    ucp_request_param_t req_param;
    ucs_status_ptr_t ucp_status;
    ucp_tag_t ucp_tag;

    ucp_tag = UCC_TL_UCP_MAKE_SEND_TAG((args->mask & UCC_COLL_ARGS_FIELD_TAG),
                                        task->tagged.tag, UCC_TL_TEAM_RANK(team),
                                        team->super.super.params.id,
                                        team->super.super.params.scope_id,
                                        team->super.super.params.scope);
    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                             UCP_OP_ATTR_FIELD_DATATYPE |
                             UCP_OP_ATTR_FIELD_USER_DATA |
                             UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                             UCP_OP_ATTR_FIELD_FLAGS;
    req_param.datatype     = ucp_dt_make_contig(msglen);
    req_param.cb.send      = ucc_tl_ucp_send_completion_cb;
    req_param.memory_type  = ucc_memtype_to_ucs[mtype];
    req_param.user_data    = (void*)task;
    req_param.flags        = UCP_AM_SEND_FLAG_EAGER |
                             UCP_AM_SEND_FLAG_COPY_HEADER;
    status = ucc_tl_ucp_get_ep(team, dest_group_rank, &ep);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    task->tagged.send_posted++;
    ucp_status = ucp_am_send_nbx(ep, 1, &ucp_tag, sizeof(ucp_tag), buffer,
                                 1, &req_param);
    if (UCS_OK != ucp_status) {
        UCC_TL_UCP_CHECK_REQ_STATUS();
    } else {
        task->tagged.send_completed++;
    }
    return UCC_OK;
}

static inline ucc_status_t
ucc_tl_ucp_check_am_recv(ucc_tlcp_ucp_example_am_msg_t **recv,
                         ucc_rank_t dest_group_rank, ucc_tl_ucp_team_t *team,
                         ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t      *args       = &TASK_ARGS(task);
    ucc_tl_ucp_context_t *tl_ucp_ctx = TASK_CTX(task);
    ucc_tlcp_ucp_example_context_t *plugin_ctx;
    ucc_tlcp_ucp_example_am_msg_t *entry;
    ucp_tag_t ucp_tag, ucp_tag_mask;

    plugin_ctx = (ucc_tlcp_ucp_example_context_t*)
        tl_ucp_ctx->super.coll_plugin_context;

    UCC_TL_UCP_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask,
                             (args->mask & UCC_COLL_ARGS_FIELD_TAG),
                             task->tagged.tag, dest_group_rank,
                             team->super.super.params.id,
                             team->super.super.params.scope_id,
                             team->super.super.params.scope);

    ucc_assert(ucp_tag_mask != 0);
    ucc_list_for_each(entry, &plugin_ctx->am_list, list_elem) {
        if (entry->tag == ucp_tag) {
            *recv = entry;
            return UCC_OK;
        }
    }
    ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    return UCC_INPROGRESS;
}

static inline void
ucc_tl_ucp_put_am_msg(ucc_tl_ucp_task_t *task,
                      ucc_tlcp_ucp_example_am_msg_t *recv)
{
    ucp_am_data_release(TASK_CTX(task)->worker.ucp_worker, recv->msg);
    ucc_list_del(&recv->list_elem);
    ucc_free(recv);
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_am_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t st;

    st = ucc_tl_ucp_coll_finalize(&task->super);
    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed finalize collective");
    }
    return st;
}

void ucc_tl_ucp_allreduce_knomial_am_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    int                    avg_pre_op = team->cfg.reduce_avg_pre_op;
    ucc_kn_radix_t         radix      = task->allreduce_kn.p.radix;
    uint8_t                node_type  = task->allreduce_kn.p.node_type;
    ucc_knomial_pattern_t *p          = &task->allreduce_kn.p;
    void                  *sbuf       = args->src.info.buffer;
    void                  *rbuf       = args->dst.info.buffer;
    ucc_memory_type_t      mem_type   = args->dst.info.mem_type;
    size_t                 count      = args->dst.info.count;
    ucc_datatype_t         dt         = args->dst.info.datatype;
    size_t                 data_size  = count * ucc_dt_size(dt);
    ucc_rank_t             rank       = task->subset.myrank;
    void                  *send_buf;
    ptrdiff_t              recv_offset;
    ucc_rank_t             peer;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    int                    is_avg, k;
    void *srcs[8];
    ucc_tlcp_ucp_example_am_msg_t *recv;

    if (UCC_IS_INPLACE(*args)) {
        sbuf = rbuf;
    }
    UCC_KN_REDUCE_GOTO_PHASE(task->allreduce_kn.phase);

    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_ep_map_eval(task->subset.map,
                               ucc_knomial_pattern_get_proxy(p, rank));
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_am(sbuf, data_size, mem_type, peer, team, task),
            task, out);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(rbuf, data_size, mem_type, peer, team, task),
            task, out);
    }

UCC_KN_PHASE_EXTRA:
    if (KN_NODE_PROXY == node_type || KN_NODE_EXTRA == node_type) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto completion;
        } else {
            peer = ucc_ep_map_eval(task->subset.map,
                                   ucc_knomial_pattern_get_extra(p, rank));
            status = ucc_tl_ucp_check_am_recv(&recv, peer, team, task);
            if (status == UCC_INPROGRESS) {
                SAVE_STATE(UCC_KN_PHASE_EXTRA);
                return;
            }
            status = ucc_dt_reduce(sbuf, recv->msg, rbuf, count, dt, args, 0, 0,
                                   task->allreduce_kn.executor,
                                   &task->allreduce_kn.etask);

            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.status = status;
                return;
            }
UCC_KN_PHASE_EXTRA_REDUCE:
            EXEC_TASK_TEST(UCC_KN_PHASE_EXTRA_REDUCE,
                           "failed to perform dt reduction",
                           task->allreduce_kn.etask);
            ucc_tl_ucp_put_am_msg(task, recv);
        }
    }
    while(!ucc_knomial_pattern_loop_done(p)) {
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, loop_step);
            if (peer == UCC_KN_PEER_NULL) {
                continue;
            }
            peer = ucc_ep_map_eval(task->subset.map, peer);
            if ((ucc_knomial_pattern_loop_first_iteration(p)) &&
                (KN_NODE_PROXY != node_type) && !UCC_IS_INPLACE(*args)) {
                send_buf = sbuf;
            } else {
                send_buf = rbuf;
            }
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_am(send_buf, data_size, mem_type, peer, team,
                                   task),
                task, out);
        }

    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return;
        }
        recv_offset = 0;
        for (loop_step = 1, k = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, loop_step);
            if (peer == UCC_KN_PEER_NULL) {
                continue;
            }
            peer = ucc_ep_map_eval(task->subset.map, peer);
            status = ucc_tl_ucp_check_am_recv(&recv, peer, team, task);
            if (status != UCC_OK) {
                SAVE_STATE(UCC_KN_PHASE_LOOP);
                return;
            }
            srcs[k] = recv->msg;
            recv_offset += data_size;
            k++;
        }

        if (task->tagged.send_posted > p->iteration * (radix - 1)) {
            if ((ucc_knomial_pattern_loop_first_iteration(p)) &&
                (KN_NODE_PROXY != node_type) && !UCC_IS_INPLACE(*args)) {
                send_buf = sbuf;
            } else {
                send_buf = rbuf;
            }
            is_avg = args->op == UCC_OP_AVG &&
                     (avg_pre_op ? ucc_knomial_pattern_loop_first_iteration(p)
                                 : ucc_knomial_pattern_loop_last_iteration(p));
            srcs[0] = send_buf;
            status = ucc_dt_reduce_vec(
                srcs, rbuf,
                task->tagged.send_posted - p->iteration * (radix - 1) + 1, count,
                dt, args,
                UCC_EEE_TASK_FLAG_REDUCE_SRCS_EXT |
                (is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0),
                AVG_ALPHA(task), task->allreduce_kn.executor,
                &task->allreduce_kn.etask);

            if (ucc_unlikely(UCC_OK != status)) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.status = status;
                return;
            }
UCC_KN_PHASE_REDUCE:
            EXEC_TASK_TEST(UCC_KN_PHASE_REDUCE,
                           "failed to perform dt reduction",
                           task->allreduce_kn.etask);
            for (loop_step = 1; loop_step < radix; loop_step++) {
                peer = ucc_knomial_pattern_get_loop_peer(p, rank, loop_step);
                if (peer == UCC_KN_PEER_NULL)
                    continue;
                peer = ucc_ep_map_eval(task->subset.map, peer);
                status = ucc_tl_ucp_check_am_recv(&recv, peer, team, task);
                ucc_tl_ucp_put_am_msg(task, recv);
            }
        }
        ucc_knomial_pattern_next_iteration(p);
    }
    if (KN_NODE_PROXY == node_type) {
        peer = ucc_ep_map_eval(task->subset.map,
                               ucc_knomial_pattern_get_extra(p, rank));
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(rbuf, data_size, mem_type, peer, team, task),
            task, out);
        goto UCC_KN_PHASE_PROXY;
    } else {
        goto completion;
    }

UCC_KN_PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(UCC_KN_PHASE_PROXY);
        return;
    }

completion:
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_kn_done", 0);
UCC_KN_PHASE_COMPLETE: /* unused label */
out:
    return;
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_am_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         size      = (ucc_rank_t)task->subset.map.ep_num;
    ucc_rank_t         rank      = task->subset.myrank;
    ucc_memory_type_t  mem_type  = TASK_ARGS(task).dst.info.mem_type;
    size_t             count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    size_t             data_size = count * ucc_dt_size(dt);
    ucc_mrange_uint_t *p         = &team->cfg.allreduce_kn_radix;
    ucc_kn_radix_t     cfg_radix;
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_kn_start", 0);
    task->allreduce_kn.phase = UCC_KN_PHASE_INIT;
    ucc_assert(UCC_IS_INPLACE(TASK_ARGS(task)) ||
               (TASK_ARGS(task).src.info.mem_type == mem_type));
    cfg_radix = ucc_tl_ucp_get_radix_from_range(team, data_size,
                                                mem_type, p);
    ucc_knomial_pattern_init(size, rank, ucc_min(cfg_radix, size),
                             &task->allreduce_kn.p);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    status =
        ucc_coll_task_get_executor(&task->super, &task->allreduce_kn.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_am_init(ucc_base_coll_args_t *coll_args,
                                                  ucc_base_team_t *tl_team,
                                                  ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_task_t *task;

    task = ucc_tl_ucp_init_task(coll_args, tl_team);
    if (!task) {
        return UCC_ERR_NO_MEMORY;
    }

    task->super.post     = ucc_tl_ucp_allreduce_knomial_am_start;
    task->super.progress = ucc_tl_ucp_allreduce_knomial_am_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_knomial_am_finalize;
    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;

    *task_h = &task->super;
    return UCC_OK;
}
