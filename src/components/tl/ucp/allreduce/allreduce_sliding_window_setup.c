/**
 * Copyright(c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "allreduce_sliding_window.h"
#include "../allgather/allgather.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_alloc_pipe(ucc_base_team_t   *team,
                                               ucc_tl_ucp_task_t *task)
{
    int                      i;
    ucc_tl_ucp_team_t       *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t               team_size = (ucc_rank_t)team->params.size;
    ucc_tl_ucp_lib_config_t *cfg       = &UCC_TL_UCP_TEAM_LIB(tl_team)->cfg;

    const size_t buf_size = cfg->allreduce_sliding_window_buf_size;
    int   put_window_size = cfg->allreduce_sliding_window_put_window_size;
    int      num_get_bufs = cfg->allreduce_sliding_window_num_get_bufs;

    ucc_tl_ucp_allreduce_sw_pipeline *pipe =
        (ucc_tl_ucp_allreduce_sw_pipeline *)ucc_malloc(
            sizeof(ucc_tl_ucp_allreduce_sw_pipeline));
    if (pipe == NULL) {
        goto err;
    }

    if (put_window_size <= 0) {
        put_window_size = team_size;
    }

    if (num_get_bufs <= 0) {
        num_get_bufs = team_size;
    }

    ucc_assert(num_get_bufs > 0);
    ucc_assert(put_window_size > 0);

    pipe->accbuf.buf = ucc_malloc(buf_size);
    if (pipe->accbuf.buf == NULL) {
        goto free_pipe;
    }
    pipe->getbuf = (ucc_tl_ucp_allreduce_sw_buf_t *)ucc_malloc(
        num_get_bufs * sizeof(ucc_tl_ucp_allreduce_sw_buf_t));
    if (pipe->getbuf == NULL) {
        goto free_acc;
    }
    for (i = 0; i < num_get_bufs; i++) {
        pipe->getbuf[i].buf = NULL;
    }
    for (i = 0; i < num_get_bufs; i++) {
        pipe->getbuf[i].buf = ucc_malloc(buf_size);
        if (pipe->getbuf[i].buf == NULL) {
            goto free_getbuf;
        }
    }

    pipe->buffer_size  = buf_size;
    pipe->num_buffers  = num_get_bufs;
    pipe->put_requests = (ucs_status_ptr_t *)ucc_malloc(
        put_window_size * sizeof(ucs_status_ptr_t));
    if (pipe->put_requests == NULL) {
        goto free_getbuf;
    }

    task->allreduce_sliding_window.pipe = pipe;

    return UCC_OK;

free_getbuf:
    for (i = 0; i < num_get_bufs; i++) {
        if (pipe->getbuf[i].buf == NULL)
            break;
        ucc_free(pipe->getbuf[i].buf);
    }
    ucc_free(pipe->getbuf);
free_acc:
    ucc_free(pipe->accbuf.buf);
free_pipe:
    ucc_free(pipe);
err:
    tl_error(UCC_TL_TEAM_LIB(tl_team), "error allocating sliding window pipe\n");
    return UCC_ERR_NO_RESOURCE;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_task_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t      *team,
                                              ucc_tl_ucp_task_t    *task)
{
    ucc_tl_ucp_allreduce_sw_host_allgather_t *allgather_data;
    void                 *src_buf   = coll_args->args.src.info.buffer;
    void                 *dst_buf   = coll_args->args.dst.info.buffer;
    ucc_tl_ucp_team_t    *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t            team_size = UCC_TL_TEAM_SIZE(tl_team);
    int                   inplace   = UCC_IS_INPLACE(coll_args->args);
    ucc_tl_ucp_allreduce_sw_global_work_buf_info_t *gwbi_p = NULL;
    size_t allgather_size = sizeof(ucc_tl_ucp_allreduce_sw_host_allgather_t);

    ucc_assert(team_size > 0);

    if (ucc_tl_ucp_allreduce_sliding_window_alloc_pipe(team, task) != UCC_OK) {
        goto err;
    }

    allgather_data = ucc_malloc(allgather_size * (team_size + 1));
    if (allgather_data == NULL) {
        goto free_pipe;
    }

    gwbi_p = coll_args->args.global_work_buffer;
    task->super.bargs.args.global_work_buffer = gwbi_p;

    task->allreduce_sliding_window.barrier_task = NULL;
    task->allreduce_sliding_window.reduce_task = NULL;

    task->allreduce_sliding_window.rbufs =
        ucc_malloc(sizeof(void *) * team_size);
    if (task->allreduce_sliding_window.rbufs == NULL) {
        goto free_allgather_data;
    }
    task->allreduce_sliding_window.dst_rkeys =
        ucc_malloc(sizeof(ucp_rkey_h) * team_size);
    if (task->allreduce_sliding_window.dst_rkeys == NULL) {
        goto free_rbufs;
    }

    task->allreduce_sliding_window.put_requests =
        task->allreduce_sliding_window.pipe->put_requests;

    task->allreduce_sliding_window.dst_ebuf =
        ucc_malloc(sizeof(struct ucc_tl_ucp_allreduce_sw_export_buf));
    if (task->allreduce_sliding_window.dst_ebuf == NULL) {
        goto free_dst_rkeys;
    }

    allgather_data->dst_buf = dst_buf;

    task->allreduce_sliding_window.allgather_data = allgather_data;
    task->allreduce_sliding_window.allgather_task = NULL;

    if (!inplace) {
        allgather_data->src_buf = src_buf;

        task->allreduce_sliding_window.sbufs =
            ucc_malloc(sizeof(void *) * team_size);
        if (task->allreduce_sliding_window.sbufs == NULL) {
            goto free_dst_ebuf;
        }
        task->allreduce_sliding_window.src_rkeys =
            ucc_malloc(sizeof(ucp_rkey_h) * team_size);
        if (task->allreduce_sliding_window.src_rkeys == NULL) {
            goto free_sbufs;
        }

        task->allreduce_sliding_window.src_ebuf =
            ucc_malloc(sizeof(struct ucc_tl_ucp_allreduce_sw_export_buf));
        if (task->allreduce_sliding_window.src_ebuf == NULL) {
            goto free_src_rkeys;
        }
    } else {
        task->allreduce_sliding_window.src_ebuf = NULL;
    }

    return UCC_OK;

free_src_rkeys:
    ucc_free(task->allreduce_sliding_window.src_rkeys);
free_sbufs:
    ucc_free(task->allreduce_sliding_window.sbufs);
free_dst_ebuf:
    ucc_free(task->allreduce_sliding_window.dst_ebuf);
free_dst_rkeys:
    ucc_free(task->allreduce_sliding_window.dst_rkeys);
free_rbufs:
    ucc_free(task->allreduce_sliding_window.rbufs);
free_allgather_data:
    ucc_free(allgather_data);
free_pipe:
    ucc_tl_ucp_allreduce_sliding_window_free_pipe(&task->super);
err:
    tl_error(UCC_TL_TEAM_LIB(tl_team), "error while allocating task");
    return UCC_ERR_NO_RESOURCE;
}

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_allgather_info_finalize(
                ucc_tl_ucp_task_t *sw_task)
{
    ucc_rank_t         i;
    ucp_ep_h           ep;
    ucp_rkey_h         src_unpacked, dst_unpacked;
    ucs_status_t       ucs_status = UCS_OK;
    ucc_base_team_t   *base_team  = sw_task->super.team;
    ucc_tl_ucp_team_t *tl_team    = ucc_derived_of(base_team, ucc_tl_ucp_team_t);
    ucc_rank_t         team_size  = base_team->params.size;
    void              *recvbuf    = sw_task->allreduce_sliding_window.
                                    allgather_task->bargs.args.dst.info.buffer;
    ucc_tl_ucp_allreduce_sw_host_allgather_t *all_host_allgather = recvbuf;
    ucc_status_t       status     = UCC_OK;
    int                inplace    = UCC_IS_INPLACE(sw_task->super.bargs.args);

    ucc_assert(team_size > 0);

    for (i = 0; i < team_size; i++) {
        status = ucc_tl_ucp_get_ep(tl_team, i, &ep);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }

        ucs_status = ucp_ep_rkey_unpack(
            ep, all_host_allgather[i].packed_dst_key, &dst_unpacked);
        if (UCS_OK != ucs_status) {
            tl_error(UCC_TL_TEAM_LIB(tl_team), "dst rkey unpack failed\n");
            return UCC_ERR_NO_RESOURCE;
        }

        sw_task->allreduce_sliding_window.rbufs[i] =
            all_host_allgather[i].dst_buf;
        sw_task->allreduce_sliding_window.dst_rkeys[i] = dst_unpacked;

        if (!inplace) {
            ucs_status = ucp_ep_rkey_unpack(
                ep, all_host_allgather[i].packed_src_key, &src_unpacked);
            if (UCS_OK != ucs_status) {
                tl_error(UCC_TL_TEAM_LIB(tl_team), "src rkey unpack failed\n");
                return UCC_ERR_NO_RESOURCE;
            }

            sw_task->allreduce_sliding_window.sbufs[i] =
                all_host_allgather[i].src_buf;
            sw_task->allreduce_sliding_window.src_rkeys[i] = src_unpacked;
        } else {
            sw_task->allreduce_sliding_window.sbufs =
                sw_task->allreduce_sliding_window.rbufs;
            sw_task->allreduce_sliding_window.src_rkeys =
                sw_task->allreduce_sliding_window.dst_rkeys;
        }
    }

    return status;
}

void
ucc_tl_ucp_allreduce_sliding_window_free_task(ucc_coll_task_t *coll_task)
{
    ucc_base_team_t      *team    = coll_task->team;
    ucc_tl_ucp_team_t    *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t    *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    int                   inplace = UCC_IS_INPLACE(coll_task->bargs.args);
    ucc_tl_ucp_context_t *tl_ctx  = UCC_TL_UCP_TEAM_CTX(tl_team);

    if (!inplace) {
        ucc_free(task->allreduce_sliding_window.sbufs);
    }

    ucc_free(task->allreduce_sliding_window.rbufs);
    ucc_free(task->allreduce_sliding_window.allgather_data);

    if (!inplace) {
        ucp_mem_unmap(tl_ctx->worker.ucp_context,
                      task->allreduce_sliding_window.src_ebuf->memh);
        ucc_free(task->allreduce_sliding_window.src_ebuf);
        ucc_free(task->allreduce_sliding_window.src_rkeys);
    }

    ucp_mem_unmap(tl_ctx->worker.ucp_context,
                  task->allreduce_sliding_window.dst_ebuf->memh);
    ucc_free(task->allreduce_sliding_window.dst_ebuf);
    ucc_free(task->allreduce_sliding_window.dst_rkeys);
}

void
ucc_tl_ucp_allreduce_sliding_window_free_pipe(ucc_coll_task_t *coll_task)
{
    int                   i;
    ucc_base_team_t      *team    = coll_task->team;
    ucc_tl_ucp_team_t    *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t    *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_allreduce_sw_pipeline *pipe =
        task->allreduce_sliding_window.pipe;
    int num_get_bufs =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.allreduce_sliding_window_num_get_bufs;

    ucc_free(pipe->accbuf.buf);
    for (i = 0; i < num_get_bufs; i++) {
        ucc_free(pipe->getbuf[i].buf);
    }
    ucc_free(pipe->getbuf);
    ucc_free(pipe->put_requests);
    ucc_free(pipe);
}
