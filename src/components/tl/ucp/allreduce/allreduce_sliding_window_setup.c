/**
 * Copyright(c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "../allgather/allgather.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"

static int ucc_tl_ucp_allreduce_sliding_window_register(
    ucp_context_h ucp_context, ucc_tl_ucp_team_t *tl_team,
    struct ucc_tl_ucp_allreduce_sw_export_buf *ebuf, void *packed_memh)
{
    ucs_status_t         ucs_status;
    ucp_mem_map_params_t params = {0};

    ebuf->ucp_context = ucp_context;

    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = packed_memh;

    ucs_status = ucp_mem_map(ucp_context, &params, &ebuf->memh);
    if (UCS_OK != ucs_status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "import using ucp_mem_map() returned error: %s\n",
                 ucs_status_string(ucs_status));
        return 0;
    }

    ucs_status = ucp_rkey_pack(ucp_context, ebuf->memh, &ebuf->packed_key,
                               &ebuf->packed_key_len);
    if (UCS_OK != ucs_status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "ucp_rkey_pack() returned error: %s\n",
                 ucs_status_string(ucs_status));
        return 0;
    }

    return 0;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_task_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *     team,
                                              ucc_tl_ucp_task_t *   task)
{
    ucc_status_t          status    = UCC_OK;
    void *                src_buf   = coll_args->args.src.info.buffer;
    void *                dst_buf   = coll_args->args.dst.info.buffer;
    ucc_rank_t            team_size = (ucc_rank_t)team->params.size;
    ucc_tl_ucp_team_t *   tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *tl_ctx    = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_tl_ucp_allreduce_sw_global_work_buf_info *gwbi_p = NULL;
    size_t allgather_size = sizeof(ucc_tl_ucp_allreduce_sw_host_allgather);
    ucc_tl_ucp_allreduce_sw_host_allgather *allgather_data;

    tl_debug(UCC_TL_TEAM_LIB(tl_team), "allocating pipe\n");
    if ((status = ucc_tl_ucp_allreduce_sliding_window_alloc_pipe(
             coll_args, team, task)) != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "error while allocating pipe");
        goto out;
    }

    allgather_data = ucc_malloc(allgather_size * (team_size + 1));

    gwbi_p = coll_args->args.global_work_buffer;
    task->super.bargs.args.global_work_buffer = gwbi_p;

    task->allreduce_sliding_window.inplace = UCC_IS_INPLACE(coll_args->args);

    task->allreduce_sliding_window.barrier_task = NULL;

    if (!task->allreduce_sliding_window.inplace) {
        task->allreduce_sliding_window.sbufs =
            ucc_malloc(sizeof(void *) * team_size);
        task->allreduce_sliding_window.src_rkeys =
            ucc_malloc(sizeof(ucp_rkey_h) * team_size);
    }

    task->allreduce_sliding_window.rbufs =
        ucc_malloc(sizeof(void *) * team_size);
    task->allreduce_sliding_window.dst_rkeys =
        ucc_malloc(sizeof(ucp_rkey_h) * team_size);
    task->allreduce_sliding_window.eps =
        ucc_malloc(sizeof(ucp_ep_h) * team_size);

    task->allreduce_sliding_window.put_requests =
        task->allreduce_sliding_window.pipe->put_requests;

    if (!task->allreduce_sliding_window.inplace) {
        task->allreduce_sliding_window.src_ebuf =
            ucc_malloc(sizeof(struct ucc_tl_ucp_allreduce_sw_export_buf));
    } else {
        task->allreduce_sliding_window.src_ebuf = NULL;
    }

    task->allreduce_sliding_window.dst_ebuf =
        ucc_malloc(sizeof(struct ucc_tl_ucp_allreduce_sw_export_buf));

    if (!task->allreduce_sliding_window.inplace)
        allgather_data->src_buf = src_buf;

    allgather_data->dst_buf = dst_buf;

    // Register the src and dst bufs
    if (!task->allreduce_sliding_window.inplace) {
        ucc_tl_ucp_allreduce_sliding_window_register(
            tl_ctx->worker.ucp_context, tl_team,
            task->allreduce_sliding_window.src_ebuf, gwbi_p->packed_src_memh);
        memcpy(allgather_data->packed_src_key,
               task->allreduce_sliding_window.src_ebuf->packed_key,
               task->allreduce_sliding_window.src_ebuf->packed_key_len);
    }

    ucc_tl_ucp_allreduce_sliding_window_register(
        tl_ctx->worker.ucp_context, tl_team,
        task->allreduce_sliding_window.dst_ebuf, gwbi_p->packed_dst_memh);
    memcpy(allgather_data->packed_dst_key,
           task->allreduce_sliding_window.dst_ebuf->packed_key,
           task->allreduce_sliding_window.dst_ebuf->packed_key_len);

    task->allreduce_sliding_window.allgather_data      = allgather_data;
    task->allreduce_sliding_window.allgather_scoll_req = NULL;

    return UCC_OK;
out:
    return status;
}

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_allgather_info_finalize(
    ucc_service_coll_req_t *scoll_req, ucc_tl_ucp_task_t *sw_task)
{
    ucc_status_t       status;
    ucc_rank_t         i;
    ucc_base_team_t *  base_team = sw_task->super.team;
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(base_team, ucc_tl_ucp_team_t);
    ucc_rank_t         team_size = base_team->params.size;

    size_t allgather_size = sizeof(ucc_tl_ucp_allreduce_sw_host_allgather);
    ucc_tl_ucp_allreduce_sw_host_allgather *all_host_allgather =
        PTR_OFFSET(scoll_req->data, allgather_size);

    for (i = 0; i < team_size; i++) {
        ucs_status_t ucs_status = UCS_OK;
        ucp_rkey_h   src_unpacked, dst_unpacked;
        ucp_ep_h     ep;

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

        if (!sw_task->allreduce_sliding_window.inplace) {
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

        sw_task->allreduce_sliding_window.eps[i] = ep;
    }

    return status;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_alloc_pipe(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_tl_ucp_task_t *   task)
{
    int                i;
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         team_size = (ucc_rank_t)team->params.size;

    const size_t buf_size =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.allreduce_sliding_window_buf_size;
    int put_window_size = UCC_TL_UCP_TEAM_LIB(tl_team)
                              ->cfg.allreduce_sliding_window_put_window_size;
    int num_get_bufs =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.allreduce_sliding_window_num_get_bufs;

    ucc_tl_ucp_allreduce_sw_pipeline *pipe =
        (ucc_tl_ucp_allreduce_sw_pipeline *)ucc_malloc(
            sizeof(ucc_tl_ucp_allreduce_sw_pipeline));
    if (pipe == NULL) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "error allocating dpu pipe\n");
        return UCC_ERR_NO_RESOURCE;
    }

    if (put_window_size <= 0)
        put_window_size = team_size;
    if (num_get_bufs <= 0)
        num_get_bufs = team_size;

    pipe->accbuf.buf = ucc_malloc(buf_size);
    if (pipe->accbuf.buf == NULL) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "error allocating accbuf\n");
        return UCC_ERR_NO_RESOURCE;
    }
    pipe->getbuf = (ucc_tl_ucp_allreduce_sw_buf *)ucc_malloc(
        num_get_bufs * sizeof(ucc_tl_ucp_allreduce_sw_buf));
    if (pipe->getbuf == NULL) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "error allocating getbuf array\n");
        return UCC_ERR_NO_RESOURCE;
    }
    for (i = 0; i < num_get_bufs; i++) {
        pipe->getbuf[i].buf = ucc_malloc(buf_size);
    }

    pipe->buffer_size  = buf_size;
    pipe->num_buffers  = num_get_bufs;
    pipe->put_requests = (ucs_status_ptr_t *)ucc_malloc(
        put_window_size * sizeof(ucs_status_ptr_t));

    task->allreduce_sliding_window.put_window_size = put_window_size;
    task->allreduce_sliding_window.num_get_bufs    = num_get_bufs;

    task->allreduce_sliding_window.pipe = pipe;

    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_free_gwbi(ucc_coll_task_t *coll_task)
{
    int                   i;
    ucc_base_team_t *     team    = coll_task->team;
    ucc_tl_ucp_team_t *   tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *   task   = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_context_t *tl_ctx = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_tl_ucp_allreduce_sw_pipeline *pipe =
        task->allreduce_sliding_window.pipe;

    if (!task->allreduce_sliding_window.inplace)
        ucc_free(task->allreduce_sliding_window.sbufs);

    ucc_free(task->allreduce_sliding_window.rbufs);
    ucc_free(task->allreduce_sliding_window.eps);
    ucc_free(task->allreduce_sliding_window.allgather_data);

    if (!task->allreduce_sliding_window.inplace) {
        ucp_mem_unmap(tl_ctx->worker.ucp_context,
                      task->allreduce_sliding_window.src_ebuf->memh);
        ucc_free(task->allreduce_sliding_window.src_ebuf);
        ucc_free(task->allreduce_sliding_window.src_rkeys);
    }

    ucp_mem_unmap(tl_ctx->worker.ucp_context,
                  task->allreduce_sliding_window.dst_ebuf->memh);
    ucc_free(task->allreduce_sliding_window.dst_ebuf);
    ucc_free(task->allreduce_sliding_window.dst_rkeys);

    ucc_free(pipe->accbuf.buf);
    for (i = 0; i < task->allreduce_sliding_window.num_get_bufs; i++) {
        ucc_free(pipe->getbuf[i].buf);
    }
    ucc_free(pipe->getbuf);
    ucc_free(pipe->put_requests);
    ucc_free(pipe);

    return UCC_OK;
}

