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
    ucc_tl_ucp_team_t                *tl_team         =
        ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t                        team_size       =
        UCC_TL_TEAM_SIZE(tl_team);
    ucc_tl_ucp_lib_config_t          *cfg             =
        &UCC_TL_UCP_TEAM_LIB(tl_team)->cfg;
    const size_t                      buf_size        =
        cfg->allreduce_sliding_window_buf_size;
    uint32_t                          put_window_size =
        cfg->allreduce_sliding_window_put_window_size;
    uint32_t                          num_get_bufs    =
        cfg->allreduce_sliding_window_num_get_bufs;
    int                               i, j;
    ucc_tl_ucp_allreduce_sw_pipeline *pipe;

    pipe = ucc_malloc(sizeof(ucc_tl_ucp_allreduce_sw_pipeline));
    if (pipe == NULL) {
        goto err;
    }

    ucc_assert(team_size > 0); // Bypass clang linter

    if (put_window_size == 0 || put_window_size > team_size) {
        put_window_size = team_size;
    }

    if (num_get_bufs == 0) {
        num_get_bufs = team_size;
    }

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
    task->allreduce_sliding_window.put_requests =
        task->allreduce_sliding_window.pipe->put_requests;

    return UCC_OK;

free_getbuf:
    for (j = 0; j < i; j++) {
        ucc_free(pipe->getbuf[j].buf);
    }
    ucc_free(pipe->getbuf);
free_acc:
    ucc_free(pipe->accbuf.buf);
free_pipe:
    ucc_free(pipe);
err:
    tl_error(UCC_TL_TEAM_LIB(tl_team), "error allocating sliding window pipe");
    return UCC_ERR_NO_RESOURCE;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_task_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t      *team,
                                              ucc_tl_ucp_task_t    *task)
{
    void                   *src_buf        = coll_args->args.src.info.buffer;
    void                   *dst_buf        = coll_args->args.dst.info.buffer;
    ucc_tl_ucp_team_t      *tl_team        = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t              team_size      = UCC_TL_TEAM_SIZE(tl_team);
    int                     inplace        = UCC_IS_INPLACE(coll_args->args);
    ucc_tl_ucp_allreduce_sw_global_work_buf_info_t
                           *gwbi_p         = NULL;
    size_t                  allgather_size =
        sizeof(ucc_tl_ucp_allreduce_sw_host_allgather_t);
    ucc_tl_ucp_allreduce_sw_host_allgather_t
                           *allgather_data;
    ucc_rank_t              i;
    void                   *buffer;
    void                   *ptr;
    size_t                  bufs_sz, allgather_data_sz, rbufs_sz, dst_rkeys_sz,
                            dst_ebuf_sz, sbufs_sz, src_rkeys_sz, src_ebuf_sz;

    ucc_assert(team_size > 0);

    bufs_sz           = sizeof(ucc_tl_ucp_dpu_offload_buf_info_t);
    allgather_data_sz = allgather_size * (team_size + 1);
    rbufs_sz          = sizeof(void *) * team_size;
    dst_rkeys_sz      = sizeof(ucp_rkey_h) * team_size;
    dst_ebuf_sz       = sizeof(struct ucc_tl_ucp_allreduce_sw_export_buf);

    if (!inplace) {
        sbufs_sz      = sizeof(void *) * team_size;
        src_rkeys_sz  = sizeof(ucp_rkey_h) * team_size;
        src_ebuf_sz   = sizeof(struct ucc_tl_ucp_allreduce_sw_export_buf);
    } else {
        sbufs_sz      = 0;
        src_rkeys_sz  = 0;
        src_ebuf_sz   = 0;
    }

    buffer = ucc_malloc(bufs_sz + allgather_data_sz + rbufs_sz +
                        dst_rkeys_sz + dst_ebuf_sz + sbufs_sz +
                        src_rkeys_sz + src_ebuf_sz);
    if (buffer == NULL) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "error while allocating task");
        return UCC_ERR_NO_RESOURCE;
    }

    ptr = buffer;

    task->allreduce_sliding_window.bufs = ptr;

    ptr = allgather_data = PTR_OFFSET(ptr, bufs_sz);
    task->allreduce_sliding_window.allgather_data = allgather_data;

    gwbi_p = coll_args->args.global_work_buffer;
    task->super.bargs.args.global_work_buffer = gwbi_p;

    task->allreduce_sliding_window.reduce_task = NULL;

    ptr = task->allreduce_sliding_window.bufs->rbufs = PTR_OFFSET(ptr, allgather_data_sz);
    ptr = task->allreduce_sliding_window.bufs->dst_rkeys = PTR_OFFSET(ptr, rbufs_sz);
    for (i = 0; i < team_size; i++) {
        task->allreduce_sliding_window.bufs->dst_rkeys[i] = NULL;
    }

    ptr = task->allreduce_sliding_window.bufs->dst_ebuf = PTR_OFFSET(ptr, dst_rkeys_sz);
    task->allreduce_sliding_window.bufs->dst_ebuf->memh = NULL;

    allgather_data->dst_buf = dst_buf;

    task->allreduce_sliding_window.allgather_data = allgather_data;
    task->allreduce_sliding_window.allgather_task = NULL;

    if (!inplace) {
        allgather_data->src_buf = src_buf;

        ptr = task->allreduce_sliding_window.bufs->sbufs = PTR_OFFSET(ptr, dst_ebuf_sz);
        ptr = task->allreduce_sliding_window.bufs->src_rkeys = PTR_OFFSET(ptr, sbufs_sz);
        for (i = 0; i < team_size; i++) {
            task->allreduce_sliding_window.bufs->src_rkeys[i] = NULL;
        }

        task->allreduce_sliding_window.bufs->src_ebuf = PTR_OFFSET(ptr, src_rkeys_sz);
        task->allreduce_sliding_window.bufs->src_ebuf->memh = NULL;
    } else {
        task->allreduce_sliding_window.bufs->src_ebuf = NULL;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_allgather_info_finalize(
                ucc_tl_ucp_task_t *sw_task)
{
    ucs_status_t       ucs_status = UCS_OK;
    ucc_base_team_t   *base_team  = sw_task->super.team;
    ucc_tl_ucp_team_t *tl_team    = ucc_derived_of(base_team, ucc_tl_ucp_team_t);
    ucc_rank_t         team_size  = UCC_TL_TEAM_SIZE(tl_team);
    void              *recvbuf    = sw_task->allreduce_sliding_window.
                                    allgather_task->bargs.args.dst.info.buffer;
    ucc_tl_ucp_allreduce_sw_host_allgather_t *all_host_allgather = recvbuf;
    ucc_status_t       status     = UCC_OK;
    int                inplace    = UCC_IS_INPLACE(TASK_ARGS(sw_task));
    ucc_rank_t         i;
    ucp_ep_h           ep;
    ucp_rkey_h         src_unpacked, dst_unpacked;

    ucc_assert(team_size > 0);

    for (i = 0; i < team_size; i++) {
        status = ucc_tl_ucp_get_ep(tl_team, i, &ep);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }

        ucs_status = ucp_ep_rkey_unpack(
            ep, all_host_allgather[i].packed_dst_key, &dst_unpacked);
        if (UCS_OK != ucs_status) {
            tl_error(UCC_TL_TEAM_LIB(tl_team), "dst rkey unpack failed");
            return ucs_status_to_ucc_status(ucs_status);
        }

        sw_task->allreduce_sliding_window.bufs->rbufs[i] =
            all_host_allgather[i].dst_buf;
        sw_task->allreduce_sliding_window.bufs->dst_rkeys[i] = dst_unpacked;

        if (!inplace) {
            ucs_status = ucp_ep_rkey_unpack(
                ep, all_host_allgather[i].packed_src_key, &src_unpacked);
            if (UCS_OK != ucs_status) {
                tl_error(UCC_TL_TEAM_LIB(tl_team), "src rkey unpack failed");
                return ucs_status_to_ucc_status(ucs_status);
            }

            sw_task->allreduce_sliding_window.bufs->sbufs[i] =
                all_host_allgather[i].src_buf;
            sw_task->allreduce_sliding_window.bufs->src_rkeys[i] = src_unpacked;
        } else {
            sw_task->allreduce_sliding_window.bufs->sbufs =
                sw_task->allreduce_sliding_window.bufs->rbufs;
            sw_task->allreduce_sliding_window.bufs->src_rkeys =
                sw_task->allreduce_sliding_window.bufs->dst_rkeys;
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

    if (task->allreduce_sliding_window.bufs) {
        if (!inplace) {
            if (task->allreduce_sliding_window.bufs->src_ebuf->memh != NULL) {
                ucp_mem_unmap(tl_ctx->worker.ucp_context,
                              task->allreduce_sliding_window.bufs->src_ebuf->memh);
                task->allreduce_sliding_window.bufs->src_ebuf->memh = NULL;
            }
        }

        if (task->allreduce_sliding_window.bufs->dst_ebuf->memh != NULL) {
            ucp_mem_unmap(tl_ctx->worker.ucp_context,
                          task->allreduce_sliding_window.bufs->dst_ebuf->memh);
        }
        ucc_free(task->allreduce_sliding_window.bufs);
    }
}

void
ucc_tl_ucp_allreduce_sliding_window_free_pipe(ucc_coll_task_t *coll_task)
{
    ucc_base_team_t                  *team         = coll_task->team;
    ucc_tl_ucp_team_t                *tl_team      =
        ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t                        team_size    = UCC_TL_TEAM_SIZE(tl_team);
    ucc_tl_ucp_task_t                *task         =
        ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_allreduce_sw_pipeline *pipe         =
        task->allreduce_sliding_window.pipe;
    int                               num_get_bufs =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.allreduce_sliding_window_num_get_bufs;
    int                               i;

    if (num_get_bufs == 0) {
        num_get_bufs = team_size;
    }

    if (pipe) {
        ucc_free(pipe->accbuf.buf);
        for (i = 0; i < num_get_bufs; i++) {
            ucc_free(pipe->getbuf[i].buf);
        }
        ucc_free(pipe->getbuf);
        ucc_free(pipe->put_requests);
        ucc_free(pipe);
    }
}
