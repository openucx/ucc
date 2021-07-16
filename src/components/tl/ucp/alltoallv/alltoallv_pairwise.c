/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_mc.h"
#include "cuda_runtime.h"

#define INTRA_PPN (8)

typedef struct {
    void  *d_ptr;
    size_t size;
    cudaIpcMemHandle_t handle;
    size_t offset;
    size_t displ[INTRA_PPN];
} mem_info_t;

static inline ucc_rank_t get_recv_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank + step) % size;
}

static inline ucc_rank_t get_send_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank - step + size) % size;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = task->team;
    ptrdiff_t          sbuf  = (ptrdiff_t)task->args.src.info_v.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)task->args.dst.info_v.buffer;
    ucc_memory_type_t  smem  = task->args.src.info_v.mem_type;
    ucc_memory_type_t  rmem  = task->args.dst.info_v.mem_type;
    ucc_rank_t         grank = team->rank;
    ucc_rank_t         gsize = team->size;
    int                polls = 0;
    ucc_rank_t         peer;
    int                posts, nreqs;//, count_stride, displ_stride;
    size_t             rdt_size, sdt_size, data_size, data_displ;
    ucc_rank_t         intra_rank_start, intra_rank_end;

    if (task->alltoall_intra.info) {
        intra_rank_start = ucs_align_down(team->rank, INTRA_PPN);
        intra_rank_end   = ucs_min(intra_rank_start + INTRA_PPN, team->size) - 1;
    } else {
        intra_rank_start = intra_rank_end = grank;
    }


    posts    = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_pairwise_num_posts;
    nreqs    = (posts > gsize || posts == 0) ? gsize : posts;
    rdt_size = ucc_dt_size(task->args.src.info_v.datatype);
    sdt_size = ucc_dt_size(task->args.dst.info_v.datatype);
    while ((task->send_posted < gsize || task->recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->ucp_worker);
        while ((task->recv_posted < gsize) &&
               ((task->recv_posted - task->recv_completed) < nreqs)) {
            peer       = get_recv_peer(intra_rank_start, gsize, task->recv_posted);
            data_size  = ucc_coll_args_get_count(&task->args,
                            task->args.dst.info_v.counts, peer) * rdt_size;
            data_displ = ucc_coll_args_get_displacement(&task->args,
                            task->args.dst.info_v.displacements,
                            peer) * rdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nz((void *)(rbuf + data_displ),
                                             data_size, rmem, peer, team, task),
                          task, out);
           // tl_warn(UCC_TL_TEAM_LIB(team), "UCX recv posted [%d:%d]", team->rank, peer);

            polls = 0;
        }
        while ((task->send_posted < gsize) &&
               ((task->send_posted - task->send_completed) < nreqs)) {
            peer       = get_send_peer(intra_rank_end, gsize, task->send_posted);
            data_size  = ucc_coll_args_get_count(&task->args,
                            task->args.src.info_v.counts, peer) * sdt_size;
            data_displ = ucc_coll_args_get_displacement(&task->args,
                            task->args.src.info_v.displacements,
                            peer) * sdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nz((void *)(sbuf + data_displ),
                                             data_size, smem, peer, team, task),
                          task, out);
            // tl_warn(UCC_TL_TEAM_LIB(team), "UCX send posted [%d:%d]", team->rank, peer);
            polls = 0;
        }
    }
    if ((task->send_posted < gsize) || (task->recv_posted < gsize)) {
        return task->super.super.status;
    }
    task->super.super.status = ucc_tl_ucp_test(task);
out:
    if (task->super.super.status != UCC_INPROGRESS) {
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                         "ucp_alltoallv_pairwise_done", 0);
    }

    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_alltoallv_pairwise_start",
                                     0);
    ucc_tl_ucp_alltoallv_pairwise_progress(&task->super);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_early_triggered_post(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;
    ptrdiff_t          sbuf  = (ptrdiff_t)task->args.src.info_v.buffer;
    ucc_rank_t intra_rank_start = ucs_align_down(team->rank, INTRA_PPN);
    ucc_rank_t intra_rank_end   = ucs_min(intra_rank_start + INTRA_PPN, team->size) - 1;
    ucc_rank_t intra_rank       = team->rank - intra_rank_start;
    size_t   sdt_size, data_size, data_displ;
    int rank, j, peer;
    ptrdiff_t dst;

    if (task->alltoall_intra.info == NULL) {
        return UCC_OK;
    }

    sdt_size = ucc_dt_size(task->args.src.info_v.datatype);

    for (j=0; j < INTRA_PPN; j++) {
        rank = team->rank + j;
        if (rank > intra_rank_end) {
            rank = intra_rank_start - 1 + ((rank - intra_rank_end) % INTRA_PPN);
        }
        peer = rank - intra_rank_start;
        mem_info_t *peer_info = &((mem_info_t *)task->alltoall_intra.info)[peer];


        if (rank == team->rank) {
            dst = (ptrdiff_t)task->args.dst.info_v.buffer +
                    + peer_info->displ[intra_rank];
        } else {
            dst = (ptrdiff_t) task->alltoall_intra.peer_map_addr[peer] +
                    peer_info->offset + peer_info->displ[intra_rank];
        }

        data_size  = ucc_coll_args_get_count(&task->args,
                            task->args.src.info_v.counts, rank) * sdt_size;
        data_displ = ucc_coll_args_get_displacement(&task->args,
                            task->args.src.info_v.displacements, rank)* sdt_size;

        //printf("SNED [%d: %d] sdispl:%ld rdispl:(%ld:%ld) size:%ld \n", team->rank, rank, data_displ, peer_info->offset, peer_info->displ[intra_rank], data_size);
        CUDACHECK(cudaMemcpyAsync((void *)dst, (void *)(sbuf + data_displ), data_size, cudaMemcpyDeviceToDevice, (cudaStream_t)coll_task->ee->ee_context));
        task->send_completed++;
        task->recv_completed++;
        task->send_posted++;
        task->recv_posted++;
    }

    ucc_free(task->alltoall_intra.info);
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = task->team;

    task->super.post     = ucc_tl_ucp_alltoallv_pairwise_start;
    task->super.progress = ucc_tl_ucp_alltoallv_pairwise_progress;
    task->super.early_triggered_post  = ucc_tl_ucp_alltoallv_pairwise_early_triggered_post;

    if ( UCC_TL_UCP_TEAM_CTX(team)->cfg.alltoall_use_ipc &&
            task->args.src.info_v.mem_type == UCC_MEMORY_TYPE_CUDA &&
            task->args.dst.info_v.mem_type == UCC_MEMORY_TYPE_CUDA) {
        ucc_status_t status;
        void *base_address;
        size_t alloc_length, rdt_size;
        int i, j;
        mem_info_t my_info;
        mem_info_t *peer_info;
        void *mapped_addr;
        ucc_rank_t intra_rank_start = ucs_align_down(team->rank, INTRA_PPN);
        ucc_rank_t intra_rank_end   = ucs_min(intra_rank_start + INTRA_PPN, team->size) - 1;


        ucc_tl_ucp_get_alloc_info(task->args.dst.info_v.buffer,
                                  (ucc_coll_args_get_total_count(&task->args,
                                   task->args.dst.info_v.counts, team->size ) *
                                   ucc_dt_size(task->args.dst.info_v.datatype)),
                                   &base_address, &alloc_length);

        CUDACHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &my_info.handle, base_address));

        rdt_size = ucc_dt_size(task->args.dst.info_v.datatype);

        my_info.d_ptr  = base_address;
        my_info.size   = alloc_length;
        my_info.offset = task->args.dst.info_v.buffer - base_address;
        for (i = intra_rank_start, j = 0; i <= intra_rank_end; i++, j++) {
            my_info.displ[j] =  ucc_coll_args_get_displacement(&task->args,
                                task->args.dst.info_v.displacements,i) * rdt_size;
            //printf("[%d: %d] offset:%ld displ:%ld\n", team->rank, i, my_info.offset, my_info.displ[j]);
        }


        ucc_tl_iface_t  *tl_iface = UCC_TL_TEAM_IFACE(team->super.super.team->service_team);
        ucc_tl_team_subset_t subset = {
            .map.type   = UCC_EP_MAP_STRIDED,
            .map.ep_num = ucc_min(team->size, INTRA_PPN),
            .map.strided.start = intra_rank_start,
            .map.strided.stride = 1,
            .myrank     = team->rank - intra_rank_start
        };
        peer_info = (mem_info_t *)ucc_malloc(sizeof(my_info) * team->super.super.team->size, "alltoall service coll");
        ucc_assert(peer_info);
        status = tl_iface->scoll.allgather(&team->super.super.team->service_team->super, (void *)&my_info, peer_info,
                                           sizeof(my_info), subset, &team->super.super.team->task);
        if (status < 0) {
            ucc_error("failed to start service allgather for memhandle exchange failed: %s",
                      ucc_status_string(status));
            return status;
        }

        while (1) {
            ucc_context_progress(team->super.super.team->contexts[0]);
            status = tl_iface->scoll.test(team->super.super.team->task);
            if (status < 0) {
                ucc_error("service allgather test failure: %s",
                        ucc_status_string(status));
                return status;
            } else if (status == UCC_OK) {
                break;
            }
        }
        tl_iface->scoll.cleanup(team->super.super.team->task);
        team->super.super.team->task = NULL;

        //printf("[%d] intra_rank_start: %d intra_rank_end:%d", team->rank, intra_rank_start, intra_rank_end);
        for (i=intra_rank_start,j = 0 ; i <= intra_rank_end; i++, j++) {
            if (i != team->rank) {
                //tl_warn(UCC_TL_TEAM_LIB(team), "opening memhandl for [%d:%d]", team->rank, i);
                status = ucc_cuda_ipc_map_memhandle(peer_info[j].d_ptr, peer_info[j].size,
                                                peer_info[j].handle, &mapped_addr,
                                                UCC_TL_UCP_TEAM_CTX(team)->ipc_cache[j]);
                if (UCC_OK != status) {
                    ucc_error("ucc_cuda_ipc_map_memhandle failed");
                    return UCC_ERR_INVALID_PARAM;
                }
                ucc_assert(j < INTRA_PPN);
                task->alltoall_intra.peer_map_addr[j] = mapped_addr;
            }
        }

        task->alltoall_intra.info = peer_info;
    } else {
        task->alltoall_intra.info = NULL;
    }

    task->n_polls = ucc_min(1, task->n_polls);
    if (UCC_TL_UCP_TEAM_CTX(team)->cfg.pre_reg_mem) {
        if (task->args.flags & UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER) {
            ucc_tl_ucp_pre_register_mem(team, task->args.src.info_v.buffer,
                                        (ucc_coll_args_get_total_count(&task->args,
                                         task->args.src.info_v.counts, team->size ) *
                                         ucc_dt_size(task->args.src.info_v.datatype)),
                                        task->args.src.info_v.mem_type);
        }

        if (task->args.flags & UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER) {
            ucc_tl_ucp_pre_register_mem(team, task->args.dst.info_v.buffer,
                                        (ucc_coll_args_get_total_count(&task->args,
                                         task->args.dst.info_v.counts, team->size) *
                                         ucc_dt_size(task->args.dst.info_v.datatype)),
                                        task->args.dst.info_v.mem_type);
        }
    }

    ucc_tl_ucp_task_reset(task);

    return UCC_OK;
}
