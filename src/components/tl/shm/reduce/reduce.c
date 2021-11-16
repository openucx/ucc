/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_shm.h"
#include "reduce.h"

ucc_status_t ucc_tl_shm_reduce_progress(ucc_coll_task_t *coll_task)
{
//    ucc_tl_shm_task_t  *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
//    ucc_tl_shm_team_t  *team = TASK_TEAM(task);
//    ucc_coll_args_t     args = TASK_ARGS(task);
//    ucc_shm_seg_data_t *shm_seg_data = team->shm_seg_data;
//    ucc_rank_t          rank = team->rank;
//    ucc_memory_type_t   mtype;
//    ucc_datatype_t      dt;
//    size_t              count, data_size;
//
//    int64_t sequence_number = shm_seg_data->seq_num;
//    shmem_sync_t *shmem;
//
//    int* ar64_radix_array       = team->shm_seg.ar64_radix_array;
//    int ar64_radix_array_length = team->shm_seg.ar64_radix_array_length;
//    int *ar2k_radix_array       = team->shm_seg.ar2k_radix_array;
//    int ar2k_radix_array_length = team->shm_seg.ar2k_radix_array_length;
//
//    int i = 0, matched = 0, radix_i = 0, partner_rank = 0, poll_probe_count = BCOL_SHMSEG_PROBE_COUNT;
//
//    if (rank == 0) {
//        count = args.dst.info.count;
//        mtype = args.dst.info.mem_type;
//        dt    = args.dst.info.datatype;
//    } else {
//        count = args.src.info.count;
//        mtype = args.src.info.mem_type;
//        dt    = args.src.info.datatype;
//    }
//    data_size = count * ucc_dt_size(dt);
//
//    if (task->use_small) {
//        /* Works up to 64bytes */
//        shmem = task->shmseg_base;
//        for (radix_i = task->step; radix_i < ar64_radix_array_length; radix_i++) {
//            matched = 0;
//            partner_rank = ar64_radix_array[radix_i];
//            for (i = 0; i < poll_probe_count; i++) {
//                if (SHMEM_STATE(shmem, partner_rank, SHMEM_LEAF) == sequence_number) {
//                    SHMSEG_ISYNC();
//                    ucc_dt_reduce(aSHMEM_STATE(shmem, partner_rank, SHMEM_DATA),
//                                   aSHMEM_STATE(shmem, rank, SHMEM_DATA),
//                                   aSHMEM_STATE(shmem, rank, SHMEM_DATA),
//                                   count, dt, mtype, &args);
//                    matched++;
//                    break;
//                }
//            }
//            if (!matched) {
//                task->step = radix_i;
//                return UCC_INPROGRESS;
//            }
//        }
//        SHMSEG_WMB();
//
//        /* coverity[copy_paste_error] */
//        if (rank > 0) {
//            SHMEM_STATE(shmem, rank, SHMEM_LEAF) = sequence_number;
//        }
//        goto completion;
//    } else {
//        /* Works for up to 2K bytes */
//        /* This flag is set at ml, and checks the following:
//         * msg_size < 2k, blocking, is !fragmented and sbgp_shmseg is used
//         */
//        shmem = team->shm_seg.ar2k_sync_shmseg;
//        void *ar2k_data_ptr = task->shmseg_base;
//
//        void *my_2k_ptr = NULL, *partner_2k_ptr = NULL;
//        int  my_2k_offset = 0, partner_2k_offset = 0;
//
//        my_2k_offset = SHMEM_STATE(shmem, rank, SHMEM_2K_OFFSET);
//        my_2k_ptr = (void *)((char *)ar2k_data_ptr + my_2k_offset);
//        /* Fan-In */
//        for (radix_i = task->step; radix_i < ar2k_radix_array_length; radix_i++) {
//            matched=0;
//            partner_rank = ar2k_radix_array[radix_i];
//            for (i = 0; i < poll_probe_count; i++) {
//                if (SHMEM_STATE(shmem, partner_rank, SHMEM_LEAF) == sequence_number) {
//                    SHMSEG_ISYNC();
//                    partner_2k_offset = SHMEM_STATE(shmem, partner_rank, SHMEM_2K_OFFSET);
//                    partner_2k_ptr = (void *) ((char *)ar2k_data_ptr + partner_2k_offset);
//
//                    ucc_dt_reduce(partner_2k_ptr, my_2k_ptr, my_2k_ptr,
//                                  count, dt, mtype, &args);
//                    matched++;
//                    break; /* Done with this tree level */
//                }
//            }
//            if(!matched) {
//                /* Partner is slow */
//                task->step = radix_i;
//                return UCC_INPROGRESS;
//            }
//        }
//        SHMSEG_WMB();
//        if (rank > 0) {
//            SHMEM_STATE(shmem, rank, SHMEM_LEAF) = (sequence_number);
//        }
//        goto completion;
//    }
//
//completion:
//    if (rank == 0) {
//        memcpy(args.dst.info.buffer, task->shmseg_dest, data_size);
//    }
//    task->super.super.status = UCC_OK;
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_reduce_start(ucc_coll_task_t *coll_task)
{
//    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
//    ucc_tl_shm_team_t *team = TASK_TEAM(task);
//    ucc_coll_args_t    args = TASK_ARGS(task);
//    void *ptr;
//    ucc_status_t status = UCC_OK;
//    ucc_rank_t rank = team->rank;
//    ucc_shm_seg_data_t *shm_seg_data = team->shm_seg_data;
//    size_t data_size, count;
//    ucc_datatype_t dt;
//
//    if (rank == 0) {
//        count = args.dst.info.count;
//        dt    = args.dst.info.datatype;
//    } else {
//        count = args.src.info.count;
//        dt    = args.src.info.datatype;
//    }
//    data_size = count * ucc_dt_size(dt);
//    shm_seg_data->seq_num++;
//
//    if (data_size <= AR_SMALL_MAX) {
//        task->use_small = 1;
//        if (shm_seg_data->seq_num & 1) {
//            ptr = (void*) ((char *) shm_seg_data->ar128b_shmseg[0] +
//                           (128 * team->rank) + 16
//                           /* first 16 reserved for sync.*/);
//            task->shmseg_base = (void *)shm_seg_data->ar128b_shmseg[0];
//        } else {
//            ptr = (void*) ((char *) shm_seg_data->ar128b_shmseg[1] +
//                           (128 * team->rank) + 16);
//            task->shmseg_base = (void *)shm_seg_data->ar128b_shmseg[1];
//        }
//    } else {
//        task->use_small = 0;
//        if (shm_seg_data->seq_num & 1) {
//            ptr = (void*) ((char *) shm_seg_data->ar2k_data_shmseg[0] +
//                           (SHMEM_2K * team->rank));
//            task->shmseg_base = (void *)shm_seg_data->ar2k_data_shmseg[0];
//        }
//        else {
//            ptr = (void*) ((char *) shm_seg_data->ar2k_data_shmseg[1] +
//                           (SHMEM_2K * team->rank));
//            task->shmseg_base = (void *)shm_seg_data->ar2k_data_shmseg[1];
//        }
//    }
//    task->step = 0;
//    task->shmseg_dest = (void *)ptr;
//    memcpy(ptr, args.src.info.buffer, data_size);
//    task->super.super.status = UCC_INPROGRESS;
//
//    status = ucc_tl_shm_reduce_progress(&task->super);
//    if (UCC_INPROGRESS == status) {
//        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
//        return UCC_OK;
//    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_shm_reduce_init(ucc_tl_shm_task_t *task)
{
    ucc_coll_args_t args = TASK_ARGS(task);
//    if (args.root != 0) {
//    	return UCC_ERR_NOT_SUPPORTED;
//    }
    task->super.post     = ucc_tl_shm_reduce_start;
    task->super.progress = ucc_tl_shm_reduce_progress;
    return UCC_OK;
}
