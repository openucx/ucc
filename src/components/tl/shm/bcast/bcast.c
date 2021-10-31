/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_shm.h"
#include "bcast.h"

ucc_status_t ucc_tl_shm_bcast_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    ucc_shm_seg_data_t *shm_seg_data = team->shm_seg_data;
    size_t data_size = args.src.info.count *
                       ucc_dt_size(args.src.info.datatype);
    int64_t sequence_number = shm_seg_data->seq_num;
    shmem_sync_t *shmem;
    int i = 0, radix_i = 0, n_ready = 0, poll_probe_count = BCOL_SHMSEG_PROBE_COUNT; //on_node_rank = team->shm_seg.on_node_rank;
    ucc_rank_t root = (ucc_rank_t) args.root, group_size = team->size, rank = team->rank;
    void *ar2k_data_ptr, *my_2k_ptr, *partner_2k_ptr;
    int  my_2k_offset, partner_2k_offset;

    if (task->use_small) {
        /* Works up to 64bytes */
        shmem = task->shmseg_base;
        if (rank == root) {
            SHMSEG_WMB();
            if (task->step == 0) {
                for (radix_i = 0; radix_i < group_size; radix_i++) {
                    if (rank == radix_i) continue;
                    SHMEM_STATE(shmem, radix_i, SHMEM_ROOT) = sequence_number;
                }
                task->step = 1;
            }
            for (i = 0; i < poll_probe_count; i++) {
                n_ready = 0;
                for (radix_i = 0; radix_i < group_size; radix_i++) {
                    if (rank == radix_i) continue;
                    if (SHMEM_STATE(shmem, radix_i, SHMEM_LEAF) == sequence_number) {
                        SHMSEG_ISYNC();
                        n_ready++;
                    }
                }
                if (n_ready == group_size - 1) {
                    goto completion;
                }
            }
            return UCC_INPROGRESS;
        } else {
            for (i = 0; i < poll_probe_count; i++) {
                if (SHMEM_STATE(shmem, rank, SHMEM_ROOT) == sequence_number) {
                    SHMSEG_ISYNC();
                    memcpy((void *)aSHMEM_STATE(shmem, rank, SHMEM_DATA),
                           (void *)aSHMEM_STATE(shmem, root, SHMEM_DATA),
                           data_size);
                    SHMSEG_WMB();
                    SHMEM_STATE(shmem, rank, SHMEM_LEAF) = sequence_number;
                    goto completion;
                }
            }
            return UCC_INPROGRESS;
        }
    } else {
        /* Works for up to 2K bytes */
        /* This flag is set at ml, and checks the following:
        * msg_size < 2k, blocking, is !fragmented and sbgp_shmseg is used
        */
        shmem = team->shm_seg.ar2k_sync_shmseg;
        ar2k_data_ptr = task->shmseg_base;

        if (rank == root) {
            SHMSEG_WMB();
            if (task->step == 0) {
                for (radix_i = 0; radix_i < group_size; radix_i++) {
                    if (rank == radix_i) continue;
                    SHMEM_STATE(shmem, radix_i, SHMEM_ROOT) = sequence_number;
                }
                task->step = 1;
            }
            for (i = 0; i < poll_probe_count; i++) {
                n_ready = 0;
                for (radix_i = 0; radix_i < group_size; radix_i++) {
                    if (rank == radix_i) continue;
                    if (SHMEM_STATE(shmem, radix_i, SHMEM_LEAF) == sequence_number) {
                    	SHMSEG_ISYNC();
                        n_ready++;
                    }
                }
                if (n_ready == group_size - 1) {
                    goto completion;
                }
            }
            return UCC_INPROGRESS;
        } else {
            for (i = 0; i < poll_probe_count; i++) {
                if (SHMEM_STATE(shmem, rank, SHMEM_ROOT) == sequence_number) {
                    SHMSEG_ISYNC();
                    my_2k_offset = SHMEM_STATE(shmem, rank, SHMEM_2K_OFFSET);
                    my_2k_ptr = (void *)((char *)ar2k_data_ptr + my_2k_offset);
   	                partner_2k_offset = SHMEM_STATE(shmem, root, SHMEM_2K_OFFSET);
   	                partner_2k_ptr = (void *)((char *)ar2k_data_ptr + partner_2k_offset);

   	                memcpy(my_2k_ptr, partner_2k_ptr, data_size);
   	                SHMSEG_WMB();
   	                SHMEM_STATE(shmem, rank, SHMEM_LEAF) = sequence_number;
   	                goto completion;
   	            }
   	        }
   	        return UCC_INPROGRESS;
   	    }
    }
completion:
    if (rank != root || args.dst.info.buffer != args.src.info.buffer) {
        memcpy(args.src.info.buffer, task->shmseg_dest, data_size);
    }
    task->super.super.status = UCC_OK;
    return UCC_OK;
}



ucc_status_t ucc_tl_shm_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
	ucc_coll_args_t    args = TASK_ARGS(task);
    void *ptr;
    ucc_status_t status;
    ucc_shm_seg_data_t *shm_seg_data = team->shm_seg_data;
    size_t data_size = args.src.info.count *
                       ucc_dt_size(args.src.info.datatype);
    ucc_rank_t rank = team->rank;
    //int on_node_rank = team->shm_seg.on_node_rank;

    shm_seg_data->seq_num++;
    task->step = 0;
    if (data_size <= AR_SMALL_MAX) {
        task->use_small = 1;
        if (shm_seg_data->seq_num & 1) {
            ptr = (void*) ((char *) shm_seg_data->ar128b_shmseg[0] +
                           (128 * rank) + 16 /* first 16 reserved for sync.*/);
            task->shmseg_base = (void *)shm_seg_data->ar128b_shmseg[0];
        } else {
            ptr = (void*) ((char *) shm_seg_data->ar128b_shmseg[1] +
                           (128 * rank) + 16);
            task->shmseg_base = (void *)shm_seg_data->ar128b_shmseg[1];
        }
    } else {
        task->use_small = 0;
        if (shm_seg_data->seq_num & 1) {
            ptr = (void*) ((char *) shm_seg_data->ar2k_data_shmseg[0] +
                           (SHMEM_2K * rank));
            task->shmseg_base = (void *)shm_seg_data->ar2k_data_shmseg[0];
        }
        else {
            ptr = (void*) ((char *) shm_seg_data->ar2k_data_shmseg[1] +
                           (SHMEM_2K * rank));
            task->shmseg_base = (void *)shm_seg_data->ar2k_data_shmseg[1];
        }
    }

    task->shmseg_dest = (void *)ptr;
    if (args.root == rank) {
    	memcpy(ptr, args.src.info.buffer, data_size);
    }
    task->super.super.status = UCC_INPROGRESS;

    status = ucc_tl_shm_bcast_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}


ucc_status_t ucc_tl_shm_bcast_init(ucc_tl_shm_task_t *task)
{
    task->super.post     = ucc_tl_shm_bcast_start;
    task->super.progress = ucc_tl_shm_bcast_progress;
    return UCC_OK;
}
