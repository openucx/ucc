/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"
#include "tl_shm_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "coll_score/ucc_coll_score.h"
#include <sys/stat.h> //make extern C?

#define UINT_LOG_X(_val, _base) ((unsigned int) ceil(log((_val)) / log((_base))))

#define SHM_MODE (IPC_CREAT|IPC_EXCL|S_IRUSR|S_IWUSR|S_IWOTH|S_IRGRP|S_IWGRP)

static void ucc_shm_reduce_preprocess(ucc_tl_shm_team_t *team) {
	int radix = SHMSEG_128B_RADIX; // TODO read from conf
	team->shm_seg.ar64_logx_group_size = UINT_LOG_X(team->size, radix);
	team->shm_seg.ar64_radix_array = (int *) ucc_malloc(team->size * sizeof(int));
	team->shm_seg.ar2k_radix_array = (int *) ucc_malloc(team->size * sizeof(int));
	team->shm_seg.ar64_bcol_to_node_group_list = (int *) ucc_malloc(team->size * sizeof(int));
	memset(team->shm_seg.ar64_radix_array, 0, team->size * sizeof(int));
	memset(team->shm_seg.ar2k_radix_array, 0, team->size * sizeof(int));
	memset(team->shm_seg.ar64_bcol_to_node_group_list, 0, team->size * sizeof(int));
	for (int i = 0; i < team->size; i++) {
	    (team->shm_seg.ar64_bcol_to_node_group_list)[i] = i;
	}

	/* Traverse the radix tree and create the 64b radix_array */
	int array_length = 0, partner_rank = 0;
	for (int level = 0; level < team->shm_seg.ar64_logx_group_size; level++) {
	    if (ROOT_AT_LEVEL(team->rank, radix, level)) {
            int k = 0;
            for (k = 0, partner_rank = team->rank + NEXT_PARTNER(radix, level);//, array_length;//, array_length = 0; // should this be array length = 0 or remove?
                 k < radix - 1 && partner_rank < team->size && array_length < team->size;
                 k++, partner_rank += NEXT_PARTNER(radix, level), array_length++) {
                (team->shm_seg.ar64_radix_array)[array_length] = (team->shm_seg.ar64_bcol_to_node_group_list)[partner_rank];
            }
        } else {
            /* Compute radix^(level+1) */
            int tmp = 1;
            for(int i = 0; i < level + 1; i++) {
                tmp*=radix;
            }
            team->shm_seg.my_ar64_node_root_rank = (unsigned int) ((team->rank / tmp) * tmp);
            team->shm_seg.my_ar64_node_root_rank = (team->shm_seg.ar64_bcol_to_node_group_list)[team->shm_seg.my_ar64_node_root_rank];
            break;
        }
    }
	team->shm_seg.ar64_radix_array_length = array_length;
    /* Setup the 2K radix array */
    radix = SHMSEG_2K_RADIX;
    team->shm_seg.ar2k_logx_group_size = UINT_LOG_X(team->size, radix);
    /* Traverse the radix tree and create the 2K radix_array */
    array_length = 0; partner_rank = 0;
    for (int level = 0; level < team->shm_seg.ar2k_logx_group_size; level++) {
        if (ROOT_AT_LEVEL(team->rank, radix, level)) {
            int k=0;
            for (k = 0, partner_rank = team->rank + NEXT_PARTNER(radix, level);
                 k < radix-1 && partner_rank < team->size && array_length < team->size;
                 k++, partner_rank += NEXT_PARTNER(radix, level), array_length++) {
                (team->shm_seg.ar2k_radix_array)[array_length] = partner_rank;
            }
        } else {
            /* Compute radix^(level+1) */
            int tmp = 1;
            for (int i = 0; i < level + 1; i++) {
                tmp*=radix;
            }
            team->shm_seg.my_ar2k_root_rank = (unsigned int) ((team->rank / tmp) * tmp);
            break;
        }
    }
    team->shm_seg.ar2k_radix_array_length = array_length;
}

static ucc_status_t ucc_shm_seg_alloc(ucc_tl_shm_team_t *team) {
	size_t shmsize = 0, shmsize_2k = 0;
	int shmid = -1, shmid_2k = -1;
    ucc_rank_t team_rank = team->rank, team_size = team->size;
    ucc_status_t status;

    team->allgather_dst = (int *) ucc_malloc(sizeof(int) * 2 * team_size);
//    memset(team->allgather_dst, 0, team_size * sizeof(int) * 2);
    team->allgather_src[0] = shmid;
    team->allgather_src[1] = shmid_2k;

    /* LOWEST on node rank  within the comm will initiate the segment creation.
     * Everyone else will attach. */
    if (team_rank == 0) {
        shmsize = (SHMEM_128b * team_size * 2) + (SHMEM_2K * team_size * 2);
        shmid = shmget(IPC_PRIVATE, shmsize, SHM_MODE);
        if (shmid < 0) {
            tl_error(team->super.super.context->lib,
                     "Root: shmget failed, shmid=%d, shmsize=%ld, errno: %s\n",
                     shmid, shmsize, strerror(errno));
            return UCC_ERR_NO_RESOURCE;
        }
        team->seg = (shmem_sync_t *) shmat(shmid, NULL, 0);
        shmctl(shmid, IPC_RMID, NULL);
        if (team->seg == (void *) -1) {
        	shmid = -1;
        	team->seg = NULL;
            tl_error(team->super.super.context->lib,
                     "Root: shmat failed, errno: %s\n", strerror(errno));
            status = team->oob.allgather(team->allgather_src, team->allgather_dst, sizeof(int) * 2,
                                         team->oob.coll_info, &team->oob_req);
            if (UCC_OK != status) {
                tl_error(team->super.super.context->lib, "allgather failed");
                return status;
            }
            return UCC_ERR_NO_RESOURCE;
        } else {
            /* Initialize to base sequence number, for 128b segments
             * The 2K area is data only, no need to initialize */
            for (int i=0; i<team_size*2; i++) {
                SHMEM_STATE((shmem_sync_t *)team->seg, i, SHMEM_ROOT) = 0;
                SHMEM_STATE((shmem_sync_t *)team->seg, i, SHMEM_LEAF) = 0;
            }
        }

        /* 2k seg */
        shmsize_2k = team_size * sizeof(shmem_sync_t);
        shmid_2k = shmget(IPC_PRIVATE, shmsize_2k, SHM_MODE);
        if (shmid_2k < 0) {
            tl_error(team->super.super.context->lib,
                "Root: shmget 2k failed, shmid_2k=%d, shmsize=%ld, errno %s\n",
                shmid_2k, shmsize_2k, strerror(errno));
            return UCC_ERR_NO_RESOURCE;
        }
        team->shm_seg.ar2k_sync_shmseg = (shmem_sync_t *) shmat(shmid_2k, NULL, 0);
        shmctl(shmid_2k, IPC_RMID, NULL);
        if (team->shm_seg.ar2k_sync_shmseg == (void *) -1) {
            shmid_2k = -1;
            team->shm_seg.ar2k_sync_shmseg = NULL;
            tl_error(team->super.super.context->lib,
                    "Root: shmat 2k failed, errno: %s\n", strerror(errno));
            status = team->oob.allgather(team->allgather_src, team->allgather_dst, sizeof(int) * 2,
                                         team->oob.coll_info, &team->oob_req);
            if (UCC_OK != status) {
                tl_error(team->super.super.context->lib, "allgather failed");
                return status;
            }
            return UCC_ERR_NO_RESOURCE;
        } else {
            for (int i=0; i<team_size; i++) {
                SHMEM_STATE((shmem_sync_t *)team->shm_seg.ar2k_sync_shmseg, i, SHMEM_ROOT) = 0;
                SHMEM_STATE((shmem_sync_t *)team->shm_seg.ar2k_sync_shmseg, i, SHMEM_LEAF) = 0;
            }
        }
    }
    team->allgather_src[0] = shmid;
    team->allgather_src[1] = shmid_2k;
    status = team->oob.allgather(team->allgather_src, team->allgather_dst, sizeof(int) * 2,
                                 team->oob.coll_info, &team->oob_req);
    if (UCC_OK != status) {
    	tl_error(team->super.super.context->lib, "allgather failed");
    	return status;
    }
    return UCC_OK;
}

UCC_CLASS_INIT_FUNC(ucc_tl_shm_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
	ucc_status_t status;
	ucc_tl_shm_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_shm_context_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params->team);

    self->seq_num = 1;
    self->status  = UCC_INPROGRESS;
    self->size    = params->params.oob.n_oob_eps;
    self->rank    = params->rank;
    self->oob     = params->params.oob;
    self->seg     = NULL;
    self->shm_seg.ar2k_sync_shmseg = NULL;
    self->shm_seg.ar64_radix_array = NULL;
    self->shm_seg.ar2k_radix_array = NULL;
    self->shm_seg.ar64_bcol_to_node_group_list = NULL;

    status = ucc_shm_seg_alloc(self);

    printf("shm_team_create\n");
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_shm_team_t)
{
    printf("shm_team_cleanup\n");
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_shm_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_shm_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_shm_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

ucc_status_t ucc_tl_shm_team_destroy(ucc_base_team_t *tl_team)
{
	ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);

	if (team->seg) {
		ucc_free(team->shm_seg_data);
		if (shmdt(team->seg) == -1) {
			tl_error(team->super.super.context->lib, "shmdt failed");
			return UCC_ERR_NO_MESSAGE;
		}
	}
	if (team->shm_seg.ar2k_sync_shmseg) {
		ucc_free(team->shm_seg.ar64_radix_array);
		ucc_free(team->shm_seg.ar2k_radix_array);
		ucc_free(team->shm_seg.ar64_bcol_to_node_group_list);
        if (shmdt(team->shm_seg.ar2k_sync_shmseg) == -1) {
            tl_error(team->super.super.context->lib, "shmdt 2k failed");
            return UCC_ERR_NO_MESSAGE;
        }
    }
	UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_shm_team_t)(tl_team);
    printf("shm_team_destroy\n");
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_team_create_test(ucc_base_team_t *tl_team)
{
	ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
	ucc_status_t status = team->oob.req_test(team->oob_req);
	if (status == UCC_INPROGRESS) {
	    return UCC_INPROGRESS;
	}
	if (status != UCC_OK) {
        team->oob.req_free(team->oob_req);
        tl_error(team->super.super.context->lib, "oob req test failed");
        return status;
    }
	status = team->oob.req_free(team->oob_req);
    if (status != UCC_OK) {
        tl_error(team->super.super.context->lib, "oob req free failed");
        return status;
    }

    /* Exchange keys */
    int shmid = team->allgather_dst[0], shmid_2k = team->allgather_dst[1];
    printf("allgather finished. rank = %d, shmid = %d\n", team->rank, shmid);

    if (team->rank > 0) {
        team->seg = (shmem_sync_t *) shmat(shmid, NULL, 0);
        if (team->seg == (void *) -1) {
            tl_error(team->super.super.context->lib,
                     "Child failed to attach to shmseg, errno: %s\n",
                     strerror(errno));
            return UCC_ERR_NO_RESOURCE;
        }

        team->shm_seg.ar2k_sync_shmseg = (shmem_sync_t *) shmat(shmid_2k, NULL, 0);
        if (team->shm_seg.ar2k_sync_shmseg == (void *) -1) {
            tl_error(team->super.super.context->lib,
                     "Child failed to attach to shmseg, errno: %s\n",
                     strerror(errno));
            return UCC_ERR_NO_RESOURCE;
        }
    }

	if (team->seg) {
        team->shm_seg_data =
            (ucc_shm_seg_data_t*) ucc_malloc(sizeof(*team->shm_seg_data));
        memset(team->shm_seg_data, 0, sizeof(*team->shm_seg_data));

        /* |--128b shmseg---|---128b shmseg---|---2K shmseg---|---2K shmseg---| */
        team->shm_seg_data->ar128b_shmseg[0] = team->seg;
        team->shm_seg_data->ar128b_shmseg[1] =
        (void *) ((char *)team->seg + (SHMEM_128b * team->size));
        team->shm_seg_data->ar2k_data_shmseg[0] =
        (void *) ((char *)team->seg + ((SHMEM_128b * team->size) * 2));
        team->shm_seg_data->ar2k_data_shmseg[1] =
        (void *)((char *)team->shm_seg_data->ar2k_data_shmseg[0] +
                 (SHMEM_2K * team->size));
//        team->shm_seg_data->seq_num = 0;
    } else {
        tl_error(team->super.super.context->lib,
                 "shm segment allocation failed\n");
        return UCC_ERR_NO_MESSAGE;
    }

	if (team->shm_seg.ar2k_sync_shmseg) {
        SHMEM_STATE(team->shm_seg.ar2k_sync_shmseg, team->rank, SHMEM_2K_OFFSET) = SHMEM_2K * team->rank;
    }
    ucc_shm_reduce_preprocess(team);
	ucc_free(team->allgather_dst);
	team->status = UCC_OK;
    printf("shm_team_create_test\n");
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p)
{
	ucc_coll_score_t *score;
	ucc_coll_score_build_default(tl_team, UCC_TL_SHM_DEFAULT_SCORE,
	                             ucc_tl_shm_coll_init,
	                             UCC_TL_SHM_SUPPORTED_COLLS, NULL, 0, &score);
	*score_p = score;
    printf("shm_team_get_scores\n");
    return UCC_OK;
}
