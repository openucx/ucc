/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"
#include "tl_shm_coll.h"
#include "tl_shm_knomial_pattern.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "coll_score/ucc_coll_score.h"
#include <sys/stat.h> //make extern C?

#define UINT_LOG_X(_val, _base) ((unsigned int) ceil(log((_val)) / log((_base))))

#define SHM_MODE (IPC_CREAT|IPC_EXCL|S_IRUSR|S_IWUSR|S_IWOTH|S_IRGRP|S_IWGRP)

#define MAX_INLINE(_CS, _data, _inline)                                       \
    do {                                                                      \
        _inline = _CS - ucc_offsetof(ucc_tl_shm_ctrl_t, _data);               \
    } while (0)

static ucc_rank_t ucc_tl_shm_team_rank_to_group_id(ucc_tl_shm_team_t *team, ucc_rank_t r)
{
    for (i = 0; i < team->n_base_groups; i++) {
        for (j = 0; j < team->base_groups[i]->group_size; j++) {
            if (r == ucc_sbgp_rank2team(team->base_groups[i], j)) {
                /* found team rank r in base group i */
                return i;
            }
        }
    }
    ucc_assert(0);
    return UCC_RANK_INVALID;
}

static ucc_status_t ucc_tl_shm_rank_group_id_map_init(ucc_tl_shm_team_t *team)
{
	ucc_rank_t  team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t *ranks = ucc_malloc(team_size * sizeof(*ranks)); // dont need to free because happens in ucc_ep_map_from_array()?
    if (!ranks) {
        return UCC_ERR_NO_MEM;
    }
    for (i = 0; i < team_size; i++) {
        ranks[i] = ucc_tl_shm_team_rank_to_group_id(team, i);
    }
    team->rank_group_id_map = ucc_ep_map_from_array(ranks, team_size, team_size, 1);
    return UCC_OK;
}

static ucc_rank_t ucc_tl_team_rank_to_group_rank(ucc_tl_shm_team_t *team, ucc_rank_t r) {
    ucc_rank_t group_id = ucc_ep_map_eval(team->rank_group_id_map, r);
    for (i = 0; i < team->base_groups[group_id]->group_size; i++) {
        if (ucc_sbgp_rank2team(team->base_groups[group_id], i) == r) {
            return i;
        }
    }
    ucc_assert(0);
    return UCC_RANK_INVALID;
}

static ucc_status_t ucc_tl_shm_group_rank_map_init(ucc_tl_shm_team_t *team)
{
    ucc_rank_t  team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t *ranks = ucc_malloc(team_size * sizeof(*ranks)); // dont need to free because happens in ucc_ep_map_from_array()?
    if (!ranks) {
        return UCC_ERR_NO_MEM;
    }
    for (i = 0; i < team_size; i++) {
        ranks[i] = tl_team_rank_to_group_rank(team, i);
    }
    team->group_rank_map = ucc_ep_map_from_array(ranks, team_size, team_size, 1);
    return UCC_OK;
}

static inline void tree_to_team_ranks(ucc_kn_tree_t *tree, ucc_ep_map_t map)
{
    if (tree->parent != UCC_RANK_INVALID) {
        tree->parent = ucc_ep_map_eval(map, tree->parent);
    }
    for (i=0; i<tree->n_childred; i++) {
        tree->children[i] = ucc_ep_map_eval(map, tree->children[i]);
    }
}


static ucc_tl_shm_init_segs(ucc_tl_shm_team_t *team) {
    void *shmseg_base;
    for (i = 0; i < team->n_concurrent; i++) {
        shmseg_base = PTR_OFFSET(team->shm_buffer, (team->ctrl_size + team->data_size) * i);
        team->segs[i].ctrl = shmseg_base;
        team->segs[i].data = PTR_OFFSET(shmseg_base, team->ctrl_size);
    }
}

static ucc_status_t ucc_shm_seg_alloc(ucc_tl_shm_team_t *team) {
    ucc_rank_t    team_rank = UCC_TL_TEAM_RANK(team), team_size = UCC_TL_TEAM_SIZE(team);
    size_t        shmsize   = team->n_concurrent *
                              (team->ctrl_size + team->data_size);
    int           shmid     = -1;
    ucc_team_oob_coll_t oob = UCC_TL_TEAM_OOB(self);
    ucc_status_t status;

    team->allgather_dst = (int *) ucc_malloc(sizeof(int) * team_size);

    /* LOWEST on node rank  within the comm will initiate the segment creation.
     * Everyone else will attach. */
    if (team_rank == 0) {
        shmid = shmget(IPC_PRIVATE, shmsize, SHM_MODE);
        if (shmid < 0) {
            tl_error(team->super.super.context->lib,
                     "Root: shmget failed, shmid=%d, shmsize=%ld, errno: %s\n",
                     shmid, shmsize, strerror(errno));
            return UCC_ERR_NO_RESOURCE;
        }
        team->shm_buffer = (void *) shmat(shmid, NULL, 0);
        if (team->shm_buffer == (void *) -1) {
        	shmid = -1;
        	team->shm_buffer = NULL;
            tl_error(team->super.super.context->lib,
                     "Root: shmat failed, errno: %s\n", strerror(errno));
            status = oob.allgather(&shmid, team->allgather_dst, sizeof(int),
                                         oob.coll_info, &team->oob_req);
            if (UCC_OK != status) {
                tl_error(team->super.super.context->lib, "allgather failed");
                return status;
            }
            return UCC_ERR_NO_RESOURCE;
        }
        memset(team->shm_buffer, 0, shmsize);
        shmctl(shmid, IPC_RMID, NULL);
    }
    status = oob.allgather(&shmid, team->allgather_dst, sizeof(int),
                                 oob.coll_info, &team->oob_req);
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
    int max_trees;
    ucc_rank_t size, rank;
    ucc_tl_shm_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_shm_context_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params->team);

    size = UCC_TL_TEAM_SIZE(self);
    rank = UCC_TL_TEAM_RANK(self);

    self->seq_num      = 1;
    self->status       = UCC_INPROGRESS;
    self->shm_buffer   = NULL;
    self->n_concurrent = UCC_TL_SHM_TEAM_LIB(self)->cfg.n_concurrent;
    self->data_size    = UCC_TL_SHM_TEAM_LIB(self)->cfg.data_size * size;

    self->segs = ucc_malloc(sizeof(ucc_tl_shm_seg_t) * self->n_concurrent;

    if (!self->segs) {
        return UCC_ERR_NO_MEM;
    }

    max_trees = UCC_TL_SHM_TEAM_LIB(self)->cfg.max_trees_cached;
    self->tree_cache = ucc_malloc(max_trees *
                                  sizeof(ucc_tl_shm_tree_cache_keys_t) +
                                  max_trees * (sizeof(ucc_tl_shm_tree_t) *
                                  size) + sizeof(size_t));
    if (!self->tree_cache) {
        return UCC_ERR_NO_MEM;
    }

    self->tree_cache.trees = PTR_OFFSET(self->tree_cache, max_trees *
                                 sizeof(ucc_tl_shm_tree_cache_keys_t));
    &self->tree_cache.size = PTR_OFFSET(self->tree_cache.trees, max_trees * //needed?
                                 (sizeof(ucc_tl_shm_tree_t) * size));

    /* Subgrouping Extensions Required :
       need a way to compute ALL socket/numa subgroups: TODO: Val
    */

    /* sbgp type gl is either SOCKET_LEADERS or NUMA_LEADERS depending on the config: grouping type */
    self->leaders_group = ucc_topo_get_sbgp(self->topo, UCC_SBGP_SOCKET_LEADERS);

    /* sbgp type is either SOCKET or NUMA depending on the config: grouping type */
    self->n_base_groups = self->leaders_group->group_size;
    ucc_assert(self->n_base_groups == self->topo->n_sockets);
//    self->base_groups[i] = ucc_team_topo_get_sockets/numas(); //TODO val //what is this?
    // the above call should return ALL socket/numa sbgps including size=1 subgroups */

    if (UCC_OK != (status = ucc_tl_shm_rank_group_id_map_init(team))) {
        return status;
    }

    if (UCC_OK != (status = ucc_tl_shm_group_rank_map_init(team))) {
        return status;
    }

    /* Build ctrl_map */
    size_t ctrl_size = 0;
    uint64_t *rank_ctrl_offsets = ucc_malloc(size * sizeof(uint64_t));
    uint64_t ctrl_offset = 0;
    uint32_t cfg_ctrl_size = UCC_TL_SHM_TEAM_LIB(team)->cfg.ctrl_size;
    size_t page_size = ucc_get_page_size();
    uint32_t group_size;
    ucc_rank_t team_rank;

    for (int i = 0; i < team->n_base_groups; i++) {
        group_size = team->base_groups[i]->group_size;
        if (i > 1) {
            ctrl_offset = ctrl_size;
        }
        ctrl_size += ucc_align_up(group_size * cfg_ctrl_size, page_size);
        for (int j=0; j<team->base_groups[i]->group_size; j++) {
            team_rank = ucc_sbgp_rank2team(team->base_groups[i], j);
            rank_ctrl_offsets[team_rank] = ctrl_offset +
                                           team_rank * cfg_ctrl_size;
        }
    }
    self->ctrl_size = ctrl_size;

    /* ucc_ep_map_from_array64 same as ucc_ep_map_from_array but for array of 8 byte values */
    self->ctrl_map = ucc_ep_map_from_array64(rank_ctrl_offsets, size, size, 1);\

    status = ucc_shm_seg_alloc(self);

    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_shm_team_t)
{
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_shm_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_shm_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_shm_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

ucc_status_t ucc_tl_shm_team_destroy(ucc_base_team_t *tl_team)
{
	ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);

	if (team->shm_buffer) {
		if (shmdt(team->shm_buffer) == -1) {
			tl_error(team->super.super.context->lib, "shmdt failed");
			return UCC_ERR_NO_MESSAGE;
		}
	}

	ucc_free(team->segs);
	UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_shm_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_team_create_test(ucc_base_team_t *tl_team)
{
	ucc_tl_shm_team_t   *team   = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
	ucc_team_oob_coll_t  oob    = UCC_TL_TEAM_OOB(team);
	ucc_status_t         status = oob.req_test(team->oob_req);
	if (status == UCC_INPROGRESS) {
	    return UCC_INPROGRESS;
	}
	if (status != UCC_OK) {
        oob.req_free(team->oob_req);
        tl_error(team->super.super.context->lib, "oob req test failed");
        return status;
    }
	status = oob.req_free(team->oob_req);
    if (status != UCC_OK) {
        tl_error(team->super.super.context->lib, "oob req free failed");
        return status;
    }

    /* Exchange keys */
    int shmid = team->allgather_dst[0];

    if (UCC_TL_TEAM_RANK(team) > 0) {
        team->shm_buffer = (shmem_sync_t *) shmat(shmid, NULL, 0);
        if (team->shm_buffer == (void *) -1) {
            tl_error(team->super.super.context->lib,
                     "Child failed to attach to shmseg, errno: %s\n",
                     strerror(errno));
            return UCC_ERR_NO_RESOURCE;
        }
    }

    ucc_tl_shm_init_segs(team);
	ucc_free(team->allgather_dst);
	team->status = UCC_OK;
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
    return UCC_OK;
}
