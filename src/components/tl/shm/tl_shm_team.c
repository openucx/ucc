/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"
#include "tl_shm_coll.h"
#include "tl_shm_knomial_pattern.h"
#include "perf/tl_shm_coll_perf_params.h"
#include "core/ucc_ee.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/ucc_sys.h"
#include "bcast/bcast.h"
#include "reduce/reduce.h"
#include "barrier/barrier.h"
#include "fanin/fanin.h"
#include "fanout/fanout.h"
#include <sys/stat.h>

#define SHM_MODE                                                               \
    (IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR | S_IWOTH | S_IRGRP | S_IWGRP)

static ucc_rank_t ucc_tl_shm_team_rank_to_group_id(ucc_tl_shm_team_t *team,
                                                   ucc_rank_t         r)
{
    int i, j;
    for (i = 0; i < team->n_base_groups; i++) {
        for (j = 0; j < team->base_groups[i].group_size; j++) {
            if (r == ucc_ep_map_eval(team->base_groups[i].map, j)) {
                /* found team rank r in base group i */
                break;
            }
        }
        if (j < team->base_groups[i].group_size) {
            break;
        }
    }
    ucc_assert(i < team->n_base_groups && j < team->base_groups[i].group_size);
    return i;
}

static ucc_status_t ucc_tl_shm_rank_group_id_map_init(ucc_tl_shm_team_t *team)
{
    ucc_rank_t  team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t *ranks = (ucc_rank_t *)ucc_malloc(team_size * sizeof(*ranks));
    int         i;

    if (!ranks) {
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < team_size; i++) {
        ranks[i] = ucc_tl_shm_team_rank_to_group_id(team, i);
    }
    team->rank_group_id_map =
        ucc_ep_map_from_array(&ranks, team_size, team_size, 1);
    return UCC_OK;
}

static ucc_rank_t ucc_tl_shm_team_rank_to_group_rank(ucc_tl_shm_team_t *team,
                                                     ucc_rank_t         r)
{
    ucc_rank_t group_id = ucc_ep_map_eval(team->rank_group_id_map, r);
    ucc_rank_t i;

    for (i = 0; i < team->base_groups[group_id].group_size; i++) {
        if (ucc_ep_map_eval(team->base_groups[group_id].map, i) == r) {
            break;
        }
    }
    ucc_assert(i < team->base_groups[group_id].group_size);
    return i;
}

static ucc_status_t ucc_tl_shm_group_rank_map_init(ucc_tl_shm_team_t *team)
{
    ucc_rank_t  team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t *ranks = (ucc_rank_t *)ucc_malloc(team_size * sizeof(*ranks));
    ucc_rank_t  i;

    if (!ranks) {
        return UCC_ERR_NO_MEMORY;
    }
    for (i = 0; i < team_size; i++) {
        ranks[i] = ucc_tl_shm_team_rank_to_group_rank(team, i);
    }
    team->group_rank_map =
        ucc_ep_map_from_array(&ranks, team_size, team_size, 1);
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_shm_set_perf_funcs(ucc_tl_shm_team_t *team)
{
    ucc_rank_t team_size = UCC_TL_TEAM_SIZE(team);
    int        i         = 0,
        max_size =
            20; // max size is general estimate,can be changed as more archs are added for perf selection
    ucc_tl_shm_perf_keys_t * perf_funcs_keys;
    ucc_tl_shm_perf_funcs_t *perf_funcs_list;
    ucc_cpu_vendor_t         vendor;
    ucc_cpu_model_t          model;

    perf_funcs_keys =
        ucc_malloc(max_size * sizeof(ucc_tl_shm_perf_keys_t), "perf keys");

    if (!perf_funcs_keys) {
        tl_error(team->super.super.context->lib,
                 "failed to allocate %zd bytes for perf_funcs_keys",
                 max_size * sizeof(ucc_tl_shm_perf_keys_t));
        return UCC_ERR_NO_MEMORY;
    }

    perf_funcs_list = (ucc_tl_shm_perf_funcs_t *)ucc_malloc(
        sizeof(ucc_tl_shm_perf_funcs_t), "perf funcs");

    if (!perf_funcs_list) {
        tl_error(team->super.super.context->lib,
                 "failed to allocate %zd bytes for perf_funcs_list",
                 max_size * sizeof(ucc_tl_shm_perf_funcs_t));
        return UCC_ERR_NO_MEMORY;
    }

    ucc_tl_shm_create_perf_func_list(team, perf_funcs_keys, perf_funcs_list);
    vendor = ucc_arch_get_cpu_vendor();
    model  = ucc_arch_get_cpu_model();

    team->perf_params_bcast  = ucc_tl_shm_perf_params_generic_bcast;
    team->perf_params_reduce = ucc_tl_shm_perf_params_generic_reduce;

    for (i = 0; i < perf_funcs_list->size; i++) {
        if (perf_funcs_list->keys[i].cpu_vendor == vendor &&
            perf_funcs_list->keys[i].cpu_model == model &&
            perf_funcs_list->keys[i].team_size == team_size) {
            team->perf_params_bcast  = perf_funcs_list->keys[i].bcast_func;
            team->perf_params_reduce = perf_funcs_list->keys[i].reduce_func;
            break;
        }
    }

    ucc_free(perf_funcs_keys);
    ucc_free(perf_funcs_list);
    return UCC_OK;
}

static void ucc_tl_shm_init_segs(ucc_tl_shm_team_t *team)
{
    void *shmseg_base;
    int   i;

    for (i = 0; i < team->n_concurrent; i++) {
        shmseg_base        = PTR_OFFSET(team->shm_buffer,
                                 (team->ctrl_size + team->data_size) * i);
        team->segs[i].ctrl = shmseg_base;
        team->segs[i].data = PTR_OFFSET(shmseg_base, team->ctrl_size);
    }
}

static ucc_status_t ucc_tl_shm_seg_alloc(ucc_tl_shm_team_t *team)
{
    ucc_rank_t team_rank = UCC_TL_TEAM_RANK(team),
               team_size = UCC_TL_TEAM_SIZE(team);
    size_t shmsize = team->n_concurrent * (team->ctrl_size + team->data_size);
    int    shmid   = -1;
    ucc_team_oob_coll_t oob = UCC_TL_TEAM_OOB(team);
    ucc_status_t        status;

    team->allgather_dst =
        (int *)ucc_malloc(sizeof(int) * (team_size + 1), "algather dst buffer");

    /* LOWEST on node rank  within the comm will initiate the segment creation.
     * Everyone else will attach. */
    if (team_rank == 0) {
        shmid = shmget(IPC_PRIVATE, shmsize, SHM_MODE);
        if (shmid < 0) {
            tl_error(team->super.super.context->lib,
                     "Root: shmget failed, shmid=%d, shmsize=%ld, errno: %s\n",
                     shmid, shmsize, strerror(errno));
            goto allgather;
        }
        team->shm_buffer = (void *)shmat(shmid, NULL, 0);
        if (team->shm_buffer == (void *)-1) {
            shmid            = -1;
            team->shm_buffer = NULL;
            tl_error(team->super.super.context->lib,
                     "Root: shmat failed, errno: %s\n", strerror(errno));
            goto allgather;
        }
        memset(team->shm_buffer, 0, shmsize);
        shmctl(shmid, IPC_RMID, NULL);
    }
allgather:
    team->allgather_dst[team_size] = shmid;
    status = oob.allgather(&team->allgather_dst[team_size], team->allgather_dst,
                           sizeof(int), oob.coll_info, &team->oob_req);

    if (UCC_OK != status) {
        tl_error(team->super.super.context->lib, "allgather failed");
        return status;
    }
    return UCC_OK;
}

UCC_CLASS_INIT_FUNC(ucc_tl_shm_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_shm_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_shm_context_t);
    ucc_status_t status;
    int          n_sbgps, i, j, max_trees;
    ucc_rank_t   team_size, team_rank;
    uint32_t     cfg_ctrl_size, group_size;
    ucc_subset_t subset;
    size_t       ctrl_size, page_size;
    uint64_t *   rank_ctrl_offsets;
    uint64_t     ctrl_offset;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);

    if (NULL == UCC_TL_CORE_CTX(self)->topo) {
        /* CORE context does not have topo information -
	 * local context mode */
        return UCC_ERR_NOT_SUPPORTED;
    }
    subset.map    = UCC_TL_TEAM_MAP(self);
    subset.myrank = UCC_TL_TEAM_RANK(self);
    team_size     = UCC_TL_TEAM_SIZE(self);
    cfg_ctrl_size = UCC_TL_SHM_TEAM_LIB(self)->cfg.ctrl_size;

    self->seq_num      = UCC_TL_SHM_TEAM_LIB(self)->cfg.n_concurrent;
    self->status       = UCC_INPROGRESS;
    self->shm_buffer   = NULL;
    self->n_concurrent = UCC_TL_SHM_TEAM_LIB(self)->cfg.n_concurrent;
    self->data_size    = UCC_TL_SHM_TEAM_LIB(self)->cfg.data_size * team_size;
    self->max_inline   = cfg_ctrl_size - ucc_offsetof(ucc_tl_shm_ctrl_t, data);

    status = ucc_topo_init(subset, UCC_TL_CORE_CTX(self)->topo, &self->topo);

    if (UCC_OK != status) {
        tl_error(ctx->super.super.lib, "failed to init team topo");
        return status;
    }

    if (!ucc_topo_is_single_node(self->topo)) {
        tl_debug(ctx->super.super.lib, "multi node team is not supported");
        ucc_topo_cleanup(self->topo);
        return UCC_ERR_INVALID_PARAM;
    }

    if (UCC_TL_CORE_CTX(self)->topo->sock_bound != 1) {
        tl_debug(ctx->super.super.lib, "sock bound is not supported");
        ucc_topo_cleanup(self->topo);
        return UCC_ERR_NOT_SUPPORTED;
    }

    self->last_posted = ucc_calloc(sizeof(*self->last_posted),
                                   self->n_concurrent, "last_posted");
    if (!self->last_posted) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for last_posted array",
                 sizeof(*self->last_posted) * self->n_concurrent);
        status = UCC_ERR_NO_MEMORY;
        goto err_last_posted;
    }

    self->segs = (ucc_tl_shm_seg_t *)ucc_malloc(
        sizeof(ucc_tl_shm_seg_t) * self->n_concurrent, "shm_segs");

    if (!self->segs) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for shm_segs",
                 sizeof(ucc_tl_shm_seg_t) * self->n_concurrent);
        status = UCC_ERR_NO_MEMORY;
        goto err_segs;
    }

    max_trees        = UCC_TL_SHM_TEAM_LIB(self)->cfg.max_trees_cached;
    self->tree_cache = (ucc_tl_shm_tree_cache_t *)ucc_malloc(
        sizeof(ucc_tl_shm_tree_cache_t), "tree_cache");

    if (!self->tree_cache) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for tree_cache",
                 sizeof(ucc_tl_shm_tree_cache_t));
        status = UCC_ERR_NO_MEMORY;
        goto err_tree;
    }

    self->tree_cache->elems = (ucc_tl_shm_tree_cache_elems_t *)ucc_malloc(
        max_trees * sizeof(ucc_tl_shm_tree_cache_elems_t), "tree_cache->elems");

    if (!self->tree_cache->elems) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for tree_cache->elems",
                 max_trees * sizeof(ucc_tl_shm_tree_cache_elems_t));
        status = UCC_ERR_NO_MEMORY;
        goto err_sockets;
    }

    self->tree_cache->size = 0;

    /* sbgp type gl is either SOCKET_LEADERS or NUMA_LEADERS
     * depending on the config: grouping type */
    self->leaders_group =
        ucc_topo_get_sbgp(self->topo, UCC_SBGP_SOCKET_LEADERS);

    if (self->leaders_group->status == UCC_SBGP_NOT_EXISTS ||
        self->leaders_group->group_size == team_size) {
        self->leaders_group->group_size = 0;
        self->base_groups   = ucc_topo_get_sbgp(self->topo, UCC_SBGP_NODE);
        self->n_base_groups = 1;
    } else {
        /* sbgp type is either SOCKET or NUMA
     * depending on the config: grouping type */
        self->n_base_groups = self->leaders_group->group_size;
        ucc_assert(self->n_base_groups == self->topo->n_sockets);

        status =
            ucc_topo_get_all_sockets(self->topo, &self->base_groups, &n_sbgps);
        if (UCC_OK != status) {
            tl_error(ctx->super.super.lib, "failed to get all base subgroups");
            goto err_sockets;
        }
    }

    /* the above call should return ALL socket/numa sbgps including size=1 subgroups */
    status = ucc_tl_shm_rank_group_id_map_init(self);
    if (UCC_OK != status) {
        goto err_sockets;
    }
    status = ucc_tl_shm_group_rank_map_init(self);
    if (UCC_OK != status) {
        goto err_group_rank_map;
    }

    self->my_group_id =
        ucc_ep_map_eval(self->rank_group_id_map, UCC_TL_TEAM_RANK(self));
    ctrl_size   = 0;
    ctrl_offset = 0;
    page_size   = ucc_get_page_size();

    rank_ctrl_offsets =
        (uint64_t *)ucc_malloc(team_size * sizeof(uint64_t), "ctrl_offsets");
    if (!rank_ctrl_offsets) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for ctrl_offsets",
                 team_size * sizeof(uint64_t));
        status = UCC_ERR_NO_MEMORY;
        goto err_offsets;
    }

    for (i = 0; i < self->n_base_groups; i++) {
        group_size = self->base_groups[i].group_size;
        if (i > 1) {
            ctrl_offset = ctrl_size;
        }
        ctrl_size += ucc_align_up(group_size * cfg_ctrl_size, page_size);
        for (j = 0; j < self->base_groups[i].group_size; j++) {
            team_rank = ucc_ep_map_eval(self->base_groups[i].map, j);
            rank_ctrl_offsets[team_rank] =
                ctrl_offset + team_rank * cfg_ctrl_size;
        }
    }
    self->ctrl_size = ctrl_size;

    /* ucc_ep_map_from_array64 same as ucc_ep_map_from_array but for array of 8 byte values */
    self->ctrl_map =
        ucc_ep_map_from_array_64(&rank_ctrl_offsets, team_size, team_size, 1);

    status = ucc_tl_shm_set_perf_funcs(self);
    if (UCC_OK != status) {
        goto err_seg_alloc;
    }

    status = ucc_tl_shm_seg_alloc(self);
    if (UCC_OK != status) {
        goto err_seg_alloc;
    }
    return UCC_OK;

err_seg_alloc:
//    ucc_free(self->ctrl_map.array.map);
err_offsets:
//    ucc_free(self->group_rank_map.array.map);
err_group_rank_map:
//    ucc_free(self->rank_group_id_map.array.map); //TODO switch to ucc_ep_map_destroy once
// it is merged to master upstream
err_sockets:
    ucc_free(self->tree_cache);
err_tree:
    ucc_free(self->segs);
err_segs:
    ucc_free(self->last_posted);
err_last_posted:
    ucc_topo_cleanup(self->topo);

    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_shm_team_t)
{
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_shm_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_shm_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_shm_team_destroy(ucc_base_team_t *tl_team)
{
    ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    int                i;

    if (team->shm_buffer) {
        if (shmdt(team->shm_buffer) == -1) {
            tl_error(team->super.super.context->lib, "shmdt failed");
            return UCC_ERR_NO_MESSAGE;
        }
    }

    for (i = 0; i < team->tree_cache->size; i++) {
        ucc_free(team->tree_cache->elems[i].tree->top_tree);
        ucc_free(team->tree_cache->elems[i].tree->base_tree);
        ucc_free(team->tree_cache->elems[i].tree);
    }
    ucc_free(team->tree_cache->elems);
    ucc_free(team->tree_cache);
    ucc_free(team->segs);
    ucc_free(team->last_posted);

    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_shm_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_shm_team_t * team   = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_team_oob_coll_t oob    = UCC_TL_TEAM_OOB(team);
    ucc_status_t        status = oob.req_test(team->oob_req);
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

    if (shmid == -1) {
        return UCC_ERR_NO_RESOURCE;
    }

    if (UCC_TL_TEAM_RANK(team) > 0) {
        team->shm_buffer = (void *)shmat(shmid, NULL, 0);
        if (team->shm_buffer == (void *)-1) {
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

ucc_status_t ucc_tl_shm_team_get_scores(ucc_base_team_t *  tl_team,
                                        ucc_coll_score_t **score_p)
{
    ucc_tl_shm_team_t * team      = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_base_lib_t *    lib       = UCC_TL_TEAM_LIB(team);
    ucc_base_context_t *ctx       = UCC_TL_TEAM_CTX(team);
    size_t              data_size = UCC_TL_SHM_TEAM_LIB(team)->cfg.data_size;
    size_t              ctrl_size = UCC_TL_SHM_TEAM_LIB(team)->cfg.ctrl_size;
    size_t              inline_size = ctrl_size - sizeof(ucc_tl_shm_ctrl_t);
    size_t              max_size    = ucc_max(inline_size, data_size);
    ucc_coll_score_t *  score;
    ucc_status_t        status;

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "faild to alloc score_t");
        return status;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_BCAST, UCC_MEMORY_TYPE_HOST, 0, max_size,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_bcast_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_REDUCE, UCC_MEMORY_TYPE_HOST, 0, max_size,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_reduce_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_FANIN, UCC_MEMORY_TYPE_HOST, 0, UCC_MSG_MAX,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_fanin_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_FANOUT, UCC_MEMORY_TYPE_HOST, 0, UCC_MSG_MAX,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_fanout_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST, 0, UCC_MSG_MAX,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_barrier_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, score, UCC_TL_TEAM_SIZE(team), ucc_tl_shm_coll_init,
            tl_team, UCC_TL_SHM_DEFAULT_SCORE, NULL);

        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
    }

    //TODO: check that collective range does not exceed data size

    *score_p = score;
    return UCC_OK;

err:
    ucc_coll_score_free(score);
    *score_p = NULL;
    return status;
}
