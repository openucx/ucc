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
#include "core/ucc_team.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/ucc_sys.h"
#include "bcast/bcast.h"
#include "reduce/reduce.h"
#include "barrier/barrier.h"
#include "fanin/fanin.h"
#include "fanout/fanout.h"
#include "allreduce/allreduce.h"
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

    //TODO opt for single group
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
    ucc_rank_t *ranks = (ucc_rank_t *)ucc_malloc(team_size * sizeof(*ranks)); /* NOLINT to supress clang_tidy check of team size == 0, already checked in core */

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
        ucc_free(perf_funcs_keys);
        return UCC_ERR_NO_MEMORY;
    }

    ucc_tl_shm_create_perf_func_list(team, perf_funcs_keys, perf_funcs_list);
    vendor = ucc_arch_get_cpu_vendor();
    model  = ucc_arch_get_cpu_model();

    for (i = 0; i < perf_funcs_list->size; i++) {
        if (perf_funcs_list->keys[i].cpu_vendor == vendor &&
            perf_funcs_list->keys[i].cpu_model == model &&
            perf_funcs_list->keys[i].team_size == team_size) {
            team->perf_params_bcast  = perf_funcs_list->keys[i].bcast_func;
            team->perf_params_reduce = perf_funcs_list->keys[i].reduce_func;
            team->layout             = perf_funcs_list->keys[i].layout;
            break;
        }
    }

    ucc_free(perf_funcs_keys);
    ucc_free(perf_funcs_list);
    return UCC_OK;
}

static void ucc_tl_shm_init_segs(ucc_tl_shm_team_t *team)
{
    size_t cfg_ctrl_size       = UCC_TL_SHM_TEAM_LIB(team)->cfg.ctrl_size;
    size_t cfg_data_size       = UCC_TL_SHM_TEAM_LIB(team)->cfg.data_size;
    ucc_tl_shm_seg_layout_t sl = team->layout;
    size_t                  page_size = ucc_get_page_size();
    size_t ctrl_offset, data_offset, grp_ctrl_size,
        grp_data_size, grp_seg_size, grp_0_data_size;
    void * ctrl, *data;
    ucc_rank_t group_size;
    int        i, j;

    for (i = 0; i < team->n_concurrent; i++) {
        ctrl_offset = 0;
        data_offset = 0;
        for (j = 0; j < team->n_base_groups; j++) {
            group_size      = team->base_groups[j].group_size;
            grp_ctrl_size   = ucc_align_up(group_size * cfg_ctrl_size,
                                           page_size);
            grp_data_size   = group_size * cfg_data_size;
            grp_0_data_size = team->base_groups[0].group_size * cfg_data_size;
            grp_seg_size    = grp_ctrl_size + grp_data_size;

            if (sl == SEG_LAYOUT_CONTIG) {
                ctrl = PTR_OFFSET(team->shm_buffers[0],
                                  (team->ctrl_size + team->data_size) * i +
                                      ctrl_offset);
                data = PTR_OFFSET(ctrl,
                                  team->ctrl_size + data_offset - ctrl_offset);
                ctrl_offset += grp_ctrl_size;
                data_offset += grp_data_size;
            } else if (sl == SEG_LAYOUT_MIXED) {
                ctrl = PTR_OFFSET(team->shm_buffers[0],
                                  (team->ctrl_size + grp_0_data_size) * i +
                                      ctrl_offset);
                if (j == 0) {
                    data = PTR_OFFSET(ctrl, team->ctrl_size - ctrl_offset);
                } else {
                    data = PTR_OFFSET(team->shm_buffers[j], grp_data_size * i);
                }
                ctrl_offset += grp_ctrl_size;
            } else {
                ctrl = PTR_OFFSET(team->shm_buffers[j], grp_seg_size * i);
                data = PTR_OFFSET(ctrl, grp_ctrl_size);
            }
            team->segs[i * team->n_base_groups + j].ctrl = ctrl;
            team->segs[i * team->n_base_groups + j].data = data;
        }
    }
}

static ucc_status_t ucc_tl_shm_seg_alloc(ucc_tl_shm_team_t *team)
{
    ucc_rank_t team_rank = UCC_TL_TEAM_RANK(team),
               team_size = UCC_TL_TEAM_SIZE(team);
    size_t     cfg_ctrl_size        = UCC_TL_SHM_TEAM_LIB(team)->cfg.ctrl_size;
    size_t     cfg_data_size        = UCC_TL_SHM_TEAM_LIB(team)->cfg.data_size;
    ucc_tl_shm_seg_layout_t sl      = team->layout;
    size_t                  shmsize = 0;
    int                     shmid   = -1;
    ucc_team_oob_coll_t     oob     = UCC_TL_TEAM_OOB(team);
    ucc_rank_t              gsize;
    ucc_status_t            status;
    size_t                  page_size;

    gsize = team->base_groups[team->my_group_id].group_size;
    team->allgather_dst =
        (int *)ucc_malloc(sizeof(int) * (team_size + 1), "algather dst buffer");

    if (sl == SEG_LAYOUT_CONTIG) {
        if (team_rank == 0) {
            shmsize = team->n_concurrent * (team->ctrl_size + team->data_size);
        }
    } else if (sl == SEG_LAYOUT_MIXED) {
        if (team_rank == 0) {
            ucc_assert(team->is_group_leader);
            shmsize =
                team->n_concurrent * (team->ctrl_size + gsize * cfg_data_size);
        } else if (team->is_group_leader) {
            shmsize = team->n_concurrent * gsize * cfg_data_size;
        }
    } else if (team->is_group_leader) {
        page_size = ucc_get_page_size();
        shmsize   = team->n_concurrent *
                  (gsize * cfg_data_size +
                   ucc_align_up(gsize * cfg_ctrl_size, page_size));
    }
    /* LOWEST on node rank  within the comm will initiate the segment creation.
     * Everyone else will attach. */
    if (shmsize != 0) {
        shmid = shmget(IPC_PRIVATE, shmsize, SHM_MODE);
        if (shmid < 0) {
            tl_error(team->super.super.context->lib,
                     "Root: shmget failed, shmid=%d, shmsize=%ld, errno: %s",
                     shmid, shmsize, strerror(errno));
            goto allgather;
        }
        team->shm_buffers[team->my_group_id] = (void *)shmat(shmid, NULL, 0);
        if (team->shm_buffers[team->my_group_id] == (void *)-1) {
            shmid                                = -2;
            team->shm_buffers[team->my_group_id] = NULL;
            tl_error(team->super.super.context->lib, "shmat failed, errno: %s",
                     strerror(errno));
            goto allgather;
        }
        memset(team->shm_buffers[team->my_group_id], 0, shmsize);
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
    int          n_sbgps, i, max_trees;
    ucc_rank_t   team_size;
    uint32_t     cfg_ctrl_size, group_size;
    ucc_subset_t subset;
    size_t       page_size;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);

    if (NULL == UCC_TL_CORE_CTX(self)->topo) {
        /* CORE context does not have topo information -
	 * local context mode */
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_ep_map_create_nested(&UCC_TL_CORE_TEAM(self)->ctx_map,
                                      &UCC_TL_TEAM_MAP(self),
                                      &self->ctx_map);
    if (UCC_OK != status) {
        tl_error(ctx->super.super.lib, "failed to create ctx map");
        return status;
    }
    subset.map    = self->ctx_map;
    subset.myrank = UCC_TL_TEAM_RANK(self);
    team_size     = UCC_TL_TEAM_SIZE(self);
    cfg_ctrl_size = UCC_TL_SHM_TEAM_LIB(self)->cfg.ctrl_size;

    self->layout             = UCC_TL_SHM_TEAM_LIB(self)->cfg.layout;
    self->perf_params_bcast  = ucc_tl_shm_perf_params_generic_bcast;
    self->perf_params_reduce = ucc_tl_shm_perf_params_generic_reduce;
    self->seq_num            = UCC_TL_SHM_TEAM_LIB(self)->cfg.max_concurrent;
    self->status             = UCC_INPROGRESS;
    self->n_concurrent       = UCC_TL_SHM_TEAM_LIB(self)->cfg.max_concurrent;
    self->data_size          = UCC_TL_SHM_TEAM_LIB(self)->cfg.data_size *
                                   team_size;
    self->max_inline         = cfg_ctrl_size - ucc_offsetof(ucc_tl_shm_ctrl_t,
                                                            data);

    status = ucc_topo_init(subset, UCC_TL_CORE_CTX(self)->topo, &self->topo);

    if (UCC_OK != status) {
        tl_error(ctx->super.super.lib, "failed to init team topo");
        goto err_topo_init;
    }

    if (!ucc_topo_is_single_node(self->topo)) {
        tl_debug(ctx->super.super.lib, "multi node team is not supported");
        status = UCC_ERR_INVALID_PARAM;
        goto err_topo_cleanup;
    }

    if (UCC_TL_CORE_CTX(self)->topo->sock_bound != 1) {
        tl_debug(ctx->super.super.lib, "sock bound is not supported");
        status = UCC_ERR_NOT_SUPPORTED;
        goto err_topo_cleanup;
    }

    self->last_posted = ucc_calloc(sizeof(*self->last_posted),
                                   self->n_concurrent, "last_posted");
    if (!self->last_posted) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for last_posted array",
                 sizeof(*self->last_posted) * self->n_concurrent);
        ucc_topo_cleanup(self->topo);
        return UCC_ERR_NO_MEMORY;
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
        goto err_elems;
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
    self->is_group_leader = 0;
    for (i = 0; i < self->n_base_groups; i++) {
        if (UCC_TL_TEAM_RANK(self) ==
            ucc_ep_map_eval(self->base_groups[i].map, 0)) {
            self->is_group_leader = 1;
        }
    }
    self->segs = (ucc_tl_shm_seg_t *)ucc_malloc(
        sizeof(ucc_tl_shm_seg_t) * self->n_base_groups * self->n_concurrent,
        "shm_segs");
    if (!self->segs) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for shm_segs",
                 sizeof(ucc_tl_shm_seg_t) * self->n_concurrent);
        status = UCC_ERR_NO_MEMORY;
        goto err_sockets;
    }

    /* the above call should return ALL socket/numa sbgps including size=1 subgroups */
    status = ucc_tl_shm_rank_group_id_map_init(self);
    if (UCC_OK != status) {
        goto err_segs;
    }

    status = ucc_tl_shm_group_rank_map_init(self);
    if (UCC_OK != status) {
        goto err_segs;
    }

    self->my_group_id =
        ucc_ep_map_eval(self->rank_group_id_map, UCC_TL_TEAM_RANK(self));

    if (UCC_TL_SHM_TEAM_LIB(self)->cfg.set_perf_params) {
        status = ucc_tl_shm_set_perf_funcs(self);
        if (UCC_OK != status) {
            goto err_segs;
        }
    }

    self->ctrl_size  = 0;
    page_size        = ucc_get_page_size();

    for (i = 0; i < self->n_base_groups; i++) {
        group_size = self->base_groups[i].group_size;
        self->ctrl_size += ucc_align_up(group_size * cfg_ctrl_size,
                                        page_size);
    }

    self->shm_buffers =
        (void *)ucc_calloc(sizeof(void *), self->n_base_groups, "shm_buffers");
    if (!self->shm_buffers) {
        status = UCC_ERR_NO_MEMORY;
        goto err_segs;
    }

    status = ucc_tl_shm_seg_alloc(self);
    if (UCC_OK != status) {
        goto err_buffers;
    }
    return UCC_OK;

    //TODO
err_buffers:
    ucc_free(self->shm_buffers);
err_segs:
    ucc_free(self->segs);
err_sockets:
    ucc_free(self->tree_cache->elems);
err_elems:
    ucc_free(self->tree_cache);
err_tree:
    ucc_free(self->last_posted);
err_topo_cleanup:
    ucc_topo_cleanup(self->topo);
err_topo_init:
    ucc_ep_map_destroy_nested(&self->ctx_map);
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

    for (i = 0; i < team->n_base_groups; i++) {
        if (team->shm_buffers[i]) {
            if (shmdt(team->shm_buffers[i]) == -1) {
                tl_error(team->super.super.context->lib, "shmdt failed");
                return UCC_ERR_NO_MESSAGE;
            }
        }
    }
    ucc_free(team->shm_buffers);
    for (i = 0; i < team->tree_cache->size; i++) {
        ucc_free(team->tree_cache->elems[i].tree->top_tree);
        ucc_free(team->tree_cache->elems[i].tree->base_tree);
        ucc_free(team->tree_cache->elems[i].tree);
    }
    ucc_free(team->tree_cache->elems);
    ucc_free(team->tree_cache);
    ucc_free(team->segs);
    ucc_free(team->last_posted);
    ucc_ep_map_destroy_nested(&team->ctx_map);
    ucc_topo_cleanup(team->topo);
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_shm_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_shm_team_t * team   = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_team_oob_coll_t oob    = UCC_TL_TEAM_OOB(team);
    ucc_status_t        status = oob.req_test(team->oob_req);
    int                 i, shmid;
    ucc_rank_t          group_leader;

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
    for (i = 0; i < team->n_base_groups; i++) {
        group_leader = ucc_ep_map_eval(team->base_groups[i].map, 0);
        shmid        = team->allgather_dst[group_leader];
        ucc_assert(group_leader != 0 || shmid != -1);

        if (shmid == -1) {
            /* no shm seg from that group leader */
            continue;
        }

        if (shmid == -2) {
            return UCC_ERR_NO_RESOURCE;
        }

        if (UCC_TL_TEAM_RANK(team) != group_leader) {
            team->shm_buffers[i] = (void *)shmat(shmid, NULL, 0);
            if (team->shm_buffers[i] == (void *)-1) {
                tl_error(team->super.super.context->lib,
                         "Child failed to attach to shmseg, errno: %s\n",
                         strerror(errno));
                return UCC_ERR_NO_RESOURCE;
            }
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

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_ALLREDUCE, UCC_MEMORY_TYPE_HOST, 0, UCC_MSG_MAX,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_allreduce_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, score, UCC_TL_TEAM_SIZE(team), NULL,
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
