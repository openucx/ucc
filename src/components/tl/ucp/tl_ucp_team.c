/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_ep.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_parser.h"
#include "utils/ucc_string.h"
#include "coll_score/ucc_coll_score.h"

static inline ucc_status_t ucc_tl_ucp_get_topo(ucc_tl_ucp_team_t *team)
{
    ucc_subset_t  subset;
    ucc_status_t  status;

    status = ucc_ep_map_create_nested(&UCC_TL_CORE_TEAM(team)->ctx_map,
                                      &UCC_TL_TEAM_MAP(team),
                                      &team->ctx_map);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to create ctx map");
        return status;
    }
    subset.map    = team->ctx_map;
    subset.myrank = UCC_TL_TEAM_RANK(team);

    status = ucc_topo_init(subset, UCC_TL_CORE_CTX(team)->topo, &team->topo);

    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init team topo");
        goto err_topo_init;
    }

    return UCC_OK;
err_topo_init:
    ucc_ep_map_destroy_nested(&team->ctx_map);
    return status;
}

UCC_CLASS_INIT_FUNC(ucc_tl_ucp_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_ucp_context_t *ctx = ucc_derived_of(tl_context,
                                               ucc_tl_ucp_context_t);
    ucc_kn_radix_t max_radix, min_radix;
    ucc_rank_t     tsize, max_ppn;
    ucc_status_t   status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);
    /* TODO: init based on ctx settings and on params: need to check
             if all the necessary ranks mappings are provided */
    self->preconnect_task = NULL;
    self->seq_num         = 0;
    self->status          = UCC_INPROGRESS;
    self->tuning_str      = "";
    self->topo            = NULL;
    self->opt_radix       = UCC_UUNITS_AUTO_RADIX;
    self->opt_radix_host  = UCC_UUNITS_AUTO_RADIX;

    status = ucc_config_clone_table(&UCC_TL_UCP_TEAM_LIB(self)->cfg, &self->cfg,
                                    ucc_tl_ucp_lib_config_table);
    if (UCC_OK != status) {
        return status;
    }

    if (ctx->topo_required) {
        status = ucc_tl_ucp_get_topo(self);
        if (UCC_OK != status) {
            return status;
        }
    }

    if (ucc_global_config.file_cfg && !UCC_TL_IS_SERVICE_TEAM(self) &&
        ctx->topo_required && tl_context->lib->use_tuning) {
        status = ucc_add_team_sections(&self->cfg, ucc_tl_ucp_lib_config_table,
                                       self->topo, &self->tuning_str,
                                       "UCC_TL_UCP_TUNE",
                                       UCC_TL_CORE_CTX(self)->lib->full_prefix,
                                       ucc_tl_ucp.super.tl_lib_config.prefix);
        if (status != UCC_OK) {
            ucc_debug("section not found");
        }
    }

    if (!self->topo && self->cfg.use_reordering) {
        tl_debug(tl_context->lib,
                 "topo is not available, disabling ranks reordering");
        self->cfg.use_reordering = 0;
    }

    if (self->topo && !UCC_TL_IS_SERVICE_TEAM(self)) {
        tsize = UCC_TL_TEAM_SIZE(self);
        max_ppn = ucc_topo_max_ppn(self->topo);

        min_radix = ucc_min(tsize, 3);
        max_radix = tsize;
        self->opt_radix = ucc_kn_get_opt_radix(tsize, min_radix, max_radix);
        if (max_ppn == 1) {
            self->opt_radix_host = self->opt_radix;
        } else {
            if (self->topo->topo->sock_bound) {
                min_radix = 2;
                max_radix = ucc_min(tsize, ucc_topo_min_socket_size(self->topo));
                self->opt_radix_host = ucc_kn_get_opt_radix(tsize, min_radix,
                                                            max_radix);
            }
        }
        tl_debug(tl_context->lib, "opt knomial radix: general %d host %d",
                 self->opt_radix, self->opt_radix_host);
    }

    tl_debug(tl_context->lib, "posted tl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_ucp_team_t)
{
    ucc_config_parser_release_opts(&self->cfg, ucc_tl_ucp_lib_config_table);
    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_ucp_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_ucp_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_ucp_team_destroy(ucc_base_team_t *tl_team)
{
    ucc_tl_ucp_team_t *team = ucc_derived_of(tl_team, ucc_tl_ucp_team_t);

    if (team->topo) {
        ucc_ep_map_destroy_nested(&team->ctx_map);
        ucc_topo_cleanup(team->topo);
    }
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_ucp_team_t)(tl_team);
    return UCC_OK;
}

static ucc_status_t ucc_tl_ucp_team_preconnect(ucc_tl_ucp_team_t *team)
{
    ucc_rank_t src, dst, size, rank;
    ucc_status_t status;
    int i;

    size = UCC_TL_TEAM_SIZE(team);
    rank = UCC_TL_TEAM_RANK(team);
    if (!team->preconnect_task) {
        team->preconnect_task             = ucc_tl_ucp_get_task(team);
        team->preconnect_task->tagged.tag = 0;
        team->preconnect_task->super.bargs.args.mask = 0;
    }
    if (UCC_INPROGRESS == ucc_tl_ucp_test(team->preconnect_task)) {
        ucp_worker_progress(team->worker->ucp_worker);
        return UCC_INPROGRESS;
    }
    for (i = team->preconnect_task->tagged.send_posted; i < size; i++) {
        src = (rank - i + size) % size;
        dst = (rank + i) % size;
        status = ucc_tl_ucp_send_nb(NULL, 0, UCC_MEMORY_TYPE_UNKNOWN, src, team,
                                    team->preconnect_task);
        if (UCC_OK != status) {
            return status;
        }
        status = ucc_tl_ucp_recv_nb(NULL, 0, UCC_MEMORY_TYPE_UNKNOWN, dst, team,
                                    team->preconnect_task);
        if (UCC_OK != status) {
            return status;
        }
        if (UCC_INPROGRESS == ucc_tl_ucp_test(team->preconnect_task)) {
            return UCC_INPROGRESS;
        }
    }
    tl_debug(UCC_TL_TEAM_LIB(team), "preconnected tl team: %p, num_eps %d",
             team, size);
    ucc_tl_ucp_put_task(team->preconnect_task);
    team->preconnect_task = NULL;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_ucp_team_t *   team = ucc_derived_of(tl_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *ctx  = UCC_TL_UCP_TEAM_CTX(team);
    int                   i;
    ucc_status_t          status;

    if (USE_SERVICE_WORKER(team)) {
        team->worker = &ctx->service_worker;
    } else {
        team->worker = &ctx->worker;
    }

    if (team->status == UCC_OK) {
        return UCC_OK;
    }

    if (UCC_TL_TEAM_SIZE(team) <= ctx->cfg.preconnect) {
        status = ucc_tl_ucp_team_preconnect(team);
        if (UCC_INPROGRESS == status) {
            return UCC_INPROGRESS;
        } else if (UCC_OK != status) {
            goto err_preconnect;
        }
    }

    if (ctx->remote_info) {
        for (i = 0; i < ctx->n_rinfo_segs; i++) {
            team->va_base[i]     = ctx->remote_info[i].va_base;
            team->base_length[i] = ctx->remote_info[i].len;
        }
    }

    tl_debug(tl_team->context->lib, "initialized tl team: %p", team);
    team->status = UCC_OK;
    return UCC_OK;

err_preconnect:
    return status;
}

ucc_status_t ucc_tl_ucp_team_get_scores(ucc_base_team_t   *tl_team,
                                        ucc_coll_score_t **score_p)
{
    ucc_tl_ucp_team_t          *team    = ucc_derived_of(tl_team,
                                                      ucc_tl_ucp_team_t);
    ucc_component_framework_t  *plugins = &ucc_tl_ucp.super.coll_plugins;
    ucc_tl_ucp_context_t       *tl_ctx  = UCC_TL_UCP_TEAM_CTX(team);
    ucc_base_context_t         *ctx     = UCC_TL_TEAM_CTX(team);
    int                         mt_n    = 0;
    ucc_memory_type_t           mem_types[UCC_MEMORY_TYPE_LAST];
    ucc_coll_score_t           *score, *tlcp_score;
    ucc_tl_coll_plugin_iface_t *tlcp;
    ucc_status_t                status;
    unsigned                    i;
    char                       *ucc_tl_ucp_default_alg_select_str
                                          [UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR];
    ucc_coll_score_team_info_t team_info;

    for (i = 0; i < UCC_MEMORY_TYPE_LAST; i++) {
        if (tl_ctx->ucp_memory_types & UCC_BIT(ucc_memtype_to_ucs[i])) {
            tl_debug(tl_team->context->lib,
                     "enable support for memory type %s",
                     ucc_memory_type_names[i]);
            mem_types[mt_n++] = (ucc_memory_type_t)i;
        }
    }

    team_info.alg_fn              = ucc_tl_ucp_alg_id_to_init;
    team_info.default_score       = UCC_TL_UCP_DEFAULT_SCORE;
    team_info.init                = ucc_tl_ucp_coll_init;
    team_info.num_mem_types       = mt_n;
    team_info.supported_mem_types = mem_types; /* all memory types supported*/
    team_info.supported_colls     = UCC_TL_UCP_SUPPORTED_COLLS;
    team_info.size                = UCC_TL_TEAM_SIZE(team);

    /* There can be a different logic for different coll_type/mem_type.
       Right now just init everything the same way. */
    status = ucc_coll_score_build_default(tl_team, UCC_TL_UCP_DEFAULT_SCORE,
                              ucc_tl_ucp_coll_init, UCC_TL_UCP_SUPPORTED_COLLS,
                              mem_types, mt_n, &score);
    if (UCC_OK != status) {
        return status;
    }
    status = ucc_tl_ucp_team_default_score_str_alloc(team,
        ucc_tl_ucp_default_alg_select_str);
    if (UCC_OK != status) {
        return status;
    }
    for (i = 0; i < UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR; i++) {
        status = ucc_coll_score_update_from_str(
            ucc_tl_ucp_default_alg_select_str[i], &team_info,
            &team->super.super, score);
        if (UCC_OK != status) {
            tl_error(tl_team->context->lib,
                     "failed to apply default coll select setting: %s",
                     ucc_tl_ucp_default_alg_select_str[i]);
            goto err;
        }
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(ctx->score_str, &team_info,
                                                &team->super.super, score);
    } else if (strlen(team->tuning_str) > 0) {
        status = ucc_coll_score_update_from_str(team->tuning_str, &team_info,
                                                &team->super.super, score);
    }
    /* If INVALID_PARAM - User provided incorrect input - try to proceed */
    if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
        (status != UCC_ERR_NOT_SUPPORTED)) {
        goto err;
    }

    for (i = 0; i < plugins->n_components; i++) {
        tlcp = ucc_derived_of(plugins->components[i],
                              ucc_tl_coll_plugin_iface_t);
        status = tlcp->get_scores(tl_team, &tlcp_score);
        if (UCC_OK != status) {
            goto err;
        }
        status = ucc_coll_score_merge_in(&score, tlcp_score);
        if (UCC_OK != status) {
            goto err;
        }
    }
    ucc_tl_ucp_team_default_score_str_free(ucc_tl_ucp_default_alg_select_str);
    *score_p = score;
    return UCC_OK;
err:
    ucc_tl_ucp_team_default_score_str_free(ucc_tl_ucp_default_alg_select_str);
    ucc_coll_score_free(score);
    return status;
}
