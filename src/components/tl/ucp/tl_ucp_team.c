/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

static inline ucc_status_t
ucc_tl_ucp_get_topo_ppn(ucc_tl_ucp_team_t *team, ucc_rank_t *ppn_min, ucc_rank_t *ppn_max) // add ppn_range
{
    ucc_ep_map_t            ctx_map;
    ucc_subset_t            subset;
    ucc_topo_t             *topo;
//    ucc_rank_t              nnodes;
    ucc_status_t            status;

    status = ucc_ep_map_create_nested(&UCC_TL_CORE_TEAM(team)->ctx_map,
                                      &UCC_TL_TEAM_MAP(team),
                                      &ctx_map);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to create ctx map");
        return status;
    }
    subset.map    = ctx_map;
    subset.myrank = UCC_TL_TEAM_RANK(team);

    status = ucc_topo_init(subset, UCC_TL_CORE_CTX(team)->topo, &topo);

    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init team perf topo");
        goto err_topo_init;
    }

    *ppn_min = ucc_topo_min_ppn(topo);
    *ppn_max = ucc_topo_max_ppn(topo);
//        nnodes = ucc_topo_nnodes(topo); // needed?
    ucc_topo_cleanup(topo);
err_topo_init:
    ucc_ep_map_destroy_nested(&ctx_map);
    return status;
}

static inline int ucc_parse_section_name(const char* name,
                                         ucc_cpu_vendor_t _vendor,
                                         ucc_cpu_model_t _model,
                                         size_t _team_size,
                                         ucc_cfg_ppn_range_t _ppn_range)
{
    char **split = ucc_str_split(name, " ");

    ucc_cpu_vendor_t vendor = ucc_get_vendor_from_str(ucc_str_split(split[0], "=")[1]);
    ucc_cpu_model_t model = ucc_get_model_from_str(ucc_str_split(split[1], "=")[1]);

    const char *team_size_range = ucc_str_split(split[2], "=")[1];
    ucc_rank_t team_size_begin = (size_t) atoi(ucc_str_split(team_size_range, "-")[0]);
    ucc_rank_t team_size_end = team_size_begin;
    if (ucc_str_split(team_size_range, "-")[1]) {
        team_size_end = (size_t) atoi(ucc_str_split(team_size_range, "-")[1]);
        printf("inside team size end\n");
    }
    printf("team_size_begin = %d, team_size_end = %d\n",team_size_begin, team_size_end);

    const char *ppn_range = ucc_str_split(split[3], "=")[1];
    ucc_rank_t ppn_min = (ucc_rank_t) atoi(ucc_str_split(ppn_range, "-")[0]);
    ucc_rank_t ppn_max = ppn_min;
    if (ucc_str_split(ppn_range, "-")[1]) {
        ppn_max = (ucc_rank_t) atoi(ucc_str_split(ppn_range, "-")[1]);
        printf("inside ppn str max\n");
    }
    printf("ppn_min_str = %d, ppn_max_str = %d\n",ppn_min, ppn_max);

    return (vendor == _vendor && model == _model &&
            _team_size >= team_size_begin && _team_size <= team_size_end &&
            _ppn_range.begin >= ppn_min && _ppn_range.end <= ppn_max);
}

static ucc_status_t ucc_tl_ucp_cfg_add_section(ucc_tl_ucp_team_t *team,
                                               ucc_file_config_t *cfg)//, const ucc_base_team_params_t *params)
{
    ucc_cpu_vendor_t          vendor = ucc_arch_get_cpu_vendor();
    ucc_cpu_model_t           model  = ucc_arch_get_cpu_model();
//    ucc_cfg_team_size_range_t team_sizes;
    size_t team_size = UCC_TL_TEAM_SIZE(team);
    ucc_cfg_ppn_range_t ppn_range;
    khash_t(ucc_sections) *sections = &cfg->sections;
    const char *sec_name;
    ucc_status_t status;
    int i;

    status = ucc_tl_ucp_get_topo_ppn(team, &ppn_range.begin, &ppn_range.end);
    if (UCC_OK != status) {
        return status;
    }
    printf("ppn_min = %d, ppn_max = %d\n", ppn_range.begin, ppn_range.end);

    for (i = kh_begin(sections); i != kh_end(sections); ++i) {
        if (!kh_exist(sections, i)) continue;
        sec_name = kh_key(sections, i);
        if (ucc_parse_section_name(sec_name, vendor, model, team_size, ppn_range)) {
            status = ucc_apply_file_cfg(&team->cfg,
                                        ucc_tl_ucp_lib_config_table, "UCC_",
                                        ucc_tl_ucp.super.tl_lib_config.prefix,
                                        sec_name);
            return status;
        }
    }
    return UCC_ERR_NOT_FOUND;
}

UCC_CLASS_INIT_FUNC(ucc_tl_ucp_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_ucp_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_ucp_context_t);
    ucc_status_t status = UCC_OK;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);
    /* TODO: init based on ctx settings and on params: need to check
             if all the necessary ranks mappings are provided */

    if (UCC_TL_TEAM_SIZE(self) < 2) {
        tl_trace(tl_context->lib,
                 "team size %d is too small, minimal size is 2",
                 UCC_TL_TEAM_SIZE(self));
        return UCC_ERR_NOT_SUPPORTED;
    }

    self->preconnect_task = NULL;
    self->seq_num         = 0;
    self->status          = UCC_INPROGRESS;
    memcpy(&self->cfg, &UCC_TL_UCP_TEAM_LIB(self)->cfg,
           sizeof(ucc_tl_ucp_team_config_t));
    if (ucc_global_config.file_cfg && !IS_SERVICE_TEAM(self) &&
        UCC_TL_CORE_CTX(self)->topo != NULL) {
        status = ucc_tl_ucp_cfg_add_section(self, ucc_global_config.file_cfg);
        if (status != UCC_OK) {
            ucc_debug("section not found");
        }
    }
    if (!IS_SERVICE_TEAM(self)) {
        printf("allreduce kn radix from cfg = %d\n", self->cfg.allreduce_kn_radix);
    }
    tl_info(tl_context->lib, "posted tl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_ucp_team_t)
{
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_ucp_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_ucp_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_ucp_team_destroy(ucc_base_team_t *tl_team)
{
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
        for (int i = 0; i < ctx->n_rinfo_segs; i++) {
            team->va_base[i]     = ctx->remote_info[i].va_base;
            team->base_length[i] = ctx->remote_info[i].len;
        }
    }

    tl_info(tl_team->context->lib, "initialized tl team: %p", team);
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

    for (i = 0; i < UCC_MEMORY_TYPE_LAST; i++) {
        if (tl_ctx->ucp_memory_types & UCC_BIT(ucc_memtype_to_ucs[i])) {
            tl_debug(tl_team->context->lib,
                     "enable support for memory type %s",
                     ucc_memory_type_names[i]);
            mem_types[mt_n++] = (ucc_memory_type_t)i;
        }
    }

    /* There can be a different logic for different coll_type/mem_type.
       Right now just init everything the same way. */
    status = ucc_coll_score_build_default(tl_team, UCC_TL_UCP_DEFAULT_SCORE,
                              ucc_tl_ucp_coll_init, UCC_TL_UCP_SUPPORTED_COLLS,
                              mem_types, mt_n, &score);
    if (UCC_OK != status) {
        return status;
    }

    for (i = 0; i < UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR; i++) {
        status = ucc_coll_score_update_from_str(
            ucc_tl_ucp_default_alg_select_str[i], score, UCC_TL_TEAM_SIZE(team),
            ucc_tl_ucp_coll_init, &team->super.super, UCC_TL_UCP_DEFAULT_SCORE,
            ucc_tl_ucp_alg_id_to_init, mem_types, mt_n);
        if (UCC_OK != status) {
            tl_error(tl_team->context->lib,
                     "failed to apply default coll select setting: %s",
                     ucc_tl_ucp_default_alg_select_str[i]);
            goto err;
        }
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, score, UCC_TL_TEAM_SIZE(team), NULL,
            &team->super.super, UCC_TL_UCP_DEFAULT_SCORE,
            ucc_tl_ucp_alg_id_to_init, mem_types, mt_n);

        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
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
    *score_p = score;
    return UCC_OK;
err:
    ucc_coll_score_free(score);
    return status;
}
