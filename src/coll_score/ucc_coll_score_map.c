/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "ucc_coll_score.h"
#include "utils/ucc_coll_utils.h"
#include "schedule/ucc_schedule.h"

typedef struct ucc_score_map {
    ucc_coll_score_t *score;
} ucc_score_map_t;

ucc_status_t ucc_coll_score_build_map(ucc_coll_score_t *score,
                                      ucc_score_map_t **map_p)
{
    ucc_score_map_t *map;
    map = ucc_malloc(sizeof(*map), "ucc_score_map");
    if (!map) {
        ucc_error("failed to allocate %zd bytes for score map", sizeof(*map));
        return UCC_ERR_NO_MEMORY;
    }
    map->score = score;
    *map_p     = map;
    return UCC_OK;
}

void ucc_coll_score_free_map(ucc_score_map_t *map)
{
    ucc_coll_score_free(map->score);
    ucc_free(map);
}

static
ucc_status_t ucc_coll_score_map_lookup(ucc_score_map_t      *map,
                                       ucc_base_coll_args_t *bargs,
                                       ucc_msg_range_t     **range)
{
    ucc_memory_type_t mt      = ucc_coll_args_mem_type(bargs);
    unsigned          ct      = ucc_ilog2(bargs->args.coll_type);
    size_t            msgsize = ucc_coll_args_msgsize(bargs);
    ucc_list_link_t  *list;
    ucc_msg_range_t  *r;

    if (mt == UCC_MEMORY_TYPE_ASSYMETRIC) {
        /* TODO */
        return UCC_ERR_NOT_SUPPORTED;
    } else if (mt == UCC_MEMORY_TYPE_NOT_APPLY) {
        /* Temporary solution: for Barrier, Fanin, Fanout - use
           "host" range list */
        mt = UCC_MEMORY_TYPE_HOST;
    }
    if (msgsize == UCC_MSG_SIZE_INVALID || msgsize == UCC_MSG_SIZE_ASSYMETRIC) {
        /* These algorithms require global communication to get the same msgsize estimation.
           Can't use msg ranges. Use msize 0 (assuming the range list should only contain 1
           range [0:inf]) */
        msgsize = 0;
    }
    list = &map->score->scores[ct][mt];
    ucc_list_for_each(r, list, super.list_elem) {
        if (msgsize >= r->start && msgsize < r->end) {
            *range = r;
            return UCC_OK;
        }
    }
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_coll_init(ucc_score_map_t      *map,
                           ucc_base_coll_args_t *bargs,
                           ucc_coll_task_t     **task)
{
    ucc_msg_range_t  *r;
    ucc_coll_entry_t *fb;
    ucc_base_team_t  *team;
    ucc_status_t      status;

    status = ucc_coll_score_map_lookup(map, bargs, &r);
    if (UCC_OK != status) {
        return status;
    }

    team   = r->super.team;
    status = r->super.init(bargs, team, task);
    fb     = ucc_list_head(&r->fallback, ucc_coll_entry_t, list_elem);

    while (&fb->list_elem != &r->fallback &&
           (status == UCC_ERR_NOT_SUPPORTED ||
            status == UCC_ERR_NOT_IMPLEMENTED)) {
        ucc_debug("coll is not supported for %s, fallback %s",
                  team->context->lib->log_component.name,
                  fb->team->context->lib->log_component.name);
        team   = fb->team;
        status = fb->init(bargs, team, task);
        fb     = ucc_list_next(&fb->list_elem, ucc_coll_entry_t, list_elem);
    }

    return status;
}
