/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_coll_score.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_string.h"
#include "schedule/ucc_schedule.h"

#include <dlfcn.h>

typedef struct ucc_score_map {
    ucc_coll_score_t *score;
    /* Size, rank of the process in the base_team associated with that
       score_map. It can be CL or TL team, which can be a subset of a
       core UCC team */
    ucc_rank_t        team_size;
    ucc_rank_t        team_rank;
} ucc_score_map_t;

ucc_status_t ucc_coll_score_build_map(ucc_coll_score_t *score,
                                      ucc_score_map_t **map_p)
{
    ucc_score_map_t *map;
    ucc_msg_range_t *range, *temp, *next;
    ucc_list_link_t *lst;
    int              i, j;

    map = ucc_calloc(1, sizeof(*map), "ucc_score_map");
    if (!map) {
        ucc_error("failed to allocate %zd bytes for score map", sizeof(*map));
        return UCC_ERR_NO_MEMORY;
    }

    /* Resolve boundary between neighbour ranges:
       if ranges share a msg size as boundary leave that msgsize
       to the range with higher score. That way components that report
       higher scores do not get overwritten at range boundary */
    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        for (j = 0; j < UCC_MEMORY_TYPE_LAST; j++) {
            lst = &score->scores[i][j];
            if (!ucc_list_is_empty(lst) && map->team_size == 0) {
                /* For a given score_map all the entries refer to the base_teams
                   (CL/TL) of the same size/rank. So we can take the first one. */
                range = ucc_list_head(lst, ucc_msg_range_t, super.list_elem);
                map->team_size = range->super.team->params.size;
                map->team_rank = range->super.team->params.rank;
            }
            ucc_list_for_each_safe(range, temp, lst, super.list_elem) {
                if (range->super.list_elem.next != lst) {
                    next = ucc_container_of(range->super.list_elem.next,
                                            ucc_msg_range_t, super.list_elem);

                    if (range->end == next->start) {
                        if (range->super.score > next->super.score) {
                            next->start++;
                        } else {
                            range->end--;
                        }
                    }
                }
            }
        }
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

static ucc_status_t ucc_coll_score_map_lookup(ucc_score_map_t *map,
                                              ucc_base_coll_args_t *bargs,
                                              ucc_msg_range_t **range)
{
    ucc_memory_type_t mt      = ucc_coll_args_mem_type(&bargs->args,
                                                       map->team_rank);
    unsigned          ct      = ucc_ilog2(bargs->args.coll_type);
    size_t            msgsize = ucc_coll_args_msgsize(&bargs->args,
                                                      map->team_rank,
                                                      map->team_size);
    ucc_list_link_t *list;
    ucc_msg_range_t *r;

    if (mt == UCC_MEMORY_TYPE_NOT_APPLY) {
        /* Temporary solution: for Barrier, Fanin, Fanout - use
           "host" range list */
        mt = UCC_MEMORY_TYPE_HOST;
    }
    ucc_assert(ucc_coll_args_is_mem_symmetric(&bargs->args, map->team_rank));
    if (msgsize == UCC_MSG_SIZE_INVALID || msgsize == UCC_MSG_SIZE_ASYMMETRIC) {
        /* These algorithms require global communication to get the same msgsize estimation.
           Can't use msg ranges. Use msize 0 (assuming the range list should only contain 1
           range [0:inf]) */
        msgsize = 0;
    }
    list = &map->score->scores[ct][mt];
    ucc_list_for_each(r, list, super.list_elem) {
        if (msgsize >= r->start && msgsize <= r->end) {
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
    if (ucc_unlikely(UCC_OK != status)) {
        ucc_debug("coll_score_map lookup failed %d (%s)",
                   status, ucc_status_string(status));
        return status;
    }

    team   = r->super.team;
    status = r->super.init(bargs, team, task);
    if (UCC_OK == status) {
        return UCC_OK;
    }

    fb = ucc_list_head(&r->fallback, ucc_coll_entry_t, list_elem);
    while (&fb->list_elem != &r->fallback &&
           (status == UCC_ERR_NOT_SUPPORTED ||
            status == UCC_ERR_NOT_IMPLEMENTED)) {
        ucc_debug("coll %s is not supported for %s, fallback %s",
                  ucc_coll_type_str(bargs->args.coll_type),
                  team->context->lib->log_component.name,
                  fb->team->context->lib->log_component.name);
        team   = fb->team;
        status = fb->init(bargs, team, task);
        fb     = ucc_list_next(&fb->list_elem, ucc_coll_entry_t, list_elem);
    }

    return status;
}

#define STR_APPEND(_str, _left, _tmp_size, _format, ...) {           \
    char _tmp[_tmp_size];                                            \
    ucc_snprintf_safe(_tmp, _tmp_size, _format, ## __VA_ARGS__ );    \
    strncat(_str, _tmp, _left - 1);                                  \
    _left = sizeof(_str)  - strlen(_str);                            \
    if (!_left) {                                                    \
        return;                                                      \
    }                                                                \
}

static const char *get_fn_name(ucc_base_coll_init_fn_t init_fn)
{
    int status;
    Dl_info info;
    const char *fn_ptr_str = "?";
    status = dladdr(init_fn, &info);
    if (status && info.dli_sname != NULL) {
        fn_ptr_str = info.dli_sname;
    }
    return fn_ptr_str;
}

void ucc_coll_score_map_print_info(const ucc_score_map_t *map, int verbosity)
{
    size_t           left;
    ucc_msg_range_t *range;
    int              i, j, all_empty;
    char             score_str[32];
    char             range_str[128];
    char             coll_str[1024];

    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        all_empty = 1;
        for (j = 0; j < UCC_MEMORY_TYPE_LAST; j++) {
            if (!ucc_list_is_empty(&map->score->scores[i][j])) {
                all_empty = 0;
                break;
            }
        }
        if (all_empty) {
            continue;
        }
        coll_str[0] = '\0';
        left        = sizeof(coll_str);
        STR_APPEND(coll_str, left, 32, "%s:\n",
                   ucc_coll_type_str((ucc_coll_type_t)UCC_BIT(i)));
        for (j = 0; j < UCC_MEMORY_TYPE_LAST; j++) {
            if (ucc_list_is_empty(&map->score->scores[i][j])) {
                continue;
            }
            STR_APPEND(coll_str, left, 32, "\t%s: ",
                       ucc_mem_type_str((ucc_memory_type_t)j));
            ucc_list_for_each(range, &map->score->scores[i][j],
                              super.list_elem) {
                ucc_memunits_range_str(range->start, range->end, range_str,
                                       sizeof(range_str));
                ucc_score_to_str(range->super.score, score_str,
                                 sizeof(score_str));
                if (verbosity >= UCC_LOG_LEVEL_DEBUG) {
                    // Get the name of the init function through dladdr
                    STR_APPEND(coll_str, left, 256, "{%s}:%s:%s=%s ",
                            range_str,
                            range->super.team->context->lib->log_component.name,
                            score_str, get_fn_name(range->super.init));
                } else {
                    STR_APPEND(coll_str, left, 256, "{%s}:%s:%s ",
                            range_str,
                            range->super.team->context->lib->log_component.name,
                            score_str);
                }
            }
            STR_APPEND(coll_str, left, 4, "\n");
        }
        ucc_info("%s", coll_str);
    }
}
