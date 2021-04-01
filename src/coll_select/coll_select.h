/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#ifndef COLL_SELECT_H_
#define COLL_SELECT_H_

#include "utils/ucc_list.h"
#include "components/base/ucc_base_iface.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_compiler_def.h"
#include <limits.h>

#define UCC_SCORE_MAX INT_MAX
#define UCC_SCORE_MIN 0
#define UCC_SCORE_INVALID -1

#define UCC_MSG_MAX UINT64_MAX

typedef struct ucc_msg_range {
    ucc_list_link_t         list_elem;
    size_t                  start;
    size_t                  end;
    ucc_score_t             score;
    ucc_base_coll_init_fn_t init;
    ucc_base_team_t        *team;
} ucc_msg_range_t;

typedef struct ucc_coll_score {
    ucc_list_link_t scores[UCC_COLL_TYPE_NUM][UCC_MEMORY_TYPE_LAST];
} ucc_coll_score_t;

typedef struct ucc_score_map ucc_score_map_t;

ucc_status_t  ucc_coll_score_alloc(ucc_coll_score_t **score);

ucc_status_t  ucc_coll_score_add_range(ucc_coll_score_t *score,
                                       ucc_coll_type_t   coll_type,
                                       ucc_memory_type_t mem_type, size_t start,
                                       size_t end, ucc_score_t msg_score,
                                       ucc_base_coll_init_fn_t init,
                                       ucc_base_team_t *team);

void          ucc_coll_score_free(ucc_coll_score_t *score);

ucc_status_t  ucc_coll_score_merge(ucc_coll_score_t * score1,
                                   ucc_coll_score_t * score2,
                                   ucc_coll_score_t **rst, int free_inputs);

ucc_status_t ucc_coll_score_alloc_from_str(const char *str,
                                           ucc_coll_score_t **score,
                                           ucc_rank_t         team_size);

ucc_status_t ucc_coll_score_update_from_str(const char *str,
                                            ucc_coll_score_t *score,
                                            ucc_rank_t        team_size);

ucc_status_t ucc_coll_score_update(ucc_coll_score_t *score,
                                   ucc_coll_score_t *update);

ucc_status_t ucc_coll_score_build_default(ucc_base_team_t *team,
                                          ucc_score_t             default_score,
                                          ucc_base_coll_init_fn_t default_init,
                                          ucc_coll_type_t         coll_types,
                                          ucc_memory_type_t      *mem_types,
                                          int mt_n, ucc_coll_score_t **score_p);

ucc_status_t ucc_coll_score_build_map(ucc_coll_score_t *score,
                                      ucc_score_map_t **map);

void         ucc_coll_score_free_map(ucc_score_map_t *map);

ucc_status_t ucc_coll_score_map_lookup(ucc_score_map_t         *map,
                                       ucc_base_coll_args_t    *args,
                                       ucc_base_coll_init_fn_t *init,
                                       ucc_base_team_t        **team);
#endif
