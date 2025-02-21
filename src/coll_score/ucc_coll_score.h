/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_COLL_SCORE_H_
#define UCC_COLL_SCORE_H_
#include "config.h"
#include "utils/ucc_list.h"
#include "components/base/ucc_base_iface.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_compiler_def.h"
#include <limits.h>

#define UCC_SCORE_MAX INT_MAX
#define UCC_SCORE_MIN 0
#define UCC_SCORE_INVALID -1

#define UCC_MSG_MAX UINT64_MAX

/* Callback that maps alg_id (int or str) to the "init" function.
   This callback is provided by the component (CL/TL) that uses
   ucc_coll_score_alloc_from_str.
   Return values:
   UCC_OK - input alg_id can be correctly mapped to the "init" fn
   UCC_ERR_NOT_SUPPORTED - CL/TL doesn't allow changing algorithms ids for
   the given coll_type, mem_type
   UCC_ERR_INVALID_PARAM - incorrect value of alg_id is provided */
typedef ucc_status_t (*ucc_alg_id_to_init_fn_t)(int alg_id,
                                                const char *alg_id_str,
                                                ucc_coll_type_t coll_type,
                                                ucc_memory_type_t mem_type,
                                                ucc_base_coll_init_fn_t *init);

typedef struct ucc_coll_score_team_info {
    ucc_score_t              default_score;
    ucc_rank_t               size;
    uint64_t                 supported_colls;
    ucc_memory_type_t       *supported_mem_types;
    int                      num_mem_types;
    ucc_base_coll_init_fn_t  init;
    ucc_alg_id_to_init_fn_t  alg_fn;
} ucc_coll_score_team_info_t;

typedef struct ucc_coll_entry {
    ucc_list_link_t          list_elem;
    ucc_score_t              score;
    ucc_base_coll_init_fn_t  init;
    ucc_base_team_t         *team;
} ucc_coll_entry_t;

typedef struct ucc_msg_range {
    ucc_coll_entry_t        super;
    ucc_list_link_t         fallback;
    size_t                  start;
    size_t                  end;
} ucc_msg_range_t;

typedef struct ucc_coll_score {
    ucc_list_link_t scores[UCC_COLL_TYPE_NUM][UCC_MEMORY_TYPE_LAST];
} ucc_coll_score_t;

typedef struct ucc_score_map ucc_score_map_t;

char *ucc_score_to_str(ucc_score_t score, char *buf, size_t max);

/* Allocates empty score data structure */
ucc_status_t  ucc_coll_score_alloc(ucc_coll_score_t **score);

/* Adds a single score range to the storage.
   "init" must be either proper base_coll_init_fn or NULL. */
ucc_status_t  ucc_coll_score_add_range(ucc_coll_score_t *score,
                                       ucc_coll_type_t   coll_type,
                                       ucc_memory_type_t mem_type, size_t start,
                                       size_t end, ucc_score_t msg_score,
                                       ucc_base_coll_init_fn_t init,
                                       ucc_base_team_t *team);

/* Releases the score data structure and all the score ranges stored
   there */
void ucc_coll_score_free(ucc_coll_score_t *score);

/* Merges 2 scores score1 and score2 into the new score "rst" selecting
   larger score. Ie.: rst will contain a range from score1 if either
   score of that range in score1 is larger than that of score2 or
   that range does not overlap with score2.

    This fn is used by CL to merge scores from multiple TLs and produce
    a score map. As a result the produced score map will select TL with
    higher score.*/
ucc_status_t ucc_coll_score_merge(ucc_coll_score_t * score1,
                                  ucc_coll_score_t * score2,
                                  ucc_coll_score_t **rst, int free_inputs);


/* Parses SCORE string (see ucc_base_iface.c for pattern description)
   and initializes score data structure. team_size is used to filter
   score ranges provided by user. "init" - default init function to be
   used if alg_id is not explicitly present in SCORE str. */
ucc_status_t ucc_coll_score_alloc_from_str(const char *            str,
                                           ucc_coll_score_t **     score,
                                           ucc_rank_t              team_size,
                                           ucc_base_coll_init_fn_t init,
                                           ucc_base_team_t *       team,
                                           ucc_alg_id_to_init_fn_t alg_fn);

/* Update existing score datastructure with the custom input specified
   in "str". Update applies the modifications specified in "str" to the
   existing "score": if some range in "str" overlaps with a range in "score"
   then the latter is modified according to "str" (in contrast to
   ucc_coll_score_merge where MAX score rule is used). If the new range
   is provided in "str" and it does not have "score" qualifier then def_score
   is used for it. If the new range is provided in "str" and it does not have
   "alg_id" qualifier than "init" fn is used otherwise "init" is taken from
   alg_fn mapper callback. "mtypes" parameter determines which memory types
   will be udpated.

   This function has 2 usages (see tl_ucp_team.c: ucc_tl_ucp_team_get_scores
   function):
   1. Construct custom score table for component based on the built-in
      selection rules represented by string. In this case "init"" can be set
      to some generic init function (ucc_tl_ucp_coll_init).
   2. Update existing score datastruct with user input: in this case
      "init" is set to NULL. User provided ranges without alg_id will not
      modify any existing "init" functions in that case and only change the
      score of existing ranges*/

ucc_status_t
ucc_coll_score_update_from_str(const char *str,
                               const ucc_coll_score_team_info_t *info,
                               ucc_base_team_t *team,
                               ucc_coll_score_t *score);

ucc_status_t ucc_coll_score_merge_in(ucc_coll_score_t **dst,
                                     ucc_coll_score_t *src);

/* Initializes the default score datastruct with a set of coll_types specified
   as a bitmap, mem_types passed as array, default score value and default init fn.
   The collective will have msg range 0-inf. */
ucc_status_t ucc_coll_score_build_default(ucc_base_team_t        *team,
                                          ucc_score_t             default_score,
                                          ucc_base_coll_init_fn_t default_init,
                                          uint64_t                coll_types,
                                          ucc_memory_type_t      *mem_types,
                                          int mt_n, ucc_coll_score_t **score_p);

/* Builds optimized representation of a score for the faster lookup */
ucc_status_t ucc_coll_score_build_map(ucc_coll_score_t *score,
                                      ucc_score_map_t **map);

void ucc_coll_score_free_map(ucc_score_map_t *map);

/* Initializes task based on args selection and score map.
   Checks fallbacks if necessary. */
ucc_status_t ucc_coll_init(ucc_score_map_t      *map,
                           ucc_base_coll_args_t *bargs,
                           ucc_coll_task_t     **task);

ucc_status_t ucc_coll_score_dup(const ucc_coll_score_t *in,
                                ucc_coll_score_t      **out);

void ucc_coll_score_set(ucc_coll_score_t *score,
                        ucc_score_t       value);

void ucc_coll_score_map_print_info(const ucc_score_map_t *score, int verbosity);

ucc_status_t ucc_coll_score_update(ucc_coll_score_t  *score,
                                   ucc_coll_score_t  *update,
                                   ucc_score_t        default_score,
                                   ucc_memory_type_t *mtypes,
                                   int                mt_n,
                                   uint64_t           colls);

#endif
