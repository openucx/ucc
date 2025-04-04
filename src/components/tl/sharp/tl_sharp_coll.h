/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHARP_COLL_H_
#define UCC_TL_SHARP_COLL_H_

#include "tl_sharp.h"

/* need to query for datatype support at runtime */
#define SHARP_DTYPE_UNKNOWN 0xFFFF

#define UCC_TL_SHARP_N_DEFAULT_ALG_SELECT_STR 2
extern const char
    *ucc_tl_sharp_default_alg_select_str[UCC_TL_SHARP_N_DEFAULT_ALG_SELECT_STR];

extern enum sharp_datatype ucc_to_sharp_dtype[];

ucc_status_t ucc_tl_sharp_allreduce_init(ucc_tl_sharp_task_t *task);

ucc_status_t ucc_tl_sharp_barrier_init(ucc_tl_sharp_task_t *task);

ucc_status_t ucc_tl_sharp_bcast_init(ucc_tl_sharp_task_t *task);

#if HAVE_DECL_SHARP_COLL_DO_REDUCE_SCATTER
ucc_status_t ucc_tl_sharp_reduce_scatter_init(ucc_tl_sharp_task_t *task);
#endif

#if HAVE_DECL_SHARP_COLL_DO_ALLGATHER
ucc_status_t ucc_tl_sharp_allgather_init(ucc_tl_sharp_task_t *task);
#endif

#endif
