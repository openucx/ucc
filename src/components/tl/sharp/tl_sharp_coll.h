/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHARP_COLL_H_
#define UCC_TL_SHARP_COLL_H_

#include "tl_sharp.h"

/* need to query for datatype support at runtime */
#define SHARP_DTYPE_UNKNOWN 0xFFFF

extern enum sharp_datatype ucc_to_sharp_dtype[];

ucc_status_t ucc_tl_sharp_allreduce_init(ucc_tl_sharp_task_t *task);

ucc_status_t ucc_tl_sharp_barrier_init(ucc_tl_sharp_task_t *task);

ucc_status_t ucc_tl_sharp_bcast_init(ucc_tl_sharp_task_t *task);

#if HAVE_DECL_SHARP_COLL_DO_REDUCE_SCATTER
ucc_status_t ucc_tl_sharp_reduce_scatter_init(ucc_tl_sharp_task_t *task);
#endif
#endif
