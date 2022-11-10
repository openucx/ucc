/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHARP_COLL_H_
#define UCC_TL_SHARP_COLL_H_

#include "tl_sharp.h"

/* need to query for datatype support at runtime */
#define SHARP_DTYPE_UNKNOWN -1

extern enum sharp_datatype ucc_to_sharp_dtype[];

ucc_status_t ucc_tl_sharp_allreduce_init(ucc_tl_sharp_task_t *task);

ucc_status_t ucc_tl_sharp_barrier_init(ucc_tl_sharp_task_t *task);

ucc_status_t ucc_tl_sharp_bcast_init(ucc_tl_sharp_task_t *task);

#endif
