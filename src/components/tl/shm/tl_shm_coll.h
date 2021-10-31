/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHM_COLL_H_
#define UCC_TL_SHM_COLL_H_

#include "tl_shm.h"

#define AR_SMALL_MAX 64
#define MAX_REQS 32

typedef struct ucc_tl_shm_task { // xccl_shmseg_collreq
	ucc_coll_task_t   super;
    union {
        struct {
            int   use_small;
            int   step;
            void* shmseg_base;
            void* shmseg_dest;
        };
    };
} ucc_tl_shm_task_t;

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_shm_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_shm_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_shm_lib_t))
#define UCC_TL_SHM_TEAM_LIB(_team)                                             \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_shm_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

#endif
