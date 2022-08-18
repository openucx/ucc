/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_EE_H_
#define UCC_EE_H_

#include "ucc/api/ucc.h"
#include "utils/ucc_datastruct.h"
#include "utils/ucc_queue.h"
#include "utils/ucc_spinlock.h"

extern const char *ucc_ee_ev_names[];

typedef struct ucc_ee {
    ucc_team_h       team;
    ucc_ee_type_t    ee_type;
    ucc_spinlock_t   lock;
    ucc_queue_head_t event_in_queue;
    ucc_queue_head_t event_out_queue;
    size_t           ee_context_size;
    char             *ee_context;
} ucc_ee_t;

typedef struct ucc_event_desc {
    ucc_queue_elem_t queue;
    ucc_ev_t ev;
} ucc_event_desc_t;

ucc_status_t ucc_ee_get_event_internal(ucc_ee_h ee, ucc_ev_t **ev, ucc_queue_head_t *queue);

ucc_status_t ucc_ee_set_event_internal(ucc_ee_h ee, ucc_ev_t *ev, ucc_queue_head_t *queue);
#endif
