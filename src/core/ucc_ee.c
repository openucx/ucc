/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team.h"
#include "ucc_ee.h"
#include "ucc_lib.h"
#include "components/cl/ucc_cl.h"
#include "components/tl/ucc_tl.h"

const char *ucc_ee_ev_names[] = {
    [UCC_EVENT_COLLECTIVE_POST]     = "COLL_POST",
    [UCC_EVENT_COLLECTIVE_COMPLETE] = "COLL_COMPLETE",
    [UCC_EVENT_COMPUTE_COMPLETE]    = "COMPUTE_COMPLETE",
    [UCC_EVENT_OVERFLOW]            = "OVERFLOW",
};

ucc_status_t ucc_ee_create(ucc_team_h team, const ucc_ee_params_t *params,
                           ucc_ee_h *ee_p)
{
    ucc_ee_t *ee;

    ee = ucc_malloc(sizeof(ucc_ee_t), "ucc execution engine");
    if (!ee) {
        ucc_error("failed to allocate %zd bytes for ucc execution engine",
                  sizeof(ucc_ee_t));
        return UCC_ERR_NO_MEMORY;
    }

    ee->team = team;
    ee->ee_type = params->ee_type;
    ee->ee_context_size = params->ee_context_size;
    ee->ee_context = params->ee_context;
    ucc_spinlock_init(&ee->lock, 0);
    ucc_queue_head_init(&ee->event_in_queue);
    ucc_queue_head_init(&ee->event_out_queue);
    *ee_p = ee;

    ucc_info("ee is created: %p ee_context: %p",
              ee, params->ee_context);

    return UCC_OK;
}

ucc_status_t ucc_ee_destroy(ucc_ee_h ee)
{
    ucc_info("ee is destroyed: %p", ee);
    ucc_spinlock_destroy(&ee->lock);
    ucc_free(ee);

    return UCC_OK;
}

ucc_status_t ucc_ee_get_event_internal(ucc_ee_h ee, ucc_ev_t **ev, ucc_queue_head_t *queue)
{
    ucc_event_desc_t *event_desc;
    ucc_queue_elem_t *elem;

    ucc_spin_lock(&ee->lock);
    if (ucc_queue_is_empty(queue)) {
        ucc_spin_unlock(&ee->lock);
        return UCC_ERR_NOT_FOUND;
    }

    elem = ucc_queue_pull(queue);
    ucc_spin_unlock(&ee->lock);

    ucc_assert(elem);

    event_desc = ucc_container_of(elem, ucc_event_desc_t, queue);
    *ev = &event_desc->ev;

    ucc_info("EE Event Get. ee:%p, queue:%p ev_type:%s ",
                ee, queue, ucc_ee_ev_names[event_desc->ev.ev_type]);
    return UCC_OK;
}

ucc_status_t ucc_ee_get_event(ucc_ee_h ee, ucc_ev_t **ev)
{
    return ucc_ee_get_event_internal(ee, ev, &ee->event_out_queue);
}

ucc_status_t ucc_ee_ack_event(ucc_ee_h ee, //NOLINT
                              ucc_ev_t *ev)
{
    ucc_event_desc_t *event_desc;

    event_desc = ucc_container_of(ev, ucc_event_desc_t, ev);
    /* TODO destroy event context */
    ucc_free(event_desc);
    return UCC_OK;
}

ucc_status_t ucc_ee_set_event_internal(ucc_ee_h ee, ucc_ev_t *ev, ucc_queue_head_t *queue)
{
    ucc_event_desc_t *event_desc;

    event_desc = ucc_malloc(sizeof(ucc_event_desc_t), "event descriptor");
    if (ucc_unlikely(!event_desc)) {
        ucc_error("failed to allocate ucc event descriptor");
        return UCC_ERR_NO_MEMORY;
    }

    event_desc->ev = *ev;

    ucc_spin_lock(&ee->lock);
    ucc_queue_push(queue, &event_desc->queue);
    ucc_spin_unlock(&ee->lock);
    ucc_info("EE Event Set. ee:%p, queue:%p ev_type:%s ",
                ee, queue, ucc_ee_ev_names[ev->ev_type]);

    return UCC_OK;
}

ucc_status_t ucc_ee_set_event(ucc_ee_h ee, ucc_ev_t *ev)
{
    return ucc_ee_set_event_internal(ee, ev, &ee->event_in_queue);
}

ucc_status_t ucc_ee_wait(ucc_ee_h ee, ucc_ev_t *ev)
{
    while(UCC_OK != ucc_ee_get_event(ee, &ev)) {
        ucc_progress_queue(ee->team->contexts[0]->pq);
    }

    return UCC_OK;

}
