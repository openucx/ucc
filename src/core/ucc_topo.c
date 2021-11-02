/*
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_topo.h"
#include "ucc_context.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_math.h"
#include <string.h>
#include <limits.h>

static int ucc_topo_compare_proc_info(const void *a, const void *b)
{
    const ucc_proc_info_t *d1 = (const ucc_proc_info_t *)a;
    const ucc_proc_info_t *d2 = (const ucc_proc_info_t *)b;

    if (d1->host_hash != d2->host_hash) {
        return d1->host_hash > d2->host_hash ? 1 : -1;
    } else if (d1->socket_id != d2->socket_id) {
        return d1->socket_id - d2->socket_id;
    } else {
        return d1->pid - d2->pid;
    }
}

static ucc_status_t ucc_topo_compute_layout(ucc_topo_t *topo, ucc_rank_t size)
{
    ucc_rank_t       current_ppn  = 1;
    ucc_rank_t       min_ppn      = UCC_RANK_MAX;
    ucc_rank_t       max_ppn      = 0;
    ucc_rank_t       nnodes       = 1;
    int              max_sockid   = 0;
    ucc_proc_info_t *sorted;
    ucc_host_id_t    current_hash, hash;
    int              i, j;

    sorted = (ucc_proc_info_t *)ucc_malloc(size * sizeof(ucc_proc_info_t),
                                           "proc_sorted");
    if (!sorted) {
        ucc_error("failed to allocate %zd bytes for proc sorted",
                  size * sizeof(ucc_proc_info_t));
        return UCC_ERR_NO_MEMORY;
    }
    memcpy(sorted, topo->procs, size * sizeof(ucc_proc_info_t));
    qsort(sorted, size, sizeof(ucc_proc_info_t), ucc_topo_compare_proc_info);
    current_hash = sorted[0].host_hash;

    for (i = 1; i < size; i++) {
        hash = sorted[i].host_hash;
        if (hash != current_hash) {
            for (j = 0; j < size; j++) {
                if (topo->procs[j].host_hash == current_hash) {
                    topo->procs[j].host_id = nnodes - 1;
                }
            }
            if (current_ppn > max_ppn)
                max_ppn = current_ppn;
            if (current_ppn < min_ppn)
                min_ppn = current_ppn;
            nnodes++;
            current_hash = hash;
            current_ppn  = 1;
        } else {
            current_ppn++;
        }
    }
    for (j = 0; j < size; j++) {
        if (topo->procs[j].socket_id > max_sockid) {
            max_sockid = topo->procs[j].socket_id;
        }
        if (topo->procs[j].host_hash == current_hash) {
            topo->procs[j].host_id = nnodes - 1;
        }
    }

    if (current_ppn > max_ppn) {
        max_ppn = current_ppn;
    }
    if (current_ppn < min_ppn) {
        min_ppn = current_ppn;
    }

    ucc_free(sorted);

    topo->nnodes        = nnodes;
    topo->min_ppn       = min_ppn;
    topo->max_ppn       = max_ppn;
    topo->max_n_sockets = max_sockid + 1;
    return UCC_OK;
}

ucc_status_t ucc_topo_init(ucc_addr_storage_t *storage, ucc_topo_t **_topo)
{
    ucc_context_addr_header_t *h;
    ucc_topo_t                *topo;
    int                        i;
    ucc_status_t               status;

    if (storage->size < 2) {
        /* We should always expect at least 2 ranks data in the storage */
        return UCC_ERR_NO_MESSAGE;
    }

    topo = ucc_malloc(sizeof(*topo), "topo");
    if (!topo) {
        ucc_error("failed to allocate %zd bytes for topo", sizeof(*topo));
        return UCC_ERR_NO_MEMORY;
    }

    topo->sock_bound = 1;
    topo->n_procs    = storage->size;
    topo->procs =
        (ucc_proc_info_t *)ucc_malloc(storage->size * sizeof(ucc_proc_info_t),
            "topo_procs");
    if (!topo->procs) {
        ucc_error("failed to allocate %zd bytes for topo_procs",
                  storage->size * sizeof(ucc_proc_info_t));
        ucc_free(topo);
        return UCC_ERR_NO_MEMORY;
    }
    for (i = 0; i < storage->size; i++) {
        h = (ucc_context_addr_header_t *)PTR_OFFSET(storage->storage,
                                                    storage->addr_len * i);
        topo->procs[i] = h->ctx_id.pi;
        if (h->ctx_id.pi.socket_id == -1) {
            topo->sock_bound = 0;
        }
    }
    status = ucc_topo_compute_layout(topo, storage->size);
    if (UCC_OK != status) {
        ucc_free(topo->procs);
        ucc_free(topo);
        return status;
    }

    *_topo = topo;
    return UCC_OK;
}

void ucc_topo_cleanup(ucc_topo_t *topo)
{
    if (topo) {
        ucc_free(topo->procs);
        ucc_free(topo);
    }
}

ucc_status_t ucc_subset_topo_init(ucc_subset_t set, ucc_topo_t *topo,
                                  ucc_subset_topo_t **_subset_topo)
{
    ucc_subset_topo_t *subset_topo = malloc(sizeof(*subset_topo));
    int              i;
    if (!subset_topo) {
        return UCC_ERR_NO_MEMORY;
    }
    subset_topo->topo = topo;
    for (i = 0; i < UCC_SBGP_LAST; i++) {
        subset_topo->sbgps[i].status = UCC_SBGP_NOT_INIT;
    }
    subset_topo->no_socket           = 0;
    subset_topo->node_leader_rank    = -1;
    subset_topo->node_leader_rank_id = 0;
    subset_topo->set                 = set;
    subset_topo->min_ppn             = UCC_RANK_MAX;
    subset_topo->max_ppn             = 0;
    *_subset_topo                    = subset_topo;
    return UCC_OK;
}

void ucc_subset_topo_cleanup(ucc_subset_topo_t *subset_topo)
{
    int i;
    if (subset_topo) {
        for (i = 0; i < UCC_SBGP_LAST; i++) {
            if (subset_topo->sbgps[i].status == UCC_SBGP_ENABLED) {
                ucc_sbgp_cleanup(&subset_topo->sbgps[i]);
            }
        }
        free(subset_topo);
    }
}

ucc_sbgp_t *ucc_subset_topo_get_sbgp(ucc_subset_topo_t *topo, ucc_sbgp_type_t type)
{
    if (topo->sbgps[type].status == UCC_SBGP_NOT_INIT) {
        if (UCC_OK != ucc_sbgp_create(topo, type)) {
            ucc_error("failed to create sbgp %s", ucc_sbgp_str(type));
            /* sbgps[type]->status is set accordingly */
        }
    }
    return &topo->sbgps[type];
}

int ucc_topo_is_single_node(ucc_subset_topo_t *topo)
{
    ucc_sbgp_t *sbgp;

    sbgp = ucc_subset_topo_get_sbgp(topo, UCC_SBGP_NODE);
    if (UCC_SBGP_ENABLED == sbgp->status &&
        sbgp->group_size == ucc_subset_size(&topo->set)) {
        return 1;
    }
    return 0;
}
