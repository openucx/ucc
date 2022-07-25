/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_topo.h"
#include "core/ucc_context.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_math.h"
#include <string.h>
#include <limits.h>

static int ucc_compare_proc_info(const void *a, const void *b)
{
    const ucc_proc_info_t *d1 = (const ucc_proc_info_t *)a;
    const ucc_proc_info_t *d2 = (const ucc_proc_info_t *)b;

    if (d1->host_hash != d2->host_hash) {
        return d1->host_hash > d2->host_hash ? 1 : -1;
    } else if (d1->socket_id != d2->socket_id) {
        return d1->socket_id - d2->socket_id;
    } else if (d1->numa_id != d2->numa_id) {
        return d1->numa_id - d2->numa_id;
    } else {
        return d1->pid - d2->pid;
    }
}

static ucc_status_t ucc_context_topo_compute_layout(ucc_context_topo_t *topo,
                                                    ucc_rank_t          size)
{
    ucc_rank_t       current_ppn = 1;
    ucc_rank_t       min_ppn     = UCC_RANK_MAX;
    ucc_rank_t       max_ppn     = 0;
    ucc_rank_t       nnodes      = 1;
    int              max_sockid  = 0;
    int              max_numaid  = 0;
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
    qsort(sorted, size, sizeof(ucc_proc_info_t), ucc_compare_proc_info);
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
        if (topo->procs[j].numa_id > max_numaid) {
            max_numaid = topo->procs[j].numa_id;
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
    topo->max_n_numas   = max_numaid + 1;
    return UCC_OK;
}

ucc_status_t ucc_context_topo_init(ucc_addr_storage_t * storage,
                                   ucc_context_topo_t **_topo)
{
    ucc_context_addr_header_t *h;
    ucc_context_topo_t        *topo;
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
    topo->numa_bound = 1;
    topo->n_procs    = storage->size;
    topo->procs      = (ucc_proc_info_t *)ucc_malloc(
        storage->size * sizeof(ucc_proc_info_t), "topo_procs");
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
        if (h->ctx_id.pi.socket_id == UCC_SOCKET_ID_INVALID) {
            topo->sock_bound = 0;
        }
        if (h->ctx_id.pi.numa_id == UCC_NUMA_ID_INVALID) {
            topo->numa_bound = 0;
        }
    }
    status = ucc_context_topo_compute_layout(topo, storage->size);
    if (UCC_OK != status) {
        ucc_free(topo->procs);
        ucc_free(topo);
        return status;
    }

    *_topo = topo;
    return UCC_OK;
}

void ucc_context_topo_cleanup(ucc_context_topo_t *topo)
{
    if (topo) {
        ucc_free(topo->procs);
        ucc_free(topo);
    }
}

ucc_status_t ucc_topo_init(ucc_subset_t set, ucc_context_topo_t *ctx_topo,
                           ucc_topo_t **_topo)
{
    ucc_topo_t *topo = ucc_malloc(sizeof(*topo), "topo");
    int         i;
    if (!topo) {
        return UCC_ERR_NO_MEMORY;
    }
    topo->topo = ctx_topo;
    for (i = 0; i < UCC_SBGP_LAST; i++) {
        topo->sbgps[i].status = UCC_SBGP_NOT_INIT;
    }
    topo->n_sockets           = -1;
    topo->node_leader_rank    = -1;
    topo->node_leader_rank_id = 0;
    topo->set                 = set;
    topo->min_ppn             = UCC_RANK_MAX;
    topo->max_ppn             = 0;
    topo->all_sockets         = NULL;
    topo->all_numas           = NULL;

    *_topo = topo;
    return UCC_OK;
}

void ucc_topo_cleanup(ucc_topo_t *topo)
{
    int i;
    if (topo) {
        for (i = 0; i < UCC_SBGP_LAST; i++) {
            if (topo->sbgps[i].status == UCC_SBGP_ENABLED) {
                ucc_sbgp_cleanup(&topo->sbgps[i]);
            }
        }
        if (topo->all_sockets) {
            for (i = 0; i < topo->n_sockets; i++) {
                if (topo->all_sockets[i].status == UCC_SBGP_ENABLED) {
                    ucc_sbgp_cleanup(&topo->all_sockets[i]);
                }
            }
            ucc_free(topo->all_sockets);
        }
        ucc_free(topo);
    }
}

ucc_sbgp_t *ucc_topo_get_sbgp(ucc_topo_t *topo, ucc_sbgp_type_t type)
{
    if (topo->sbgps[type].status == UCC_SBGP_NOT_INIT) {
        if (UCC_OK != ucc_sbgp_create(topo, type)) {
            ucc_error("failed to create sbgp %s", ucc_sbgp_str(type));
            /* sbgps[type]->status is set accordingly */
        }
    }
    return &topo->sbgps[type];
}

int ucc_topo_is_single_node(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp;

    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    if (UCC_SBGP_ENABLED == sbgp->status &&
        sbgp->group_size == ucc_subset_size(&topo->set)) {
        return 1;
    }
    return 0;
}

ucc_status_t ucc_topo_get_all_sockets(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                      int *n_sbgps)
{
    ucc_status_t status = UCC_OK;

    if (!topo->all_sockets) {
        status = ucc_sbgp_create_all_sockets(topo, &topo->all_sockets, n_sbgps);
    }

    *sbgps   = topo->all_sockets;
    *n_sbgps = topo->n_sockets;

    return status;
}

ucc_status_t ucc_topo_get_all_numas(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                    int *n_sbgps)
{
    ucc_status_t status = UCC_OK;

    if (!topo->all_numas) {
        status = ucc_sbgp_create_all_numas(topo, &topo->all_numas, n_sbgps);
    }

    *sbgps   = topo->all_numas;
    *n_sbgps = topo->n_numas;

    return status;
}
