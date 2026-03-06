/*
 * Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_topo.h"
#include "core/ucc_context.h"
#include "ucc/api/ucc_status.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_proc_info.h"
#include <string.h>
#include <limits.h>
#include <stdlib.h>

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

static ucc_status_t ucc_context_topo_compute_layout(
    ucc_context_topo_t *topo, ucc_rank_t size)
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

ucc_status_t ucc_context_topo_init(
    ucc_addr_storage_t *storage, ucc_context_topo_t **_topo)
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

    topo->hosts = (ucc_host_info_t *)ucc_malloc(
        storage->size * sizeof(ucc_host_info_t), "topo_hosts");
    if (!topo->hosts) {
        ucc_error(
            "failed to allocate %zd bytes for topo_hosts",
            storage->size * sizeof(ucc_host_info_t));
        ucc_free(topo->procs);
        ucc_free(topo);
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < storage->size; i++) {
        h = (ucc_context_addr_header_t *)PTR_OFFSET(
            storage->storage, storage->addr_len * i);
        topo->procs[i] = h->ctx_id.pi;
        topo->hosts[i] = h->host_info;
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
        ucc_free(topo->hosts);
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
        ucc_free(topo->hosts);
        ucc_free(topo);
    }
}

ucc_status_t ucc_topo_init(
    ucc_subset_t set, ucc_context_topo_t *ctx_topo, ucc_topo_t **_topo)
{
    ucc_topo_t *topo = ucc_malloc(sizeof(*topo), "topo");
    ucc_rank_t  size = ucc_subset_size(&set);
    ucc_rank_t  ctx_rank;
    int         i, all_valid = 1;

    if (!topo) {
        return UCC_ERR_NO_MEMORY;
    }

    if (!ctx_topo) {
        ucc_free(topo);
        return UCC_ERR_INVALID_PARAM;
    }

    topo->topo = ctx_topo;
    for (i = 0; i < UCC_SBGP_LAST; i++) {
        topo->sbgps[i].status = UCC_SBGP_NOT_INIT;
    }

    topo->n_sockets             = -1;
    topo->node_leader_rank      = UCC_RANK_INVALID;
    topo->node_leader_rank_id   = 0;
    topo->set                   = set;
    topo->min_ppn               = UCC_RANK_MAX;
    topo->max_ppn               = 0;
    topo->min_socket_size       = UCC_RANK_MAX;
    topo->max_socket_size       = 0;
    topo->min_numa_size         = UCC_RANK_MAX;
    topo->max_numa_size         = 0;
    topo->all_sockets           = NULL;
    topo->all_numas             = NULL;
    topo->all_nodes             = NULL;
    topo->node_leaders          = NULL;
    topo->device_map.device_ids = NULL;
    topo->device_map.n_ranks    = 0;
    topo->n_nodes               = 0;

    for (i = 0; i < size; i++) {

        ctx_rank = ucc_ep_map_eval(set.map, i);
        if (!ctx_topo->hosts[ctx_rank].visible_gpus) {
            all_valid = 0;
            break;
        }
    }

    if (all_valid) {
        topo->device_map.device_ids =
            ucc_malloc(size * sizeof(ucc_device_id_t), "topo_device_map");
        if (!topo->device_map.device_ids) {
            ucc_free(topo);
            return UCC_ERR_NO_MEMORY;
        }
        topo->device_map.n_ranks = size;
        for (i = 0; i < size; i++) {
            ctx_rank                       = ucc_ep_map_eval(set.map, i);
            topo->device_map.device_ids[i] = ucc_ilog2(
                ctx_topo->hosts[ctx_rank].visible_gpus);
        }
    }

    *_topo                    = topo;
    return UCC_OK;
}

void ucc_topo_cleanup(ucc_topo_t *topo)
{
    int i;

    if (topo) {
        for (i = 0; i < UCC_SBGP_LAST; i++) {
            if (topo->sbgps[i].status == UCC_SBGP_ENABLED ||
                topo->sbgps[i].status == UCC_SBGP_DISABLED) {
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

        if (topo->all_numas) {
            for (i = 0; i < topo->n_numas; i++) {
                if (topo->all_numas[i].status == UCC_SBGP_ENABLED) {
                    ucc_sbgp_cleanup(&topo->all_numas[i]);
                }
            }
            ucc_free(topo->all_numas);
        }

        if (topo->all_nodes) {
            for (i = 0; i < topo->n_nodes; i++) {
                if (topo->all_nodes[i].status == UCC_SBGP_ENABLED) {
                    ucc_sbgp_cleanup(&topo->all_nodes[i]);
                }
            }
            ucc_free(topo->all_nodes);
        }

        if (topo->node_leaders) {
            ucc_free(topo->node_leaders);
        }
        if (topo->device_map.device_ids) {
            ucc_free(topo->device_map.device_ids);
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

/* Returns invalid param if there's only one node in the team (leader sbgp does
   not exist). Otherwise, creates a node sbgp for every node. One or more sbgps
   may be UCC_SBGP_NOT_EXISTS if they have only one rank */
ucc_status_t ucc_sbgp_create_all_nodes(ucc_topo_t *topo, ucc_sbgp_t **_sbgps,
                                      int *n_sbgps)
{
    ucc_rank_t   myrank      = topo->set.myrank;
    ucc_rank_t   leader_rank = UCC_RANK_INVALID;
    ucc_sbgp_t  *sbgps, *leader_sbgp;
    ucc_rank_t   i;
    ucc_status_t status;
    ucc_rank_t   nnodes;

    leader_sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);

    if (leader_sbgp->status == UCC_SBGP_NOT_INIT ||
        leader_sbgp->status == UCC_SBGP_NOT_EXISTS) {
        ucc_debug("could not create all_nodes subgroups, leader subgroup "
                  "does not exist for topo");
        return UCC_ERR_INVALID_PARAM;
    }

    nnodes = leader_sbgp->group_size;

    sbgps = ucc_calloc(nnodes, sizeof(ucc_sbgp_t), "sbgps");
    if (!sbgps) {
        ucc_error("failed to allocate %zd bytes for sbgps array",
                  nnodes * sizeof(ucc_sbgp_t));
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < nnodes; i++) {
        ucc_rank_t ldr_team_rank = ucc_ep_map_eval(leader_sbgp->map, i);
        sbgps[i].type = UCC_SBGP_NODE;
        topo->set.myrank = ldr_team_rank;
        status = ucc_sbgp_create_node(topo, &sbgps[i]);
        if (status == UCC_ERR_NOT_FOUND) {
            /* ucc_sbgp_create_node returned because 0 == node_size */
            sbgps[i].status = UCC_SBGP_NOT_EXISTS;
            continue;
        } else if (status != UCC_OK) {
            ucc_error("failed to create all_node subgroup %d", i);
            goto error;
        }
        if (ucc_rank_on_local_node(myrank, topo)) {
            leader_rank = topo->node_leader_rank;
        }

        if (sbgps[i].status == UCC_SBGP_NOT_EXISTS) {
            /* Single-member subgroup: free rank_map and skip map creation
               to avoid a dangling pointer in map.array.map */
            if (sbgps[i].rank_map) {
                ucc_free(sbgps[i].rank_map);
                sbgps[i].rank_map = NULL;
            }
        } else if (sbgps[i].rank_map && sbgps[i].type != UCC_SBGP_FULL) {
            sbgps[i].map = ucc_ep_map_from_array(
                                &sbgps[i].rank_map, sbgps[i].group_size,
                                ucc_subset_size(&topo->set), 1);
        }
    }

    /* Reset myrank and node_leader_rank because the calls to
       ucc_sbgp_create_node above will have changed them */
    topo->set.myrank = myrank;
    ucc_assert(leader_rank != UCC_RANK_INVALID);
    topo->node_leader_rank = leader_rank;

    *_sbgps  = sbgps;
    *n_sbgps = nnodes;

    return UCC_OK;
error:
    topo->set.myrank = myrank;
    for (i = 0; i < nnodes; i++) {
        if (sbgps[i].rank_map) {
            ucc_free(sbgps[i].rank_map);
        }
    }
    ucc_free(sbgps);
    return status;
}

ucc_status_t ucc_topo_get_all_nodes(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                    int *n_sbgps)
{
    ucc_status_t status = UCC_OK;

    if (!topo->all_nodes) {
        status = ucc_sbgp_create_all_nodes(topo, &topo->all_nodes, &topo->n_nodes);
    }

    *sbgps = topo->all_nodes;
    *n_sbgps = topo->n_nodes;

    return status;
}

static int ucc_topo_rank_device_info(const ucc_topo_t *topo, ucc_rank_t rank,
                                     const ucc_host_info_t **host,
                                     ucc_device_id_t *device_id)
{
    ucc_rank_t ctx_rank = ucc_ep_map_eval(topo->set.map, rank);

    *host = &topo->topo->hosts[ctx_rank];
    if (!(*host)->visible_gpus) {
        return 0;
    }

    *device_id = ucc_ilog2((*host)->visible_gpus);

    return 1;
}

int ucc_topo_has_device_info(const ucc_topo_t *topo)
{
    ucc_host_info_t *host;
    ucc_rank_t       i;

    if (!topo->topo) {
        return 0;
    }

    for (i = 0; i < ucc_subset_size(&topo->set); i++) {
        host = &topo->topo->hosts[ucc_ep_map_eval(topo->set.map, i)];
        if (host->visible_gpus == 0) {
            return 0;
        }
    }
    return 1;
}

static int ucc_topo_ranks_on_node_nvlink(const ucc_topo_t *topo, ucc_rank_t rank1,
                                         ucc_rank_t rank2)
{
    ucc_rank_t             ctx_rank1 =  ucc_ep_map_eval(topo->set.map, rank1);
    ucc_rank_t             ctx_rank2 =  ucc_ep_map_eval(topo->set.map, rank2);
    ucc_device_id_t        dev1;
    ucc_device_id_t        dev2;
    const ucc_host_info_t *host1;
    const ucc_host_info_t *host2;
    const ucc_gpu_info_t  *gpu1;
    const ucc_gpu_info_t  *gpu2;

    if (!ucc_topo_rank_device_info(topo, rank1, &host1, &dev1) ||
        !ucc_topo_rank_device_info(topo, rank2, &host2, &dev2)) {
        return 0;
    }

    if (topo->topo->procs[ctx_rank1].host_hash !=
        topo->topo->procs[ctx_rank2].host_hash) {
        return 0;
    }

    if (dev1 == dev2) {
        return 1;
    }

    gpu1 = &host1->gpus[dev1];
    gpu2 = &host2->gpus[dev2];

    if (gpu1->fabric_clique_id != 0 &&
        gpu1->fabric_clique_id == gpu2->fabric_clique_id) {
        return 1;
    }

    if (gpu1->nvswitch_connected && gpu2->nvswitch_connected) {
        return 1;
    }

    return host1->nvlink_matrix[dev1][dev2] > 0;
}

int ucc_topo_is_nvlink_fully_connected(
    const ucc_topo_t *topo, const ucc_sbgp_t *sbgp)
{
    ucc_rank_t size;
    ucc_rank_t i, j;
    ucc_rank_t rank_i;

    if (!sbgp) {
        return 0;
    }

    size = sbgp->group_size;
    if (size <= 1) {
        return 1;
    }

    for (i = 0; i < size; i++) {
        rank_i = ucc_ep_map_eval(sbgp->map, i);
        for (j = i + 1; j < size; j++) {
            ucc_rank_t rank_j = ucc_ep_map_eval(sbgp->map, j);

            if (!ucc_topo_ranks_on_node_nvlink(topo, rank_i, rank_j)) {
                return 0;
            }
        }
    }

    return 1;
}

ucc_status_t ucc_topo_get_node_leaders(ucc_topo_t *topo, ucc_rank_t **node_leaders_out)
{
    ucc_subset_t *set    = &topo->set;
    ucc_rank_t    size   = ucc_subset_size(set);
    ucc_rank_t    nnodes = topo->topo->nnodes;
    ucc_rank_t    i;
    ucc_rank_t   *ranks_seen_per_node;
    ucc_rank_t   *per_node_leaders;
    ucc_rank_t   *node_leaders;

    if (topo->node_leaders) {
        *node_leaders_out = topo->node_leaders;
        return UCC_OK;
    }

    ucc_assert(nnodes > 1);

    /* Allocate arrays */
    node_leaders = ucc_malloc(sizeof(ucc_rank_t) * size, "node_leaders");
    if (!node_leaders) {
        ucc_error("failed to allocate %zd bytes for node_leaders array",
                  size * sizeof(ucc_rank_t));
        return UCC_ERR_NO_MEMORY;
    }

    ranks_seen_per_node = ucc_calloc(nnodes, sizeof(ucc_rank_t), "ranks_seen_per_node");
    if (!ranks_seen_per_node) {
        ucc_error("failed to allocate %zd bytes for ranks_seen_per_node array",
                  nnodes * sizeof(ucc_rank_t));
        ucc_free(node_leaders);
        return UCC_ERR_NO_MEMORY;
    }

    per_node_leaders = ucc_calloc(nnodes, sizeof(ucc_rank_t), "per_node_leaders");
    if (!per_node_leaders) {
        ucc_error("failed to allocate %zd bytes for per_node_leaders array",
                  nnodes * sizeof(ucc_rank_t));
        ucc_free(node_leaders);
        ucc_free(ranks_seen_per_node);
        return UCC_ERR_NO_MEMORY;
    }

    /* First pass: identify node leaders */
    for (i = 0; i < size; i++) {
        ucc_rank_t ctx_rank = ucc_ep_map_eval(set->map, i);
        ucc_host_id_t current_host = topo->topo->procs[ctx_rank].host_id;

        /* Count ranks on this node */
        ranks_seen_per_node[current_host]++;

        /* If this is the rank we want as leader for this node, mark it */
        if (ranks_seen_per_node[current_host] == topo->node_leader_rank_id + 1) {
            per_node_leaders[current_host] = i;
        }
    }

    /* Second pass: propagate node leaders to all ranks */
    for (i = 0; i < size; i++) {
        ucc_rank_t ctx_rank = ucc_ep_map_eval(set->map, i);
        ucc_host_id_t current_host = topo->topo->procs[ctx_rank].host_id;

        node_leaders[i] = per_node_leaders[current_host];
    }

    topo->node_leaders = node_leaders;
    //NOLINTNEXTLINE
    *node_leaders_out = node_leaders;
    ucc_free(ranks_seen_per_node);
    ucc_free(per_node_leaders);
    return UCC_OK;
}
