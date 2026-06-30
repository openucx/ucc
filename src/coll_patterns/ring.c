/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ring.h"
#include "components/topo/ucc_topo.h"
#include "ucc/api/ucc.h"

static ucc_status_t ucc_ring_pattern_init_topo_host(
    ucc_topo_t *topo, ucc_ring_pattern_t *p)
{
    ucc_rank_t    size = ucc_subset_size(&topo->set);
    ucc_sbgp_t   *sbgp;
    ucc_ep_map_t *maps;
    ucc_rank_t   *rings_buf;
    ucc_rank_t    i;


    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_FULL_HOST_ORDERED);
    if (sbgp->status != UCC_SBGP_ENABLED) {
        return UCC_ERR_NOT_FOUND;
    }

    maps = ucc_malloc(
        sizeof(*maps) + size * sizeof(ucc_rank_t), "nvlink_ring_maps");
    if (!maps) {
        return UCC_ERR_NO_MEMORY;
    }

    rings_buf               = (ucc_rank_t *)(maps + 1);
    maps[0].type            = UCC_EP_MAP_ARRAY;
    maps[0].ep_num          = size;
    maps[0].array.map       = rings_buf;
    maps[0].array.elem_size = sizeof(ucc_rank_t);
    /* Use ucc_ep_map_eval to get ranks from sbgp->map since rank_map
       may be NULL after ucc_ep_map_from_array took ownership */
    for (i = 0; i < size; i++) {
        rings_buf[i] = ucc_ep_map_eval(sbgp->map, i);
    }
    ucc_ring_pattern_init_map(maps, 1, p);
    return UCC_OK;
}

static ucc_status_t ucc_topo_build_sbgp_ring(
    ucc_topo_t *topo, const ucc_sbgp_t *sbgp, ucc_rank_t **ring_buf,
    unsigned *num_rings)
{
    ucc_rank_t       gsize = sbgp->group_size;
    ucc_status_t     status = UCC_OK;
    ucc_device_id_t *sbgp_nics;
    ucc_device_id_t *sbgp_devs;
    ucc_rank_t      *sbgp_ranks;
    ucc_rank_t       i, j, k, r, path_len, rank;
    unsigned         num_sbgp_nics;
    int              dist, best_dist;
    ucc_gpu_info_t  *gpu_info;
    ucc_nic_info_t  *nic_info;
    ucc_host_info_t *host_info;
    ucc_device_id_t  best_dev;

    if (sbgp->status != UCC_SBGP_ENABLED) {
        return UCC_ERR_INVALID_PARAM;
    }

    if (sbgp->type != UCC_SBGP_NODE) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!ucc_topo_is_nvlink_fully_connected(topo, sbgp)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    sbgp_nics = ucc_malloc(
        sbgp->group_size * sizeof(*sbgp_nics), "nvlink_sbgp_nics");
    if (!sbgp_nics) {
        status = UCC_ERR_NO_MEMORY;
        goto exit;
    }

    sbgp_devs = ucc_malloc(
        sbgp->group_size * sizeof(*sbgp_devs), "nvlink_sbgp_devs");
    if (!sbgp_devs) {
        status = UCC_ERR_NO_MEMORY;
        goto free_sbgp_nics;
    }

    sbgp_ranks = ucc_malloc(
        sbgp->group_size * sizeof(*sbgp_ranks), "nvlink_sbgp_ranks");
    if (!sbgp_ranks) {
        status = UCC_ERR_NO_MEMORY;
        goto free_sbgp_devs;
    }

    num_sbgp_nics = 0;
    for (j = 0; j < gsize; j++) {
        rank      = ucc_ep_map_eval(sbgp->map, j);
        host_info = &topo->topo->hosts[rank];
        gpu_info  = &host_info->gpus[ucc_ilog2(host_info->visible_gpus)];

        best_dist = INT_MAX;
        best_dev  = UCC_DEVICE_ID_INVALID;

        ucc_for_each_bit (i, host_info->visible_nics) {
            nic_info = &host_info->nics[i];
            dist     = ucc_pci_distance(&gpu_info->pci, &nic_info->pci);
            if (dist < best_dist) {
                /* check if the nic is already in the sbgp_nics */
                for (k = 0; k < j; k++) {
                    if (sbgp_nics[k] == UCC_DEVICE_ID_INVALID) {
                        continue;
                    }
                    if (ucc_compare_pci_info(
                            &topo->topo->hosts[sbgp_ranks[k]]
                                 .nics[sbgp_nics[k]]
                                 .pci,
                            &nic_info->pci)) {
                        break;
                    }
                }

                if (k == j) {
                    /* no duplicate nic found, update the best dist and dev */
                    best_dist = dist;
                    best_dev  = i;
                }
            }
        }

        if (best_dev != UCC_DEVICE_ID_INVALID) {
            num_sbgp_nics++;
        }

        r         = j;
        best_dist = INT_MAX;
        for (i = 0; i < j; i++) {
            dist = ucc_pci_distance(
                &gpu_info->pci,
                &topo->topo->hosts[sbgp_ranks[i]].gpus[sbgp_devs[i]].pci);
            if (dist < best_dist) {
                best_dist = dist;
                r         = i + 1;
            }
        }

        for (i = j; i > r; i--) {
            sbgp_devs[i]  = sbgp_devs[i - 1];
            sbgp_nics[i]  = sbgp_nics[i - 1];
            sbgp_ranks[i] = sbgp_ranks[i - 1];
        }

        sbgp_devs[r]  = ucc_ilog2(host_info->visible_gpus);
        sbgp_nics[r]  = best_dev;
        sbgp_ranks[r] = rank;
    }

    ucc_debug("found %d nvlink rings for sbgp %p", num_sbgp_nics, sbgp);
    *num_rings = num_sbgp_nics;
    if (*num_rings == 0) {
        status = UCC_ERR_NOT_FOUND;
        goto free_sbgp_ranks;
    }

    *ring_buf  = ucc_malloc(
        *num_rings * gsize * sizeof(ucc_rank_t), "nvlink_sbgp_ring");
    if (!(*ring_buf)) {
        status = UCC_ERR_NO_MEMORY;
        goto free_sbgp_ranks;
    }

    for (j = 0, r = 0; j < gsize; j++) {
        if (sbgp_nics[j] == UCC_DEVICE_ID_INVALID) {
            continue;
        }

        /* find next valid NIC in ring order after j */
        k = (j + 1) % gsize;
        while (k != j && sbgp_nics[k] == UCC_DEVICE_ID_INVALID) {
            k = (k + 1) % gsize;
        }

        path_len = 1;
        (*ring_buf)[r*gsize] = sbgp_ranks[j];
        i = (j - 1 + gsize) % gsize;
        while (i != j) {
            if (i != k) {
                (*ring_buf)[r*gsize + path_len++] = sbgp_ranks[i];
            }
            i = (i - 1 + gsize) % gsize;
        }
        (*ring_buf)[r*gsize + path_len++] = sbgp_ranks[k];
        r++;
    }

free_sbgp_ranks:
    ucc_free(sbgp_ranks);
free_sbgp_devs:
    ucc_free(sbgp_devs);
free_sbgp_nics:
    ucc_free(sbgp_nics);
exit:
    return status;
}

static ucc_status_t ucc_ring_pattern_init_topo_cuda(
    ucc_topo_t *topo, unsigned n_rings, ucc_ring_pattern_t *p)
{
    ucc_rank_t    size    = ucc_subset_size(&topo->set);
    ucc_sbgp_t   *sbgps   = NULL;
    int           n_sbgps = 0;
    ucc_rank_t    i;
    ucc_ep_map_t *maps;
    ucc_rank_t   *rings_buf;
    unsigned      rings_avail;
    ucc_rank_t  **sbgp_rings;
    ucc_status_t  status;
    ucc_rank_t   *ring;
    ucc_rank_t    sbgp_id, sbgp_size;

    if (!ucc_topo_has_device_info(topo)) {
        goto fallback_host_ring;
    }

    status = ucc_topo_get_all_nodes(topo, &sbgps, &n_sbgps);
    if (status != UCC_OK) {
        if (status == UCC_ERR_INVALID_PARAM || status == UCC_ERR_NOT_FOUND) {
            goto fallback_host_ring;
        }
        return status;
    }

    sbgp_rings = ucc_calloc(
        n_sbgps, sizeof(*sbgp_rings), "nvlink_sbgp_rings_buf");
    if (!sbgp_rings) {
        return UCC_ERR_NO_MEMORY;
    }

    ucc_debug("building nvlink rings for %d sbgps", n_sbgps);
    for (i = 0; i < n_sbgps; i++) {
        status = ucc_topo_build_sbgp_ring(
            topo, &sbgps[i], &sbgp_rings[i], &rings_avail);
        if (status != UCC_OK) {
            ucc_debug(
                "failed to build sbgp ring for sbgp %d (status=%d)", i, status);
            goto free_node_ring_info;
        }

        if (rings_avail == 0) {
            ucc_debug("no nvlink rings available for sbgp %d", i);
            goto free_node_ring_info;
        }

        n_rings = ucc_min(n_rings, rings_avail);
    }

    maps = ucc_malloc(
        n_rings * sizeof(*maps) + n_rings * size * sizeof(ucc_rank_t),
        "nvlink_ring_maps");
    if (!maps) {
        goto free_node_ring_info;
    }

    rings_buf = (ucc_rank_t *)(maps + n_rings);
    for (i = 0; i < n_rings; i++) {
        maps[i].type            = UCC_EP_MAP_ARRAY;
        maps[i].ep_num          = size;
        maps[i].array.map       = rings_buf + (size_t)i * size;
        maps[i].array.elem_size = sizeof(ucc_rank_t);
    }

    for (i = 0; i < n_rings; i++) {
        ring = (ucc_rank_t *)maps[i].array.map;
        for (sbgp_id = 0; sbgp_id < n_sbgps; sbgp_id++) {
            sbgp_size = sbgps[sbgp_id].group_size;
            memcpy(
                ring,
                PTR_OFFSET(
                    sbgp_rings[sbgp_id], i * sbgp_size * sizeof(ucc_rank_t)),
                sbgp_size * sizeof(ucc_rank_t));
            ring += sbgp_size;
        }
    }

    for (sbgp_id = 0; sbgp_id < n_sbgps; sbgp_id++) {
        ucc_free(sbgp_rings[sbgp_id]);
    }
    ucc_free(sbgp_rings);

    ucc_ring_pattern_init_map(maps, n_rings, p);
    return UCC_OK;

free_node_ring_info:
    for (sbgp_id = 0; sbgp_id < n_sbgps; sbgp_id++) {
        ucc_free(sbgp_rings[sbgp_id]);
    }
    ucc_free(sbgp_rings);
fallback_host_ring:
    ucc_debug("fallback to host ring for cuda ring");
    return ucc_ring_pattern_init_topo_host(topo, p);
}

ucc_status_t ucc_ring_pattern_init_topo(
    ucc_topo_t *topo, ucc_memory_type_t mt, unsigned num_rings,
    ucc_ring_pattern_t *p)
{
    if (!topo || !topo->topo || num_rings == 0) {
        return UCC_ERR_INVALID_PARAM;
    }

    switch (mt) {
        case UCC_MEMORY_TYPE_HOST:
            return ucc_ring_pattern_init_topo_host(topo, p);
        case UCC_MEMORY_TYPE_CUDA:
            return ucc_ring_pattern_init_topo_cuda(topo, num_rings, p);
        default:
            return UCC_ERR_NOT_SUPPORTED;
    }
}
