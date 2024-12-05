/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_team_topo.h"
#include "tl_cuda.h"

#define UCC_TL_CUDA_TEAM_TOPO_SAME_DEVICE ((ucc_rank_t)(UCC_RANK_MAX))

static ucc_status_t
ucc_tl_cuda_team_topo_add_ring(const ucc_tl_cuda_team_t *team,
                               ucc_tl_cuda_team_topo_t *topo,
                               ucc_tl_cuda_ring_t *ring,
                               int invert, int num_dups)
{
    ucc_rank_t size = UCC_TL_TEAM_SIZE(team);
    ucc_tl_cuda_ring_t *new_ring;
    ucc_status_t status;
    int i, j;

    ucc_assert(size > 1);
    for (i = 0; i < num_dups; i++) {
        topo->rings[topo->num_rings + i].ring  = NULL;
        topo->rings[topo->num_rings + i].iring = NULL;
    }

    for (i = 0; i < num_dups; i++) {
        new_ring = &topo->rings[topo->num_rings + i];
        new_ring->ring  = (ucc_rank_t*)ucc_malloc(2 * size * sizeof(ucc_rank_t),
                                                  "cuda_topo_ring");
        new_ring->iring = PTR_OFFSET(new_ring->ring, size * sizeof(ucc_rank_t));
        if (!new_ring->ring) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to allocate topo ring");
            status = UCC_ERR_NO_MEMORY;
            goto free_rings;
        }
        for (j = 0; j < size; j++) {
            if (invert) {
                new_ring->ring[j] = ring->ring[size - j - 1];
            } else {
                new_ring->ring[j] = ring->ring[j];
            }
        }
        for (j = 0; j < size; j++) {
            new_ring->iring[new_ring->ring[j]] = j;
        }
    }
    topo->num_rings += num_dups;

    return UCC_OK;
free_rings:
    for (i = 0; i < num_dups; i++) {
        ucc_free(topo->rings[topo->num_rings + i].ring);
    }
    return status;
}

static ucc_status_t
ucc_tl_cuda_team_topo_build_ring(const ucc_tl_cuda_team_t *team,
                                 const ucc_rank_t *graph,
                                 ucc_tl_cuda_ring_t *ring,
                                 ucc_rank_t pos,
                                 int width)
{
    ucc_rank_t size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t i, j;
    int in_ring;
    int links;
    ucc_status_t status;

    if (pos == size) {
        links = graph[ring->ring[pos - 1] * size + ring->ring[0]];
        if ((links == UCC_TL_CUDA_TEAM_TOPO_SAME_DEVICE) || (links >= width)) {
            return UCC_OK;
        } else {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    for (i = 0; i < size; i++) {
        links = graph[ring->ring[pos - 1] * size + i];
        if ((links < width) && (links != UCC_TL_CUDA_TEAM_TOPO_SAME_DEVICE)) {
            continue;
        }
        in_ring = 0;
        for (j = 0; j < pos; j++) {
            if (ring->ring[j] == i) {
                in_ring = 1;
                break;
            }
        }
        if (in_ring) {
            continue;
        }
        ring->ring[pos] = i;
        status = ucc_tl_cuda_team_topo_build_ring(team, graph, ring, pos + 1,
                                                  width);
        if (status == UCC_OK) {
            return UCC_OK;
        }
    }
    return UCC_ERR_NOT_SUPPORTED;
}

/* TODO: simple algorithm to find NVLink rings.
 * 1. Find number of rings and minimal ring width
 * 2. Try to find ring of width W in given team topo.
 * 3. If found
 *    3.1. Duplicate ring W/min_width times to get rings of width min_width
 *    3.2. Duplicate inverted ring W times because NVLink is full duplex
 *    3.3. Remove ring from topology
 * 3. W = W/2 goto 2
 */

static ucc_status_t
ucc_tl_cuda_team_topo_init_rings(const ucc_tl_cuda_team_t *team,
                                 ucc_tl_cuda_team_topo_t *topo)
{
    ucc_rank_t size = UCC_TL_TEAM_SIZE(team);
    ucc_tl_cuda_ring_t ring;
    int i, width, nr, num_rings, min_width;
    ucc_status_t status;
    ucc_rank_t *graph;

    ucc_assert(size > 1);
    topo->num_rings = 0;
    ring.ring = (ucc_rank_t*) ucc_malloc(size * sizeof(ucc_rank_t),
                                         "cuda_topo_ring");
    if (!ring.ring) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to allocate topo ring");
        return UCC_ERR_NO_MEMORY;
    }

    graph = (ucc_rank_t*) ucc_malloc(size * size * sizeof(ucc_rank_t),
                                     "cuda_topo_graph");
    if (!graph) {
        status = UCC_ERR_NO_MEMORY;
        tl_error(UCC_TL_TEAM_LIB(team), "failed to allocate topo graph");
        goto free_ring;
    }

    memcpy(graph, topo->matrix, size * size * sizeof(ucc_rank_t));

    num_rings = 0;
    min_width = 4;
    for (width = min_width; width > 0; width >>= 1) {
        ring.ring[0] = 0;
        status = ucc_tl_cuda_team_topo_build_ring(team, graph, &ring, 1,
                                                  width);
        if (status == UCC_OK) {
            num_rings += 2*width;
            if (width < min_width) {
                min_width = width;
            }
            for (i = 0; i < size; i++) {
                if (graph[ring.ring[i] * size + ring.ring[(i+1)%size]] !=
                    UCC_TL_CUDA_TEAM_TOPO_SAME_DEVICE) {
                    graph[ring.ring[i] * size + ring.ring[(i+1)%size]] -= width;
                    graph[ring.ring[(i+1)%size] * size + ring.ring[i]] -= width;
                }
            }
        }
    }

    if (num_rings == 0) {
        status = UCC_ERR_NOT_SUPPORTED;
        tl_debug(UCC_TL_TEAM_LIB(team), "no rings found");
        goto free_graph;
    }

    topo->rings = (ucc_tl_cuda_ring_t*)ucc_malloc(num_rings * sizeof(*topo->rings),
                                                  "cuda_topo_rings");
    if (!topo->rings) {
        status = UCC_ERR_NO_MEMORY;
        tl_error(UCC_TL_TEAM_LIB(team), "failed to allocate topo rings array");
        goto free_graph;
    }

    for (i = 0; i < size * size; i++) {
        graph[i] = topo->matrix[i];
    }
    for (width = 4; width > 0; width >>= 1) {
        ring.ring[0] = 0;
        status = ucc_tl_cuda_team_topo_build_ring(team, graph, &ring, 1,
                                                  width);
        if (status == UCC_OK) {
            nr = width / min_width;
            status = ucc_tl_cuda_team_topo_add_ring(team, topo, &ring, 0, nr);
            if (status != UCC_OK) {
                goto free_rings;
            }

            if (size > 2) {
                status = ucc_tl_cuda_team_topo_add_ring(team, topo, &ring, 1, nr);
                if (status != UCC_OK) {
                    goto free_rings;
                }
            }

            for (i = 0; i < size; i++) {
                if (graph[ring.ring[i] * size + ring.ring[(i+1)%size]] !=
                    UCC_TL_CUDA_TEAM_TOPO_SAME_DEVICE) {
                    graph[ring.ring[i] * size + ring.ring[(i+1)%size]] -= width;
                    graph[ring.ring[(i+1)%size] * size + ring.ring[i]] -= width;
                }
            }
        }
    }

    ucc_free(graph);
    ucc_free(ring.ring);
    return UCC_OK;

free_rings:
    for (i = 0; i < topo->num_rings; i++) {
        ucc_free(topo->rings[i].ring);
    }
    ucc_free(topo->rings);
free_graph:
    ucc_free(graph);
free_ring:
    ucc_free(ring.ring);
    return status;
}

static ucc_status_t
ucc_tl_cuda_team_topo_init_proxies(const ucc_tl_cuda_team_t *team,
                                   ucc_tl_cuda_team_topo_t *topo)
{
    ucc_rank_t size        = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t rank        = UCC_TL_TEAM_RANK(team);
    ucc_rank_t num_proxies = 0;
    ucc_rank_t i, j, k, proxy;
    float *data;
    float score, min_score;
    ucc_status_t status;
    char pci_str[2][MAX_PCI_BUS_ID_STR];

    topo->proxy_needed = 0;

    for (i = 0; i < size * size; i++) {
        if (topo->matrix[i] == 0) {
            num_proxies++;
        }
    }

    topo->num_proxies = num_proxies;
    topo->is_fully_connected = (num_proxies == 0) ? 1 : 0;
    if (num_proxies == 0) {
        return UCC_OK;
    }

    topo->proxies = (ucc_tl_cuda_proxy_t*)ucc_malloc(
            num_proxies * sizeof(ucc_tl_cuda_proxy_t), "cuda_topo_proxies");
    if (!topo->proxies) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to alloc cuda topo proxies");
        return UCC_ERR_NO_MEMORY;
    }

    data = (float*)ucc_malloc(size * size * sizeof(float),
                              "cuda topo proxies data");
    if (!data) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to alloc cuda topo work array");
        status = UCC_ERR_NO_MEMORY;
        goto free_proxy;
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            if (ucc_tl_cuda_team_topo_is_direct(&team->super, topo, i, j)) {
                data[i * size + j] = 1.0;
            } else {
                data[i * size + j] = 0.0;
            }
        }
    }

    num_proxies = 0;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            if (ucc_tl_cuda_team_topo_is_direct(&team->super, topo, i, j)) {
                continue;
            }
            if ((i == rank) || (j == rank)) {
                topo->proxy_needed = 1;
            }
            proxy = UCC_RANK_INVALID;
            min_score = (float)(UCC_RANK_MAX);
            for (k = 0; k < size; k++) {
                if (ucc_tl_cuda_team_topo_is_direct(&team->super, topo, i, k) &&
                    ucc_tl_cuda_team_topo_is_direct(&team->super, topo, k, j)) {
                    ucc_assert((topo->matrix[i * size + k] > 0) &&
                               (topo->matrix[k * size + j] > 0));
                    score = ucc_max((data[i * size + k] + 1.0) /
                                    topo->matrix[i * size + k],
                                    (data[k * size + j] + 1.0) /
                                    topo->matrix[k * size + j]);
                    if (score >= min_score) {
                        continue;
                    }
                    proxy = k;
                    min_score = score;
                }
            }
            if (proxy == UCC_RANK_INVALID) {
                ucc_tl_cuda_topo_pci_id_to_str(&team->ids[i].pci_id,
                                                pci_str[0], MAX_PCI_BUS_ID_STR);
                ucc_tl_cuda_topo_pci_id_to_str(&team->ids[j].pci_id,
                                                pci_str[1], MAX_PCI_BUS_ID_STR);
                tl_debug(UCC_TL_TEAM_LIB(team), "no proxy found between "
                         "dev %s (%d) and dev %s (%d), "
                         "cuda topology is not supported",
                        pci_str[0], i, pci_str[1], j);
                status = UCC_ERR_NOT_SUPPORTED;
                goto free_data;
            }
            if (proxy == rank) {
                topo->proxies[num_proxies].src   = i;
                topo->proxies[num_proxies].dst   = j;
                topo->proxies[num_proxies].proxy = proxy;
                num_proxies++;
            }
            data[i * size + proxy] += 1.0;
            data[proxy * size + j] += 1.0;
        }
    }
    topo->num_proxies = num_proxies;

    ucc_free(data);
    return UCC_OK;
free_data:
    ucc_free(data);
free_proxy:
    ucc_free(topo->proxies);
    return status;
}

static ucc_status_t
ucc_tl_cuda_team_topo_init_matrix(const ucc_tl_cuda_team_t *team,
                                  ucc_rank_t *matrix)
{
    ucc_tl_cuda_topo_t *topo = UCC_TL_CUDA_TEAM_CTX(team)->topo;
    int                 size = UCC_TL_TEAM_SIZE(team);
    ucc_status_t status;
    int i, j;

    for (i = 0; i < size; i++) {
        matrix[i + i*size] = UCC_TL_CUDA_TEAM_TOPO_SAME_DEVICE;
        for (j = i + 1; j < size; j++) {
            if (ucc_tl_cuda_topo_device_id_equal(&team->ids[i].pci_id,
                                                 &team->ids[j].pci_id)) {
                matrix[i + j*size] = UCC_TL_CUDA_TEAM_TOPO_SAME_DEVICE;
            } else {
                status = ucc_tl_cuda_topo_num_links(topo,
                                                    &team->ids[i].pci_id,
                                                    &team->ids[j].pci_id,
                                                    &matrix[i + j*size]);
                if (status != UCC_OK) {
                    return status;
                }
            }
            matrix[j + i*size] = matrix[i +j*size];
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_team_topo_create(const ucc_tl_team_t *cuda_team,
                                          ucc_tl_cuda_team_topo_t **team_topo)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(cuda_team, ucc_tl_cuda_team_t);
    ucc_rank_t          size = UCC_TL_TEAM_SIZE(team);
    ucc_tl_cuda_team_topo_t *topo;
    ucc_status_t status;

    topo = (ucc_tl_cuda_team_topo_t*)ucc_malloc(sizeof(*topo), "cuda_team_topo");
    if (!topo) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to alloc cuda team topo");
        return UCC_ERR_NO_MEMORY;
    }

    topo->matrix = (ucc_rank_t*)ucc_malloc(size * size * sizeof(ucc_rank_t),
                                           "cuda_topo_matrix");
    if (!topo->matrix) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to alloc cuda team topo matrix");
        status = UCC_ERR_NO_MEMORY;
        goto free_topo;
    }
    status = ucc_tl_cuda_team_topo_init_matrix(team, topo->matrix);
    if (status != UCC_OK) {
        goto free_matrix;
    }

    status = ucc_tl_cuda_team_topo_init_proxies(team, topo);
    if (status != UCC_OK) {
        if (status != UCC_ERR_NOT_SUPPORTED) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to init cuda topo proxy");
        }
        goto free_matrix;
    }

    status = ucc_tl_cuda_team_topo_init_rings(team, topo);
    if (status != UCC_OK) {
        if (status != UCC_ERR_NOT_SUPPORTED) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to init cuda topo rings");
        }
        goto free_proxy;
    }

    *team_topo = topo;
    return UCC_OK;
free_proxy:
    if (topo->num_proxies > 0) {
        ucc_free(topo->proxies);
    }
free_matrix:
    ucc_free(topo->matrix);
free_topo:
    ucc_free(topo);
    return status;
}

void ucc_tl_cuda_team_topo_print_proxies(const ucc_tl_team_t *tl_team,
                                         const ucc_tl_cuda_team_topo_t *topo)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_rank_t          size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t          rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t i;
    char pci_str[3][MAX_PCI_BUS_ID_STR];

    for (i = 0; i < size; i++) {
        if (ucc_tl_cuda_team_topo_is_direct(tl_team, topo, rank, i)) {
            ucc_tl_cuda_topo_pci_id_to_str(&team->ids[rank].pci_id,
                                            pci_str[0], MAX_PCI_BUS_ID_STR);
            ucc_tl_cuda_topo_pci_id_to_str(&team->ids[i].pci_id,
                                            pci_str[1], MAX_PCI_BUS_ID_STR);
            if (topo->matrix[rank * size +i] == UCC_TL_CUDA_TEAM_TOPO_SAME_DEVICE)
            {
                tl_debug(UCC_TL_TEAM_LIB(team),
                        "dev %s (%d) to dev %s (%d): same device",
                        pci_str[0], rank, pci_str[1], i);

            } else {
                tl_debug(UCC_TL_TEAM_LIB(team),
                        "dev %s (%d) to dev %s (%d): %d direct links",
                        pci_str[0], rank, pci_str[1], i,
                        topo->matrix[rank * size + i]);
            }
        }
    }

    for (i = 0; i < topo->num_proxies; i++) {
        ucc_tl_cuda_topo_pci_id_to_str(&team->ids[topo->proxies[i].src].pci_id,
                                        pci_str[0], MAX_PCI_BUS_ID_STR);
        ucc_tl_cuda_topo_pci_id_to_str(&team->ids[topo->proxies[i].dst].pci_id,
                                        pci_str[1], MAX_PCI_BUS_ID_STR);
        ucc_tl_cuda_topo_pci_id_to_str(&team->ids[topo->proxies[i].proxy].pci_id,
                                        pci_str[2], MAX_PCI_BUS_ID_STR);
        tl_debug(UCC_TL_TEAM_LIB(team),
                 "dev %s (%d) to dev %s (%d): proxy dev %s (%d)",
                 pci_str[0], topo->proxies[i].src,
                 pci_str[1], topo->proxies[i].dst,
                 pci_str[2], topo->proxies[i].proxy);
    }
}

void ucc_tl_cuda_team_topo_print_rings(const ucc_tl_team_t *tl_team,
                                       const ucc_tl_cuda_team_topo_t *topo)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_rank_t          rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          size = UCC_TL_TEAM_SIZE(team);
    int i, j;

    for (i = 0; i < topo->num_rings; i++) {
        for (j = 0; j < size; j++) {
            if (topo->rings[i].ring[j] == rank) {
                tl_debug(UCC_TL_TEAM_LIB(team), "ring %d: %d send to %d",
                         i, rank, topo->rings[i].ring[(j + 1) % size]);
            }
        }
    }
}

ucc_status_t ucc_tl_cuda_team_topo_destroy(ucc_tl_cuda_team_topo_t *team_topo)
{
    int i;

    for (i = 0; i < team_topo->num_rings; i++) {
        ucc_free(team_topo->rings[i].ring);
    }
    ucc_free(team_topo->rings);
    if (team_topo->num_proxies > 0) {
        ucc_free(team_topo->proxies);
    }
    ucc_free(team_topo->matrix);
    ucc_free(team_topo);
    return UCC_OK;
}
