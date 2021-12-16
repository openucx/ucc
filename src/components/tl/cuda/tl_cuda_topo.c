/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_topo.h"

static ucc_status_t ucc_tl_cuda_topo_init_matrix(ucc_tl_cuda_team_t *team,
                                                 ucc_tl_cuda_topo_t *cuda_topo)
{
    ucc_rank_t size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t i, j;
    int dev, peer_dev, peer_access;
    ucc_status_t status;

    for (i = 0; i < size; i++) {
       dev = team->ids[i].device;
       cuda_topo->matrix[i*size + i] = 1;
        for (j = i + 1; j < size; j++) {
            peer_dev = team->ids[j].device;
            CUDACHECK_GOTO(cudaDeviceCanAccessPeer(&peer_access, dev, peer_dev),
                           exit_err, status, UCC_TL_TEAM_LIB(team));
            cuda_topo->matrix[i*size + j] = peer_access;
            cuda_topo->matrix[j*size + i] = peer_access;
        }
    }
    return UCC_OK;
exit_err:
    return status;
}

static ucc_status_t ucc_tl_cuda_topo_init_proxy(ucc_tl_cuda_team_t *team,
                                                ucc_tl_cuda_topo_t *topo)
{
    ucc_rank_t size        = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t num_proxies = 0;
    ucc_rank_t i, j, k, p;
    ucc_status_t status;

    for (i = 0; i < size * size; i++) {
        if (topo->matrix[i] == 0) {
            num_proxies ++;
        }
    }

    topo->num_proxies = num_proxies;
    if (num_proxies == 0) {
        return UCC_OK;
    }
    topo->proxies = (ucc_tl_cuda_proxy_t*)ucc_malloc(
            num_proxies * sizeof(ucc_tl_cuda_proxy_t), "cuda_topo_proxies");
    if (!topo->proxies) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to alloc cuda topo proxy");
        return UCC_ERR_NO_MEMORY;
    }

    p = 0;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            if (ucc_tl_cuda_topo_is_direct(team, topo, i, j)) {
                continue;
            }
            for (k = 0; k < size; k++) {
                if (ucc_tl_cuda_topo_is_direct(team, topo, i, k) &&
                    ucc_tl_cuda_topo_is_direct(team, topo, k, j)) {
                    topo->proxies[p].src   = i;
                    topo->proxies[p].dst   = j;
                    topo->proxies[p].proxy = k;
                    break;
                }
            }
            if (k == size) {
                tl_info(UCC_TL_TEAM_LIB(team), "no proxy found between"
                        "dev %d rank %d and dev %d rank %d, "
                        "cuda topology is not supported",
                        i, team->ids[i].device, j, team->ids[j].device);
                status = UCC_ERR_NOT_SUPPORTED;
                goto free_proxy;
            }
            p++;
        }
    }
    return UCC_OK;

free_proxy:
    ucc_free(topo->proxies);
    return status;
}

ucc_status_t ucc_tl_cuda_topo_create(ucc_tl_cuda_team_t *team,
                                     ucc_tl_cuda_topo_t **cuda_topo)
{
    ucc_rank_t size = UCC_TL_TEAM_SIZE(team);
    ucc_tl_cuda_topo_t *topo;
    ucc_status_t status;

    topo = (ucc_tl_cuda_topo_t*)ucc_malloc(sizeof(*topo), "cuda_topo");
    if (!topo) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to alloc cuda topo");
        status = UCC_ERR_NO_MEMORY;
        goto exit_err;
    }

    topo->proxies = NULL;
    topo->matrix  = (int*)ucc_malloc(size * size * sizeof(int),
                                     "cuda_topo_matrix");
    if (!topo->matrix) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to alloc cuda topo matrix");
        status = UCC_ERR_NO_MEMORY;
        goto free_topo;
    }

    status = ucc_tl_cuda_topo_init_matrix(team, topo);
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init cuda topo matrix");
        goto free_topo_matrix;
    }

    status = ucc_tl_cuda_topo_init_proxy(team, topo);
    if (status != UCC_OK) {
        if (status != UCC_ERR_NOT_SUPPORTED) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to init cuda topo proxy");
        }
        goto free_topo_matrix;
    }

    *cuda_topo = topo;
    return UCC_OK;
free_topo_matrix:
    ucc_free(topo->matrix);
free_topo:
    ucc_free(topo);
exit_err:
    return status;
}

void ucc_tl_cuda_topo_print(ucc_tl_cuda_team_t *team,
                            ucc_tl_cuda_topo_t *cuda_topo)
{
    ucc_rank_t size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t i, j;

    for (i = 0; i < size; i++) {
        if (ucc_tl_cuda_topo_is_direct(team, cuda_topo, rank, i)) {
            tl_debug(UCC_TL_TEAM_LIB(team),
                     "dev %d rank %d to dev %d rank %d: direct",
                     team->ids[rank].device, rank, team->ids[i].device, i);
        } else {
            for (j = 0 ; j < cuda_topo->num_proxies; j++) {
                if ((cuda_topo->proxies[j].src == rank) &&
                    (cuda_topo->proxies[j].dst == i)) {
                    tl_debug(UCC_TL_TEAM_LIB(team),
                             "dev %d rank %d to dev %d rank %d: "
                             "proxy dev %d rank %d",
                             team->ids[rank].device, rank,
                             team->ids[i].device, i,
                             team->ids[cuda_topo->proxies[j].proxy].device,
                             cuda_topo->proxies[j].proxy);
                    break;
                }
            }
        }
    }
}

ucc_status_t ucc_tl_cuda_topo_destroy(ucc_tl_cuda_topo_t *cuda_topo)
{
    ucc_free(cuda_topo->matrix);
    if (cuda_topo->proxies) {
        ucc_free(cuda_topo->proxies);
    }
    ucc_free(cuda_topo);
    return UCC_OK;
}
