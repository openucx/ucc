#include "tl_cuda_team_topo.h"
#include "tl_cuda.h"

static ucc_status_t
ucc_tl_cuda_team_topo_init_proxy(const ucc_tl_cuda_team_t *team,
                                 ucc_tl_cuda_team_topo_t *topo)
{
    ucc_rank_t size        = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t num_proxies = 0;
    ucc_rank_t i, j, p, k;
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
            if (ucc_tl_cuda_team_topo_is_direct(&team->super, topo, i, j)) {
                continue;
            }
            for (k = 0; k < size; k++) {
                if (ucc_tl_cuda_team_topo_is_direct(&team->super, topo, i, k) &&
                    ucc_tl_cuda_team_topo_is_direct(&team->super, topo, k, j)) {
                    topo->proxies[p].src   = i;
                    topo->proxies[p].dst   = j;
                    topo->proxies[p].proxy = k;
                    break;
                }
            }
            if (k == size) {
                tl_info(UCC_TL_TEAM_LIB(team), "no proxy found between "
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

static ucc_status_t
ucc_tl_cuda_team_topo_init_matrix(const ucc_tl_cuda_team_t *team,
                                  int *matrix)
{
    ucc_tl_cuda_topo_t *topo = UCC_TL_CUDA_TEAM_CTX(team)->topo;
    int                 size = UCC_TL_TEAM_SIZE(team);
    ucc_status_t status;
    int i, j;

    for (i = 0; i < size; i++) {
        matrix[i + i*size] = 1;
        for (j = i + 1; j < size; j++) {
            status = ucc_tl_cuda_topo_num_links(topo,
                                                &team->ids[i].pci_id,
                                                &team->ids[j].pci_id,
                                                &matrix[i + j*size]);
            if (status != UCC_OK) {
                return status;
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

    topo->matrix = (int*)ucc_malloc(size * size * sizeof(int),
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

    status = ucc_tl_cuda_team_topo_init_proxy(team, topo);
    if (status != UCC_OK) {
        if (status != UCC_ERR_NOT_SUPPORTED) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to init cuda topo proxy");
        }
        goto free_matrix;
    }

    *team_topo = topo;
    return UCC_OK;
free_matrix:
    ucc_free(topo->matrix);
free_topo:
    ucc_free(team_topo);
    return status;
}

void ucc_tl_cuda_team_topo_print(const ucc_tl_team_t *tl_team,
                                 const ucc_tl_cuda_team_topo_t *topo)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_rank_t size          = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t rank          = UCC_TL_TEAM_RANK(team);
    ucc_rank_t i, j;

    for (i = 0; i < size; i++) {
        if (ucc_tl_cuda_team_topo_is_direct(tl_team, topo, rank, i)) {
            tl_debug(UCC_TL_TEAM_LIB(team),
                     "dev %d rank %d to dev %d rank %d: %d direct links",
                     team->ids[rank].device, rank, team->ids[i].device, i,
                     topo->matrix[rank * size + i]);
        } else {
            for (j = 0 ; j < topo->num_proxies; j++) {
                if ((topo->proxies[j].src == rank) &&
                    (topo->proxies[j].dst == i)) {
                    tl_debug(UCC_TL_TEAM_LIB(team),
                             "dev %d rank %d to dev %d rank %d: "
                             "proxy dev %d rank %d",
                             team->ids[rank].device, rank,
                             team->ids[i].device, i,
                             team->ids[topo->proxies[j].proxy].device,
                             topo->proxies[j].proxy);
                    break;
                }
            }
        }
    }
}

ucc_status_t ucc_tl_cuda_team_topo_destroy(ucc_tl_cuda_team_topo_t *team_topo)
{
    if (team_topo->num_proxies) {
        ucc_free(team_topo->proxies);
    }
    ucc_free(team_topo->matrix);
    ucc_free(team_topo);
    return UCC_OK;
}
