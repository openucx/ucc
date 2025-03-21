/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_sbgp.h"
#include "ucc_topo.h"
#include "utils/ucc_log.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_math.h"
#include "utils/ucc_compiler_def.h"
#include <limits.h>

static char *ucc_sbgp_type_str[UCC_SBGP_LAST] = {
    "numa", "socket",         "node",         "node_leaders",
    "net",  "socket_leaders", "numa_leaders", "flat"};

const char* ucc_sbgp_str(ucc_sbgp_type_t type)
{
    return ucc_sbgp_type_str[type];
}

#define UCC_TOPO_IS_BOUND(_topo, _sbgp_type)                    \
    (UCC_SBGP_SOCKET         == (_sbgp_type) ||                 \
     UCC_SBGP_SOCKET_LEADERS == (_sbgp_type)) ?                 \
        (_topo)->topo->sock_bound : (_topo)->topo->numa_bound

static inline int ucc_ranks_on_local_sn(ucc_rank_t rank1, ucc_rank_t rank2,
                                        ucc_topo_t *topo, ucc_sbgp_type_t type)
{
    ucc_rank_t       ctx_rank1 = ucc_ep_map_eval(topo->set.map, rank1);
    ucc_rank_t       ctx_rank2 = ucc_ep_map_eval(topo->set.map, rank2);
    ucc_proc_info_t *proc1     = &topo->topo->procs[ctx_rank1];
    ucc_proc_info_t *proc2     = &topo->topo->procs[ctx_rank2];
    int              bound     = UCC_TOPO_IS_BOUND(topo, type);

    if (!bound) {
        return 0;
    }
    return proc1->host_hash == proc2->host_hash &&
           ((UCC_SBGP_SOCKET == type) ? proc1->socket_id == proc2->socket_id
                                      : proc1->numa_id == proc2->numa_id);
}

static inline ucc_status_t sbgp_create_sn(ucc_topo_t *topo, ucc_sbgp_t *sbgp,
                                          ucc_rank_t group_rank,
                                          int        allow_size_1)
{
    ucc_sbgp_t *node_sbgp = &topo->sbgps[UCC_SBGP_NODE];
    ucc_rank_t  nlr       = topo->node_leader_rank;
    ucc_rank_t  sn_rank = 0, sn_size = 0;
    int         i, r, nlr_pos;
    ucc_rank_t *local_ranks;

    ucc_assert(node_sbgp->status == UCC_SBGP_ENABLED);
    local_ranks =
        ucc_malloc(node_sbgp->group_size * sizeof(ucc_rank_t), "local_ranks");
    if (!local_ranks) {
        ucc_error("failed to allocate %zd bytes for local_ranks array",
                  node_sbgp->group_size * sizeof(ucc_rank_t));
        return UCC_ERR_NO_MEMORY;
    }
    for (i = 0; i < node_sbgp->group_size; i++) {
        r = ucc_ep_map_eval(node_sbgp->map, i);
        if (ucc_ranks_on_local_sn(r, group_rank, topo, sbgp->type)) {
            local_ranks[sn_size] = r;
            if (r == group_rank) {
                sn_rank = sn_size;
            }
            sn_size++;
        }
    }
    sbgp->group_size = sn_size;
    sbgp->group_rank = sn_rank;
    nlr_pos          = -1;
    for (i = 0; i < sn_size; i++) {
        if (nlr == local_ranks[i]) {
            nlr_pos = i;
            break;
        }
    }
    if (nlr_pos > 0) {
        if (sn_rank == 0)
            sbgp->group_rank = nlr_pos;
        if (sn_rank == nlr_pos)
            sbgp->group_rank = 0;
        SWAP(local_ranks[nlr_pos], local_ranks[0], int);
    }
    if (sn_size > 1 || allow_size_1) {
        sbgp->status   = UCC_SBGP_ENABLED;
        sbgp->rank_map = local_ranks;
    } else {
        sbgp->status = UCC_SBGP_NOT_EXISTS;
        ucc_free(local_ranks);
    }
    return UCC_OK;
}

ucc_status_t ucc_sbgp_create_node(ucc_topo_t *topo, ucc_sbgp_t *sbgp)
{
    ucc_subset_t *set            = &topo->set;
    ucc_rank_t    group_size     = ucc_subset_size(set);
    ucc_rank_t    group_rank     = set->myrank;
    ucc_rank_t    max_local_size = 256;
    ucc_rank_t    ctx_nlr        = topo->node_leader_rank_id;
    ucc_rank_t    node_rank = 0, node_size = 0;
    int           i;
    ucc_rank_t   *local_ranks, *tmp;
    local_ranks =
        ucc_malloc(max_local_size * sizeof(ucc_rank_t), "local_ranks");
    if (!local_ranks) {
        ucc_error("failed to allocate %zd bytes for local_ranks array",
                  max_local_size * sizeof(ucc_rank_t));
        return UCC_ERR_NO_MEMORY;
    }
    for (i = 0; i < group_size; i++) {
        if (ucc_rank_on_local_node(i, topo)) {
            if (node_size == max_local_size) {
                max_local_size *= 2;
                tmp = ucc_realloc(local_ranks,
                                  max_local_size * sizeof(ucc_rank_t));
                if (!tmp) {
                    ucc_error(
                        "failed to allocate %zd bytes for local_ranks array",
                        max_local_size * sizeof(ucc_rank_t));
                    ucc_free(local_ranks);
                    return UCC_ERR_NO_MEMORY;
                }
                local_ranks = tmp;
            }
            local_ranks[node_size] = i;

            if (i == group_rank) {
                node_rank = node_size;
            }
            node_size++;
        }
    }
    if (0 == node_size) {
        /* We should always have at least 1 local rank */
        ucc_free(local_ranks);
        return UCC_ERR_NOT_FOUND;
    }
    sbgp->group_size = node_size;
    sbgp->group_rank = node_rank;
    sbgp->rank_map   = local_ranks;
    if (0 < ctx_nlr && ctx_nlr < node_size) {
        /* Rotate local_ranks array so that node_leader_rank_id becomes first
           in that array */
        sbgp->rank_map = ucc_malloc(node_size * sizeof(ucc_rank_t), "rank_map");
        if (!sbgp->rank_map) {
            ucc_error("failed to allocate %zd bytes for rank_map array",
                      node_size * sizeof(ucc_rank_t));
            ucc_free(local_ranks);
            return UCC_ERR_NO_MEMORY;
        }
        for (i = ctx_nlr; i < node_size; i++) {
            sbgp->rank_map[i - ctx_nlr] = local_ranks[i];
        }

        for (i = 0; i < ctx_nlr; i++) {
            sbgp->rank_map[node_size - ctx_nlr + i] = local_ranks[i];
        }
        sbgp->group_rank = (node_rank + node_size - ctx_nlr) % node_size;
        ucc_free(local_ranks);
    }
    topo->node_leader_rank = sbgp->rank_map[0];
    if (node_size > 1) {
        sbgp->status = UCC_SBGP_ENABLED;
    } else {
        sbgp->status = UCC_SBGP_NOT_EXISTS;
    }
    return UCC_OK;
}

static ucc_status_t sbgp_create_node_leaders(ucc_topo_t *topo, ucc_sbgp_t *sbgp,
                                             int ctx_nlr)
{
    ucc_subset_t *set               = &topo->set;
    ucc_rank_t    comm_size         = ucc_subset_size(set);
    ucc_rank_t    comm_rank         = set->myrank;
    ucc_rank_t    min_sbgp_size     = UCC_RANK_MAX;
    ucc_rank_t    max_sbgp_size     = 0;
    ucc_rank_t    max_ctx_sbgp_size = 0;
    ucc_rank_t   *nl_array_3        = NULL;
    int           i_am_node_leader  = 0;
    int           socket_bound      = topo->topo->sock_bound;
    int           numa_bound        = topo->topo->numa_bound;
    int           bound             = socket_bound || numa_bound;
    ucc_rank_t    nnodes            = topo->topo->nnodes;
    ucc_rank_t    n_node_leaders, ctx_rank, i;
    ucc_rank_t   *nl_array_1, *nl_array_2;
    ucc_host_id_t host_id;
    uint8_t       sbgp_id;

    ucc_assert(comm_size != 0 && nnodes != 0);

    if (topo->min_ppn != UCC_RANK_MAX && ctx_nlr >= topo->min_ppn) {
        sbgp->status = UCC_SBGP_NOT_EXISTS;
        return UCC_OK;
    }
    nl_array_1 = ucc_malloc(nnodes * sizeof(ucc_rank_t), "nl_array_1");
    if (!nl_array_1) {
        ucc_error("failed to allocate %zd bytes for nl_array_1",
                  nnodes * sizeof(ucc_rank_t));
        return UCC_ERR_NO_MEMORY;
    }
    nl_array_2 = ucc_malloc(nnodes * sizeof(ucc_rank_t), "nl_array_2");
    if (!nl_array_2) {
        ucc_error("failed to allocate %zd bytes for nl_array_2",
                  nnodes * sizeof(ucc_rank_t));
        ucc_free(nl_array_1);
        return UCC_ERR_NO_MEMORY;
    }

    if (bound) {
        max_ctx_sbgp_size = socket_bound ? topo->topo->max_n_sockets :
                                           topo->topo->max_n_numas;
        nl_array_3 = ucc_malloc(max_ctx_sbgp_size * nnodes *
                                sizeof(ucc_rank_t), "nl_array_3");
        if (!nl_array_3) {
            ucc_error("failed to allocate %zd bytes for nl_array_3",
                      max_ctx_sbgp_size * nnodes * sizeof(ucc_rank_t));
            ucc_free(nl_array_1);
            ucc_free(nl_array_2);
            return UCC_ERR_NO_MEMORY;
        }

        memset(nl_array_3, 0, max_ctx_sbgp_size * nnodes * sizeof(ucc_rank_t));
    }

    for (i = 0; i < nnodes; i++) {
        nl_array_1[i] = 0;
        nl_array_2[i] = UCC_RANK_MAX;
    }

    for (i = 0; i < comm_size; i++) {
        ctx_rank  = ucc_ep_map_eval(set->map, i);
        host_id   = topo->topo->procs[ctx_rank].host_id;
        if (bound) {
            sbgp_id = socket_bound ? topo->topo->procs[ctx_rank].socket_id :
                                     topo->topo->procs[ctx_rank].numa_id;
            nl_array_3[sbgp_id + host_id * max_ctx_sbgp_size]++;
        }

        /* Find the first rank that maps to this node, store in nl_array_2 */
        if (nl_array_1[host_id] == 0 || nl_array_1[host_id] == ctx_nlr) {
            nl_array_2[host_id] = i;
        }
        nl_array_1[host_id]++;
    }

    for (i = 0; i < nnodes; i++) {
        if (nl_array_1[i] > topo->max_ppn) {
            topo->max_ppn = nl_array_1[i];
        }
        if (nl_array_1[i] != 0 && nl_array_1[i] < topo->min_ppn) {
            topo->min_ppn = nl_array_1[i];
        }
    }

    if (bound) {
        for (i = 0; i < nnodes * max_ctx_sbgp_size; i++) {
            if (nl_array_3[i] == 0) {
                continue;
            }
            min_sbgp_size = ucc_min(min_sbgp_size, nl_array_3[i]);
            max_sbgp_size = ucc_max(max_sbgp_size, nl_array_3[i]);
        }
        if (socket_bound) {
            topo->min_socket_size = min_sbgp_size;
            topo->max_socket_size = max_sbgp_size;
        } else {
            topo->min_numa_size = min_sbgp_size;
            topo->max_numa_size = max_sbgp_size;
        }
        ucc_free(nl_array_3);
    }

    n_node_leaders = 0;
    if (ctx_nlr >= topo->min_ppn) {
        /* at least one node has less number of local ranks than
           ctx_nlr - can't build NET sbgp */
        goto skip;
    }

    for (i = 0; i < nnodes; i++) {
        if (nl_array_2[i] != UCC_RANK_MAX) {
            if (comm_rank == nl_array_2[i]) {
                i_am_node_leader = 1;
                sbgp->group_rank = n_node_leaders;
            }
            nl_array_1[n_node_leaders++] = nl_array_2[i];
        }
    }
skip:
    ucc_free(nl_array_2);

    if (n_node_leaders > 1) {
        sbgp->group_size = n_node_leaders;
        if (i_am_node_leader) {
            sbgp->status = UCC_SBGP_ENABLED;
        } else {
            sbgp->status = UCC_SBGP_DISABLED;
        }
        sbgp->rank_map = nl_array_1;
    } else {
        ucc_free(nl_array_1);
        sbgp->status = UCC_SBGP_NOT_EXISTS;
    }
    return UCC_OK;
}

#define GET_SN_ID(_topo, _proc, _type)                                         \
    (((_type) == UCC_SBGP_SOCKET_LEADERS)                                      \
         ? (_topo)->topo->procs[(_proc)].socket_id                             \
         : (_topo)->topo->procs[(_proc)].numa_id)

static ucc_status_t sbgp_create_sn_leaders(ucc_topo_t *topo, ucc_sbgp_t *sbgp)
{
    ucc_subset_t *  set            = &topo->set;
    ucc_sbgp_t *    node_sbgp      = &topo->sbgps[UCC_SBGP_NODE];
    ucc_rank_t      comm_rank      = set->myrank;
    ucc_rank_t      nlr            = topo->node_leader_rank;
    int             i_am_sn_leader = (nlr == comm_rank);
    ucc_rank_t      n_sn_leaders   = 1;
    int             max_n_sns      = (sbgp->type == UCC_SBGP_SOCKET_LEADERS)
                                         ? topo->topo->max_n_sockets
                                         : topo->topo->max_n_numas;
    ucc_rank_t     *sl_array;
    ucc_socket_id_t nlr_sock_id;
    int             i;

    sl_array = ucc_malloc(max_n_sns * sizeof(ucc_rank_t), "sl_array");
    if (!sl_array) {
        ucc_error("failed to allocate %zd bytes for sl_array",
                  max_n_sns * sizeof(ucc_rank_t));
        return UCC_ERR_NO_MEMORY;
    }
    for (i = 0; i < max_n_sns; i++) {
        sl_array[i] = UCC_RANK_MAX;
    }
    nlr_sock_id = GET_SN_ID(topo, ucc_ep_map_eval(set->map, nlr), sbgp->type);
    sl_array[nlr_sock_id] = nlr;

    for (i = 0; i < node_sbgp->group_size; i++) {
        ucc_rank_t      r        = ucc_ep_map_eval(node_sbgp->map, i);
        ucc_rank_t      ctx_rank = ucc_ep_map_eval(set->map, r);
        ucc_socket_id_t sn_id    = GET_SN_ID(topo, ctx_rank, sbgp->type);
        if (sl_array[sn_id] == UCC_RANK_MAX) {
            n_sn_leaders++;
            sl_array[sn_id] = r;
            if (r == comm_rank) {
                i_am_sn_leader = 1;
            }
        }
    }
    if (UCC_SBGP_SOCKET_LEADERS == sbgp->type) {
        topo->n_sockets = n_sn_leaders;
    } else {
        ucc_assert(UCC_SBGP_NUMA_LEADERS == sbgp->type);
        topo->n_numas = n_sn_leaders;
    }
    if (n_sn_leaders > 1) {
        ucc_rank_t sl_rank = UCC_RANK_INVALID;
        sbgp->rank_map =
            ucc_malloc(sizeof(ucc_rank_t) * n_sn_leaders, "rank_map");
        if (!sbgp->rank_map) {
            ucc_error("failed to allocate %zd bytes for rank_map",
                      n_sn_leaders * sizeof(ucc_rank_t));
            ucc_free(sl_array);
            return UCC_ERR_NO_MEMORY;
        }
        n_sn_leaders = 0;
        for (i = 0; i < max_n_sns; i++) {
            if (sl_array[i] != UCC_RANK_MAX) {
                sbgp->rank_map[n_sn_leaders] = sl_array[i];
                if (comm_rank == sl_array[i]) {
                    sl_rank = n_sn_leaders;
                }
                n_sn_leaders++;
            }
        }
        int nlr_pos = -1;
        for (i = 0; i < n_sn_leaders; i++) {
            if (sbgp->rank_map[i] == nlr) {
                nlr_pos = i;
                break;
            }
        }
        ucc_assert(nlr_pos >= 0);
        sbgp->group_rank = sl_rank;
        if (nlr_pos > 0) {
            if (sl_rank == 0)
                sbgp->group_rank = nlr_pos;
            if (sl_rank == nlr_pos)
                sbgp->group_rank = 0;
            SWAP(sbgp->rank_map[nlr_pos], sbgp->rank_map[0], int);
        }

        sbgp->group_size = n_sn_leaders;
        if (i_am_sn_leader) {
            sbgp->status = UCC_SBGP_ENABLED;
        } else {
            sbgp->status = UCC_SBGP_DISABLED;
        }
    } else {
        sbgp->status = UCC_SBGP_NOT_EXISTS;
    }
    ucc_free(sl_array);
    return UCC_OK;
}

static inline ucc_status_t sbgp_create_full(ucc_topo_t *topo, ucc_sbgp_t *sbgp)
{
    sbgp->status     = UCC_SBGP_ENABLED;
    sbgp->group_size = ucc_subset_size(&topo->set);
    sbgp->group_rank = topo->set.myrank;
    sbgp->map.type   = UCC_EP_MAP_FULL;
    sbgp->map.ep_num = ucc_subset_size(&topo->set);

    return UCC_OK;
}

typedef struct proc_info_id {
    ucc_proc_info_t info;
    ucc_rank_t      id;
} proc_info_id_t;

static int ucc_compare_proc_info_id(const void *a, const void *b)
{
    const ucc_proc_info_t *d1 = &((const proc_info_id_t *)a)->info;
    const ucc_proc_info_t *d2 = &((const proc_info_id_t *)b)->info;

    if (d1->host_hash != d2->host_hash) {
        return d1->host_hash > d2->host_hash ? 1 : -1;
    } else if (d1->socket_id != d2->socket_id) {
        return d1->socket_id - d2->socket_id;
    } else if (d1->numa_id != d2->numa_id) {
        return d1->numa_id - d2->numa_id;
    } else {
        return 0;
    }
}

static ucc_status_t sbgp_create_full_ordered(ucc_topo_t *topo, ucc_sbgp_t *sbgp)
{
    ucc_rank_t       gsize = ucc_subset_size(&topo->set);
    ucc_proc_info_t *pinfo = topo->topo->procs;
    ucc_host_id_t   *visited;
    proc_info_id_t  *sorted;
    ucc_rank_t       i, j, num_visited;
    int              is_sorted, d;

    ucc_assert(gsize > 0);
    sbgp->status     = UCC_SBGP_ENABLED;
    sbgp->group_size = gsize;
    sbgp->group_rank = topo->set.myrank;
    sbgp->rank_map   = ucc_malloc(sizeof(ucc_rank_t) * gsize, "rank_map");
    if (ucc_unlikely(!sbgp->rank_map)) {
        ucc_error("failed to allocate %zd bytes for rank_map",
                  gsize * sizeof(ucc_rank_t));
        return UCC_ERR_NO_MEMORY;
    }

    visited = (ucc_host_id_t *)ucc_malloc(gsize * sizeof(ucc_host_id_t),
                                          "visited host");
    if (ucc_unlikely(!visited)) {
        ucc_error("failed to allocate %zd bytes for list of visited nodes",
                  gsize * sizeof(ucc_host_id_t));
        ucc_free(sbgp->rank_map);
        return UCC_ERR_NO_MEMORY;
    }

    is_sorted   = 1;
    num_visited = 1;
    visited[0] = pinfo[0].host_hash;
    for (i = 1; i < gsize; i++) {
        if (pinfo[i].host_hash != pinfo[i-1].host_hash) {
            /* check if we saw that host_has before*/
            for (j = 0; j < num_visited; j++) {
                if (visited[j] == pinfo[i].host_hash) {
                    break;
                }
            }
            if (j < num_visited) {
                /* this host was present already, ranks are not ordered */
                is_sorted = 0;
                break;
            }
            /* add new host to the list of visited */
            visited[num_visited++] = pinfo[i].host_hash;
        } else {
            d = ucc_compare_proc_info_id(&pinfo[i - 1].host_hash,
                                         &pinfo[i].host_hash);

            if (d > 0) {
                is_sorted = 0;
                break;
            }
        }
    }
    ucc_free(visited);

    if (is_sorted) {
        for (i = 0; i < gsize; i++) {
            sbgp->rank_map[i] = i;
        }
        return UCC_OK;
    }

    sorted = (proc_info_id_t *)ucc_malloc(gsize * sizeof(proc_info_id_t),
                                          "proc_sorted");
    if (ucc_unlikely(!sorted)) {
        ucc_error("failed to allocate %zd bytes for sorted proc info",
                  gsize * sizeof(proc_info_id_t));
        ucc_free(sbgp->rank_map);
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < gsize; i++) {
        sorted[i].info = topo->topo->procs[i];
        sorted[i].id   = i;
    }

    qsort(sorted, gsize, sizeof(proc_info_id_t), ucc_compare_proc_info_id);
    for (i = 0; i < gsize; i++) {
        if (sorted[i].id == topo->set.myrank) {
            sbgp->group_rank = i;
        }
        sbgp->rank_map[i] = sorted[i].id;
    }

    /*TODO: try to detect map by numa,socket,node and use UCC_EP_MAP_CB to save
     *      memory
     */

    ucc_free(sorted);
    return UCC_OK;
}

ucc_status_t ucc_sbgp_create(ucc_topo_t *topo, ucc_sbgp_type_t type)
{
    ucc_status_t status   = UCC_OK;
    ucc_sbgp_t * sbgp     = &topo->sbgps[type];
    int          sn_bound = UCC_TOPO_IS_BOUND(topo, type);

    sbgp->type     = type;
    sbgp->status   = UCC_SBGP_NOT_EXISTS;
    sbgp->rank_map = NULL;
    switch (type) {
    case UCC_SBGP_NODE:
        status = ucc_sbgp_create_node(topo, sbgp);
        break;
    case UCC_SBGP_FULL:
        status = sbgp_create_full(topo, sbgp);
        break;
    case UCC_SBGP_FULL_HOST_ORDERED:
        status = sbgp_create_full_ordered(topo, sbgp);
        break;
    case UCC_SBGP_SOCKET:
    case UCC_SBGP_NUMA:
        if (!sn_bound) {
            break;
        }
        if (topo->sbgps[UCC_SBGP_NODE].status == UCC_SBGP_NOT_INIT) {
            ucc_sbgp_create(topo, UCC_SBGP_NODE);
        }
        if (topo->sbgps[UCC_SBGP_NODE].status == UCC_SBGP_ENABLED) {
            status = sbgp_create_sn(topo, sbgp, topo->set.myrank, 0);
        }
        break;
    case UCC_SBGP_NODE_LEADERS:
        ucc_assert(UCC_SBGP_DISABLED != topo->sbgps[UCC_SBGP_NODE].status);
        status =
            sbgp_create_node_leaders(topo, sbgp, topo->node_leader_rank_id);
        break;
    case UCC_SBGP_NET:
        if (topo->sbgps[UCC_SBGP_NODE].status == UCC_SBGP_NOT_INIT) {
            ucc_sbgp_create(topo, UCC_SBGP_NODE);
        }
        ucc_assert(UCC_SBGP_DISABLED != topo->sbgps[UCC_SBGP_NODE].status);
        status = sbgp_create_node_leaders(
            topo, sbgp, topo->sbgps[UCC_SBGP_NODE].group_rank);
        break;
    case UCC_SBGP_SOCKET_LEADERS:
    case UCC_SBGP_NUMA_LEADERS:
        if (!sn_bound) {
            break;
        }
        if (topo->sbgps[UCC_SBGP_NODE].status == UCC_SBGP_NOT_INIT) {
            ucc_sbgp_create(topo, UCC_SBGP_NODE);
        }
        if (topo->sbgps[UCC_SBGP_NODE].status == UCC_SBGP_ENABLED) {
            status = sbgp_create_sn_leaders(topo, sbgp);
        }
        break;
    default:
        status = UCC_ERR_NOT_IMPLEMENTED;
        break;
    };
    if (sbgp->rank_map && sbgp->type != UCC_SBGP_FULL) {
        sbgp->map = ucc_ep_map_from_array(&sbgp->rank_map, sbgp->group_size,
                                          ucc_subset_size(&topo->set), 1);
    }
    if (sbgp->rank_map && sbgp->status == UCC_SBGP_NOT_EXISTS) {
        ucc_free(sbgp->rank_map);
    }
    return status;
}

ucc_status_t ucc_sbgp_cleanup(ucc_sbgp_t *sbgp)
{
    if (sbgp->rank_map) {
        ucc_free(sbgp->rank_map);
        sbgp->rank_map = NULL;
    }
    return UCC_OK;
}

void ucc_sbgp_print(ucc_sbgp_t *sbgp)
{
    int i;
    if (sbgp->group_rank == 0 && sbgp->status == UCC_SBGP_ENABLED) {
        printf("sbgp: %15s: group_size %4d, team_ranks=[ ",
               ucc_sbgp_str(sbgp->type), sbgp->group_size);
        for (i = 0; i < sbgp->group_size; i++) {
            printf("%d ", sbgp->rank_map[i]);
        }
        printf("]");
        printf("\n");
    }
}

ucc_status_t ucc_sbgp_create_all_sns(ucc_topo_t *topo, ucc_sbgp_t **_sbgps,
                                     int *n_sbgps, ucc_sbgp_type_t type)
{
    int          sn_bound = UCC_TOPO_IS_BOUND(topo, type);
    ucc_sbgp_t  *sbgps;
    ucc_sbgp_t * sn_leaders_sbgp;
    int          n_sn_groups, i;
    ucc_rank_t   sl_rank;
    ucc_status_t status;

    if (!sn_bound) {
        return UCC_ERR_NOT_FOUND;
    }

    sn_leaders_sbgp = ucc_topo_get_sbgp(topo, (UCC_SBGP_SOCKET == type)
                                                  ? UCC_SBGP_SOCKET_LEADERS
                                                  : UCC_SBGP_NUMA_LEADERS);
    n_sn_groups = (UCC_SBGP_SOCKET == type) ? topo->n_sockets : topo->n_numas;
    ucc_assert(n_sn_groups >= 1);

    if (topo->sbgps[UCC_SBGP_NODE].status != UCC_SBGP_ENABLED ||
        topo->sbgps[UCC_SBGP_NODE].group_size < 1) {
        /* second conditional is to suppress LINTER */
        return UCC_ERR_NOT_FOUND;
    }

    sbgps = ucc_calloc(n_sn_groups, sizeof(ucc_sbgp_t), "sn_sbgps");
    if (!sbgps) {
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < n_sn_groups; i++) {
        sbgps[i].type = type;
        sl_rank       = (n_sn_groups > 1)
                            ? ucc_ep_map_eval(sn_leaders_sbgp->map, i)
                            : ucc_ep_map_eval(topo->sbgps[UCC_SBGP_NODE].map, 0);
        status        = sbgp_create_sn(topo, &sbgps[i], sl_rank, 1);
        if (UCC_OK != status) {
            ucc_error("failed to create socket sbgp for sl_rank %d:%u", i,
                      sl_rank);
            goto error;
        }
        if (sbgps[i].rank_map) {
            sbgps[i].map =
                ucc_ep_map_from_array(&sbgps[i].rank_map, sbgps[i].group_size,
                                      ucc_subset_size(&topo->set), 1);
        }
    }
    *_sbgps  = sbgps;
    *n_sbgps = n_sn_groups;
    return UCC_OK;
error:
    ucc_free(sbgps);
    return status;
}

ucc_status_t ucc_sbgp_create_all_sockets(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                         int *n_sbgps)
{
    return ucc_sbgp_create_all_sns(topo, sbgps, n_sbgps, UCC_SBGP_SOCKET);
}

ucc_status_t ucc_sbgp_create_all_numas(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                       int *n_sbgps)
{
    return ucc_sbgp_create_all_sns(topo, sbgps, n_sbgps, UCC_SBGP_NUMA);
}
