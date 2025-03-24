/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#ifndef UCC_TOPO_H_
#define UCC_TOPO_H_
#include "ucc_sbgp.h"
#include "utils/ucc_proc_info.h"

/* Topo data structure initialized per UCC context.
   key element of this struct is the array of ucc_proc_info_t
   that is constructed during ucc_context_topo_init using
   ucc_addr_storage_t. In other words the topo struct can
   be initialized after the exchange of addresses is performed. */

typedef struct ucc_context_topo {
    ucc_proc_info_t *procs;
    ucc_rank_t       n_procs;
    ucc_rank_t       nnodes;
    ucc_rank_t       min_ppn;       /*< smallest ppn value across the nodes
                                        spanned by the group of processes
                                        defined by ucc_addr_storage_t */
    ucc_rank_t       max_ppn;       /*< biggest ppn across the nodes */
    ucc_rank_t       max_n_sockets; /*< max number of different sockets
                                        on a node */
    uint32_t         sock_bound;    /*< global flag, 1 if processes are bound
                                        to sockets */
    ucc_rank_t       max_n_numas;   /*< max number of different numa domains
                                        on a node */
    uint32_t         numa_bound;    /*< global flag, 1 if processes are bound
                                        to numa nodes */
} ucc_context_topo_t;

typedef struct ucc_addr_storage ucc_addr_storage_t;

/* This topo structure is initialized over a SUBSET of processes
   from ucc_context_topo_t.

   For example, if ucc_context_t is global then address exchange
   is performed during ucc_context_create and we have ctx wide
   ucc_addr_storage_t. So, we init ucc_context_topo_t on ucc_context.

   Then, ucc_team is a subset of ucc_context mapped via team->ctx_map.
   It represents a subset of ranks and we can initialize ucc_topo_t
   for that subset, ie for a team. */
typedef struct ucc_topo {
    ucc_context_topo_t *topo;         /*< Cached pointer of the ctx topo */
    ucc_sbgp_t  sbgps[UCC_SBGP_LAST]; /*< LOCAL sbgps initialized on demand */
    ucc_sbgp_t *all_sockets;          /*< array of socket sbgps, init on demand */
    int         n_sockets;
    ucc_sbgp_t *all_numas;            /*< array of numa sbgps, init on demand */
    int         n_numas;
    ucc_sbgp_t *all_nodes;            /*< array of node sbgps, init on demand */
    int         n_nodes;
    ucc_rank_t  node_leader_rank_id;  /*< defines which rank on a node will be
                                          node leader. Similar to local node rank.
                                          currently set to 0, can be selected differently
                                          in the future */
    ucc_rank_t   node_leader_rank;    /*< actual rank of the node leader in the original
                                          (ucc_team) ranking */
    ucc_rank_t  *node_leaders;        /*< array mapping each rank to its node leader in the original
                                          (ucc_team) ranking, initialized on demand */
    ucc_subset_t set;     /*< subset of procs from the ucc_context_topo.
                         for ucc_team topo it is team->ctx_map */
    ucc_rank_t   min_ppn; /*< min ppn across the nodes for a team */
    ucc_rank_t   max_ppn; /*< max ppn across the nodes for a team */
    ucc_rank_t   min_socket_size; /*< min number of processes on a socket,
                                      across all nodes of a team */
    ucc_rank_t   max_socket_size; /*< max number of processes on a socket,
                                      across all nodes of a team */
    ucc_rank_t   min_numa_size; /*< min number of processes on a numa,
                                    across all nodes of a team */
    ucc_rank_t   max_numa_size; /*< max number of processes on a numa,
                                    across all nodes of a team */
} ucc_topo_t;

/* Initializes ctx level topo structure using addr_storage.
   Each address contains ucc_proc_info_t which is extracted and placed
   into array for each participating proc. The array is then sorted
   (see ucc_compare_proc_info in ucc_topo.c) that allows O(N) local
   subgroup discoveries */
ucc_status_t ucc_context_topo_init(ucc_addr_storage_t * storage,
                                   ucc_context_topo_t **topo);
void         ucc_context_topo_cleanup(ucc_context_topo_t *topo);

/* Initializes topo structure for a subset, e.g. for a team */
ucc_status_t ucc_topo_init(ucc_subset_t set, ucc_context_topo_t *topo,
                           ucc_topo_t **subset_topo);
void         ucc_topo_cleanup(ucc_topo_t *subset_topo);

ucc_sbgp_t *ucc_topo_get_sbgp(ucc_topo_t *topo, ucc_sbgp_type_t type);

int ucc_topo_is_single_node(ucc_topo_t *topo);
/* Returns the array of ALL existing socket subgroups of given topo */
ucc_status_t ucc_topo_get_all_sockets(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                      int *n_sbgps);

/* Returns the array of ALL existing numa subgroups of given topo */
ucc_status_t ucc_topo_get_all_numas(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                    int *n_sbgps);

/* Returns the array of ALL existing node subgroups of given topo */
ucc_status_t ucc_topo_get_all_nodes(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                    int *n_sbgps);

static inline int ucc_rank_on_local_node(ucc_rank_t team_rank, ucc_topo_t *topo)
{
    ucc_proc_info_t *procs    = topo->topo->procs;
    ucc_rank_t       ctx_rank = ucc_ep_map_eval(topo->set.map, team_rank);
    ucc_rank_t my_ctx_rank = ucc_ep_map_eval(topo->set.map, topo->set.myrank);

    return procs[ctx_rank].host_hash == procs[my_ctx_rank].host_hash;
}

/* Returns min ppn value across the nodes */
static inline ucc_rank_t ucc_topo_min_ppn(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);

    if (sbgp->status == UCC_SBGP_NOT_EXISTS) {
        ucc_assert(ucc_topo_is_single_node(topo));
        return ucc_subset_size(&topo->set);
    }
    return topo->min_ppn;
}

/* Returns max ppn value across the nodes */
static inline ucc_rank_t ucc_topo_max_ppn(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);

    if (sbgp->status == UCC_SBGP_NOT_EXISTS) {
        ucc_assert(ucc_topo_is_single_node(topo));
        return ucc_subset_size(&topo->set);
    }
    return topo->max_ppn;
}

/* Returns true if PPN is the same across all the nodes */
static inline int ucc_topo_isoppn(ucc_topo_t *topo)
{
    return ucc_topo_max_ppn(topo) == ucc_topo_min_ppn(topo);
}

/* Returns min socket size across the nodes.
 * If not set will return UCC_RANK_MAX,
 * in case of error will return UCC_RANK_INVALID.
 */
static inline ucc_rank_t ucc_topo_min_socket_size(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp;

    ucc_assert(topo->topo->sock_bound);
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);

    if (sbgp->status == UCC_SBGP_NOT_INIT) {
        return UCC_RANK_INVALID;
    }

    return topo->min_socket_size;
}

/* Returns max socket size across the nodes.
 * If not set will return 0,
 * in case of error will return UCC_RANK_INVALID.
 */
static inline ucc_rank_t ucc_topo_max_socket_size(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp;

    ucc_assert(topo->topo->sock_bound);
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);

    if (sbgp->status == UCC_SBGP_NOT_INIT) {
        return UCC_RANK_INVALID;
    }

    return topo->max_socket_size;
}

/* Returns min numa size across the nodes.
 * If not set will return UCC_RANK_MAX,
 * in case of error will return UCC_RANK_INVALID.
 */
static inline ucc_rank_t ucc_topo_min_numa_size(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp;

    ucc_assert(topo->topo->numa_bound);
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);

    if (sbgp->status == UCC_SBGP_NOT_INIT) {
        return UCC_RANK_INVALID;
    }

    return topo->min_numa_size;
}

/* Returns max numa size across the nodes.
 * If not set will return 0,
 * in case of error will return UCC_RANK_INVALID.
 */
static inline ucc_rank_t ucc_topo_max_numa_size(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp;

    ucc_assert(topo->topo->numa_bound);
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);

    if (sbgp->status == UCC_SBGP_NOT_INIT) {
        return UCC_RANK_INVALID;
    }

    return topo->max_numa_size;
}

static inline int ucc_topo_n_sockets(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp;

    if (!topo->topo->sock_bound) {
        return 0;
    }
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET_LEADERS);
    if (sbgp->status == UCC_SBGP_NOT_EXISTS) {
        return 1;
    }
    return sbgp->group_size;
}

static inline int ucc_topo_n_numas(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp;

    if (!topo->topo->numa_bound) {
        return 0;
    }
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NUMA_LEADERS);
    if (sbgp->status == UCC_SBGP_NOT_EXISTS) {
        return 1;
    }
    return sbgp->group_size;
}

static inline ucc_rank_t ucc_topo_nnodes(ucc_topo_t *topo)
{
    ucc_sbgp_t *sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);

    if (sbgp->status == UCC_SBGP_NOT_EXISTS) {
        ucc_assert(ucc_topo_is_single_node(topo));
        return 1;
    }
    return sbgp->group_size;
}

/* Returns an array mapping each rank to its node leader.
   The array is cached in topo->node_leaders. */
ucc_status_t ucc_topo_get_node_leaders(ucc_topo_t *topo,
                                       ucc_rank_t **node_leaders_out);

#endif
