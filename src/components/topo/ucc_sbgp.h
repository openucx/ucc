/**
 * Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#ifndef UCC_SBGP_H_
#define UCC_SBGP_H_
#include "ucc/api/ucc.h"
#include "utils/ucc_datastruct.h"
#include "utils/ucc_coll_utils.h"

typedef enum ucc_sbgp_type_t
{
    UCC_SBGP_NUMA,               /* Group of ranks on the same NUMA domain.
                                    This group does not exist if processes are
                                    not bound to a single NUMA node. */
    UCC_SBGP_SOCKET,             /* Group of ranks on the same SOCKET.
                                    This group does not exist if processes are
                                    not bound to a single SOCKET node. */
    UCC_SBGP_NODE,               /* Group of ranks on the same NODE. */
    UCC_SBGP_NODE_LEADERS,       /* Group of ranks with local_node_rank = 0.
                                    This group EXISTS when team spans at least 2
                                    nodes. This group is ENABLED for procs with
                                    local_node_rank = 0. This group is DISABLED but
                                    EXISTS for procs with local_node_rank != 0*/
    UCC_SBGP_NET,                /* Group of ranks with the same local_node_rank.
                                    This group EXISTS when team spans at least 2
                                    nodes AND the team has equal PPN across all the
                                    nodes. If EXISTS this group is ENABLED for all
                                    procs. */
    UCC_SBGP_SOCKET_LEADERS,     /* Group of ranks with local_socket_rank = 0.
                                    This group EXISTS when team spans at least 2
                                    sockets. This group is ENABLED for procs with
                                    local_socket_rank = 0. This group is DISABLED
                                    but EXISTS for procs with local_socket_rank != 0 */
    UCC_SBGP_NUMA_LEADERS,       /* Same as SOCKET_LEADERS but for NUMA grouping */
    UCC_SBGP_FULL,               /* Group contains ALL the ranks of the team */
    UCC_SBGP_FULL_HOST_ORDERED,  /* Group contains ALL the ranks of the team ordered
                                    by host, socket, numa */
    UCC_SBGP_LAST
} ucc_sbgp_type_t;

typedef enum ucc_sbgp_status_t
{
    UCC_SBGP_NOT_INIT,
    UCC_SBGP_DISABLED,
    UCC_SBGP_ENABLED,
    UCC_SBGP_NOT_EXISTS,
} ucc_sbgp_status_t;

typedef struct ucc_topo ucc_topo_t;
typedef struct ucc_sbgp_t {
    ucc_sbgp_type_t   type;
    ucc_sbgp_status_t status;
    ucc_rank_t        group_size;
    ucc_rank_t        group_rank;
    ucc_rank_t       *rank_map;
    ucc_ep_map_t      map;
} ucc_sbgp_t;

const char* ucc_sbgp_str(ucc_sbgp_type_t type);

/* The call creates a required subgroup specified by @in type in
   the topo->sbgps[type]. The created sbgp can be in either of 3 states:
   - NOT_EXISTS: means for a given topo (ucc team layout) there is no such
     grouping or the result group is of size 1. Example: 1) type == SBGP_SOCKET
     but processes are not bound to sockets at all; 2) type == SBGP_NODE_LEADERS
     but team is entirely on single node.
   - ENABLED: means the sbgp size >= 2 and calling process is part of that subgroup
   - DISABLED: means the subgrouping exists for the given ucc team (topo) but
     the calling process is NOT part of it.

     Note: this function returns subgroup LOCAL to calling process when multiple
     groups of the same time exist for a given topo. E.g., when team spans several
     sockets there exist several socket subgroups but the function below will
     initialize the sbgp which belongs to the calling process. */
ucc_status_t ucc_sbgp_create(ucc_topo_t *topo, ucc_sbgp_type_t type);

ucc_status_t ucc_sbgp_cleanup(ucc_sbgp_t *sbgp);

/* Returns ALL existing socket subgroups on the node of the calling process.
   If processes are not bound UCC_ERR_NOT_FOUND is returned.
   Also returns subgroups of size 1 in contrast to ucc_sbgp_create. */
ucc_status_t ucc_sbgp_create_all_sockets(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                         int *n_sbgps);

ucc_status_t ucc_sbgp_create_all_numas(ucc_topo_t *topo, ucc_sbgp_t **sbgps,
                                       int *n_sbgps);

ucc_status_t ucc_sbgp_create_node(ucc_topo_t *topo, ucc_sbgp_t *sbgp);

static inline ucc_subset_t ucc_sbgp_to_subset(ucc_sbgp_t *sbgp)
{
    ucc_subset_t s = {
        .map    = sbgp->map,
        .myrank = sbgp->group_rank
    };
    return s;
}

void ucc_sbgp_print(ucc_sbgp_t *sbgp);
#endif
