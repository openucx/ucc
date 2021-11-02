/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_SBGP_H_
#define UCC_SBGP_H_
#include "ucc/api/ucc.h"
#include "utils/ucc_datastruct.h"
#include "utils/ucc_coll_utils.h"

typedef enum ucc_sbgp_type_t {
    UCC_SBGP_NUMA,           /* Group of ranks on the same NUMA domain.
                                This group does not exist if processes are
                                not bound to a single NUMA node. */
    UCC_SBGP_SOCKET,         /* Group of ranks on the same SOCKET.
                                This group does not exist if processes are
                                not bound to a single SOCKET node. */
    UCC_SBGP_NODE,           /* Group of ranks on the same NODE. */
    UCC_SBGP_NODE_LEADERS,   /* Group of ranks with local_node_rank = 0.
                                This group EXISTS when team spans at least 2
                                nodes. This group is ENABLED for procs with
                                local_node_rank = 0. This group is DISABLED but
                                EXISTS for procs with local_node_rank != 0*/
    UCC_SBGP_NET,            /* Group of ranks with the same local_node_rank.
                                This group EXISTS when team spans at least 2
                                nodes AND the team has equal PPN across all the
                                nodes. If EXISTS this group is ENABLED for all
                                procs. */
    UCC_SBGP_SOCKET_LEADERS, /* Group of ranks with local_socket_rank = 0.
                                This group EXISTS when team spans at least 2
                                sockets. This group is ENABLED for procs with
                                local_socket_rank = 0. This group is DISABLED
                                but EXISTS for procs with local_socket_rank != 0 */
    UCC_SBGP_NUMA_LEADERS,   /* Same as SOCKET_LEADERS but for NUMA grouping */
    UCC_SBGP_FLAT,           /* Group contains ALL the ranks of the team */
    UCC_SBGP_LAST
} ucc_sbgp_type_t;

typedef enum ucc_sbgp_status_t {
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

const char*  ucc_sbgp_str(ucc_sbgp_type_t type);
ucc_status_t ucc_sbgp_create(ucc_topo_t *topo, ucc_sbgp_type_t type);
ucc_status_t ucc_sbgp_cleanup(ucc_sbgp_t *sbgp);

static inline int ucc_sbgp_rank2team(ucc_sbgp_t *sbgp, int rank)
{
    return ucc_ep_map_eval(sbgp->map, rank);
}

ucc_status_t ucc_sbgp_create_all_sockets(ucc_topo_t *topo, ucc_sbgp_t **sbgps);

void ucc_sbgp_print(ucc_sbgp_t *sbgp);
#endif
