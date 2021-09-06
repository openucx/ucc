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
    UCC_SBGP_NUMA,
    UCC_SBGP_SOCKET,
    UCC_SBGP_NODE,
    UCC_SBGP_NODE_LEADERS,
    UCC_SBGP_NET,
    UCC_SBGP_SOCKET_LEADERS,
    UCC_SBGP_NUMA_LEADERS,
    UCC_SBGP_FLAT,
    UCC_SBGP_LAST
} ucc_sbgp_type_t;

typedef enum ucc_sbgp_status_t {
    UCC_SBGP_NOT_INIT,
    UCC_SBGP_DISABLED,
    UCC_SBGP_ENABLED,
    UCC_SBGP_NOT_EXISTS,
} ucc_sbgp_status_t;

typedef struct ucc_team      ucc_team_t;
typedef struct ucc_team_topo ucc_team_topo_t;
typedef struct ucc_sbgp_t {
    ucc_sbgp_type_t   type;
    ucc_sbgp_status_t status;
    ucc_rank_t        group_size;
    ucc_rank_t        group_rank;
    ucc_rank_t       *rank_map;
    ucc_team_t       *team;
    ucc_ep_map_t      map;
} ucc_sbgp_t;

const char*  ucc_sbgp_str(ucc_sbgp_type_t type);
ucc_status_t ucc_sbgp_create(ucc_team_topo_t *topo, ucc_sbgp_type_t type);
ucc_status_t ucc_sbgp_cleanup(ucc_sbgp_t *sbgp);

static inline int ucc_sbgp_rank2team(ucc_sbgp_t *sbgp, int rank)
{
    return ucc_ep_map_eval(sbgp->map, rank);
}

void ucc_sbgp_print(ucc_sbgp_t *sbgp);

#endif
