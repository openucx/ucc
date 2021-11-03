/*
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_TOPO_H_
#define UCC_TOPO_H_
#include "ucc_sbgp.h"
#include "utils/ucc_proc_info.h"

typedef struct ucc_context_topo {
    ucc_proc_info_t *procs;
    ucc_rank_t       n_procs;
    ucc_rank_t       nnodes;
    ucc_rank_t       min_ppn;
    ucc_rank_t       max_ppn;
    ucc_rank_t       max_n_sockets;
    uint32_t         sock_bound;
} ucc_context_topo_t;

typedef struct ucc_addr_storage ucc_addr_storage_t;
typedef struct ucc_topo {
    ucc_context_topo_t   *topo;
    ucc_sbgp_t    sbgps[UCC_SBGP_LAST];
    ucc_sbgp_t   *all_sockets;
    int           n_sockets;
    ucc_rank_t    node_leader_rank;
    ucc_rank_t    node_leader_rank_id;
    ucc_subset_t  set;
    ucc_rank_t    min_ppn;
    ucc_rank_t    max_ppn;
} ucc_topo_t;

ucc_status_t ucc_context_topo_init(ucc_addr_storage_t *storage,
                                   ucc_context_topo_t **topo);
void         ucc_context_topo_cleanup(ucc_context_topo_t *topo);

ucc_status_t ucc_topo_init(ucc_subset_t set, ucc_context_topo_t *topo,
                                  ucc_topo_t **subset_topo);
void         ucc_topo_cleanup(ucc_topo_t *subset_topo);
ucc_sbgp_t *ucc_topo_get_sbgp(ucc_topo_t *topo, ucc_sbgp_type_t type);
int         ucc_topo_is_single_node(ucc_topo_t *topo);
ucc_status_t ucc_topo_get_all_sockets(ucc_topo_t *topo, ucc_sbgp_t **sbgps, int *n_sbgps);
#endif
