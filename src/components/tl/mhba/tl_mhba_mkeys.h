/*
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MHBA_MKEYS_H
#define UCC_TL_MHBA_MKEYS_H

typedef struct ucc_tl_mhba_context  ucc_tl_mhba_context_t;
typedef struct ucc_tl_mhba_node     ucc_tl_mhba_node_t;
typedef struct ucc_tl_mhba_schedule     ucc_tl_mhba_schedule_t;
typedef struct ucc_tl_mhba_team     ucc_tl_mhba_team_t;
typedef struct ucc_tl_mhba_lib      ucc_tl_mhba_lib_t;
typedef struct ucc_base_lib      ucc_base_lib_t;

#define UMR_CQ_SIZE 8 //todo check

ucc_status_t ucc_tl_mhba_init_umr(ucc_tl_mhba_context_t *ctx,
                                  ucc_tl_mhba_node_t *   node);

ucc_status_t ucc_tl_mhba_init_mkeys(ucc_tl_mhba_team_t *team);

ucc_status_t ucc_tl_mhba_populate_send_recv_mkeys(ucc_tl_mhba_team_t *    team,
                                                  ucc_tl_mhba_schedule_t *req);

ucc_status_t ucc_tl_mhba_update_mkeys_entries(ucc_tl_mhba_node_t *    node,
                                              ucc_tl_mhba_schedule_t *req,
                                              ucc_tl_mhba_lib_t *     lib);

ucc_status_t ucc_tl_mhba_destroy_umr(ucc_tl_mhba_node_t *node,
		ucc_base_lib_t * lib);

ucc_status_t ucc_tl_mhba_destroy_mkeys(ucc_tl_mhba_team_t *team,
                                       int                 error_mode);
#endif
