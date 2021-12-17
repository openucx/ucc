/*
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef TL_MHBA_PD_H
#define TL_MHBA_PD_H

#include <sys/un.h>

typedef struct ucc_tl_mhba_team ucc_tl_mhba_team_t;

ucc_status_t ucc_tl_mhba_share_ctx_pd(ucc_tl_mhba_context_t *ctx,
                                      const char *       sock_path,
                                      ucc_rank_t group_size,
                                      int is_asr, int asr_sock);

ucc_status_t ucc_tl_mhba_remove_shared_ctx_pd(ucc_tl_mhba_context_t *ctx);

#endif
