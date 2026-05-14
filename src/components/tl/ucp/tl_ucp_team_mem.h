/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_TEAM_MEM_H_
#define UCC_TL_UCP_TEAM_MEM_H_

#include "tl_ucp.h"

/* Exchange packed_key_len values to find max key size across all ranks.
 * Registers local segments and posts size-only allgather. Sets mem_map_phase=1. */
ucc_status_t ucc_tl_ucp_team_mem_map_size_exch(ucc_tl_ucp_team_t *team);

/* Exchange actual memory keys and metadata after determining max key size.
 * Reads phase-1 allgather results, posts full data allgather. Sets mem_map_phase=2. */
ucc_status_t ucc_tl_ucp_team_mem_map_data_exch(ucc_tl_ucp_team_t *team);

/* Called once data exchange task completes. Unpacks all remote rkeys, populates
 * team_remote_va/len, frees allgather scratch, finalizes task. */
ucc_status_t ucc_tl_ucp_team_mem_map_finalize(ucc_tl_ucp_team_t *team);

/*
 * Called from ucc_tl_ucp_team_destroy to release all team-segment resources:
 * unpacked rkeys, UCP memory handles, packed key buffers, and remote VA/len arrays.
 * Safe to call when n_mem_segs == 0.
 */
void ucc_tl_ucp_team_mem_map_destroy(ucc_tl_ucp_team_t *team);

#endif
