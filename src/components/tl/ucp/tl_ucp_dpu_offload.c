/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp_dpu_offload.h"

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_register(
    ucp_context_h ucp_context, ucc_tl_ucp_team_t *tl_team,
    struct ucc_tl_ucp_allreduce_sw_export_buf *ebuf, void *packed_memh)
{
    ucp_mem_map_params_t params = {0};
    ucs_status_t         ucs_status, unmap_status;

    ebuf->ucp_context = ucp_context;

    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = packed_memh;

    ucs_status = ucp_mem_map(ucp_context, &params, &ebuf->memh);
    if (UCS_OK != ucs_status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "import using ucp_mem_map() returned error: %s",
                 ucs_status_string(ucs_status));
        return ucs_status_to_ucc_status(ucs_status);
    }

    ucs_status = ucp_rkey_pack(ucp_context, ebuf->memh, &ebuf->packed_key,
                               &ebuf->packed_key_len);
    if (UCS_OK != ucs_status) {
        unmap_status = ucp_mem_unmap(ucp_context, ebuf->memh);
        tl_error(UCC_TL_TEAM_LIB(tl_team),
            "ucp_rkey_pack() returned error: %s%s",
            ucs_status_string(ucs_status),
            unmap_status == UCS_OK ? "" : 
            ". While handling this error, unmapping the memh had an error");
        return ucs_status_to_ucc_status(ucs_status);
    }

    return UCC_OK;
}
