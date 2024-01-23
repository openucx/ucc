#ifndef TEST_ALLREDUCE_SW_H
#define TEST_ALLREDUCE_SW_H

#include "components/tl/ucp/allreduce/allreduce.h"

struct export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void *        packed_memh;
    size_t        packed_memh_len;
    uint64_t      memh_id;
};

typedef struct ucp_info {
    ucp_context_h     ucp_ctx;
    struct export_buf src_ebuf;
    struct export_buf dst_ebuf;
} ucp_info_t;

void free_gwbi(int n_procs, UccCollCtxVec &ctxs, ucp_info_t *ucp_infos,
               bool inplace);
void setup_gwbi(int n_procs, UccCollCtxVec &ctxs,
                ucp_info_t **ucp_infos_p /* out */, bool inplace);
int  buffer_export_ucc(ucp_context_h ucp_context, void *buf, size_t len,
                       struct export_buf *ebuf);
void ep_err_cb(void *arg, ucp_ep_h ep, ucs_status_t ucs_status);

int ucp_init_ex(ucp_context_h *ucp_ctx);

#endif
