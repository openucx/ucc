#ifndef TEST_ALLREDUCE_SW_H
#define TEST_ALLREDUCE_SW_H

#include "common/test_ucc.h"

#ifdef HAVE_UCX

#include <ucp/api/ucp.h>

typedef struct global_work_buf_info {
    void *packed_src_memh;
    void *packed_dst_memh;
} global_work_buf_info;

struct export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void *        packed_memh;
    size_t        packed_memh_len;
    uint64_t      memh_id;
};

typedef struct test_ucp_info_t {
    ucp_context_h     ucp_ctx;
    ucp_config_t     *ucp_config;
    struct export_buf src_ebuf;
    struct export_buf dst_ebuf;
} test_ucp_info_t;

void free_gwbi(int n_procs, UccCollCtxVec &ctxs, test_ucp_info_t *ucp_infos,
               bool inplace);
ucs_status_t setup_gwbi(int n_procs, UccCollCtxVec &ctxs,
                test_ucp_info_t **ucp_infos_p /* out */, bool inplace);

#endif
#endif
