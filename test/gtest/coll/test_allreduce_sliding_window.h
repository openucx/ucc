/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
