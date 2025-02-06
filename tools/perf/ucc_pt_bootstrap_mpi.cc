/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucc_pt_bootstrap_mpi.h"

static ucc_status_t mpi_oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                      void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)(uintptr_t)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)(uintptr_t)request;
    return UCC_OK;
}

static ucc_status_t mpi_oob_allgather_test(void *req)
{
    MPI_Request request = (MPI_Request)(uintptr_t)req;
    int         completed;
    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t mpi_oob_allgather_free(void *req)
{
    return UCC_OK;
}

ucc_pt_bootstrap_mpi::ucc_pt_bootstrap_mpi()
{
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    context_oob.coll_info = (void*)(uintptr_t)MPI_COMM_WORLD;
    context_oob.allgather = mpi_oob_allgather;
    context_oob.req_test  = mpi_oob_allgather_test;
    context_oob.req_free  = mpi_oob_allgather_free;
    context_oob.n_oob_eps = size;
    context_oob.oob_ep    = rank;

    team_oob.coll_info = (void*)(uintptr_t)MPI_COMM_WORLD;
    team_oob.allgather = mpi_oob_allgather;
    team_oob.req_test  = mpi_oob_allgather_test;
    team_oob.req_free  = mpi_oob_allgather_free;
    team_oob.n_oob_eps = size;
    team_oob.oob_ep    = rank;
}

int ucc_pt_bootstrap_mpi::get_rank()
{
    return rank;
}

int ucc_pt_bootstrap_mpi::get_size()
{
    return size;
}

ucc_pt_bootstrap_mpi::~ucc_pt_bootstrap_mpi()
{
    MPI_Finalize();
}
