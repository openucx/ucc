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

#include <ucc/api/ucc.h>
#include "ucc_pt_comm.h"
#include "ucc_pt_config.h"
#include "ucc_pt_coll.h"
#include "ucc_pt_cuda.h"
#include "ucc_pt_rocm.h"
#include "ucc_pt_benchmark.h"

int main(int argc, char *argv[])
{
    ucc_pt_config pt_config;
    ucc_pt_comm *comm;
    ucc_pt_benchmark *bench;
    ucc_status_t st;

    pt_config.process_args(argc, argv);
    ucc_pt_cuda_init();
    ucc_pt_rocm_init();
    try {
        comm = new ucc_pt_comm(pt_config.comm);
    } catch(std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::exit(1);
    }
    st = comm->init();
    if (st != UCC_OK) {
        delete comm;
        std::exit(1);
    }
    try {
        bench = new ucc_pt_benchmark(pt_config.bench, comm);
    } catch(std::exception &e) {
        std::cerr << e.what() << std::endl;
        comm->finalize();
        delete comm;
        std::exit(1);
    }
    bench->run_bench();
    delete bench;
    comm->finalize();
    delete comm;
    return 0;
}
