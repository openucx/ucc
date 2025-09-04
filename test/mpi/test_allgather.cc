/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "core/ucc_dt.h"
#include "test_mpi.h"
#include "mpi_util.h"
#include "ucc/api/ucc.h"

TestAllgather::TestAllgather(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_ALLGATHER, params)
{
    int    rank, size;
    size_t dt_size, single_rank_count;

    dt                = params.dt;
    dt_size           = ucc_dt_size(dt);
    single_rank_count = msgsize / dt_size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (msgsize*size),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize * size, mem_type));
    rbuf      = rbuf_mc_header->addr;
    check_buf = ucc_malloc(msgsize * size, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    if (!inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize, mem_type));
        sbuf                   = sbuf_mc_header->addr;
        args.src.info.buffer   = sbuf;
        args.src.info.count    = single_rank_count;
        args.src.info.datatype = dt;
        args.src.info.mem_type = mem_type;
    }
    args.dst.info.buffer   = rbuf;
    args.dst.info.count    = single_rank_count * size;
    args.dst.info.datatype = dt;
    args.dst.info.mem_type = mem_type;
    if (local_registration) {
        ucc_mem_map_t segments[1];
        ucc_mem_map_params_t mem_map_params;

        mem_map_params.n_segments          = 1;
        mem_map_params.segments            = segments;

        if (!inplace) {
            mem_map_params.segments[0].address = args.src.info.buffer;
            mem_map_params.segments[0].len     = args.src.info.count *
                                                 ucc_dt_size(args.src.info.datatype);

            UCC_CHECK(ucc_mem_map(team.ctx, UCC_MEM_MAP_MODE_EXPORT,
                                  &mem_map_params, &src_memh_size, &src_memh));
            args.src_memh.local_memh = src_memh;
            args.mask |= UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH;
        }

        mem_map_params.segments[0].address = args.dst.info.buffer;
        mem_map_params.segments[0].len     = args.dst.info.count *
                                             ucc_dt_size(args.dst.info.datatype);
        UCC_CHECK(ucc_mem_map(team.ctx, UCC_MEM_MAP_MODE_EXPORT,
                              &mem_map_params, &dst_memh_size, &dst_memh));
        args.dst_memh.local_memh = dst_memh;
        args.mask |= UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;
    }
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAllgather::set_input(int iter_persistent)
{
    size_t dt_size           = ucc_dt_size(dt);
    size_t single_rank_count = msgsize / dt_size;
    size_t single_rank_size  = single_rank_count * dt_size;
    int    rank;
    void  *buf;

    this->iter_persistent = iter_persistent;
    MPI_Comm_rank(team.comm, &rank);
    if (inplace) {
        buf   = PTR_OFFSET(rbuf, rank * single_rank_size);
    } else {
        buf   = sbuf;
    }

    init_buffer(buf, single_rank_count, dt, mem_type,
                rank * (iter_persistent + 1));
    return UCC_OK;
}

ucc_status_t TestAllgather::check()
{
    size_t dt_size, single_rank_count;
    int    size, i;

    MPI_Comm_size(team.comm, &size);
    single_rank_count = args.dst.info.count / size;
    dt_size = ucc_dt_size(dt);
    for (i = 0; i < size; i++) {
        init_buffer(PTR_OFFSET(check_buf, i * single_rank_count * dt_size),
                    single_rank_count, dt, UCC_MEMORY_TYPE_HOST,
                    i * (iter_persistent + 1));
    }

    return compare_buffers(rbuf, check_buf, single_rank_count * size, dt,
                           mem_type);
}
