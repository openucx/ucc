/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>
#include <string>
#include <ifstream>
#include <vector>


ucc_pt_coll_alltoallv::ucc_pt_coll_alltoallv(ucc_datatype_t dt,
                         ucc_memory_type mt, bool is_inplace,
                         bool is_persistent,
                         ucc_pt_comm *communicator) : ucc_pt_coll(communicator)
{
    has_inplace_   = true;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = false;
    root_shift_    = 0;

    coll_args.mask                = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_args.coll_type           = UCC_COLL_TYPE_ALLTOALLV;
    coll_args.src.info_v.datatype = dt;
    coll_args.src.info_v.mem_type = mt;
    coll_args.dst.info_v.datatype = dt;
    coll_args.dst.info_v.mem_type = mt;
    coll_args.flags               = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                                    UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
    if (is_inplace) {
        coll_args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    if (is_persistent) {
        coll_args.flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
    }

}

double parse_transfer_matrix_token(string token)
{
    size_t size;
    double val;
    std::stod(token, &size);
    if (size < token.size())
    {
        switch(s[size])
        {
            case 'M':
                val *= 1e6;
                break;
            case 'G':
                val *= 1e9;
                break;
            default:
                throw std::runtime_error("Unknown suffix from transfer matrix: " + s[size]);
        }
    }
}

std::vector<std::vector<double>> ucc_pt_coll_alltoallv::fill_transfer_matrix(std::vector<std::vector<double>>& transfer_matrix)
{
    ifstream f;
    string line, token;
    double val;
    int row = col = 0;
    int N = transfer_matrix.size();
    
    f.open(std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRIX_FILE"));
    if (!f.is_open())
        throw std::runtime_error(std::format("Couldn't open transfer matrix file: {}", std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRIX_FILE")));

    while (getline(f, line)){
        if (row >= N)
            throw std::runtime_error(std::format("Transfer matrix rows number exceed expected number of {}", N));

        col = 0;
        while (line >> token){
            if (col >= N)
                throw std::runtime_error(std::format("Transfer matrix columns of row {} exceed expected number of {}", row, N));
            transfer_matrix[row][col++] = parse_transfer_matrix_token(token);
        }

        if (col != N-1)
            throw std::runtime_error(std::format("Transfer matrix row {} doesn't contain {} elements as expected.", row+1, N));

        row++;
    }

    if (row != N-1)
        throw std::runtime_error(std::format("Transfer matrix is expected to have {} rows but only have {}.", N, row+1));

    return transfer_matrix;
}


ucc_status_t ucc_pt_coll_alltoallv::init_args(size_t count,
                                              ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t                     &args      = test_args.coll_args;
    int                                 comm_size = comm->get_size();
    int                                 comm_rank = comm->get_rank();
    size_t                              dt_size   = ucc_dt_size(coll_args.src.info_v.datatype);
    size_t                              size      = comm_size * count * dt_size;
    ucc_status_t                        st        = UCC_OK;
    int                                 src_displacement = dst_displacement = 0;
    std::vector<std::vector<double>>    transfer_matrix(comm_size, std::vector<double>(comm_size, 0));

    if (std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRIX_FILE"))
        fill_transfer_matrix(transfer_matrix);

    args = coll_args;
    args.src.info_v.counts = (ucc_count_t *) ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK_GOTO(args.src.info_v.counts, exit, st);
    args.src.info_v.displacements = (ucc_aint_t *) ucc_malloc(comm_size * sizeof(uint32_t), "displacements buf");
    UCC_MALLOC_CHECK_GOTO(args.src.info_v.displacements, free_src_count, st);
    args.dst.info_v.counts = (ucc_count_t *) ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK_GOTO(args.dst.info_v.counts, free_src_displ, st);
    args.dst.info_v.displacements = (ucc_aint_t *) ucc_malloc(comm_size * sizeof(uint32_t), "displacements buf");
    UCC_MALLOC_CHECK_GOTO(args.dst.info_v.displacements, free_dst_count, st);
    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, size, args.dst.info_v.mem_type),
                  free_dst_displ, st);
    args.dst.info_v.buffer = dst_header->addr;
    if (!UCC_IS_INPLACE(args)) {
        UCCCHECK_GOTO(ucc_pt_alloc(&src_header, size, args.src.info_v.mem_type),
                      free_dst, st);
        args.src.info_v.buffer = src_header->addr;
    }

    for (int i = 0; i < comm_size; i++) {
        ((uint32_t*)args.src.info_v.counts)[i] = transfer_matrix[comm_rank][i];
        ((uint32_t*)args.src.info_v.displacements)[i] = src_displacement;
        ((uint32_t*)args.dst.info_v.counts)[i] = transfer_matrix[i][comm_rank];
        ((uint32_t*)args.dst.info_v.displacements)[i] = dst_displacement;

        src_displacement += transfer_matrix[comm_rank][i];
        dst_displacement += transfer_matrix[i][comm_rank];
    }
    return UCC_OK;
free_dst:
    ucc_pt_free(dst_header);
free_dst_displ:
    ucc_free(args.dst.info_v.displacements);
free_dst_count:
    ucc_free(args.dst.info_v.counts);
free_src_displ:
    ucc_free(args.src.info_v.displacements);
free_src_count:
    ucc_free(args.src.info_v.counts);
exit:
    return st;
}

void ucc_pt_coll_alltoallv::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;

    if (!UCC_IS_INPLACE(args)) {
        ucc_pt_free(src_header);
    }
    ucc_pt_free(dst_header);
    ucc_free(args.dst.info_v.counts);
    ucc_free(args.dst.info_v.displacements);
    ucc_free(args.src.info_v.counts);
    ucc_free(args.src.info_v.displacements);
}
