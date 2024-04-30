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
#include <fstream>
#include <iostream>
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

double parse_transfer_matrix_token(std::string token)
{
    size_t size;
    double val;
    try { 
        val = std::stod(token, &size);
    }
    catch (...) {
        throw std::invalid_argument("Invalid element in transfer matrix: " + token);
    }

    if (size < token.size())
    {
        switch(token[size])
        {
            case 'M':
                val *= 1e6;
                break;
            case 'G':
                val *= 1e9;
                break;
            default:
                throw std::invalid_argument("Unknown suffix from transfer matrix: " + token[size]);
        }
    }
        return val;
}


/**
* Fill a matrix using the file provided in the env var UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRIX_FILE
* The file should contain square matrices of size comm_size.
* The rows of the matrix should be lines and the elements should be separated by a single space.
* The element (i,j) represents the number of bytes rank i will send to rank j. \
* The notation support convenient unit notation for gigabytes and megabytes, e.g 3G or 10M.
* Multiple matrices is supported using the <offset> parameter, they should be separated by a single empty line.
* Be careful when using multiple matrices, if you have N matrices, set -b 1 and -e 2**N
*
* @param offset - The offset of the matrix to read, in case the file contains more than one matrix,
*       the function will read the matrix at offset <offset>, the first matrix has offset 0.
*/
void fill_transfer_matrix(std::vector<std::vector<double>>& transfer_matrix, int offset=0)
{
    std::ifstream f;
    std::string line, token;
    std::istringstream linestream;
    int col = 0;
    int N = transfer_matrix.size();
    int lines_offset = offset*(N+1);

    char* transfer_matrix_fn = std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRIX_FILE");
    f.open(transfer_matrix_fn);
    if (!f.is_open())
        throw std::invalid_argument("Couldn't open transfer matrix file: " + (transfer_matrix_fn ? std::string(transfer_matrix_fn) : ""));

    for (int i=0; i<lines_offset; i++)
        if (!getline(f, line))
            throw std::invalid_argument("Offset is " + std::to_string(offset) + " but the file contains less matrices than that");

    for (int row=0; row < N; row++){
        if (!getline(f, line))
            throw std::invalid_argument("Transfer matrix is expected to have " + std::to_string(N) + " rows but only have " + std::to_string(row+1));

        if (row >= N)
            throw std::invalid_argument("Transfer matrix rows number exceed expected number of " + std::to_string(N));

        linestream.str(line);
        linestream.clear();
        col = 0;
        while (linestream >> token){
            if (col >= N)
                throw std::invalid_argument("Transfer matrix columns of row " + std::to_string(row+1) + " exceed expected number of " + std::to_string(N));

            transfer_matrix[row][col] = parse_transfer_matrix_token(token);
            col++;
        }

        if (col != N)
            throw std::invalid_argument("Transfer matrix row " + std::to_string(row+1) + " doesn't contain " + std::to_string(N) + " elements as expected.");
    }
}


ucc_status_t ucc_pt_coll_alltoallv::init_args(size_t count,
                                              ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t                     &args      = test_args.coll_args;
    int                                 comm_size = comm->get_size();
    int                                 comm_rank = comm->get_rank();
    size_t                              dt_size   = ucc_dt_size(coll_args.src.info_v.datatype);
    std::vector<std::vector<double>>    transfer_matrix(comm_size, std::vector<double>(comm_size, count*dt_size));
    size_t                              dst_header_size, src_header_size;
    ucc_status_t                        st        = UCC_OK;
    int                                 src_displacement = 0;
    int                                 dst_displacement = 0;
    int                                 send_count, recv_count;

    if (std::getenv("UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRIX_FILE")){
        fill_transfer_matrix(transfer_matrix, test_args.iter);
    }

    src_header_size = dst_header_size = 0;
    for (size_t i=0; i < transfer_matrix.size(); i++){
        src_header_size += transfer_matrix[comm_rank][i];
    }

    for (size_t i=0; i < transfer_matrix.size(); i++){
        dst_header_size += transfer_matrix[i][comm_rank];
    }

    args = coll_args;
    args.src.info_v.counts = (ucc_count_t *) ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK_GOTO(args.src.info_v.counts, exit, st);
    args.src.info_v.displacements = (ucc_aint_t *) ucc_malloc(comm_size * sizeof(uint32_t), "displacements buf");
    UCC_MALLOC_CHECK_GOTO(args.src.info_v.displacements, free_src_count, st);
    args.dst.info_v.counts = (ucc_count_t *) ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK_GOTO(args.dst.info_v.counts, free_src_displ, st);
    args.dst.info_v.displacements = (ucc_aint_t *) ucc_malloc(comm_size * sizeof(uint32_t), "displacements buf");
    UCC_MALLOC_CHECK_GOTO(args.dst.info_v.displacements, free_dst_count, st);
    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, dst_header_size, args.dst.info_v.mem_type),
                  free_dst_displ, st);
    args.dst.info_v.buffer = dst_header->addr;
    if (!UCC_IS_INPLACE(args)) {
        UCCCHECK_GOTO(ucc_pt_alloc(&src_header, src_header_size, args.src.info_v.mem_type),
                      free_dst, st);
        args.src.info_v.buffer = src_header->addr;
    }

    for (int i = 0; i < comm_size; i++) {
        send_count = std::floor(transfer_matrix[comm_rank][i] / dt_size);
        recv_count = std::floor(transfer_matrix[i][comm_rank] / dt_size);

        ((uint32_t*)args.src.info_v.counts)[i] = send_count;
        ((uint32_t*)args.src.info_v.displacements)[i] = src_displacement;
        ((uint32_t*)args.dst.info_v.counts)[i] = recv_count;
        ((uint32_t*)args.dst.info_v.displacements)[i] = dst_displacement;

        src_displacement += send_count;
        dst_displacement += recv_count;
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
