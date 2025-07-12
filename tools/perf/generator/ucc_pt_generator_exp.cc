#include "ucc_pt_generator.h"

ucc_pt_generator_exponential::ucc_pt_generator_exponential(size_t min, size_t max,
                                                           size_t factor,
                                                           uint32_t gsize,
                                                           ucc_pt_op_type_t type)
{
    min_count = min;
    max_count = max;
    mult_factor = factor;
    current_count = min;
    comm_size = gsize;
    op_type = type;
}

bool ucc_pt_generator_exponential::has_next()
{
    return current_count <= max_count;
}

void ucc_pt_generator_exponential::next()
{
    if (!has_next()) {
        return;
    }
    current_count *= mult_factor;
}

void ucc_pt_generator_exponential::reset()
{
    current_count = min_count;
}

size_t ucc_pt_generator_exponential::get_src_count()
{
    switch (op_type) {
    case UCC_PT_OP_TYPE_ALLGATHER:
    case UCC_PT_OP_TYPE_ALLGATHERV:
    case UCC_PT_OP_TYPE_ALLREDUCE:
    case UCC_PT_OP_TYPE_BCAST:
    case UCC_PT_OP_TYPE_GATHER:
    case UCC_PT_OP_TYPE_GATHERV:
    case UCC_PT_OP_TYPE_REDUCE:
    case UCC_PT_OP_TYPE_MEMCPY:
    case UCC_PT_OP_TYPE_REDUCEDT:
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
        return current_count;
    case UCC_PT_OP_TYPE_ALLTOALL:
    case UCC_PT_OP_TYPE_ALLTOALLV:
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
    case UCC_PT_OP_TYPE_SCATTER:
    case UCC_PT_OP_TYPE_SCATTERV:
        return current_count * comm_size;
    case UCC_PT_OP_TYPE_BARRIER:
    case UCC_PT_OP_TYPE_FANIN:
    case UCC_PT_OP_TYPE_FANOUT:
        return 0;
    default:
        throw std::runtime_error("Operation type not supported");
    }
}

size_t ucc_pt_generator_exponential::get_dst_count()
{
    switch (op_type) {
    case UCC_PT_OP_TYPE_ALLGATHER:
    case UCC_PT_OP_TYPE_ALLGATHERV:
    case UCC_PT_OP_TYPE_GATHER:
    case UCC_PT_OP_TYPE_GATHERV:
    case UCC_PT_OP_TYPE_ALLTOALL:
    case UCC_PT_OP_TYPE_ALLTOALLV:
        return current_count * comm_size;
    case UCC_PT_OP_TYPE_ALLREDUCE:
    case UCC_PT_OP_TYPE_REDUCE:
    case UCC_PT_OP_TYPE_MEMCPY:
    case UCC_PT_OP_TYPE_REDUCEDT:
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
    case UCC_PT_OP_TYPE_SCATTER:
    case UCC_PT_OP_TYPE_SCATTERV:
        return current_count;
    case UCC_PT_OP_TYPE_BARRIER:
    case UCC_PT_OP_TYPE_BCAST:
    case UCC_PT_OP_TYPE_FANIN:
    case UCC_PT_OP_TYPE_FANOUT:
        return 0;
    default:
        throw std::runtime_error("Operation type not supported");
    }
}

size_t *ucc_pt_generator_exponential::get_src_counts()
{
    switch (op_type) {
    case UCC_PT_OP_TYPE_ALLTOALLV:
    case UCC_PT_OP_TYPE_SCATTERV:
        src_counts = std::vector<uint32_t>(comm_size, current_count);
        return (ucc_count_t *)src_counts.data();
    case UCC_PT_OP_TYPE_ALLGATHER:
    case UCC_PT_OP_TYPE_ALLGATHERV:
    case UCC_PT_OP_TYPE_ALLREDUCE:
    case UCC_PT_OP_TYPE_ALLTOALL:
    case UCC_PT_OP_TYPE_BARRIER:
    case UCC_PT_OP_TYPE_BCAST:
    case UCC_PT_OP_TYPE_FANIN:
    case UCC_PT_OP_TYPE_FANOUT:
    case UCC_PT_OP_TYPE_GATHER:
    case UCC_PT_OP_TYPE_GATHERV:
    case UCC_PT_OP_TYPE_REDUCE:
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
    case UCC_PT_OP_TYPE_SCATTER:
    case UCC_PT_OP_TYPE_MEMCPY:
    case UCC_PT_OP_TYPE_REDUCEDT:
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
    default:
        throw std::runtime_error("Operation type not supported");
    }
}

size_t *ucc_pt_generator_exponential::get_src_displs()
{
    switch (op_type) {
    case UCC_PT_OP_TYPE_ALLTOALLV:
    case UCC_PT_OP_TYPE_SCATTERV:
        src_displs = std::vector<uint32_t>(comm_size, 0);
        for (size_t i = 0; i < comm_size; i++) {
            src_displs[i] = current_count * i;
        }
        return (ucc_aint_t *)src_displs.data();
    case UCC_PT_OP_TYPE_ALLGATHER:
    case UCC_PT_OP_TYPE_ALLGATHERV:
    case UCC_PT_OP_TYPE_ALLREDUCE:
    case UCC_PT_OP_TYPE_ALLTOALL:
    case UCC_PT_OP_TYPE_BARRIER:
    case UCC_PT_OP_TYPE_BCAST:
    case UCC_PT_OP_TYPE_FANIN:
    case UCC_PT_OP_TYPE_FANOUT:
    case UCC_PT_OP_TYPE_GATHER:
    case UCC_PT_OP_TYPE_GATHERV:
    case UCC_PT_OP_TYPE_REDUCE:
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
    case UCC_PT_OP_TYPE_SCATTER:
    case UCC_PT_OP_TYPE_MEMCPY:
    case UCC_PT_OP_TYPE_REDUCEDT:
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
    default:
        throw std::runtime_error("Operation type not supported");
    }
}

size_t *ucc_pt_generator_exponential::get_dst_counts()
{
    switch (op_type) {
    case UCC_PT_OP_TYPE_ALLGATHERV:
    case UCC_PT_OP_TYPE_ALLTOALLV:
    case UCC_PT_OP_TYPE_GATHERV:
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
        dst_counts = std::vector<uint32_t>(comm_size, current_count);
        return (ucc_count_t *)dst_counts.data();
    case UCC_PT_OP_TYPE_ALLGATHER:
    case UCC_PT_OP_TYPE_ALLTOALL:
    case UCC_PT_OP_TYPE_ALLREDUCE:
    case UCC_PT_OP_TYPE_BARRIER:
    case UCC_PT_OP_TYPE_BCAST:
    case UCC_PT_OP_TYPE_FANIN:
    case UCC_PT_OP_TYPE_FANOUT:
    case UCC_PT_OP_TYPE_GATHER:
    case UCC_PT_OP_TYPE_REDUCE:
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
    case UCC_PT_OP_TYPE_SCATTER:
    case UCC_PT_OP_TYPE_MEMCPY:
    case UCC_PT_OP_TYPE_REDUCEDT:
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
    case UCC_PT_OP_TYPE_SCATTERV:
    default:
        throw std::runtime_error("Operation type not supported");
    }
}

size_t *ucc_pt_generator_exponential::get_dst_displs()
{
    switch (op_type) {
    case UCC_PT_OP_TYPE_ALLGATHERV:
    case UCC_PT_OP_TYPE_ALLTOALLV:
    case UCC_PT_OP_TYPE_GATHERV:
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
        dst_displs = std::vector<uint32_t>(comm_size, 0);
        for (size_t i = 0; i < comm_size; i++) {
            dst_displs[i] = current_count * i;
        }
        return (ucc_aint_t *)dst_displs.data();
    case UCC_PT_OP_TYPE_ALLGATHER:
    case UCC_PT_OP_TYPE_ALLTOALL:
    case UCC_PT_OP_TYPE_ALLREDUCE:
    case UCC_PT_OP_TYPE_BARRIER:
    case UCC_PT_OP_TYPE_BCAST:
    case UCC_PT_OP_TYPE_FANIN:
    case UCC_PT_OP_TYPE_FANOUT:
    case UCC_PT_OP_TYPE_GATHER:
    case UCC_PT_OP_TYPE_REDUCE:
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
    case UCC_PT_OP_TYPE_SCATTER:
    case UCC_PT_OP_TYPE_MEMCPY:
    case UCC_PT_OP_TYPE_REDUCEDT:
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
    case UCC_PT_OP_TYPE_SCATTERV:
    default:
        throw std::runtime_error("Operation type not supported");
    }
}

size_t ucc_pt_generator_exponential::get_src_count_max()
{
    switch (op_type) {
    case UCC_PT_OP_TYPE_ALLGATHER:
    case UCC_PT_OP_TYPE_ALLGATHERV:
    case UCC_PT_OP_TYPE_ALLREDUCE:
    case UCC_PT_OP_TYPE_BCAST:
    case UCC_PT_OP_TYPE_GATHER:
    case UCC_PT_OP_TYPE_GATHERV:
    case UCC_PT_OP_TYPE_REDUCE:
    case UCC_PT_OP_TYPE_MEMCPY:
    case UCC_PT_OP_TYPE_REDUCEDT:
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
        return max_count;
    case UCC_PT_OP_TYPE_ALLTOALL:
    case UCC_PT_OP_TYPE_ALLTOALLV:
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
    case UCC_PT_OP_TYPE_SCATTER:
    case UCC_PT_OP_TYPE_SCATTERV:
        return max_count * comm_size;
    case UCC_PT_OP_TYPE_BARRIER:
    case UCC_PT_OP_TYPE_FANIN:
    case UCC_PT_OP_TYPE_FANOUT:
        return 0;
    default:
        throw std::runtime_error("Operation type not supported");
    }
}

size_t ucc_pt_generator_exponential::get_dst_count_max()
{
    switch (op_type) {
    case UCC_PT_OP_TYPE_ALLGATHER:
    case UCC_PT_OP_TYPE_ALLGATHERV:
    case UCC_PT_OP_TYPE_GATHER:
    case UCC_PT_OP_TYPE_GATHERV:
    case UCC_PT_OP_TYPE_ALLTOALL:
    case UCC_PT_OP_TYPE_ALLTOALLV:
        return max_count * comm_size;
    case UCC_PT_OP_TYPE_ALLREDUCE:
    case UCC_PT_OP_TYPE_REDUCE:
    case UCC_PT_OP_TYPE_MEMCPY:
    case UCC_PT_OP_TYPE_REDUCEDT:
    case UCC_PT_OP_TYPE_REDUCEDT_STRIDED:
    case UCC_PT_OP_TYPE_REDUCE_SCATTER:
    case UCC_PT_OP_TYPE_REDUCE_SCATTERV:
    case UCC_PT_OP_TYPE_SCATTER:
    case UCC_PT_OP_TYPE_SCATTERV:
        return max_count;
    case UCC_PT_OP_TYPE_BARRIER:
    case UCC_PT_OP_TYPE_BCAST:
    case UCC_PT_OP_TYPE_FANIN:
    case UCC_PT_OP_TYPE_FANOUT:
        return 0;
    default:
        throw std::runtime_error("Operation type not supported");
    }
}

size_t ucc_pt_generator_exponential::get_count_max()
{
    return std::max(get_src_count(), get_dst_count());
}