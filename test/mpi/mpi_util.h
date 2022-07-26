/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef MPI_UTIL_H
#define MPI_UTIL_H
#include "test_mpi.h"


static inline MPI_Datatype ucc_dt_to_mpi(ucc_datatype_t dt) {
    switch (dt) {
    case UCC_DT_INT8:
        return MPI_INT8_T;
    case UCC_DT_UINT8:
        return MPI_UINT8_T;
    case UCC_DT_INT16:
        return MPI_INT16_T;
    case UCC_DT_UINT16:
        return MPI_UINT16_T;
    case UCC_DT_INT32:
        return MPI_INT32_T;
    case UCC_DT_UINT32:
        return MPI_UINT32_T;
    case UCC_DT_FLOAT32:
        return MPI_FLOAT;
    case UCC_DT_INT64:
        return MPI_INT64_T;
    case UCC_DT_UINT64:
        return MPI_UINT64_T;
    case UCC_DT_FLOAT64:
        return MPI_DOUBLE;
    case UCC_DT_FLOAT128:
        return MPI_LONG_DOUBLE;
    case UCC_DT_FLOAT32_COMPLEX:
        return MPI_C_FLOAT_COMPLEX;
    case UCC_DT_FLOAT64_COMPLEX:
        return MPI_C_DOUBLE_COMPLEX;
    case UCC_DT_FLOAT128_COMPLEX:
        return MPI_C_LONG_DOUBLE_COMPLEX;
    case UCC_DT_FLOAT16:
    case UCC_DT_INT128:
    case UCC_DT_UINT128:
    case UCC_DT_BFLOAT16:
    default:
        std::cerr << "Unsupported dt\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return MPI_DATATYPE_NULL;
}

static inline MPI_Op ucc_op_to_mpi(ucc_reduction_op_t op)
{
    switch(op) {
    case UCC_OP_SUM:
        return MPI_SUM;
    case UCC_OP_PROD:
        return MPI_PROD;
    case UCC_OP_MAX:
        return MPI_MAX;
    case UCC_OP_MIN:
        return MPI_MIN;
    case UCC_OP_LAND:
        return MPI_LAND;
    case UCC_OP_LOR:
        return MPI_LOR;
    case UCC_OP_LXOR:
        return MPI_LXOR;
    case UCC_OP_BAND:
        return MPI_BAND;
    case UCC_OP_BOR:
        return MPI_BOR;
    case UCC_OP_BXOR:
        return MPI_BXOR;
    case UCC_OP_MAXLOC:
        return MPI_MAXLOC;
    case UCC_OP_MINLOC:
        return MPI_MINLOC;
    default:
        std::cerr << "Unsupported op\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return MPI_OP_NULL;
}

MPI_Comm create_mpi_comm(ucc_test_mpi_team_t t);
#endif
