/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_LOG_H_
#define UCC_LOG_H_

#include "config.h"
#include "core/ucc_global_opts.h"
#include "core/ucc_dt.h"
#include "utils/ucc_mem_type.h"
#include "utils/debug/log_def.h"

/* Collective trace logger */
#define ucc_log_component_collective_trace(_level, fmt, ...)                   \
    ucc_log_component(_level, &ucc_global_config.coll_trace, fmt,              \
                      ##__VA_ARGS__)

#define ucc_coll_trace_info(_fmt, ...)                                         \
    ucc_log_component_collective_trace(UCC_LOG_LEVEL_INFO, _fmt, ##__VA_ARGS__)
#define ucc_coll_trace_debug(_fmt, ...)                                        \
    ucc_log_component_collective_trace(UCC_LOG_LEVEL_DEBUG, _fmt, ##__VA_ARGS__)

static inline const char* ucc_coll_type_str(ucc_coll_type_t ct)
{
    switch(ct) {
    case UCC_COLL_TYPE_BARRIER:
        return "Barrier";
    case UCC_COLL_TYPE_BCAST:
        return "Bcast";
    case UCC_COLL_TYPE_ALLREDUCE:
        return "Allreduce";
    case UCC_COLL_TYPE_REDUCE:
        return "Reduce";
    case UCC_COLL_TYPE_ALLTOALL:
        return "Alltoall";
    case UCC_COLL_TYPE_ALLTOALLV:
        return "Alltoallv";
    case UCC_COLL_TYPE_ALLGATHER:
        return "Allgather";
    case UCC_COLL_TYPE_ALLGATHERV:
        return "Allgatherv";
    case UCC_COLL_TYPE_GATHER:
        return "Gather";
    case UCC_COLL_TYPE_GATHERV:
        return "Gatherv";
    case UCC_COLL_TYPE_SCATTER:
        return "Scatter";
    case UCC_COLL_TYPE_SCATTERV:
        return "Scatterv";
    case UCC_COLL_TYPE_FANIN:
        return "Fanin";
    case UCC_COLL_TYPE_FANOUT:
        return "Fanout";
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return "Reduce_scatter";
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return "Reduce_scatterv";
    default:
        break;
    }
    return "";
}

static inline const char* ucc_datatype_str(ucc_datatype_t dt)
{
    switch (dt) {
    case UCC_DT_INT8:
        return "int8";
    case UCC_DT_UINT8:
        return "uint8";
    case UCC_DT_INT16:
        return "int16";
    case UCC_DT_UINT16:
        return "uint16";
    case UCC_DT_FLOAT16:
        return "float16";
    case UCC_DT_BFLOAT16:
        return "bfloat16";
    case UCC_DT_INT32:
        return "int32";
    case UCC_DT_UINT32:
        return "uint32";
    case UCC_DT_FLOAT32:
        return "float32";
    case UCC_DT_INT64:
        return "int64";
    case UCC_DT_UINT64:
        return "uint64";
    case UCC_DT_FLOAT64:
        return "float64";
    case UCC_DT_FLOAT128:
        return "float128";
    case UCC_DT_INT128:
        return "int128";
    case UCC_DT_UINT128:
        return "uint128";
    case UCC_DT_FLOAT32_COMPLEX:
        return "float32_complex";
    case UCC_DT_FLOAT64_COMPLEX:
        return "float64_complex";
    case UCC_DT_FLOAT128_COMPLEX:
        return "float128_complex";
    default:
        return "userdefined";
    }
}

static inline const char* ucc_reduction_op_str(ucc_reduction_op_t op)
{
    switch(op) {
    case UCC_OP_SUM:
        return "sum";
    case UCC_OP_PROD:
        return "prod";
    case UCC_OP_MAX:
        return "max";
    case UCC_OP_MIN:
        return "min";
    case UCC_OP_LAND:
        return "land";
    case UCC_OP_LOR:
        return "lor";
    case UCC_OP_LXOR:
        return "lxor";
    case UCC_OP_BAND:
        return "band";
    case UCC_OP_BOR:
        return "bor";
    case UCC_OP_BXOR:
        return "bxor";
    case UCC_OP_MAXLOC:
        return "maxloc";
    case UCC_OP_MINLOC:
        return "minloc";
    case UCC_OP_AVG:
        return "avg";
    default:
        return NULL;
    }
}

static inline const char* ucc_mem_type_str(ucc_memory_type_t ct)
{
    switch((int)ct) {
    case UCC_MEMORY_TYPE_HOST:
        return "Host";
    case UCC_MEMORY_TYPE_CUDA:
        return "Cuda";
    case UCC_MEMORY_TYPE_CUDA_MANAGED:
        return "CudaManaged";
    case UCC_MEMORY_TYPE_ROCM:
        return "Rocm";
    case UCC_MEMORY_TYPE_ROCM_MANAGED:
        return "RocmManaged";
    case UCC_MEMORY_TYPE_ASYMMETRIC:
        return "asymmetric";
    case UCC_MEMORY_TYPE_NOT_APPLY:
        return "n/a";
    default:
        break;
    }
    return "invalid";
}

static inline const char* ucc_thread_mode_str(ucc_thread_mode_t tm)
{
    switch(tm) {
    case UCC_THREAD_SINGLE:
        return "single";
    case UCC_THREAD_FUNNELED:
        return "funneled";
    case UCC_THREAD_MULTIPLE:
        return "multiple";
    }

    return "invalid";
}

static inline const char* ucc_context_type_str(ucc_context_type_t ct)
{
    switch(ct) {
    case UCC_CONTEXT_EXCLUSIVE:
        return "exclusive";
    case UCC_CONTEXT_SHARED:
        return "shared";
    }

    return "invalid";
}

#endif
