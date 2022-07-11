/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_LOG_H_
#define UCC_LOG_H_

#include "config.h"
#include "core/ucc_global_opts.h"
#include "core/ucc_dt.h"
#include <ucs/debug/log_def.h>

#define UCC_LOG_LEVEL_ERROR UCS_LOG_LEVEL_ERROR
#define UCC_LOG_LEVEL_WARN  UCS_LOG_LEVEL_WARN
#define UCC_LOG_LEVEL_INFO  UCS_LOG_LEVEL_INFO
#define UCC_LOG_LEVEL_DEBUG UCS_LOG_LEVEL_DEBUG
#define UCC_LOG_LEVEL_TRACE UCS_LOG_LEVEL_TRACE

/* Generic wrapper macro to invoke ucs logging backend */
#define ucc_log_component(_level, _component, _fmt, ...)                       \
    do {                                                                       \
        ucs_log_component(_level, &_component, _fmt, ##__VA_ARGS__);           \
    } while (0)

/* Global logger: to be used anywhere when special log level settings are not required */
#define ucc_log_component_global(_level, fmt, ...)                             \
    ucc_log_component(_level, ucc_global_config.log_component, fmt,            \
                      ##__VA_ARGS__)
#define ucc_error(_fmt, ...)                                                   \
    ucc_log_component_global(UCS_LOG_LEVEL_ERROR, _fmt, ##__VA_ARGS__)
#define ucc_warn(_fmt, ...)                                                    \
    ucc_log_component_global(UCS_LOG_LEVEL_WARN, _fmt, ##__VA_ARGS__)
#define ucc_info(_fmt, ...)                                                    \
    ucc_log_component_global(UCS_LOG_LEVEL_INFO, _fmt, ##__VA_ARGS__)
#define ucc_debug(_fmt, ...)                                                   \
    ucc_log_component_global(UCS_LOG_LEVEL_DEBUG, _fmt, ##__VA_ARGS__)
#define ucc_trace(_fmt, ...)                                                   \
    ucc_log_component_global(UCS_LOG_LEVEL_TRACE, _fmt, ##__VA_ARGS__)
#define ucc_trace_req(_fmt, ...)                                               \
    ucc_log_component_global(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ##__VA_ARGS__)
#define ucc_trace_data(_fmt, ...)                                              \
    ucc_log_component_global(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ##__VA_ARGS__)
#define ucc_trace_async(_fmt, ...)                                             \
    ucc_log_component_global(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ##__VA_ARGS__)
#define ucc_trace_func(_fmt, ...)                                              \
    ucc_log_component_global(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")",         \
                             __FUNCTION__, ##__VA_ARGS__)
#define ucc_trace_poll(_fmt, ...)                                              \
    ucc_log_component_global(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ##__VA_ARGS__)


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
    return 0;
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

#endif
