/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "ucc_coll_utils.h"
#include "components/base/ucc_base_iface.h"
#include "core/ucc_team.h"
#define STR_TYPE_CHECK(_str, _p, _prefix)                                      \
    do {                                                                       \
        if ((0 == strcasecmp(_UCC_PP_MAKE_STRING(_p), _str))) {                \
            return _prefix##_p;                                                \
        }                                                                      \
    } while (0)

#define STR_COLL_TYPE_CHECK(_str, _p) STR_TYPE_CHECK(_str, _p, UCC_COLL_TYPE_)

ucc_coll_type_t ucc_coll_type_from_str(const char *str)
{
    STR_COLL_TYPE_CHECK(str, ALLGATHER);
    STR_COLL_TYPE_CHECK(str, ALLGATHERV);
    STR_COLL_TYPE_CHECK(str, ALLREDUCE);
    STR_COLL_TYPE_CHECK(str, ALLTOALL);
    STR_COLL_TYPE_CHECK(str, ALLTOALLV);
    STR_COLL_TYPE_CHECK(str, BARRIER);
    STR_COLL_TYPE_CHECK(str, BCAST);
    STR_COLL_TYPE_CHECK(str, FANIN);
    STR_COLL_TYPE_CHECK(str, FANOUT);
    STR_COLL_TYPE_CHECK(str, GATHER);
    STR_COLL_TYPE_CHECK(str, GATHERV);
    STR_COLL_TYPE_CHECK(str, REDUCE);
    STR_COLL_TYPE_CHECK(str, REDUCE_SCATTER);
    STR_COLL_TYPE_CHECK(str, REDUCE_SCATTERV);
    STR_COLL_TYPE_CHECK(str, SCATTER);
    STR_COLL_TYPE_CHECK(str, SCATTERV);
    return UCC_COLL_TYPE_LAST;
}

#define STR_MEM_TYPE_CHECK(_str, _p) STR_TYPE_CHECK(_str, _p,UCC_MEMORY_TYPE_)


ucc_memory_type_t ucc_mem_type_from_str(const char *str)
{
    STR_MEM_TYPE_CHECK(str, HOST);
    STR_MEM_TYPE_CHECK(str, CUDA);
    STR_MEM_TYPE_CHECK(str, CUDA_MANAGED);
    STR_MEM_TYPE_CHECK(str, ROCM);
    STR_MEM_TYPE_CHECK(str, ROCM_MANAGED);
    return UCC_MEMORY_TYPE_LAST;
}

static inline int
ucc_coll_args_is_mem_symmetric(const ucc_base_coll_args_t *bargs)
{
    const ucc_coll_args_t *args = &bargs->args;
    ucc_team_t            *team = bargs->team;
    ucc_rank_t             root = args->root;
    if (UCC_IS_INPLACE(*args)) {
        return 1;
    }
    switch (args->coll_type) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_BCAST:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return 1;
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return args->dst.info.mem_type == args->src.info.mem_type;
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return args->dst.info_v.mem_type == args->src.info.mem_type;
    case UCC_COLL_TYPE_ALLTOALLV:
        return args->dst.info_v.mem_type == args->src.info_v.mem_type;
    case UCC_COLL_TYPE_REDUCE:
    case UCC_COLL_TYPE_GATHER:
    case UCC_COLL_TYPE_SCATTER:
        return root != team->rank ||
               (args->dst.info.mem_type == args->src.info.mem_type);
    case UCC_COLL_TYPE_GATHERV:
        return root != team->rank ||
               (args->dst.info_v.mem_type == args->src.info.mem_type);
    case UCC_COLL_TYPE_SCATTERV:
        return root != team->rank ||
               (args->dst.info.mem_type == args->src.info_v.mem_type);
    default:
        break;
    }
    return 0;
}

ucc_memory_type_t ucc_coll_args_mem_type(const ucc_base_coll_args_t *bargs)
{
    const ucc_coll_args_t *args = &bargs->args;
    ucc_team_t            *team = bargs->team;
    ucc_rank_t             root = args->root;

    if (!ucc_coll_args_is_mem_symmetric(bargs)) {
        return UCC_MEMORY_TYPE_ASSYMETRIC;
    }
    switch (args->coll_type) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return UCC_MEMORY_TYPE_NOT_APPLY;
    case UCC_COLL_TYPE_BCAST:
        return args->src.info.mem_type;
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return args->dst.info.mem_type;
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return args->dst.info_v.mem_type;
    case UCC_COLL_TYPE_ALLTOALLV:
        return args->dst.info_v.mem_type;
    case UCC_COLL_TYPE_REDUCE:
    case UCC_COLL_TYPE_GATHER:
        return (root == team->rank) ? args->dst.info.mem_type
                                    : args->src.info.mem_type;
    case UCC_COLL_TYPE_SCATTER:
        return (root == team->rank) ? args->src.info.mem_type
                                    : args->dst.info.mem_type;
    case UCC_COLL_TYPE_GATHERV:
        return (root == team->rank) ? args->dst.info_v.mem_type
                                    : args->src.info.mem_type;
    case UCC_COLL_TYPE_SCATTERV:
        return (root == team->rank) ? args->src.info_v.mem_type
                                    : args->dst.info.mem_type;
    default:
        break;
    }
    return UCC_MEMORY_TYPE_UNKNOWN;
}

size_t ucc_coll_args_msgsize(const ucc_base_coll_args_t *bargs)
{
    const ucc_coll_args_t *args = &bargs->args;
    ucc_team_t            *team = bargs->team;
    ucc_rank_t             root = args->root;

    switch (args->coll_type) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return 0;
    case UCC_COLL_TYPE_BCAST:
        return args->src.info.count * ucc_dt_size(args->src.info.datatype);
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return args->dst.info.count * ucc_dt_size(args->dst.info.datatype);
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                             team->size) *
               ucc_dt_size(args->dst.info_v.datatype);
    case UCC_COLL_TYPE_ALLTOALLV:
    case UCC_COLL_TYPE_GATHERV:
    case UCC_COLL_TYPE_SCATTERV:
        /* This means all team members can not know the msg size estimate w/o communication.
           Local args information is not enough.
           This prohibits algorithm selection based on msg size thresholds w/o additinoal exchange.
        */
        return UCC_MSG_SIZE_ASSYMETRIC;
    case UCC_COLL_TYPE_REDUCE:
        return (root == team->rank)
                   ? args->dst.info.count * ucc_dt_size(args->dst.info.datatype)
                   : args->src.info.count *
                         ucc_dt_size(args->src.info.datatype);
    case UCC_COLL_TYPE_GATHER:
        return (root == team->rank)
                 ? args->dst.info.count * ucc_dt_size(args->dst.info.datatype)
                 : args->src.info.count * ucc_dt_size(args->src.info.datatype) *
                   team->size;
    case UCC_COLL_TYPE_SCATTER:
        return (root == team->rank)
                 ? args->src.info.count * ucc_dt_size(args->src.info.datatype)
                 : args->dst.info.count * ucc_dt_size(args->dst.info.datatype) *
                   team->size;
    default:
        break;
    }
    return 0;
}

ucc_ep_map_t ucc_ep_map_from_array(ucc_rank_t **array, ucc_rank_t size,
                                   ucc_rank_t full_size, int need_free)
{
    int          is_const_stride = 0;
    ucc_ep_map_t map;
    int64_t      stride;
    ucc_rank_t   i;

    map.ep_num = size;
    if (size > 1) {
        /* try to detect strided pattern */
        stride          = (int64_t)(*array)[1] - (int64_t)(*array)[0];
        is_const_stride = 1;
        for (i = 2; i < size; i++) {
            if (((int64_t)(*array)[i] - (int64_t)(*array)[i - 1]) != stride) {
                is_const_stride = 0;
                break;
            }
        }
    }
    if (is_const_stride) {
        if ((stride == 1) && (size == full_size)) {
            map.type = UCC_EP_MAP_FULL;
        }
        else {
            map.type           = UCC_EP_MAP_STRIDED;
            map.strided.start  = (uint64_t)(*array)[0];
            map.strided.stride = stride;
        }
        if (need_free) {
            ucc_free(*array);
            *array = NULL;
        }
    } else {
        map.type            = UCC_EP_MAP_ARRAY;
        map.array.map       = (void *)(*array);
        map.array.elem_size = sizeof(ucc_rank_t);
    }
    return map;
}

static inline int
ucc_coll_args_is_rooted(const ucc_base_coll_args_t *bargs)
{
    ucc_coll_type_t ct = bargs->args.coll_type;
    if (ct == UCC_COLL_TYPE_REDUCE || ct == UCC_COLL_TYPE_BCAST ||
        ct == UCC_COLL_TYPE_GATHER || ct == UCC_COLL_TYPE_SCATTER ||
        ct == UCC_COLL_TYPE_FANIN || ct == UCC_COLL_TYPE_FANOUT ||
        ct == UCC_COLL_TYPE_GATHERV || ct == UCC_COLL_TYPE_SCATTERV) {
        return 1;
    }
    return 0;
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
    case UCC_MEMORY_TYPE_ASSYMETRIC:
        return "assymetric";
    case UCC_MEMORY_TYPE_NOT_APPLY:
        return "n/a";
    default:
        break;
    }
    return "invalid";
}

void ucc_coll_str(const ucc_base_coll_args_t *args, char *str, size_t len)
{
    ucc_team_t     *team  = args->team;
    ucc_coll_type_t ct    = args->args.coll_type;
    size_t          left  = len;
    char            tmp[64];

    ucc_snprintf_safe(str, left, "team id %u size %u rank %u ctx_rank %u: %s %s inplace=%u",
                      team->id, team->size, team->rank,
                      ucc_ep_map_eval(team->ctx_map, team->rank),
                      ucc_coll_type_str(ct),
                      ucc_mem_type_str(ucc_coll_args_mem_type(args)),
                      UCC_IS_INPLACE(args->args));

    if (ucc_coll_args_is_rooted(args)) {
        ucc_snprintf_safe(tmp, sizeof(tmp), " root=%u", (ucc_rank_t)args->args.root);
        left = len - strlen(str);
        strncat(str, tmp, left);
    }

    if (ct ==  UCC_COLL_TYPE_ALLTOALLV) {
        size_t sbytes = ucc_coll_args_get_total_count(&args->args,
                           args->args.src.info_v.counts, team->size ) *
                           ucc_dt_size(args->args.src.info_v.datatype);
        size_t rbytes = ucc_coll_args_get_total_count(&args->args,
                           args->args.dst.info_v.counts, team->size ) *
                           ucc_dt_size(args->args.dst.info_v.datatype);
        ucc_snprintf_safe(tmp, sizeof(tmp), " sbytes=%zd rbytes=%zd",
                          sbytes, rbytes);
    } else {
        ucc_snprintf_safe(tmp, sizeof(tmp), " bytes=%zd",
                          ucc_coll_args_msgsize(args));
    }
    left = len - strlen(str);
    strncat(str, tmp, left);

    if (ct == UCC_COLL_TYPE_ALLREDUCE ||
        ct == UCC_COLL_TYPE_REDUCE) {
        ucc_snprintf_safe(tmp, sizeof(tmp), " %s %s",
                          ucc_datatype_str(args->args.src.info.datatype),
                          (args->args.mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) ?
                          "userdefined" : ucc_reduction_op_str(args->args.reduce.predefined_op));
        left = len - strlen(str);
        strncat(str, tmp, left);
    }
}
