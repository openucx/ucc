/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

static inline int64_t ucc_ep_map_get_elem(void **array, int i, int is64)
{
    if (is64) {
        return (int64_t) (*(uint64_t **)(array))[i];
    } else {
        return (int64_t) (*(ucc_rank_t **)(array))[i];
    }
}

static inline ucc_ep_map_t
ucc_ep_map_from_array_generic(void **array, ucc_rank_t size,
                              ucc_rank_t full_size, int need_free, int is64)
{
    int          is_const_stride = 0;
    ucc_ep_map_t map;
    int64_t      stride;
    ucc_rank_t   i;

    map.ep_num = size;
    if (size > 1) {
        /* try to detect strided pattern */
        stride          = ucc_ep_map_get_elem(array, 1, is64) -
                          ucc_ep_map_get_elem(array, 0, is64);
        is_const_stride = 1;
        for (i = 2; i < size; i++) {
            if ((ucc_ep_map_get_elem(array, i, is64) -
                 ucc_ep_map_get_elem(array, i - 1, is64)) != stride) {
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
            map.strided.start  = (uint64_t) ucc_ep_map_get_elem(array, 0,
                                                                is64);
            map.strided.stride = stride;
        }
        if (need_free) {
            ucc_free(*array);
            *array = NULL;
        }
    } else {
        map.type            = UCC_EP_MAP_ARRAY;
        map.array.map       = (void *)(*array);
        map.array.elem_size = is64 ? sizeof(uint64_t) : sizeof(ucc_rank_t);
    }
    return map;
}

ucc_ep_map_t ucc_ep_map_from_array(ucc_rank_t **array, ucc_rank_t size,
                                   ucc_rank_t full_size, int need_free)
{
    return ucc_ep_map_from_array_generic((void **) array, size, full_size,
                                         need_free, 0);
}

ucc_ep_map_t ucc_ep_map_from_array_64(uint64_t **array, ucc_rank_t size,
                                      ucc_rank_t full_size, int need_free)
{
    return ucc_ep_map_from_array_generic((void **) array, size, full_size,
                                         need_free, 1);
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

void ucc_coll_str(const ucc_coll_task_t *task, char *str, size_t len)
{
    const ucc_base_coll_args_t *args  = &task->bargs;
    ucc_team_t                *team  = args->team;
    ucc_coll_type_t            ct    = args->args.coll_type;
    size_t                     left  = len;
    char                       tmp[64];

    ucc_snprintf_safe(str, left, "req %p, seq_num %u, %s, team_id %u, size %u, "
                      "rank %u, ctx_rank %u: %s %s inplace=%u",
                      task, task->seq_num,
                      task->team->context->lib->log_component.name,
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
                          ucc_reduction_op_str(args->args.op));
        left = len - strlen(str);
        strncat(str, tmp, left);
    }
}

typedef struct ucc_ep_map_nested {
    ucc_ep_map_t *base_map;
    ucc_ep_map_t *sub_map;
} ucc_ep_map_nested_t;

uint64_t ucc_ep_map_nested_cb(uint64_t ep, void *cb_ctx)
{
    ucc_ep_map_nested_t *nested = cb_ctx;
    ucc_rank_t           r;

    r = ucc_ep_map_eval(*nested->sub_map, (ucc_rank_t)ep);
    return (uint64_t)ucc_ep_map_eval(*nested->base_map, r);
}

ucc_status_t ucc_ep_map_create_nested(ucc_ep_map_t *base_map,
                                      ucc_ep_map_t *sub_map,
                                      ucc_ep_map_t *out)
{
    ucc_ep_map_nested_t *nested;

    nested = ucc_malloc(sizeof(*nested), "nested_map");
    if (ucc_unlikely(!nested)) {
        ucc_error("failed to allocate %zd bytes for nested map",
                  sizeof(*nested));
        return UCC_ERR_NO_MEMORY;
    }
    nested->base_map = base_map;
    nested->sub_map  = sub_map;
    out->type        = UCC_EP_MAP_CB;
    out->ep_num      = sub_map->ep_num;
    out->cb.cb       = ucc_ep_map_nested_cb;
    out->cb.cb_ctx   = nested;

    return UCC_OK;
}

void ucc_ep_map_destroy_nested(ucc_ep_map_t *out)
{
    ucc_free(out->cb.cb_ctx);
}

ucc_ep_map_t ucc_ep_map_create_reverse(ucc_rank_t size)
{
    ucc_ep_map_t map = {.type           = UCC_EP_MAP_STRIDED,
                        .ep_num         = size,
                        .strided.start  = size - 1,
                        .strided.stride = -1};
    return map;
}

static inline int ucc_ep_map_is_reverse(ucc_ep_map_t *map)
{
    return (map->type == UCC_EP_MAP_STRIDED) &&
           (map->strided.start == map->ep_num - 1) &&
           (map->strided.stride == -1);
}

ucc_status_t ucc_ep_map_create_inverse(ucc_ep_map_t map, ucc_ep_map_t *inv_map)
{
    ucc_ep_map_t inv;
    ucc_rank_t   i, r;
    ucc_rank_t   max_rank;

    if (ucc_ep_map_is_reverse(&map)) {
        inv = map;
    } else {
        inv.type            = UCC_EP_MAP_ARRAY;
        inv.ep_num          = map.ep_num;
        inv.array.elem_size = sizeof(ucc_rank_t);
        max_rank            = 0;
        for (i = 0; i < map.ep_num; i++) {
            r = (ucc_rank_t)ucc_ep_map_eval(map, i);
            if (r > max_rank) {
                max_rank = r;
            }
        }
        inv.array.map =
            ucc_malloc(sizeof(ucc_rank_t) * (max_rank + 1), "inv_map");
        if (!inv.array.map) {
            ucc_error("failed to allocate %zd bytes for inv map\n",
                      sizeof(ucc_rank_t) * map.ep_num);
            return UCC_ERR_NO_MEMORY;
        }
        for (i = 0; i < map.ep_num; i++) {
            r = (ucc_rank_t)ucc_ep_map_eval(map, i);
            *((ucc_rank_t *)PTR_OFFSET(inv.array.map, sizeof(ucc_rank_t) * r)) =
                i;
        }
    }
    *inv_map = inv;
    return UCC_OK;
}

void ucc_ep_map_destroy(ucc_ep_map_t *map)
{
    if (map->type == UCC_EP_MAP_ARRAY) {
        ucc_free(map->array.map);
    }
}
