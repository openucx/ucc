/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_coll_utils.h"
#include "components/base/ucc_base_iface.h"
#include "core/ucc_team.h"
#include "schedule/ucc_schedule_pipelined.h"

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

#define STR_MEM_TYPE_CHECK(_str, _p)                                           \
    do {                                                                       \
        if (0 == strcasecmp("CudaManaged", _str)) {                            \
            STR_TYPE_CHECK("cuda_managed", _p, UCC_MEMORY_TYPE_);              \
        } else if (0 == strcasecmp("RocmManaged", _str)) {                     \
            STR_TYPE_CHECK("rocm_managed", _p, UCC_MEMORY_TYPE_);              \
        }                                                                      \
        STR_TYPE_CHECK(_str, _p, UCC_MEMORY_TYPE_);                            \
    } while (0)

ucc_memory_type_t ucc_mem_type_from_str(const char *str)
{
    STR_MEM_TYPE_CHECK(str, HOST);
    STR_MEM_TYPE_CHECK(str, CUDA);
    STR_MEM_TYPE_CHECK(str, CUDA_MANAGED);
    STR_MEM_TYPE_CHECK(str, ROCM);
    STR_MEM_TYPE_CHECK(str, ROCM_MANAGED);
    return UCC_MEMORY_TYPE_LAST;
}

int
ucc_coll_args_is_mem_symmetric(const ucc_coll_args_t *args,
                               ucc_rank_t rank)
{
    ucc_rank_t root = args->root;

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
        return root != rank ||
               (args->dst.info.mem_type == args->src.info.mem_type);
    case UCC_COLL_TYPE_GATHERV:
        return root != rank ||
               (args->dst.info_v.mem_type == args->src.info.mem_type);
    case UCC_COLL_TYPE_SCATTERV:
        return root != rank ||
               (args->dst.info.mem_type == args->src.info_v.mem_type);
    default:
        break;
    }
    return 0;
}


/* If this is the root and the src/dst buffers are asymmetric, one buffer needs
   to have a new allocation to make the mem types match. If that buffer was the
   dst buffer, copy the result back into the old dst on task completion */
ucc_status_t
ucc_coll_args_init_asymmetric_buffer(ucc_coll_args_t *args,
                                     ucc_team_h team,
                                     ucc_buffer_info_asymmetric_memtype_t *save_info)
{
    ucc_status_t status = UCC_OK;

    if (UCC_IS_INPLACE(*args)) {
        return UCC_ERR_INVALID_PARAM;
    }
    switch (args->coll_type) {
    case UCC_COLL_TYPE_REDUCE:
    case UCC_COLL_TYPE_GATHER:
    {
        ucc_memory_type_t mem_type = args->src.info.mem_type;
        if (args->coll_type == UCC_COLL_TYPE_SCATTERV) {
            mem_type = args->src.info_v.mem_type;
        }
        memcpy(&save_info->old_asymmetric_buffer.info,
               &args->dst.info, sizeof(ucc_coll_buffer_info_t));
        status = ucc_mc_alloc(&save_info->scratch,
                              ucc_dt_size(args->dst.info.datatype) *
                                          args->dst.info.count,
                                          mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            ucc_error("failed to allocate replacement "
                      "memory for asymmetric buffer");
            return status;
        }
        args->dst.info.buffer = save_info->scratch->addr;
        args->dst.info.mem_type = mem_type;
        return UCC_OK;
    }
    case UCC_COLL_TYPE_GATHERV:
    {
        memcpy(&save_info->old_asymmetric_buffer.info_v,
               &args->dst.info_v, sizeof(ucc_coll_buffer_info_v_t));
        status = ucc_mc_alloc(&save_info->scratch,
                                ucc_coll_args_get_v_buffer_size(args,
                                args->dst.info_v.counts,
                                args->dst.info_v.displacements,
                                team->size),
                                args->src.info.mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            ucc_error("failed to allocate replacement "
                      "memory for asymmetric buffer");
            return status;
        }
        args->dst.info_v.buffer   = save_info->scratch->addr;
        args->dst.info_v.mem_type = args->src.info.mem_type;
        return UCC_OK;
    }
    case UCC_COLL_TYPE_SCATTER:
    {
        ucc_memory_type_t mem_type = args->dst.info.mem_type;
        memcpy(&save_info->old_asymmetric_buffer.info,
               &args->src.info, sizeof(ucc_coll_buffer_info_t));
        status = ucc_mc_alloc(&save_info->scratch,
                              ucc_dt_size(args->src.info.datatype) * args->src.info.count,
                              mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            ucc_error("failed to allocate replacement "
                      "memory for asymmetric buffer");
            return status;
        }
        args->src.info.buffer   = save_info->scratch->addr;
        args->src.info.mem_type = mem_type;
        return UCC_OK;
    }
    case UCC_COLL_TYPE_SCATTERV:
    {
        ucc_memory_type_t mem_type = args->dst.info.mem_type;
        memcpy(&save_info->old_asymmetric_buffer.info_v,
               &args->src.info_v, sizeof(ucc_coll_buffer_info_v_t));
        status = ucc_mc_alloc(&save_info->scratch,
                              ucc_coll_args_get_v_buffer_size(args,
                                args->src.info_v.counts,
                                args->src.info_v.displacements,
                                team->size),
                              mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            ucc_error("failed to allocate replacement "
                      "memory for asymmetric buffer");
            return status;
        }
        args->src.info_v.buffer   = save_info->scratch->addr;
        args->src.info_v.mem_type = mem_type;
        return UCC_OK;
    }
    default:
        break;
    }
    return UCC_ERR_INVALID_PARAM;
}

ucc_status_t
ucc_coll_args_free_asymmetric_buffer(ucc_coll_task_t *task)
{
    ucc_status_t status                        = UCC_OK;
    ucc_buffer_info_asymmetric_memtype_t *save = &task->bargs.asymmetric_save_info;

    if (UCC_IS_INPLACE(task->bargs.args)) {
        return UCC_ERR_INVALID_PARAM;
    }

    if (save->scratch == NULL) {
        ucc_error("failure trying to free NULL asymmetric buffer");
    }

    status = ucc_mc_free(save->scratch);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("error freeing scratch asymmetric buffer: %s",
                    ucc_status_string(status));
    }
    save->scratch = NULL;

    return status;
}

ucc_status_t ucc_copy_asymmetric_buffer(ucc_coll_task_t *task)
{
    ucc_status_t                          status    = UCC_OK;
    ucc_coll_args_t                      *coll_args = &task->bargs.args;
    ucc_buffer_info_asymmetric_memtype_t *save      = &task->bargs.asymmetric_save_info;
    ucc_rank_t                            size      = task->team->params.size;

    if(task->bargs.args.coll_type == UCC_COLL_TYPE_SCATTERV) {
        // copy in
        status = ucc_mc_memcpy(save->scratch->addr,
                        save->old_asymmetric_buffer.info_v.buffer,
                        ucc_coll_args_get_v_buffer_size(coll_args,
                            save->old_asymmetric_buffer.info_v.counts,
                            save->old_asymmetric_buffer.info_v.displacements,
                            size),
                        save->scratch->mt,
                        save->old_asymmetric_buffer.info_v.mem_type);
    } else if(task->bargs.args.coll_type == UCC_COLL_TYPE_SCATTER) {
        // copy in
        status = ucc_mc_memcpy(save->scratch->addr,
                        save->old_asymmetric_buffer.info.buffer,
                        ucc_dt_size(save->old_asymmetric_buffer.info.datatype) *
                            save->old_asymmetric_buffer.info.count,
                        save->scratch->mt,
                        save->old_asymmetric_buffer.info.mem_type);
    } else if(task->bargs.args.coll_type == UCC_COLL_TYPE_GATHERV) {
        // copy out
        status = ucc_mc_memcpy(save->old_asymmetric_buffer.info_v.buffer,
                        save->scratch->addr,
                        ucc_coll_args_get_v_buffer_size(coll_args,
                            save->old_asymmetric_buffer.info_v.counts,
                            save->old_asymmetric_buffer.info_v.displacements,
                            size),
                        save->old_asymmetric_buffer.info_v.mem_type,
                        save->scratch->mt);
    } else {
        // copy out
        status = ucc_mc_memcpy(save->old_asymmetric_buffer.info.buffer,
                        save->scratch->addr,
                        ucc_dt_size(save->old_asymmetric_buffer.info.datatype) *
                            save->old_asymmetric_buffer.info.count,
                        save->old_asymmetric_buffer.info.mem_type,
                        save->scratch->mt);
    }
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("error copying back to old asymmetric buffer: %s",
                    ucc_status_string(status));
    }
    return status;
}

int ucc_coll_args_is_predefined_dt(const ucc_coll_args_t *args, ucc_rank_t rank)
{
    switch (args->coll_type) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return 1;
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_ALLTOALL:
        return UCC_DT_IS_PREDEFINED(args->dst.info.datatype) &&
               (UCC_IS_INPLACE(*args) ||
                UCC_DT_IS_PREDEFINED(args->src.info.datatype));
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return UCC_DT_IS_PREDEFINED(args->dst.info_v.datatype) &&
               (UCC_IS_INPLACE(*args) ||
                UCC_DT_IS_PREDEFINED(args->src.info.datatype));
    case UCC_COLL_TYPE_ALLTOALLV:
        return UCC_DT_IS_PREDEFINED(args->dst.info_v.datatype) &&
               (UCC_IS_INPLACE(*args) ||
                UCC_DT_IS_PREDEFINED(args->src.info_v.datatype));
    case UCC_COLL_TYPE_BCAST:
        return UCC_DT_IS_PREDEFINED(args->src.info.datatype);
    case UCC_COLL_TYPE_GATHER:
    case UCC_COLL_TYPE_REDUCE:
        if (UCC_IS_ROOT(*args, rank)) {
           return UCC_DT_IS_PREDEFINED(args->dst.info.datatype) &&
                  (UCC_IS_INPLACE(*args) ||
                   UCC_DT_IS_PREDEFINED(args->src.info.datatype));
        } else {
            return UCC_DT_IS_PREDEFINED(args->src.info.datatype);
        }
    case UCC_COLL_TYPE_GATHERV:
        if (UCC_IS_ROOT(*args, rank)) {
           return UCC_DT_IS_PREDEFINED(args->dst.info_v.datatype) &&
                  (UCC_IS_INPLACE(*args) ||
                   UCC_DT_IS_PREDEFINED(args->src.info.datatype));
        } else {
            return UCC_DT_IS_PREDEFINED(args->src.info.datatype);
        }
    case UCC_COLL_TYPE_SCATTER:
        if (UCC_IS_ROOT(*args, rank)) {
           return UCC_DT_IS_PREDEFINED(args->src.info.datatype) &&
                  (UCC_IS_INPLACE(*args) ||
                   UCC_DT_IS_PREDEFINED(args->dst.info.datatype));
        } else {
            return UCC_DT_IS_PREDEFINED(args->dst.info.datatype);
        }
    case UCC_COLL_TYPE_SCATTERV:
        if (UCC_IS_ROOT(*args, rank)) {
           return UCC_DT_IS_PREDEFINED(args->src.info_v.datatype) &&
                  (UCC_IS_INPLACE(*args) ||
                   UCC_DT_IS_PREDEFINED(args->dst.info.datatype));
        } else {
            return UCC_DT_IS_PREDEFINED(args->dst.info.datatype);
        }
    default:
        ucc_error("invalid collective type %d", args->coll_type);
        return -1;
    }
}

ucc_memory_type_t ucc_coll_args_mem_type(const ucc_coll_args_t *args,
                                         ucc_rank_t rank)
{
    ucc_rank_t root = args->root;

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
    case UCC_COLL_TYPE_ALLTOALLV:
        return args->dst.info_v.mem_type;
    case UCC_COLL_TYPE_REDUCE:
    case UCC_COLL_TYPE_GATHER:
        return (root == rank) ? args->dst.info.mem_type
                              : args->src.info.mem_type;
    case UCC_COLL_TYPE_SCATTER:
        return (root == rank) ? args->src.info.mem_type
                              : args->dst.info.mem_type;
    case UCC_COLL_TYPE_GATHERV:
        return (root == rank) ? args->dst.info_v.mem_type
                              : args->src.info.mem_type;
    case UCC_COLL_TYPE_SCATTERV:
        return (root == rank) ? args->src.info_v.mem_type
                              : args->dst.info.mem_type;
    default:
        break;
    }
    return UCC_MEMORY_TYPE_UNKNOWN;
}

size_t ucc_coll_args_msgsize(const ucc_coll_args_t *args, ucc_rank_t rank,
                             ucc_rank_t size)
{
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
                                             size) *
               ucc_dt_size(args->dst.info_v.datatype);
    case UCC_COLL_TYPE_ALLTOALLV:
    case UCC_COLL_TYPE_GATHERV:
    case UCC_COLL_TYPE_SCATTERV:
        /* This means all team members can not know the msg size estimate w/o communication.
           Local args information is not enough.
           This prohibits algorithm selection based on msg size thresholds w/o additinoal exchange.
        */
        return UCC_MSG_SIZE_ASYMMETRIC;
    case UCC_COLL_TYPE_REDUCE:
        return (root == rank)
                   ? args->dst.info.count * ucc_dt_size(args->dst.info.datatype)
                   : args->src.info.count *
                         ucc_dt_size(args->src.info.datatype);
    case UCC_COLL_TYPE_GATHER:
        return (root == rank)
                 ? args->dst.info.count * ucc_dt_size(args->dst.info.datatype)
                 : args->src.info.count * ucc_dt_size(args->src.info.datatype) *
                   size;
    case UCC_COLL_TYPE_SCATTER:
        return (root == rank)
                 ? args->src.info.count * ucc_dt_size(args->src.info.datatype)
                 : args->dst.info.count * ucc_dt_size(args->dst.info.datatype) *
                   size;
    default:
        ucc_assert(args->coll_type == UCC_COLL_TYPE_LAST);
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

    map.type   = (ucc_ep_map_type_t)0;
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

int ucc_coll_args_is_rooted(ucc_coll_type_t ct)
{
    if (ct == UCC_COLL_TYPE_REDUCE || ct == UCC_COLL_TYPE_BCAST ||
        ct == UCC_COLL_TYPE_GATHER || ct == UCC_COLL_TYPE_SCATTER ||
        ct == UCC_COLL_TYPE_FANIN || ct == UCC_COLL_TYPE_FANOUT ||
        ct == UCC_COLL_TYPE_GATHERV || ct == UCC_COLL_TYPE_SCATTERV) {
        return 1;
    }
    return 0;
}

#define COLL_ARGS_HEADER_STR_MAX_SIZE 32
void ucc_coll_args_str(const ucc_coll_args_t *args, ucc_rank_t trank,
                       ucc_rank_t tsize, char *str, size_t len)
{
    ucc_coll_type_t        ct                                 = args->coll_type;
    ucc_rank_t             root                               = args->root;
    ucc_coll_buffer_info_t src_info                           = {0};
    ucc_coll_buffer_info_t dst_info                           = {0};
    char                   hdr[COLL_ARGS_HEADER_STR_MAX_SIZE] = "";
    char tmp[32];
    size_t count;
    int left, has_src, has_dst;

    ucc_snprintf_safe(hdr, COLL_ARGS_HEADER_STR_MAX_SIZE, "%s",
                      ucc_coll_type_str(ct));
    if (ucc_coll_args_is_reduction(ct)) {
        ucc_snprintf_safe(tmp, sizeof(tmp), " %s",
                          ucc_reduction_op_str(args->op));
        left = COLL_ARGS_HEADER_STR_MAX_SIZE - strlen(hdr);
        strncat(hdr, tmp, left);
    }

    if (UCC_IS_INPLACE(*args)) {
        ucc_snprintf_safe(tmp, sizeof(tmp), " inplace");
        left = COLL_ARGS_HEADER_STR_MAX_SIZE - strlen(hdr);
        strncat(hdr, tmp, left);
    }

    if (UCC_IS_PERSISTENT(*args)) {
        ucc_snprintf_safe(tmp, sizeof(tmp), " persistent");
        left = COLL_ARGS_HEADER_STR_MAX_SIZE - strlen(hdr);
        strncat(hdr, tmp, left);
    }

    if (ucc_coll_args_is_rooted(ct)) {
        ucc_snprintf_safe(tmp, sizeof(tmp), " root %u", root);
        left = COLL_ARGS_HEADER_STR_MAX_SIZE - strlen(hdr);
        strncat(hdr, tmp, left);
    }

    has_src = has_dst = 0;
    switch (ct) {
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        dst_info = args->dst.info;
        has_dst = 1;
        if (!UCC_IS_INPLACE(*args)) {
            src_info = args->src.info;
            has_src = 1;
        }
        break;
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        count = ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                              tsize);
        dst_info.buffer   = args->dst.info_v.buffer;
        dst_info.count    = count;
        dst_info.datatype = args->dst.info_v.datatype;
        dst_info.mem_type = args->dst.info_v.mem_type;
        has_dst = 1;
        if (!UCC_IS_INPLACE(*args)) {
            src_info = args->src.info;
            has_src = 1;
        }
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        count = ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                              tsize);
        dst_info.buffer   = args->dst.info_v.buffer;
        dst_info.count    = count;
        dst_info.datatype = args->dst.info_v.datatype;
        dst_info.mem_type = args->dst.info_v.mem_type;
        has_dst = 1;
        if (!UCC_IS_INPLACE(*args)) {
            count = ucc_coll_args_get_total_count(args, args->src.info_v.counts,
                                                  tsize);
            src_info.buffer   = args->src.info_v.buffer;
            src_info.count    = count;
            src_info.datatype = args->src.info_v.datatype;
            src_info.mem_type = args->src.info_v.mem_type;
            has_src = 1;
        }
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        break;
    case UCC_COLL_TYPE_BCAST:
        src_info = args->src.info;
        has_src = 1;
        break;
    case UCC_COLL_TYPE_GATHER:
    case UCC_COLL_TYPE_REDUCE:
        if (UCC_IS_ROOT(*args, trank)) {
            dst_info = args->dst.info;
            has_dst = 1;
        }
        if (!UCC_IS_ROOT(*args, trank) || !UCC_IS_INPLACE(*args)) {
            src_info = args->src.info;
            has_src = 1;
        }
        break;
    case UCC_COLL_TYPE_GATHERV:
        if (UCC_IS_ROOT(*args, trank)) {
            count = ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                                  tsize);
            dst_info.buffer   = args->dst.info_v.buffer;
            dst_info.count    = count;
            dst_info.datatype = args->dst.info_v.datatype;
            dst_info.mem_type = args->dst.info_v.mem_type;
            has_dst = 1;
        }
        if (!UCC_IS_ROOT(*args, trank) || !UCC_IS_INPLACE(*args)) {
            src_info = args->src.info;
            has_src = 1;
        }
        break;
    case UCC_COLL_TYPE_SCATTER:
        if (UCC_IS_ROOT(*args, trank)) {
            src_info = args->src.info;
            has_src = 1;
        }
        if (!UCC_IS_ROOT(*args, trank) || !UCC_IS_INPLACE(*args)) {
            dst_info = args->dst.info;
            has_dst = 1;
        }
        break;
    case UCC_COLL_TYPE_SCATTERV:
        if (UCC_IS_ROOT(*args, trank)) {
            count = ucc_coll_args_get_total_count(args, args->src.info_v.counts,
                                                  tsize);
            src_info.buffer   = args->src.info_v.buffer;
            src_info.count    = count;
            src_info.datatype = args->src.info_v.datatype;
            src_info.mem_type = args->src.info_v.mem_type;
            has_src = 1;
        }
        if (!UCC_IS_ROOT(*args, trank) || !UCC_IS_INPLACE(*args)) {
            dst_info = args->dst.info;
            has_dst = 1;
        }
        break;
    default:
        ucc_assert(args->coll_type == UCC_COLL_TYPE_LAST);
        return;
    }

    if (has_src && has_dst) {
        ucc_snprintf_safe(str, len,
                          "%s: src={%p, %zd, %s, %s}, dst={%p, %zd, %s, %s}",
                          hdr, src_info.buffer, src_info.count,
                          ucc_datatype_str(src_info.datatype),
                          ucc_mem_type_str(src_info.mem_type),
                          dst_info.buffer, dst_info.count,
                          ucc_datatype_str(dst_info.datatype),
                          ucc_mem_type_str(dst_info.mem_type));
    } else if (has_src && !has_dst) {
        ucc_snprintf_safe(str, len,
                          "%s: src={%p, %zd, %s, %s}",
                          hdr, src_info.buffer, src_info.count,
                          ucc_datatype_str(src_info.datatype),
                          ucc_mem_type_str(src_info.mem_type));
    } else if (!has_src && has_dst) {
        ucc_snprintf_safe(str, len,
                          "%s: dst={%p, %zd, %s, %s}",
                          hdr, dst_info.buffer, dst_info.count,
                          ucc_datatype_str(dst_info.datatype),
                          ucc_mem_type_str(dst_info.mem_type));
    } else {
        ucc_snprintf_safe(str, len, "%s", hdr);
    }
}

void ucc_coll_task_components_str(const ucc_coll_task_t *task, char *str,
                                  size_t *len)
{
    ucc_schedule_t *schedule;
    ucc_schedule_pipelined_t *schedule_pipelined;
    int i;

    if (task->flags & UCC_COLL_TASK_FLAG_IS_PIPELINED_SCHEDULE) {
        schedule_pipelined = ucc_derived_of(task, ucc_schedule_pipelined_t);
        for (i = 0; i < schedule_pipelined->n_frags; i++) {
            ucc_coll_task_components_str(&schedule_pipelined->frags[i]->super,
                                         str, len);
        }
    } else if (task->flags & UCC_COLL_TASK_FLAG_IS_SCHEDULE) {
        schedule = ucc_derived_of(task, ucc_schedule_t);
        for (i = 0; i < schedule->n_tasks; i++) {
            ucc_coll_task_components_str(schedule->tasks[i], str, len);
        }
    } else {
        if (!strstr(str, task->team->context->lib->log_component.name)) {
            if (*len == 0) {
                sprintf(str + *len, "%s",
                        task->team->context->lib->log_component.name);
                *len = strlen(task->team->context->lib->log_component.name) +
                       *len;
            } else {
                sprintf(str + *len, ", %s",
                        task->team->context->lib->log_component.name);
                *len = strlen(task->team->context->lib->log_component.name) +
                       *len + 2;
            }
        }
    }
}

void ucc_coll_str(const ucc_coll_task_t *task, char *str, size_t len,
                  int verbosity)
{
    ucc_team_t *team  = task->bargs.team;
    int rc;

    if (verbosity >= UCC_LOG_LEVEL_DIAG) {
        ucc_coll_args_str(&task->bargs.args, team->rank, team->size, str, len);
    }

    if (verbosity >= UCC_LOG_LEVEL_INFO) {
        size_t tl_info_len = 0;
        char task_info[64], cl_info[16], tl_info[32];

        if (!task->team) {
            /* zero size collective, no CL or TL */
            strncpy(cl_info, "NoOp", sizeof(cl_info));
            strncpy(tl_info, "NoOp", sizeof(tl_info));
        }
        else if (task->team->context->lib->log_component.name[0] == 'C') {
            /* it's not CL BASIC task */
            ucc_strncpy_safe(cl_info,
                             task->team->context->lib->log_component.name,
                             sizeof(cl_info));
            ucc_coll_task_components_str(task, tl_info, &tl_info_len);
        } else {
            ucc_strncpy_safe(cl_info, "CL_BASIC", sizeof(cl_info));
            ucc_strncpy_safe(tl_info,
                             task->team->context->lib->log_component.name,
                             sizeof(tl_info));
        }
        ucc_coll_args_str(&task->bargs.args, team->rank, team->size, str, len);
        rc = ucc_snprintf_safe(task_info, sizeof(task_info),
                               "; %s {%s}, team_id %d",
                               cl_info, tl_info, team->id);
        if (rc < 0) {
            return;
        }
        strncat(str, task_info, len - strlen(str));
    }

    if (verbosity >= UCC_LOG_LEVEL_DEBUG) {
        char task_info[64];
        ucc_snprintf_safe(task_info, sizeof(task_info),
                          " rank %u, ctx_rank %u, seq_num %d, req %p",
                          team->rank,
                          ucc_ep_map_eval(team->ctx_map, team->rank),
                          task->seq_num, task);
        strncat(str, task_info, len - strlen(str));
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

int ucc_ep_map_is_identity(const ucc_ep_map_t *map)
{
    if ((map->type == UCC_EP_MAP_FULL) ||
        ((map->type == UCC_EP_MAP_STRIDED) &&
        (map->strided.start == 0) &&
        (map->strided.stride == 1))) {
        return 1;
    } else {
        return 0;
    }
}

static inline int ucc_ep_map_is_reverse(ucc_ep_map_t *map,
                                        int reversed_reordered_flag)
{
    return (((map->type == UCC_EP_MAP_STRIDED) &&
           (map->strided.start == map->ep_num - 1) &&
           (map->strided.stride == -1)) || reversed_reordered_flag);
}

ucc_status_t ucc_ep_map_create_inverse(ucc_ep_map_t map, ucc_ep_map_t *inv_map,
                                       int reversed_reordered_flag)
{
    ucc_ep_map_t inv;
    ucc_rank_t   i, r;
    ucc_rank_t   max_rank;

    if (ucc_ep_map_is_reverse(&map, reversed_reordered_flag)) {
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
