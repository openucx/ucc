/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_COLL_UTILS_H_
#define UCC_COLL_UTILS_H_

#include "config.h"
#include "ucc_datastruct.h"
#include "ucc_math.h"
#include <string.h>
#include "utils/ucc_time.h"

#define UCC_COLL_TYPE_NUM (ucc_ilog2(UCC_COLL_TYPE_LAST - 1) + 1)

#define UCC_COLL_TYPE_ALL ((UCC_COLL_TYPE_LAST << 1) - 3)

#define UCC_MEMORY_TYPE_ASSYMETRIC                                             \
    ((ucc_memory_type_t)((int)UCC_MEMORY_TYPE_LAST + 1))

#define UCC_MEMORY_TYPE_NOT_APPLY                                              \
    ((ucc_memory_type_t)((int)UCC_MEMORY_TYPE_LAST + 2))

#define UCC_MSG_SIZE_INVALID SIZE_MAX

#define UCC_MSG_SIZE_ASSYMETRIC (UCC_MSG_SIZE_INVALID - 1)

#define UCC_IS_INPLACE(_args) \
    (((_args).mask & UCC_COLL_ARGS_FIELD_FLAGS) && \
     ((_args).flags & UCC_COLL_ARGS_FLAG_IN_PLACE))

#define UCC_IS_PERSISTENT(_args) \
    (((_args).mask & UCC_COLL_ARGS_FIELD_FLAGS) && \
     ((_args).flags & UCC_COLL_ARGS_FLAG_PERSISTENT))

#define UCC_COLL_TIMEOUT_REQUIRED(_task)                       \
    (((_task)->bargs.args.mask & UCC_COLL_ARGS_FIELD_FLAGS) && \
     ((_task)->bargs.args.flags & UCC_COLL_ARGS_FLAG_TIMEOUT))

#define UCC_COLL_SET_TIMEOUT(_task, _timeout) do {                 \
        (_task)->bargs.args.mask   |= UCC_COLL_ARGS_FIELD_FLAGS;   \
        (_task)->bargs.args.flags  |= UCC_COLL_ARGS_FLAG_TIMEOUT;  \
        (_task)->bargs.args.timeout = _timeout;                    \
        (_task)->start_time   = ucc_get_time();                    \
    } while(0)

#define UCC_COLL_ARGS_COUNT64(_args)                                           \
    (((_args)->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&                            \
     ((_args)->flags & UCC_COLL_ARGS_FLAG_COUNT_64BIT))

#define UCC_COLL_ARGS_DISPL64(_args)                                           \
    (((_args)->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&                            \
     ((_args)->flags & UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT))

static inline size_t
ucc_coll_args_get_count(const ucc_coll_args_t *args, const ucc_count_t *counts,
                        ucc_rank_t idx)
{
    if (UCC_COLL_ARGS_COUNT64(args)) {
        return ((uint64_t *)counts)[idx];
    }
    return ((uint32_t *)counts)[idx];
}

static inline size_t
ucc_coll_args_get_displacement(const ucc_coll_args_t *args,
                               const ucc_aint_t *displacements, ucc_rank_t idx)
{
    if (UCC_COLL_ARGS_DISPL64(args)) {
        return ((uint64_t *)displacements)[idx];
    }
    return ((uint32_t *)displacements)[idx];
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

static inline size_t
ucc_coll_args_get_total_count(const ucc_coll_args_t *args,
                              const ucc_count_t *counts, ucc_rank_t size)
{
    size_t count = 0;
    ucc_rank_t i;
    // TODO switch to base args and cache total count there - can we do it ?
    if ((args->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
        (args->flags & UCC_COLL_ARGS_FLAG_COUNT_64BIT)) {
        for (i = 0; i < size; i++) {
            count += ((uint64_t *)counts)[i];
        }
    } else {
        for (i = 0; i < size; i++) {
            count += ((uint32_t *)counts)[i];
        }
    }

    return count;
}
typedef struct ucc_base_coll_args ucc_base_coll_args_t;

ucc_coll_type_t   ucc_coll_type_from_str(const char *str);

ucc_memory_type_t ucc_mem_type_from_str(const char *str);

size_t            ucc_coll_args_msgsize(const ucc_base_coll_args_t *bargs);

ucc_memory_type_t ucc_coll_args_mem_type(const ucc_base_coll_args_t *bargs);


static inline ucc_rank_t ucc_ep_map_eval(ucc_ep_map_t map, ucc_rank_t rank)
{
    ucc_rank_t r;
    switch(map.type) {
    case UCC_EP_MAP_FULL:
        r = rank;
        break;
    case UCC_EP_MAP_STRIDED:
        r = map.strided.start + rank*map.strided.stride;
        break;
    case UCC_EP_MAP_ARRAY:
        r = *((ucc_rank_t*)((ptrdiff_t)map.array.map + rank*map.array.elem_size));
        break;
    case UCC_EP_MAP_CB:
        r = (ucc_rank_t)map.cb.cb(rank, map.cb.cb_ctx);
        break;
    default:
        r = -1;
    }
    return r;
}

/* Builds ucc_ep_map_t from the array of ucc_rank_t. The routine tries
   to search for a strided pattern to optimize storage and map lookup.
   @param [in] array       pointer to the array to build the map from
   @param [in] size        size of the array
   @param [in] full_size   if size == full_size and stride=1 is detected
                           the map can be optimized to be FULL
   @param [in] need_free   if set to 1 the input @array is freed and set
                           to NULL in the case of strided pattern.
                           User must check and free the array otherwise. */
ucc_ep_map_t ucc_ep_map_from_array(ucc_rank_t **array, ucc_rank_t size,
                                   ucc_rank_t full_size, int need_free);

/* Builds ucc_ep_map_t from the array of uint64_t. The routine tries
   to search for a strided pattern to optimize storage and map lookup.
   @param [in] array       pointer to the array to build the map from
   @param [in] size        size of the array
   @param [in] full_size   if size == full_size and stride=1 is detected
                           the map can be optimized to be FULL
   @param [in] need_free   if set to 1 the input @array is freed and set
                           to NULL in the case of strided pattern.
                           User must check and free the array otherwise. */
ucc_ep_map_t ucc_ep_map_from_array_64(uint64_t **array, ucc_rank_t size,
                                   ucc_rank_t full_size, int need_free);

typedef struct ucc_coll_task ucc_coll_task_t;
void ucc_coll_str(const ucc_coll_task_t *task, char *str, size_t len);

/* Creates a rank map that reverses rank order, ie
   rank r -> size - 1 - r */
ucc_ep_map_t ucc_ep_map_create_reverse(ucc_rank_t size);

/* Creates an inverse mapping for a given map */
ucc_status_t ucc_ep_map_create_inverse(ucc_ep_map_t map, ucc_ep_map_t *inv_map);

void ucc_ep_map_destroy(ucc_ep_map_t *map);

/* The two helper routines below are used to partition a buffer
   consisiting of total_count elements into blocks.
   This is used, e.g., in ReduceScatter or during fragmentation
   process.
   First total_count is devided into n_blocks. If the devision
   has remainder then it is evenly distributed among first blocks.
*/
static inline size_t ucc_buffer_block_count(size_t     total_count,
                                            ucc_rank_t n_blocks,
                                            ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;

    return (block < left) ? block_count + 1 : block_count;
}

static inline size_t ucc_buffer_block_offset(size_t     total_count,
                                             ucc_rank_t n_blocks,
                                             ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;
    size_t offset      = block * block_count + left;

    return (block < left) ? offset - (left - block) : offset;
}
#endif
