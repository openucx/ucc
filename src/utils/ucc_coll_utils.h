/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_COLL_UTILS_H_
#define UCC_COLL_UTILS_H_

#include "config.h"
#include "ucc_datastruct.h"
#include "ucc_math.h"
#include "utils/ucc_time.h"
#include "utils/ucc_assert.h"
#include <string.h>

#define UCC_COLL_TYPE_NUM (ucc_ilog2(UCC_COLL_TYPE_LAST - 1) + 1)

#define UCC_COLL_TYPE_ALL ((UCC_COLL_TYPE_LAST << 1) - 3)

#define UCC_MEMORY_TYPE_ASYMMETRIC                                             \
    ((ucc_memory_type_t)((int)UCC_MEMORY_TYPE_LAST + 1))

#define UCC_MEMORY_TYPE_NOT_APPLY                                              \
    ((ucc_memory_type_t)((int)UCC_MEMORY_TYPE_LAST + 2))

#define UCC_MSG_SIZE_INVALID SIZE_MAX

#define UCC_MSG_SIZE_ASYMMETRIC (UCC_MSG_SIZE_INVALID - 1)

#define UCC_IS_INPLACE(_args)                                                  \
    (((_args).mask & UCC_COLL_ARGS_FIELD_FLAGS) &&                             \
     ((_args).flags & UCC_COLL_ARGS_FLAG_IN_PLACE))

#define UCC_IS_PERSISTENT(_args)                                               \
    (((_args).mask & UCC_COLL_ARGS_FIELD_FLAGS) &&                             \
     ((_args).flags & UCC_COLL_ARGS_FLAG_PERSISTENT))

#define UCC_IS_ROOT(_args, _myrank) ((_args).root == (_myrank))

#define UCC_COLL_TIMEOUT_REQUIRED(_task)                                       \
    (((_task)->bargs.args.mask & UCC_COLL_ARGS_FIELD_FLAGS) &&                 \
     ((_task)->bargs.args.flags & UCC_COLL_ARGS_FLAG_TIMEOUT))

#define UCC_COLL_SET_TIMEOUT(_task, _timeout) do {                             \
        (_task)->bargs.args.mask   |= UCC_COLL_ARGS_FIELD_FLAGS;               \
        (_task)->bargs.args.flags  |= UCC_COLL_ARGS_FLAG_TIMEOUT;              \
        (_task)->bargs.args.timeout = _timeout;                                \
        (_task)->start_time   = ucc_get_time();                                \
    } while(0)

#define UCC_COLL_ARGS_COUNT64(_args)                                           \
    (((_args)->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&                            \
     ((_args)->flags & UCC_COLL_ARGS_FLAG_COUNT_64BIT))

#define UCC_COLL_ARGS_DISPL64(_args)                                           \
    (((_args)->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&                            \
     ((_args)->flags & UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT))

#define UCC_COLL_IS_SRC_CONTIG(_args)                                          \
    (((_args)->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&                            \
     ((_args)->flags & UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER))

#define UCC_COLL_IS_DST_CONTIG(_args)                                          \
    (((_args)->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&                            \
     ((_args)->flags & UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER))

#define UCC_COLL_ARGS_CONTIG_BUFFER(_args)                                     \
    (UCC_COLL_IS_SRC_CONTIG(_args) && UCC_COLL_IS_DST_CONTIG(_args))

#define UCC_COLL_ARGS_ACTIVE_SET(_args)                                        \
    ((_args)->mask & UCC_COLL_ARGS_FIELD_ACTIVE_SET)

#define UCC_MEM_TYPE_MASK_FULL (UCC_BIT(UCC_MEMORY_TYPE_HOST) |                \
                                UCC_BIT(UCC_MEMORY_TYPE_CUDA) |                \
                                UCC_BIT(UCC_MEMORY_TYPE_CUDA_MANAGED) |        \
                                UCC_BIT(UCC_MEMORY_TYPE_ROCM) |                \
                                UCC_BIT(UCC_MEMORY_TYPE_ROCM_MANAGED))

static inline int ucc_coll_args_is_reduction(ucc_coll_type_t ct)
{
    if (ct == UCC_COLL_TYPE_ALLREDUCE || ct == UCC_COLL_TYPE_REDUCE ||
        ct == UCC_COLL_TYPE_REDUCE_SCATTER ||
        ct == UCC_COLL_TYPE_REDUCE_SCATTERV) {
        return 1;
    }
    return 0;
}

static inline size_t
ucc_coll_args_get_count(const ucc_coll_args_t *args, const ucc_count_t *counts,
                        ucc_rank_t idx)
{
    if (UCC_COLL_ARGS_COUNT64(args)) {
        return ((uint64_t *)counts)[idx];
    }
    return ((uint32_t *)counts)[idx];
}

static inline void
ucc_coll_args_set_count(const ucc_coll_args_t *args, const ucc_count_t *counts,
                        ucc_rank_t idx, size_t val)
{
    if (UCC_COLL_ARGS_COUNT64(args)) {
        ((uint64_t *)counts)[idx] = (uint64_t)val;
    } else {
        ((uint32_t *)counts)[idx] = (uint32_t)val;
    }
}

static inline size_t ucc_coll_args_get_max_count(const ucc_coll_args_t *args,
                                                 const ucc_count_t *    counts,
                                                 ucc_rank_t             size)
{
    size_t max_count = 0, c;
    int    i;

    for (i = 0; i < size; i++) {
        c = ucc_coll_args_get_count(args, counts, i);
        if (c > max_count) {
            max_count = c;
        }
    }
    return max_count;
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

static inline void
ucc_coll_args_set_displacement(const ucc_coll_args_t *args,
                               const ucc_aint_t *displacements, ucc_rank_t idx,
                               size_t val)
{
    if (UCC_COLL_ARGS_DISPL64(args)) {
        ((uint64_t *)displacements)[idx] = (uint64_t)val;
    } else {
        ((uint32_t *)displacements)[idx] = (uint32_t)val;
    }
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

/* Check if the displacements are contig, whether or not the user passed
   UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER in coll args */
static inline int ucc_coll_args_is_disp_contig(const ucc_coll_args_t *args,
                                               ucc_rank_t size)
{
    size_t     count_accumulator = 0;
    size_t     disps;
    ucc_rank_t i;

    if (!UCC_COLL_IS_DST_CONTIG(args)) {
        for (i = 0; i < size; i++) {
            disps = ucc_coll_args_get_displacement(
                        args, args->dst.info_v.displacements, i);
            if (disps != count_accumulator) {
                return 0;
            }
            count_accumulator += ucc_coll_args_get_count(
                                    args, args->dst.info_v.counts, i);
        }
    }

    return 1;
}

static inline size_t
ucc_coll_args_get_v_buffer_size(const ucc_coll_args_t *args,
                                const ucc_count_t *counts,
                                const ucc_aint_t *displacements,
                                ucc_rank_t size)
{
    ucc_rank_t i;
    size_t max_disp, idx_count;

    max_disp = ucc_coll_args_get_displacement(args, displacements, 0);
    idx_count = ucc_coll_args_get_count(args, counts, 0);
    for (i = 1; i < size; i++) {
        size_t disp_i = ucc_coll_args_get_displacement(args, displacements, i);
        if (disp_i > max_disp) {
            max_disp = disp_i;
            idx_count = ucc_coll_args_get_count(args, counts, i);
        }
    }

    return max_disp + idx_count;
}
typedef struct ucc_base_coll_args ucc_base_coll_args_t;

ucc_coll_type_t   ucc_coll_type_from_str(const char *str);

ucc_memory_type_t ucc_mem_type_from_str(const char *str);

size_t            ucc_coll_args_msgsize(const ucc_coll_args_t *args,
                                        ucc_rank_t rank, ucc_rank_t size);

ucc_memory_type_t ucc_coll_args_mem_type(const ucc_coll_args_t *args,
                                         ucc_rank_t rank);

/* Convert rank from subset space to rank space (UCC team space) */
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
        r = UCC_RANK_INVALID;
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
void ucc_coll_str(const ucc_coll_task_t *task, char *str, size_t len,
                  int verbosity);

void ucc_coll_args_str(const ucc_coll_args_t *args, ucc_rank_t trank,
                       ucc_rank_t tsize, char *str, size_t len);

/* Creates a rank map that reverses rank order, ie
   rank r -> size - 1 - r */
ucc_ep_map_t ucc_ep_map_create_reverse(ucc_rank_t size);

/* Creates an inverse mapping for a given map,
   only if given map is not a reverse map
   or a reordered map within the reverse direction task.
   @param [in] map                       given map
   @param [in] inv_map                   output map
   @param [in] reversed_reordered_flag   1 if is reverse direction task and
                                         reorder ranks is configured as yes,
                                         0 otherwise. */
ucc_status_t ucc_ep_map_create_inverse(ucc_ep_map_t map, ucc_ep_map_t *inv_map,
                                       int reversed_reordered_flag);

ucc_status_t ucc_ep_map_create_nested(ucc_ep_map_t *base_map,
                                      ucc_ep_map_t *sub_map,
                                      ucc_ep_map_t *out);

int ucc_ep_map_is_identity(const ucc_ep_map_t *map);

void ucc_ep_map_destroy_nested(ucc_ep_map_t *out);

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

static inline size_t ucc_buffer_vector_block_offset(ucc_count_t *counts,
                                                    int is64,
                                                    ucc_rank_t rank)
{
    size_t offset = 0;
    ucc_rank_t i;

    if (is64) {
        for (i = 0; i < rank; i++) {
            offset += ((uint64_t *)counts)[i];
        }
    } else {
        for (i = 0; i < rank; i++) {
            offset += ((uint32_t *)counts)[i];
        }
    }
    return offset;
}

/* Given the rank space A (e.g. core ucc team), a subset B (e.g. active set
   within the core team), the ep_map that maps ranks from the subset B to A,
   and the rank of a process within A. The function below computes the local
   rank of the process within subset B.
   i.e., convert from rank space (UCC team) to subset space */
static inline ucc_rank_t ucc_ep_map_local_rank(ucc_ep_map_t map,
                                               ucc_rank_t   rank)
{
    ucc_rank_t i, local_rank = UCC_RANK_INVALID;
    int64_t    vrank;

    switch(map.type) {
    case UCC_EP_MAP_FULL:
        local_rank = rank;
        break;
    case UCC_EP_MAP_STRIDED:
        vrank = (int64_t)rank - (int64_t)map.strided.start;
        vrank /= map.strided.stride;
        ucc_assert(vrank >= 0 && vrank < map.ep_num);
        local_rank = (ucc_rank_t)vrank;
        break;
    case UCC_EP_MAP_ARRAY:
    case UCC_EP_MAP_CB:
        for (i = 0; i < map.ep_num; i++) {
            if (rank == ucc_ep_map_eval(map, i)) {
                local_rank = i;
                break;
            }
        }
    default:
        break;
    }
    return local_rank;
}

static inline ucc_ep_map_t ucc_active_set_to_ep_map(ucc_coll_args_t *args)
{
    ucc_ep_map_t map;

    map.type           = UCC_EP_MAP_STRIDED;
    map.ep_num         = args->active_set.size;
    map.strided.start  = args->active_set.start;
    map.strided.stride = args->active_set.stride;
    return map;
}

static inline size_t ucc_buffer_block_count_aligned(size_t total_count,
                                                    ucc_rank_t n_blocks,
                                                    ucc_rank_t block,
                                                    int alignment)
{
    size_t block_count = ucc_align_up_pow2(ucc_max(total_count / n_blocks, 1),
                                           alignment);
    size_t offset      = block_count * block;

    return (total_count < offset) ? 0: ucc_min(total_count - offset, block_count);
}

static inline size_t ucc_buffer_block_offset_aligned(size_t total_count,
                                                     ucc_rank_t n_blocks,
                                                     ucc_rank_t block,
                                                     int alignment)
{
    size_t block_count = ucc_align_up_pow2(ucc_max(total_count / n_blocks, 1),
                                           alignment);
    size_t offset      = block_count * block;

    return ucc_min(offset, total_count);
}

/* Returns non-zero if collective defined by args operates on predefined dt.
   @param [in] args        pointer to the collective args.
   @param [in] rank        rank to check, used only for rooted collective
                           operations. */
int ucc_coll_args_is_predefined_dt(const ucc_coll_args_t *args, ucc_rank_t rank);

int ucc_coll_args_is_mem_symmetric(const ucc_coll_args_t *args, ucc_rank_t rank);

int ucc_coll_args_is_rooted(ucc_coll_type_t ct);

typedef struct ucc_buffer_info_asymmetric_memtype ucc_buffer_info_asymmetric_memtype_t;
typedef struct ucc_mc_buffer_header ucc_mc_buffer_header_t;

ucc_status_t
ucc_coll_args_init_asymmetric_buffer(ucc_coll_args_t *args,
                                       ucc_team_h team,
                                       ucc_buffer_info_asymmetric_memtype_t *save_info);

ucc_status_t
ucc_coll_args_free_asymmetric_buffer(ucc_coll_task_t *task);

ucc_status_t ucc_copy_asymmetric_buffer(ucc_coll_task_t *task);

#endif
