/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_COLL_H_
#define UCC_TL_UCP_COLL_H_

#include "tl_ucp.h"
#include "tl_ucp_task.h"
#include "coll_patterns/recursive_knomial.h"

#define UCC_UUNITS_AUTO_RADIX 4
#define UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR 9

ucc_status_t ucc_tl_ucp_team_default_score_str_alloc(ucc_tl_ucp_team_t *team,
    char *default_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR]);

void ucc_tl_ucp_team_default_score_str_free(
    char *default_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR]);

#define CALC_KN_TREE_DIST(_size, _radix, _dist)                               \
    do {                                                                      \
        _dist = 1;                                                            \
        while (_dist * _radix < _size) {                                      \
            _dist *= _radix;                                                  \
        }                                                                     \
    } while (0)

#define VRANK(_rank, _root, _team_size)                                       \
    (((_rank) - (_root) + (_team_size)) % (_team_size))

#define INV_VRANK(_rank, _root, _team_size)                                   \
    (((_rank) + (_root)) % (_team_size))

#define EXEC_TASK_TEST(_phase, _errmsg, _etask) do {                           \
    if (_etask != NULL) {                                                      \
        status = ucc_ee_executor_task_test(_etask);                            \
        if (status > 0) {                                                      \
            task->super.status = UCC_INPROGRESS;                               \
            SAVE_STATE(_phase);                                                \
            return;                                                            \
        }                                                                      \
        ucc_ee_executor_task_finalize(_etask);                                 \
        _etask = NULL;                                                         \
        if (ucc_unlikely(status < 0)) {                                        \
            tl_error(UCC_TASK_LIB(task), _errmsg);                             \
            task->super.status = status;                                       \
            return;                                                            \
        }                                                                      \
    }                                                                          \
} while(0)

#define EXEC_TASK_WAIT(_etask, ...)                                            \
    do {                                                                       \
        if (_etask != NULL) {                                                  \
            do {                                                               \
                status = ucc_ee_executor_task_test(_etask);                    \
            } while (status > 0);                                              \
            if (status < 0) {                                                  \
                tl_error(UCC_TASK_LIB(task), "failure in ee task ee task");    \
                task->super.status = status;                                   \
                return __VA_ARGS__;                                            \
            }                                                                  \
            ucc_ee_executor_task_finalize(_etask);                             \
            if (ucc_unlikely(status < 0)) {                                    \
                tl_error(UCC_TASK_LIB(task), "failed to finalize ee task");    \
                task->super.status = status;                                   \
                return __VA_ARGS__;                                            \
            }                                                                  \
        }                                                                      \
    } while (0)

typedef char* (*ucc_tl_ucp_score_str_get_fn_t)(ucc_tl_ucp_team_t *team);
typedef struct ucc_tl_ucp_default_alg_desc {
    char                          *select_str;
    ucc_tl_ucp_score_str_get_fn_t  str_get_fn;
} ucc_tl_ucp_default_alg_desc_t;

<<<<<<< HEAD
=======
enum ucc_tl_ucp_task_flags {
    /*indicates whether subset field of tl_ucp_task is set*/
    UCC_TL_UCP_TASK_FLAG_SUBSET      = UCC_BIT(0),
    /* indicates usage of dynamic segments */
    UCC_TL_UCP_TASK_FLAG_USE_DYN_SEG = UCC_BIT(1),
    /* indicates onesided operations have been started */
    UCC_TL_UCP_TASK_FLAG_OPS_STARTED = UCC_BIT(2),
};

typedef struct ucc_tl_ucp_allreduce_sw_pipeline
    ucc_tl_ucp_allreduce_sw_pipeline;
typedef struct ucc_tl_ucp_allreduce_sw_host_allgather
    ucc_tl_ucp_allreduce_sw_host_allgather;
typedef struct ucc_tl_ucp_dpu_offload_buf_info
    ucc_tl_ucp_dpu_offload_buf_info_t;

/* Structure to hold dynamic segment exchange parameters and buffers */
typedef struct {
    ucc_tl_ucp_task_t  *task;
    void               *src_pack_buffer;
    void               *dst_pack_buffer;
    size_t              src_pack_size;
    size_t              dst_pack_size;
    size_t              max_individual_pack_size;
    size_t              exchange_size;
    ucc_mem_map_memh_t *src_memh_pack;
    ucc_mem_map_memh_t *dst_memh_pack;
    void               *exchange_buffer;
    ucc_mem_map_memh_t *src_memh_local;
    ucc_mem_map_memh_t *dst_memh_local;
    size_t             *global_sizes;
} ucc_tl_ucp_dyn_seg_args_t;

typedef struct ucc_tl_ucp_task {
    ucc_coll_task_t super;
    uint32_t        flags;
    union {
        struct {
            uint32_t        send_posted;
            uint32_t        send_completed;
            uint32_t        recv_posted;
            uint32_t        recv_completed;
            uint32_t        tag;
        } tagged;
        struct {
            uint32_t        put_posted;
            uint32_t        put_completed;
            uint32_t        get_posted;
            uint32_t        get_completed;
        } onesided;
    };
    uint32_t        n_polls;
    ucc_subset_t    subset;
    union {
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
        } barrier;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            void                   *scratch;
            void                   *reduce_bufs[UCC_EE_EXECUTOR_NUM_BUFS];
            ucc_mc_buffer_header_t *scratch_mc_header;
            ucc_ee_executor_task_t *etask;
            ucc_ee_executor_t      *executor;
        } allreduce_kn;
        struct {
            ucc_tl_ucp_allreduce_sw_pipeline          *pipe;
            ucs_status_ptr_t                          *put_requests;
            ucc_tl_ucp_allreduce_sw_host_allgather    *allgather_data;
            ucc_coll_task_t                           *allgather_task;
            ucc_ee_executor_task_t                    *reduce_task;
            ucc_tl_ucp_dpu_offload_buf_info_t         *bufs;
        } allreduce_sliding_window;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            void                   *scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
            ucc_ee_executor_task_t *etask;
            ucc_ee_executor_t      *executor;
            size_t                  max_seg;
        } reduce_scatter_kn;
        struct {
            void                   *scratch;
            size_t                  max_block_count;
            ucc_ep_map_t            inv_map;
            int                     n_frags;
            int                     frag;
            char                    s_scratch_busy[2];
            ucc_ee_executor_task_t *etask;
            ucc_ee_executor_t      *executor;
        } reduce_scatter_ring;
        struct {
            void                   *scratch;
            size_t                  max_block_count;
            ucc_ep_map_t            inv_map;
            int                     n_frags;
            int                     frag;
            char                    s_scratch_busy[2];
            ucc_ee_executor_task_t *etask;
            ucc_ee_executor_t      *executor;
        } reduce_scatterv_ring;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            ucc_rank_t              recv_dist;
            ptrdiff_t               send_offset;
            ptrdiff_t               recv_offset;
            size_t                  recv_size;
        } scatter_kn;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            void                   *sbuf;
            ucc_tl_ucp_copy_task_t *copy_task;
            ucc_rank_t              recv_dist;
        } allgather_kn;
        struct {
            /*
             * get send/recv block depends on subset type being used.
             * For service allgather we need to get context endpoints but keep
             * subset numbering.
             * For regular allgather with rank reordering both endpoints
             * and blocks permutation are necessary.
             */
            ucc_rank_t (*get_send_block)(ucc_subset_t *subset,
                                         ucc_rank_t trank,
                                         ucc_rank_t tsize,
                                         int step);
            ucc_rank_t (*get_recv_block)(ucc_subset_t *subset,
                                         ucc_rank_t trank,
                                         ucc_rank_t tsize,
                                         int step);
        } allgather_ring;
        struct {
            int                     nreqs; // number of send/recv requests in progress
            ucc_tl_ucp_copy_task_t *copy_task;
        } allgather_linear;
        struct {
            ucc_mc_buffer_header_t *scratch_header;
            size_t                  scratch_size;
        } allgather_bruck;
        struct {
            uint32_t                i;
            int                     data_expected;
        } allgather_sparbit;
        struct {
            ucc_rank_t              dist;
            uint32_t                radix;
        } bcast_kn;
        struct {
            ucc_dbt_single_tree_t   t1;
            ucc_dbt_single_tree_t   t2;
            int                     state;
        } bcast_dbt;
        struct {
            ucc_rank_t              dist;
            ucc_rank_t              max_dist;
            int                     children_per_cycle;
            uint32_t                radix;
            int                     phase;
            void                   *scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
            ucc_ee_executor_task_t *etask;
            ucc_ee_executor_t      *executor;
        } reduce_kn;
        struct {
            int                     state;
            ucc_dbt_single_tree_t   trees[2];
            int                     reduction_comp[2];
            int                     send_comp[2];
            void                   *scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
            ucc_ee_executor_task_t *etask;
            ucc_ee_executor_t      *executor;
        } reduce_dbt;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            ucc_rank_t              dist;
            ucc_rank_t              max_dist;
            uint32_t                radix;
            void *                  scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
        } gather_kn;
        struct {
            size_t                  merge_buf_size;
            ucc_mc_buffer_header_t *scratch_mc_header;
            size_t                  byte_send_limit;
            int                     phase;
            uint32_t                radix;
            uint32_t                cur_radix;
            uint32_t                iteration;
            ucc_rank_t              cur_out;
            size_t                  traffic_in;
            size_t                  traffic_out;
            ucc_rank_t              num_in;
            ucc_rank_t              num2send;
            ucc_rank_t              num2recv;
        } alltoallv_hybrid;
        struct {
            ucc_mc_buffer_header_t *scratch_mc_header;
            ucc_ee_executor_task_t *etask;
            void                   *src;
            void                   *dst;
            ucc_rank_t              iteration;
            int                     phase;
        } alltoall_bruck;
        char                        plugin_data[UCC_TL_UCP_TASK_PLUGIN_MAX_DATA];
    };
    struct {
        ucc_mem_map_memh_t        *src_local;
        ucc_mem_map_memh_t        *dst_local;
        ucc_mem_map_memh_t       **src_global;
        ucc_mem_map_memh_t       **dst_global;
        ucc_tl_ucp_dyn_seg_args_t *exchange_args;
        void                      *global_buffer;
        ucc_service_coll_req_t    *scoll_req_sizes; /* For sizes allgather */
        ucc_service_coll_req_t    *scoll_req_data; /* For data ex allgather */
        int                        exchange_step;
        ucc_status_t               exchange_status;
    } dynamic_segments;
} ucc_tl_ucp_task_t;

typedef struct ucc_tl_ucp_schedule {
    ucc_schedule_pipelined_t super;
    ucc_mc_buffer_header_t  *scratch_mc_header;
    union {
        ptrdiff_t frag_offset;
    } reduce_srg_kn;
} ucc_tl_ucp_schedule_t;

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_ucp_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_ucp_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_ucp_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

>>>>>>> fc580dc2 (TL/UCP: enable nonblocking dynamic segments)
#define AVG_ALPHA(_task) (1.0 / (double)UCC_TL_TEAM_SIZE(TASK_TEAM(_task)))

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *     team,
                                  ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task);

static inline ucc_tl_ucp_task_t *ucc_tl_ucp_init_task(ucc_base_coll_args_t *coll_args,
                                                      ucc_base_team_t *team)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task    = ucc_tl_ucp_get_task(tl_team);

    if (ucc_unlikely(!task)) {
        return NULL;
    }

    ucc_coll_task_init(&task->super, coll_args, team);

    if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args)) {
        task->tagged.tag = (coll_args->args.mask & UCC_COLL_ARGS_FIELD_TAG)
            ? coll_args->args.tag : UCC_TL_UCP_ACTIVE_SET_TAG;
        task->flags        |= UCC_TL_UCP_TASK_FLAG_SUBSET;
        task->subset.map    = ucc_active_set_to_ep_map(&coll_args->args);
        task->subset.myrank =
            ucc_ep_map_local_rank(task->subset.map,
                                  UCC_TL_TEAM_RANK(tl_team));
        ucc_assert(coll_args->args.coll_type == UCC_COLL_TYPE_BCAST);
    } else {
        if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_TAG) {
            task->tagged.tag = coll_args->args.tag;
        } else {
            tl_team->seq_num = (tl_team->seq_num + 1) % UCC_TL_UCP_MAX_COLL_TAG;
            task->tagged.tag = tl_team->seq_num;
        }
    }

    task->super.finalize       = ucc_tl_ucp_coll_finalize;
    return task;
}

#define UCC_TL_UCP_TASK_P2P_COMPLETE(_task)                                    \
    (((_task)->tagged.send_posted == (_task)->tagged.send_completed) &&        \
     ((_task)->tagged.recv_posted == (_task)->tagged.recv_completed))

static inline ucc_status_t ucc_tl_ucp_test(ucc_tl_ucp_task_t *task)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_P2P_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_P2P_COMPLETE(task)) {
            return UCC_OK;
        }
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    }
    return UCC_INPROGRESS;
}

#define UCC_TL_UCP_TASK_RECV_COMPLETE(_task)                                   \
    (((_task)->tagged.recv_posted == (_task)->tagged.recv_completed))

#define UCC_TL_UCP_TASK_SEND_COMPLETE(_task)                                   \
    (((_task)->tagged.send_posted == (_task)->tagged.send_completed))

static inline ucc_status_t ucc_tl_ucp_test_recv(ucc_tl_ucp_task_t *task)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_RECV_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_RECV_COMPLETE(task)) {
            return UCC_OK;
        }
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    }
    return UCC_INPROGRESS;
}

static inline ucc_status_t ucc_tl_ucp_test_send(ucc_tl_ucp_task_t *task)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_SEND_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_SEND_COMPLETE(task)) {
            return UCC_OK;
        }
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    }
    return UCC_INPROGRESS;
}

#define UCC_TL_UCP_TASK_RING_P2P_COMPLETE(_task)                               \
    ((((_task)->tagged.send_posted - (_task)->tagged.send_completed) <= 1) &&  \
     ((_task)->tagged.recv_posted == (_task)->tagged.recv_completed))

static inline ucc_status_t ucc_tl_ucp_test_ring(ucc_tl_ucp_task_t *task)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_RING_P2P_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_RING_P2P_COMPLETE(task)) {
            return UCC_OK;
        }
        ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
    }
    return UCC_INPROGRESS;
}

#define UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(_task)                           \
    (((_task)->onesided.put_posted == (_task)->onesided.put_completed) &&      \
     ((_task)->onesided.get_posted == (_task)->onesided.get_completed) &&      \
     ((_task)->flush_posted == (_task)->flush_completed))

#define UCC_TL_UCP_TASK_ONESIDED_SYNC_COMPLETE(_task, _end)                    \
    (*((long *)(TASK_ARGS(_task).global_work_buffer)) == _end)

static inline ucc_status_t ucc_tl_ucp_test_onesided(ucc_tl_ucp_task_t *task,
                                                    int                sync_end)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task) &&
        UCC_TL_UCP_TASK_ONESIDED_SYNC_COMPLETE(task, sync_end)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task) &&
            UCC_TL_UCP_TASK_ONESIDED_SYNC_COMPLETE(task, sync_end)) {
            return UCC_OK;
        }
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    }
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_ucp_alg_id_to_init(int alg_id, const char *alg_id_str,
                                       ucc_coll_type_t          coll_type,
                                       ucc_memory_type_t        mem_type,
                                       ucc_base_coll_init_fn_t *init);

static inline unsigned
ucc_tl_ucp_get_radix_from_range(ucc_tl_ucp_team_t *team,
                                size_t             msgsize,
                                ucc_memory_type_t  mem_type,
                                ucc_mrange_uint_t *p,
                                ucc_rank_t         default_value)
{
    unsigned radix;

    radix = ucc_mrange_uint_get(p, msgsize, mem_type);

    if (UCC_UUNITS_AUTO == radix) {
        return default_value;
    }
    return radix;
}

/*
 * Get the radix for knomial patterns.
 * If need_scratch is true, the radix is the minimum radix that can be used to fit into scratch buffer.
 * Otherwise, the radix is the minimum radix that can be used to fit into team size.
 */
static inline unsigned ucc_tl_ucp_get_knomial_radix(ucc_tl_ucp_team_t *team,
                                                    size_t             count,
                                                    ucc_datatype_t     dtype,
                                                    ucc_memory_type_t  mem_type,
                                                    ucc_mrange_uint_t *p,
                                                    int need_scratch)
{
    size_t msgsize = count * ucc_dt_size(dtype);
    unsigned opt_radix, cfg_radix, radix;

    opt_radix = (mem_type == UCC_MEMORY_TYPE_HOST) ? team->opt_radix_host :
                                                    team->opt_radix;
    cfg_radix = ucc_tl_ucp_get_radix_from_range(team, msgsize, mem_type, p,
                                                opt_radix);
    if (need_scratch) {
        radix = ucc_knomial_pattern_get_min_radix(cfg_radix, UCC_TL_TEAM_SIZE(team), count);
    } else {
        radix = ucc_min(cfg_radix, UCC_TL_TEAM_SIZE(team));

    }
    return radix;
}

ucc_status_t ucc_tl_ucp_coll_dynamic_segment_init(ucc_coll_args_t *coll_args,
                                                  ucc_tl_ucp_task_t   *task);

ucc_status_t ucc_tl_ucp_coll_dynamic_segment_exchange(ucc_tl_ucp_task_t *task);
ucc_status_t ucc_tl_ucp_coll_dynamic_segment_exchange_nb(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_coll_dynamic_segment_finalize(ucc_tl_ucp_task_t *task);

static inline ucc_status_t ucc_tl_ucp_test_dynamic_segment(ucc_tl_ucp_task_t *task)
{
    if (!(task->flags & UCC_TL_UCP_TASK_FLAG_USE_DYN_SEG)) {
        return UCC_OK;
    }

    if (task->dynamic_segments.exchange_step < 5) {
        return ucc_tl_ucp_coll_dynamic_segment_exchange_nb(task);
    }

    return UCC_OK;
}

#endif
