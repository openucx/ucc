/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

 #ifndef UCC_TL_UCP_TASK_H_
 #define UCC_TL_UCP_TASK_H_

 #include "tl_ucp.h"
 #include "tl_ucp_tag.h"
 #include "coll_patterns/recursive_knomial.h"
 #include "coll_patterns/double_binary_tree.h"
 #include "components/mc/base/ucc_mc_base.h"

#define UCC_TL_UCP_TASK_PLUGIN_MAX_DATA 128

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_ucp_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_ucp_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_ucp_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

typedef struct ucc_tl_ucp_allreduce_sw_pipeline
    ucc_tl_ucp_allreduce_sw_pipeline;
typedef struct ucc_tl_ucp_allreduce_sw_host_allgather
    ucc_tl_ucp_allreduce_sw_host_allgather;
typedef struct ucc_tl_ucp_dpu_offload_buf_info
    ucc_tl_ucp_dpu_offload_buf_info_t;

enum ucc_tl_ucp_task_flags {
    /*indicates whether subset field of tl_ucp_task is set*/
    UCC_TL_UCP_TASK_FLAG_SUBSET = UCC_BIT(0),
};

typedef struct ucc_tl_ucp_task {
    ucc_coll_task_t super;
    uint32_t        flags;
    union {
        struct {
            uint32_t                    send_posted;
            uint32_t                    send_completed;
            uint32_t                    recv_posted;
            uint32_t                    recv_completed;
            uint32_t                    tag;
        } tagged;
        struct {
            uint32_t                    put_posted;
            uint32_t                    put_completed;
            uint32_t                    get_posted;
            uint32_t                    get_completed;
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
            ucp_mem_h               dst_memh;
            ucp_mem_h               src_memh;
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
        struct {
            uint32_t                tokens;
            uint32_t                npolls;
        } alltoall_onesided;
        char                        plugin_data[UCC_TL_UCP_TASK_PLUGIN_MAX_DATA];
    };
    uint32_t flush_posted;
    uint32_t flush_completed;
} ucc_tl_ucp_task_t;

static inline void ucc_tl_ucp_task_reset(ucc_tl_ucp_task_t *task,
                                         ucc_status_t status)
{
    task->tagged.send_posted    = 0;
    task->tagged.send_completed = 0;
    task->tagged.recv_posted    = 0;
    task->tagged.recv_completed = 0;
    task->flush_posted          = 0;
    task->flush_completed       = 0;
    task->super.status          = status;
}

static inline ucc_tl_ucp_task_t *ucc_tl_ucp_get_task(ucc_tl_ucp_team_t *team)
{
    ucc_tl_ucp_context_t *ctx  = UCC_TL_UCP_TEAM_CTX(team);
    ucc_tl_ucp_task_t    *task = ucc_mpool_get(&ctx->req_mp);;

    UCC_TL_UCP_PROFILE_REQUEST_NEW(task, "tl_ucp_task", 0);
    task->super.flags       = 0;
    task->flags             = 0;
    task->n_polls           = ctx->cfg.n_polls;
    task->super.team        = &team->super.super;
    task->subset.map.type   = UCC_EP_MAP_FULL;
    task->subset.map.ep_num = UCC_TL_TEAM_SIZE(team);
    task->subset.myrank     = UCC_TL_TEAM_RANK(team);
    ucc_tl_ucp_task_reset(task, UCC_OPERATION_INITIALIZED);
    return task;
}

static inline void ucc_tl_ucp_put_task(ucc_tl_ucp_task_t *task)
{
    UCC_TL_UCP_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
}

typedef struct ucc_tl_ucp_schedule {
    ucc_schedule_pipelined_t super;
    ucc_mc_buffer_header_t  *scratch_mc_header;
    union {
        ptrdiff_t frag_offset;
    } reduce_srg_kn;
} ucc_tl_ucp_schedule_t;

static inline ucc_status_t ucc_tl_ucp_get_schedule(ucc_tl_ucp_team_t *team,
                                                   ucc_base_coll_args_t *args,
                                                   ucc_tl_ucp_schedule_t **schedule)
{
    ucc_tl_ucp_context_t  *ctx = UCC_TL_UCP_TEAM_CTX(team);

    *schedule = ucc_mpool_get(&ctx->req_mp);
    if (ucc_unlikely(!(*schedule))) {
        return UCC_ERR_NO_MEMORY;
    }

    UCC_TL_UCP_PROFILE_REQUEST_NEW(schedule, "tl_ucp_sched", 0);
    return ucc_schedule_init(&((*schedule)->super.super), args,
                             &team->super.super);
}

static inline void ucc_tl_ucp_put_schedule(ucc_schedule_t *schedule)
{
    UCC_TL_UCP_PROFILE_REQUEST_FREE(schedule);
    ucc_mpool_put(schedule);
}

#endif
