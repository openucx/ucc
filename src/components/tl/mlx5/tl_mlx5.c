/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"

ucc_status_t ucc_tl_mlx5_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t * base_attr);
ucc_status_t ucc_tl_mlx5_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t *     base_attr);

ucc_status_t ucc_tl_mlx5_get_lib_properties(ucc_base_lib_properties_t *prop);

static ucc_config_field_t ucc_tl_mlx5_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_mlx5_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"ASR_BARRIER", "0", "Boolean - use  service barrier or p2p sync of ASRs",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, asr_barrier), UCC_CONFIG_TYPE_BOOL},

    {"DM_BUF_SIZE", "8k", "Size of the pre-allocated DeviceMemory buffer",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, dm_buf_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"DM_BUF_NUM", "auto", "Number of DM buffers to alloc",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, dm_buf_num),
     UCC_CONFIG_TYPE_ULUNITS},

    {"BLOCK_SIZE", "0",
     "Size of the blocks that are sent using blocked AlltoAll Algorithm",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, block_size), UCC_CONFIG_TYPE_UINT},

    {"NUM_DCI_QPS", "16",
     "Number of parallel DCI QPs that will be used per team",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, num_dci_qps), UCC_CONFIG_TYPE_UINT},

    {"DC_THRESHOLD", "128",
     "If number of nodes >= DC_THRESHOLD then DC QPs "
     "are used instead of RC",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, dc_threshold),
     UCC_CONFIG_TYPE_UINT},

    {"DM_HOST", "n",
     "Use host registered memory instead of DM for "
     "transpose staging",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, dm_host), UCC_CONFIG_TYPE_BOOL},

    {"QP_RNR_RETRY", "7", "IB QP rnr retry count",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, qp_conf.qp_rnr_retry),
     UCC_CONFIG_TYPE_UINT},

    {"QP_RNR_TIMER", "20", "IB QP rnr timer",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, qp_conf.qp_rnr_timer),
     UCC_CONFIG_TYPE_UINT},

    {"QP_RETRY_COUNT", "7", "IB QP retry count",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, qp_conf.qp_retry_cnt),
     UCC_CONFIG_TYPE_UINT},

    {"QP_TIMEOUT", "18", "IB QP timeout",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, qp_conf.qp_timeout),
     UCC_CONFIG_TYPE_UINT},

    {"QP_MAX_ATOMIC", "1", "max num of outstanding atomics in IB QP",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, qp_conf.qp_max_atomic),
     UCC_CONFIG_TYPE_UINT},

    {"MCAST_SX_DEPTH", "512", "Send context depth of the Mcast comm",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, mcast_conf.sx_depth),
     UCC_CONFIG_TYPE_INT},

    {"MCAST_SX_INLINE", "128", "Minimal size for inline data send in Mcast",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, mcast_conf.sx_inline),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"MCAST_RX_DEPTH", "4096", "Recv context depth of the Mcast comm",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, mcast_conf.rx_depth),
     UCC_CONFIG_TYPE_INT},

    {"MCAST_POST_RECV_THRESH", "64",
        "Threshold for posting recv into rx ctx of the Mcast comm",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, mcast_conf.post_recv_thresh),
     UCC_CONFIG_TYPE_INT},

    {"MCAST_WINDOW_SIZE", "64", "Reliability Mcast window size",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, mcast_conf.wsize),
     UCC_CONFIG_TYPE_INT},

    {"MCAST_MAX_PUSH_SEND", "16", "Max number of concurrent send wq for mcast based allgather",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, mcast_conf.max_push_send),
     UCC_CONFIG_TYPE_INT},

    {"MCAST_MAX_EAGER", "65536", "Max msg size to be used for Mcast with the eager protocol",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, mcast_conf.max_eager),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"MCAST_CUDA_MEM_ENABLE", "0", "Enable GPU CUDA memory support for Mcast. GPUDirect RDMA must be enabled",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, mcast_conf.cuda_mem_enabled),
     UCC_CONFIG_TYPE_BOOL},

    {"MCAST_ONE_SIDED_RELIABILITY_ENABLE", "1", "Enable one sided reliability for mcast",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, mcast_conf.one_sided_reliability_enable),
     UCC_CONFIG_TYPE_BOOL},

    {"FANIN_KN_RADIX", "4", "Radix of the knomial tree fanin algorithm",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, fanin_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"SEND_BATCH_SIZE", "8", "number of blocks that are transposed "
    "on the NIC before being sent as a batch to a remote peer",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, block_batch_size),
     UCC_CONFIG_TYPE_UINT},

    {"NBR_SERIALIZED_BATCHES", "2", "number of block batches "
    "(within the set of blocks to be sent to a given remote peer)"
     "serialized on the same device memory chunk",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, nbr_serialized_batches),
     UCC_CONFIG_TYPE_UINT},

    {"NBR_BATCHES_PER_PASSAGE", "32", "",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, nbr_batches_per_passage),
     UCC_CONFIG_TYPE_UINT},

    {NULL}};

static ucc_config_field_t ucc_tl_mlx5_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_mlx5_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {"NET_DEVICES", "", "Specifies which network device(s) to use",
     ucc_offsetof(ucc_tl_mlx5_context_config_t, devices),
     UCC_CONFIG_TYPE_STRING_ARRAY},

    {"MCAST_TIMEOUT", "10000", "Timeout [usec] for the reliability NACK in Mcast",
     ucc_offsetof(ucc_tl_mlx5_context_config_t, mcast_ctx_conf.timeout),
     UCC_CONFIG_TYPE_INT},

    {"MCAST_ENABLE", "0", "Enable Mcast",
     ucc_offsetof(ucc_tl_mlx5_context_config_t, mcast_ctx_conf.mcast_enabled),
     UCC_CONFIG_TYPE_INT},

    {"MCAST_NET_DEVICE", "", "Specifies which network device to use for Mcast",
     ucc_offsetof(ucc_tl_mlx5_context_config_t, mcast_ctx_conf.ib_dev_name),
     UCC_CONFIG_TYPE_STRING},
    {"FANIN_NPOLLS", "1000",
     "Number of shared memory polling before returning UCC_INPROGRESS during internode FANIN",
     ucc_offsetof(ucc_tl_mlx5_context_config_t, npolls), UCC_CONFIG_TYPE_UINT},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_mlx5_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mlx5_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_mlx5_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mlx5_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_mlx5_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_mlx5_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_mlx5_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_mlx5_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     team,
                                   ucc_coll_task_t **    task);

ucc_status_t ucc_tl_mlx5_team_get_scores(ucc_base_team_t *  tl_team,
                                         ucc_coll_score_t **score);

UCC_TL_IFACE_DECLARE(mlx5, MLX5);

ucc_status_t ucc_tl_mlx5_context_create_epilog(ucc_base_context_t *context);

__attribute__((constructor)) static void tl_mlx5_iface_init(void)
{
    ucc_tl_mlx5.super.context.create_epilog = ucc_tl_mlx5_context_create_epilog;
}
