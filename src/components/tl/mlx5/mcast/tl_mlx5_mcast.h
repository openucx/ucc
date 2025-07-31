/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MCAST_H
#define UCC_MCAST_H

#include <infiniband/ib.h>
#include <infiniband/umad.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_verbs.h>
#include "utils/ucc_list.h"
#include "utils/ucc_mpool.h"
#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_rcache.h"
#include "core/ucc_service_coll.h"
#include "components/mc/ucc_mc.h"

#define POLL_PACKED       16
#define REL_DONE          ((void*)-1)
#define NB_POLL           8
#define NB_POLL_LARGE     32
#define MULTICAST_QPN     0xFFFFFF
/* default parameters during modify QP */
#define DEF_QKEY          0x1a1a1a1a
#define DEF_PKEY          0xffff
#define DEF_PSN           0
#define DEF_SL            0
#define DEF_SRC_PATH_BITS 0
#define GRH_LENGTH        40
#define DROP_THRESHOLD    10000
#define MAX_COMM_POW2     32
#define MAX_GROUP_COUNT   64

/* Allgather RDMA-based reliability designs */
#define ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE 1024u
#define ONE_SIDED_SLOTS_COUNT               2           /* number of memory slots during async design */
#define ONE_SIDED_SLOTS_INFO_SIZE           sizeof(int) /* size of metadata prepended to each slots in bytes */
#define ONE_SIDED_MAX_ZCOPY_COLL_COUNTER    32u
#define ONE_SIDED_MAX_CONCURRENT_LEVEL      64

enum ucc_tl_mlx5_mcast_one_sided_slot_states {
    ONE_SIDED_INVALID = -4,
    ONE_SIDED_VALID,
    ONE_SIDED_PENDING_INFO,
    ONE_SIDED_PENDING_DATA,
};

enum ucc_tl_mlx5_mcast_one_sided_reliability_scheme {
    ONE_SIDED_NO_RELIABILITY = 0,
    ONE_SIDED_SYNCHRONOUS_PROTO,
    ONE_SIDED_ASYNCHRONOUS_PROTO
};

#define CUDA_MEM_MCAST_BCAST_MAX_MSG 4096

enum {
    MCAST_PROTO_EAGER,     /* Internal staging buffers */
    MCAST_PROTO_ZCOPY
};

enum {
    MCAST_P2P_NACK,
    MCAST_P2P_ACK,
    MCAST_P2P_NEED_NACK_SEND,
    MCAST_P2P_NACK_SEND_PENDING
};

enum {
    MCAST_RECV_WR = 1,
    MCAST_WAIT_RECV_WR,
    MCAST_SEND_WR,
    MCAST_CALC_WR,
    MCAST_BCASTRECV_WR,
    MCAST_BCASTSEND_WR,
    MCAST_AG_RDMA_READ_INFO_WR,
    MCAST_AG_RDMA_READ_WR,
};

struct ucc_tl_mlx5_mcast_p2p_completion_obj;
typedef int (*ucc_tl_mlx5_mcast_p2p_completion_cb_fn_t)(struct ucc_tl_mlx5_mcast_p2p_completion_obj *obj);
typedef struct ucc_tl_mlx5_mcast_p2p_completion_obj {
    ucc_list_link_t                          super;
    ucc_tl_mlx5_mcast_p2p_completion_cb_fn_t compl_cb;
    uint64_t                                 data[3];
    ucc_coll_req_h                           req;
} ucc_tl_mlx5_mcast_p2p_completion_obj_t;

typedef int (*ucc_tl_mlx5_mcast_p2p_wait_cb_fn_t)(void *wait_arg);

typedef ucc_status_t (*ucc_tl_mlx5_mcast_p2p_send_nb_fn_t)(void* src, size_t size,
                                                           ucc_rank_t rank, ucc_memory_type_t mem_type, void *context,
                                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t *compl_obj);


typedef ucc_status_t (*ucc_tl_mlx5_mcast_p2p_recv_nb_fn_t)(void* src, size_t size,
                                                           ucc_rank_t rank, ucc_memory_type_t mem_type, void *context,
                                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t *compl_obj);

typedef struct ucc_tl_mlx5_mcast_p2p_interface {
    ucc_tl_mlx5_mcast_p2p_send_nb_fn_t  send_nb;
    ucc_tl_mlx5_mcast_p2p_recv_nb_fn_t  recv_nb;
} ucc_tl_mlx5_mcast_p2p_interface_t;

typedef struct ucc_tl_mlx5_mcast_coll_comm_init_spec {
    ucc_tl_mlx5_mcast_p2p_interface_t p2p_iface;
    int                               sx_depth;
    int                               rx_depth;
    int                               sx_sge;
    int                               rx_sge;
    int                               sx_inline;
    int                               post_recv_thresh;
    int                               scq_moderation;
    int                               wsize;
    int                               mcast_group_count;
    int                               max_push_send;
    int                               max_eager;
    int                               cuda_mem_enabled;
    int                               one_sided_reliability_enable;
    int                               reliability_scheme_msg_threshold;
    int                               truly_zero_copy_allgather_enabled;
    int                               truly_zero_copy_bcast_enabled;
    int                               truly_zero_copy_coll_min_msg;
    int                               mcast_prepost_bucket_size;
    int                               hca_copy_enabled;
    void                             *oob;
} ucc_tl_mlx5_mcast_coll_comm_init_spec_t;

typedef struct ucc_tl_mlx5_mcast_context_config {
    ucc_tl_context_config_t  super;
    char                    *dev_list;
    int                      use_rcache;
    size_t                   reg_threshold;
    unsigned int             rand_seed;
    unsigned int             uprogress_num_polls;
    int                      context_per_team;
} ucc_tl_mlx5_mcast_context_config_t;

typedef struct ucc_tl_mlx5_mcast_oob_ctx {
    void               *ctx;
    union {
        ucc_oob_coll_t *oob;
        ucc_subset_t    subset;
    };
} ucc_tl_mlx5_mcast_oob_ctx_t;

typedef struct ucc_tl_mlx5_mcast_reg {
    void *mr;
} ucc_tl_mlx5_mcast_reg_t;

typedef struct ucc_tl_mlx5_mcast_rcache_region {
    ucc_rcache_region_t     super;
    ucc_tl_mlx5_mcast_reg_t reg;
} ucc_tl_mlx5_mcast_rcache_region_t;

typedef struct ucc_tl_mlx5_mcast_ctx_params {
    ucc_ternary_auto_value_t        mcast_enabled;
    char                           *ib_dev_name;
    int                             print_nack_stats;
    int                             timeout;
    int                             mcast_bcast_enabled;
    int                             mcast_allgather_enabled;
} ucc_tl_mlx5_mcast_ctx_params_t;

typedef struct ucc_tl_mlx5_mcast_coll_context {
    struct ibv_context            *ctx;
    struct ibv_pd                 *pd;
    char                          *devname;
    int                            max_qp_wr;
    int                            user_provided_ib;
    int                            ib_port;
    int                            pkey_index;
    int                            mtu;
    uint16_t                       port_lid;
    struct rdma_cm_id             *id;
    struct rdma_event_channel     *channel;
    ucc_mpool_t                    compl_objects_mp;
    ucc_mpool_t                    mcast_req_mp;
    ucc_list_link_t                pending_nacks_list;
    ucc_rcache_t                  *rcache;
    ucc_tl_mlx5_mcast_ctx_params_t params;
    ucc_base_lib_t                *lib;
} ucc_tl_mlx5_mcast_coll_context_t;

typedef struct ucc_tl_mlx5_mcast_join_info_t {
    ucc_status_t  status;
    uint16_t      dlid[MAX_GROUP_COUNT];
    union ibv_gid dgid[MAX_GROUP_COUNT];
} ucc_tl_mlx5_mcast_join_info_t;

typedef struct ucc_tl_mlx5_mcast_context {
    ucc_thread_mode_t                   tm;
    ucc_tl_mlx5_mcast_coll_context_t    mcast_context;
    ucc_tl_mlx5_mcast_context_config_t  cfg;
    ucc_ternary_auto_value_t            mcast_enabled;
    int                                 mcast_ctx_ready;
    ucc_tl_mlx5_mcast_oob_ctx_t         oob_ctx;
    int                                 mcast_bcast_enabled;
    int                                 mcast_allgather_enabled;
} ucc_tl_mlx5_mcast_context_t;

struct pp_packet {
    ucc_list_link_t super;
    uint32_t        psn;
    int             length;
    int             packet_counter;
    uintptr_t       context;
    int             qp_id;
    uintptr_t       buf; // buffer address, initialized once
};

struct mcast_group {
    struct ibv_qp       *qp;
    struct ibv_ah       *ah;
    uint16_t             lid;
    union ibv_gid        mgid;
    struct sockaddr_in6  mcast_addr;
};

struct mcast_ctx {
    struct ibv_send_wr  swr;
    struct ibv_sge      ssg;
    struct ibv_cq      *scq;
    struct ibv_cq      *rcq;
    struct ibv_srq     *srq;
    struct mcast_group  groups[MAX_GROUP_COUNT];
    // RC connection info for supporing one-sided based relibality
    struct ibv_qp     **rc_qp;
};

struct packet {
    int        type;
    ucc_rank_t from;
    uint32_t   psn;
    int        comm_id;
};

typedef struct ucc_tl_mlx5_mcast_slot_mem_info {
    uint64_t remote_addr;
    uint32_t rkey;
} ucc_tl_mlx5_mcast_slot_mem_info_t;

typedef struct ucc_tl_mlx5_one_sided_reliable_team_info {
    ucc_tl_mlx5_mcast_slot_mem_info_t slot_mem;
    uint16_t                          port_lid;
    uint32_t                          rc_qp_num[ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE];
} ucc_tl_mlx5_one_sided_reliable_team_info_t;

typedef struct ucc_tl_mlx5_mcast_per_qp_posted_recv_info {
    ucc_list_link_t posted_recv_bufs;
    int             posted_recvs_count;
} ucc_tl_mlx5_mcast_per_qp_posted_recv_info_t;

typedef struct ucc_tl_mlx5_mcast_one_sided_reliability_comm {
    /* all the info required for establishing a reliable connection as
     * well as temp slots memkeys that all processes in the team need
     * to be aware of*/
    ucc_tl_mlx5_one_sided_reliable_team_info_t *info;
    /* holds all the remote-addr/rkey of sendbuf from processes in the team
     * used in sync design. it needs to be set during each mcast-allgather call
     * after sendbuf registration */
    ucc_tl_mlx5_mcast_slot_mem_info_t           *sendbuf_memkey_list;
    /* counter for each target recv packet */
    uint32_t                                    *recvd_pkts_tracker;
    /* holds the remote targets' collective call counter. it is used to check
     * if remote temp slot is ready for RDMA READ in async design */
    int                                         *remote_slot_info;
    struct ibv_mr                               *remote_slot_info_mr;
    int                                          reliability_scheme_msg_threshold;
    /* mem address and mem keys of the temp slots in async design */
    char                                        *slots_buffer;
    struct ibv_mr                               *slots_mr;
    /* size of a temp slot in async design */
    int                                          slot_size;
    /* coll req that is used during the oob service calls */
    ucc_service_coll_req_t                      *reliability_req;
    int                                          reliability_enabled;
    int                                          reliability_ready;
    int                                          pending_reads;
    int                                          hca_copy_enabled;
    enum ucc_tl_mlx5_mcast_one_sided_slot_states slots_state;
    ucc_tl_mlx5_mcast_per_qp_posted_recv_info_t  posted_recv[MAX_GROUP_COUNT];
} ucc_tl_mlx5_mcast_one_sided_reliability_comm_t;

typedef struct ucc_tl_mlx5_mcast_service_coll {
    ucc_status_t (*bcast_post)      (void*, void*, size_t, ucc_rank_t, ucc_service_coll_req_t**);
    ucc_status_t (*allgather_post)  (void*, void*, void*, size_t, ucc_service_coll_req_t**);
    ucc_status_t (*allreduce_post)  (void*, void*, void*, size_t, ucc_datatype_t, ucc_reduction_op_t, ucc_service_coll_req_t**);
    ucc_status_t (*barrier_post)    (void*, ucc_service_coll_req_t**);
    ucc_status_t (*coll_test)       (ucc_service_coll_req_t*);
} ucc_tl_mlx5_mcast_service_coll_t;

typedef struct ucc_tl_mlx5_mcast_allgather_comm {
    uint32_t under_progress_counter;
    uint32_t coll_counter;
    uint32_t max_num_packets;
    uint32_t max_push_send;
    uint8_t  truly_zero_copy_allgather_enabled;
    uint32_t mcast_prepost_bucket_size;
} ucc_tl_mlx5_mcast_allgather_comm_t;

typedef struct ucc_tl_mlx5_mcast_bcast_comm {
    uint32_t      under_progress_counter;
    uint32_t      coll_counter;
    uint32_t      last_psn;
    uint32_t      racks_n;
    uint32_t      sacks_n;
    uint32_t      last_acked;
    uint32_t      child_n;
    uint32_t      parent_n;
    struct packet p2p_pkt[MAX_COMM_POW2];
    struct packet p2p_spkt[MAX_COMM_POW2];
    int           reliable_in_progress;
    int           recv_drop_packet_in_progress;
    ucc_rank_t    parents[MAX_COMM_POW2];
    ucc_rank_t    children[MAX_COMM_POW2];
    int           nack_requests;
    int           nacks_counter;
    int           n_mcast_reliable;
    int           wsize;
    uint32_t      mcast_prepost_bucket_size;
    uint8_t       truly_zero_copy_bcast_enabled;
    uint32_t      max_push_send;
} ucc_tl_mlx5_mcast_bcast_comm_t;

typedef struct ucc_tl_mlx5_mcast_coll_comm {
    struct pp_packet                                dummy_packet;
    ucc_tl_mlx5_mcast_coll_context_t               *ctx;
    ucc_tl_mlx5_mcast_coll_comm_init_spec_t         params;
    ucc_tl_mlx5_mcast_p2p_interface_t               p2p;
    int                                             tx;
    ucc_rank_t                                      rank;
    ucc_rank_t                                      commsize;
    char                                           *grh_buf;
    struct ibv_mr                                  *grh_mr;
    unsigned                                        max_inline;
    size_t                                          max_eager;
    int                                             max_per_packet;
    int                                             pending_send;
    int                                             pending_recv;
    struct ibv_mr                                  *pp_mr;
    char                                           *pp_buf;
    ucc_mc_buffer_header_t                         *pp_buf_header;
    struct pp_packet                               *pp;
    uint32_t                                        psn;
    int                                             buf_n;
    ucc_list_link_t                                 bpool;
    ucc_list_link_t                                 pending_q;
    struct mcast_ctx                                mcast;
    struct ibv_recv_wr                             *call_rwr;
    struct ibv_sge                                 *call_rsgs;
    uint64_t                                        timer;
    int                                             stalled;
    int                                             comm_id;
    void                                           *p2p_ctx;
    ucc_base_lib_t                                 *lib;
    int                                             cuda_mem_enabled;
    ucc_tl_mlx5_mcast_join_info_t                  *group_setup_info;
    ucc_service_coll_req_t                         *group_setup_info_req;
    int                                             mcast_transport_ready;
    int                                             transport_ready_global;
    ucc_service_coll_req_t                         *transport_ready_req;
    ucc_tl_mlx5_mcast_service_coll_t                service_coll;
    struct rdma_cm_event                           *event;
    ucc_tl_mlx5_mcast_one_sided_reliability_comm_t  one_sided;
    int                                             mcast_group_count;
    ucc_tl_mlx5_mcast_allgather_comm_t              allgather_comm;
    ucc_tl_mlx5_mcast_bcast_comm_t                  bcast_comm;
    uint32_t                                        truly_zero_copy_coll_min_msg;
    ucc_tl_mlx5_mcast_context_t                    *context;
    /* Dedicated HCA copy resources - separate from mcast operations */
    struct ibv_cq                                  *hca_copy_cq;  /* Dedicated CQ for HCA copy */
    struct ibv_qp                                  *hca_copy_qp;  /* Dedicated QP for HCA copy */
    struct pp_packet                               *r_window[1]; // note: do not add any new variable after here
} ucc_tl_mlx5_mcast_coll_comm_t;

typedef struct ucc_tl_mlx5_mcast_team {
    ucc_tl_mlx5_mcast_context_t        *mcast_context;
    struct ucc_tl_mlx5_mcast_coll_comm *mcast_comm;
    ucc_tl_mlx5_mcast_oob_ctx_t         oob_ctx;
} ucc_tl_mlx5_mcast_team_t;


typedef struct ucc_tl_mlx5_mcast_nack_req {
    ucc_list_link_t                super;
    int                            pkt_id;
    ucc_tl_mlx5_mcast_coll_comm_t *comm;
} ucc_tl_mlx5_mcast_nack_req_t;

#define PSN_IS_IN_RANGE(_psn, _call, _comm)                                                       \
                             (                                                                    \
                                ((_psn >= _call->start_psn) &&                                    \
                                (_psn < _call->start_psn + _call->num_packets) &&                 \
                                (_psn >= _comm->bcast_comm.last_acked) &&                         \
                                (_psn < _comm->bcast_comm.last_acked + _comm->bcast_comm.wsize))  \
                             )

#define PSN_TO_RECV_OFFSET(_psn, _call, _comm)                                                    \
                             (                                                                    \
                                ((ptrdiff_t)((_psn - _call->start_psn)                            \
                                 * (_comm->max_per_packet)))                                      \
                             )

#define PSN_TO_RECV_LEN(_psn, _call, _comm)                                                       \
                             (                                                                    \
                                ((_psn - _call->start_psn + 1) %                                  \
                                 _call->num_packets == 0 ? _call->last_pkt_len :                  \
                                 _comm->max_per_packet)                                           \
                             )

#define PSN_RECEIVED(_psn, _comm)                                                                 \
                             (                                                                    \
                                (_comm->r_window[(_psn) %                                         \
                                 _comm->bcast_comm.wsize]->psn == (_psn))                         \
                             )

typedef struct ucc_tl_mlx5_mcast_tensor {
    int    group_id;
    size_t offset;
    size_t offset_left;
    int    root;
    int    count;
    int    to_recv;
    int    to_send_left;
} ucc_tl_mlx5_mcast_tensor_t;

typedef struct ucc_tl_mlx5_mcast_pipelined_ag_schedule {
   ucc_tl_mlx5_mcast_tensor_t multicast_op[ONE_SIDED_MAX_CONCURRENT_LEVEL];
   ucc_tl_mlx5_mcast_tensor_t prepost_buf_op[ONE_SIDED_MAX_CONCURRENT_LEVEL];
   int                        prepost_buf_op_done;
   int                        multicast_op_done;
   int                        total_steps;
   int                        num_recvd;
   int                        to_recv;
   int                        to_send;
} ucc_tl_mlx5_mcast_pipelined_ag_schedule_t;

typedef struct ucc_tl_mlx5_mcast_coll_req {
    ucc_tl_mlx5_mcast_coll_comm_t                      *comm;
    size_t                                              length;
    int                                                 proto;
    struct ibv_mr                                      *mr;
    struct ibv_mr                                      *recv_mr;
    struct ibv_recv_wr                                 *rwr;
    struct ibv_sge                                     *rsgs;
    void                                               *rreg;
    char                                               *ptr;
    char                                               *rptr;
    int                                                 am_root;
    ucc_rank_t                                          root;
    void                                              **rbufs;
    int                                                 first_send_psn;
    int                                                 to_send;
    int                                                 to_recv;
    ucc_rank_t                                          parent;
    uint32_t                                            start_psn;
    int                                                 num_packets;
    int                                                 last_pkt_len;
    int                                                 offset;
    ucc_memory_type_t                                   buf_mem_type;
    enum ucc_tl_mlx5_mcast_one_sided_reliability_scheme one_sided_reliability_scheme;
    uint32_t                                            ag_counter;
    int                                                 concurrency_level;
    int                                                 mcast_prepost_bucket_size;
    int                                                 state;
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t          *ag_schedule;
    int                                                 total_steps;
    int                                                 step;
    ucc_service_coll_req_t                             *allgather_rkeys_req;
    ucc_service_coll_req_t                             *bcast_rkeys_req;
    ucc_service_coll_req_t                             *barrier_req;
    void                                               *recv_rreg;
    ucc_ee_executor_task_t                             *exec_task;
    ucc_coll_task_t                                    *coll_task;
    ucc_status_t (*progress)                           (void *req);
    /* Scratch buffer for efficient CUDA memory assembly */
    char                                               *scratch_buf;
    ucc_mc_buffer_header_t                             *scratch_buf_header;
    int                                                 scratch_packets_received;
} ucc_tl_mlx5_mcast_coll_req_t;

typedef struct ucc_tl_mlx5_mcast_oob_p2p_context {
    ucc_context_h   base_ctx;
    ucc_team_h      base_team;
    ucc_rank_t      my_team_rank;
    ucc_subset_t    subset;
    ucc_base_lib_t *lib;
    int             tmp_sbuf;
    int             tmp_rbuf;
} ucc_tl_mlx5_mcast_oob_p2p_context_t;

static inline struct pp_packet* ucc_tl_mlx5_mcast_buf_get_free(ucc_tl_mlx5_mcast_coll_comm_t* comm)
{
    struct pp_packet* pp;

    pp = ucc_list_extract_head(&comm->bpool, struct pp_packet, super);

    ucc_assert(pp == NULL || pp->context == 0);

    return pp;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_post_recv_buffers(ucc_tl_mlx5_mcast_coll_comm_t* comm)
{
    struct ibv_recv_wr *bad_wr = NULL;
    struct ibv_recv_wr *rwr    = comm->call_rwr;
    struct ibv_sge     *sge    = comm->call_rsgs;
    struct pp_packet   *pp     = NULL;
    int                 i;
    int                 count;
    int                 count_per_qp;

    count = comm->params.rx_depth - comm->pending_recv;
    if (comm->allgather_comm.truly_zero_copy_allgather_enabled ||
        comm->bcast_comm.truly_zero_copy_bcast_enabled ||
        count <= comm->params.post_recv_thresh) {
        return UCC_OK;
    }

    count_per_qp = count / comm->mcast_group_count;
    for (i = 0; i < count_per_qp; i++) {
        if (NULL == (pp = ucc_tl_mlx5_mcast_buf_get_free(comm))) {
            break;
        }
        rwr[i].wr_id = ((uint64_t) pp);
        rwr[i].next = &rwr[i+1];
        sge[2*i + 1].addr = pp->buf;
        assert((uint64_t)comm->pp <= rwr[i].wr_id
                && ((uint64_t)comm->pp + comm->buf_n * sizeof(struct pp_packet)) > rwr[i].wr_id);
    }

    if (i > 0) {
        rwr[i-1].next = NULL;
        if (ibv_post_recv(comm->mcast.groups[0].qp, &rwr[0], &bad_wr)) {
            tl_error(comm->lib, "Failed to prepost recvs: errno %d qp index %d buffer count %d",
                    errno, 0, i);
            return UCC_ERR_NO_RESOURCE;
        }
        comm->pending_recv += i;
    }

    return UCC_OK;
}

static inline uint64_t ucc_tl_mlx5_mcast_get_timer(void)
{
    double t_second = ucc_get_time();
    return (uint64_t) (t_second * 1000000);
}

static inline ucc_status_t ucc_tl_mlx5_mcast_post_user_recv_buffers(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                                    ucc_tl_mlx5_mcast_coll_req_t  *req,
                                                                    int group_id, ucc_rank_t root,
                                                                    int coll_type,
                                                                    int count,
                                                                    size_t offset)
{
    struct ibv_recv_wr *bad_wr = NULL;
    struct ibv_recv_wr *rwr    = comm->call_rwr;
    struct ibv_sge     *sge    = comm->call_rsgs;
    struct pp_packet   *pp     = NULL;
    uint32_t            i;

    for (i = 0; i < count; i++) {
        if (NULL == (pp = ucc_tl_mlx5_mcast_buf_get_free(comm))) {
            tl_error(comm->lib, "not enought free pp packets to cover the entire message");
            return UCC_ERR_NO_RESOURCE;
        }

        if (comm->one_sided.reliability_enabled &&
            (comm->allgather_comm.truly_zero_copy_allgather_enabled ||
             comm->bcast_comm.truly_zero_copy_bcast_enabled)) {
            /* need to keep track to cancel the posted recv if needed */
            ucc_list_add_tail(&comm->one_sided.posted_recv[group_id].posted_recv_bufs,
                              &pp->super);
        }

        assert(offset % comm->max_per_packet == 0);
        pp->packet_counter  = offset / comm->max_per_packet;
        pp->qp_id           = group_id;
        rwr[i].wr_id        = ((uint64_t) pp);
        if (coll_type == UCC_COLL_TYPE_ALLGATHER) {
            sge[2*i + 1].addr = (uint64_t)req->rptr + root * req->length + offset;
        } else {
            ucc_assert(UCC_COLL_TYPE_BCAST == coll_type);
            sge[2*i + 1].addr = (uint64_t)req->ptr + offset;
        }
        sge[2*i + 1].lkey   = req->recv_mr->lkey;
        offset             += comm->max_per_packet;
        sge[2*i + 1].length = comm->max_per_packet;
        if (i < count - 1) {
            rwr[i].next = &rwr[i+1];
        }
    }

    if (i > 0) {
        rwr[i-1].next = NULL;
        if (ibv_post_recv(comm->mcast.groups[group_id].qp, &rwr[0], &bad_wr)) {
            tl_error(comm->lib, "failed to prepost recvs: errno %d buffer count %d",
                    errno, i);
            return UCC_ERR_NO_RESOURCE;
        }
        comm->pending_recv += i;
        comm->one_sided.posted_recv[group_id].posted_recvs_count += i;
        tl_trace(comm->lib, "posted %d buffers into recv queue with root %d mcast group %d"
                 " with packet counter ranging from %ld to %ld",
                i, root, group_id, (offset-(i*comm->max_per_packet))/comm->max_per_packet,
                offset/comm->max_per_packet);
    }

    return UCC_OK;
}

#define EXEC_TASK_TEST(_errmsg, _etask, _lib) do {                             \
    if (_etask != NULL) {                                                      \
        status = ucc_ee_executor_task_test(_etask);                            \
        if (status > 0) {                                                      \
            return status;                                                     \
        }                                                                      \
        ucc_ee_executor_task_finalize(_etask);                                 \
        _etask = NULL;                                                         \
        if (ucc_unlikely(status < 0)) {                                        \
            tl_error(_lib, _errmsg);                                           \
            return status;                                                     \
        }                                                                      \
    }                                                                          \
} while(0)

ucc_status_t ucc_tl_mlx5_mcast_team_init(ucc_base_context_t *tl_context,
                                         ucc_tl_mlx5_mcast_team_t **mcast_team,
                                         ucc_tl_mlx5_mcast_context_t *ctx,
                                         const ucc_base_team_params_t *params,
                                         ucc_tl_mlx5_mcast_coll_comm_init_spec_t *mcast_conf);

ucc_status_t ucc_tl_mlx5_mcast_team_test(ucc_base_team_t *team);

ucc_status_t ucc_tl_mlx5_mcast_coll_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *team,
                                         ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_mlx5_mcast_context_init(ucc_tl_mlx5_mcast_context_t *mcast_ctx,
                                            ucc_tl_mlx5_mcast_ctx_params_t *mcast_ctx_conf);


ucc_status_t ucc_tl_mlx5_mcast_clean_ctx(ucc_tl_mlx5_mcast_coll_context_t *ctx);

/* Conditional logging macro for mcast operations */
#define tl_mlx5_mcast_log(_mcast_enabled, _lib, _intended_level, _fmt, ...) \
    do { \
        if ((_mcast_enabled) == UCC_YES) { \
            ucc_log_component((_intended_level), (_lib)->log_component, (_fmt), ##__VA_ARGS__); \
        } else if ((_mcast_enabled) == UCC_TRY) { \
            ucc_log_component(UCC_LOG_LEVEL_DEBUG, (_lib)->log_component, (_fmt), ##__VA_ARGS__); \
        } \
    } while(0)

#endif
