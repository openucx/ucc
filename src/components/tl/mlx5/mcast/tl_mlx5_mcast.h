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
#define DROP_THRESHOLD    1000000
#define MAX_COMM_POW2     32

enum {
    MCAST_PROTO_EAGER,     /* Internal staging buffers */
    MCAST_PROTO_ZCOPY
};

enum {
    MCAST_P2P_NACK,
    MCAST_P2P_ACK,
    MCAST_P2P_NEED_NACK_SEND
};

enum {
    MCAST_RECV_WR = 1,
    MCAST_WAIT_RECV_WR,
    MCAST_SEND_WR,
    MCAST_CALC_WR,
    MCAST_BCASTRECV_WR,
    MCAST_BCASTSEND_WR,
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
                                                           ucc_rank_t rank, void *context,
                                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t *compl_obj);


typedef ucc_status_t (*ucc_tl_mlx5_mcast_p2p_recv_nb_fn_t)(void* src, size_t size,
                                                           ucc_rank_t rank, void *context,
                                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t *compl_obj);

typedef struct ucc_tl_mlx5_mcast_p2p_interface {
    ucc_tl_mlx5_mcast_p2p_send_nb_fn_t  send_nb;
    ucc_tl_mlx5_mcast_p2p_recv_nb_fn_t  recv_nb;
} ucc_tl_mlx5_mcast_p2p_interface_t;

typedef struct mcast_coll_comm_init_spec {
    ucc_tl_mlx5_mcast_p2p_interface_t p2p_iface;
    int                               sx_depth;
    int                               rx_depth;
    int                               sx_sge;
    int                               rx_sge;
    int                               sx_inline;
    int                               post_recv_thresh;
    int                               scq_moderation;
    int                               wsize;
    int                               max_eager;
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
    int      mcast_enabled;
    char    *ib_dev_name;
    int      print_nack_stats;
    int      timeout;
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
    struct rdma_cm_id             *id;
    struct rdma_event_channel     *channel;
    ucc_mpool_t                    compl_objects_mp;
    ucc_mpool_t                    nack_reqs_mp;
    ucc_list_link_t                pending_nacks_list;
    ucc_rcache_t                  *rcache;
    ucc_tl_mlx5_mcast_ctx_params_t params;
    ucc_base_lib_t                *lib;
} ucc_tl_mlx5_mcast_coll_context_t;

typedef struct ucc_tl_mlx5_mcast_join_info_t {
    ucc_status_t  status;
    uint16_t      dlid;
    union ibv_gid dgid;
} ucc_tl_mlx5_mcast_join_info_t;

typedef struct ucc_tl_mlx5_mcast_context {
    ucc_thread_mode_t                  tm;
    ucc_tl_mlx5_mcast_coll_context_t   mcast_context;
    ucc_tl_mlx5_mcast_context_config_t cfg;
    ucc_mpool_t                        req_mp;
    int                                mcast_enabled;
    int                                mcast_ready;
    ucc_tl_mlx5_mcast_oob_ctx_t        oob_ctx;
} ucc_tl_mlx5_mcast_context_t;

struct pp_packet {
    ucc_list_link_t super;
    uint32_t        psn;
    int             length;
    uintptr_t       context;
    uintptr_t       buf;
};

struct mcast_ctx {
    struct ibv_qp     *qp;
    struct ibv_ah     *ah;
    struct ibv_send_wr swr;
    struct ibv_sge     ssg;
};

struct packet {
    int        type;
    ucc_rank_t from;
    uint32_t   psn;
    int        comm_id;
};

typedef struct ucc_tl_mlx5_mcast_coll_comm {
    struct pp_packet                        dummy_packet;
    ucc_tl_mlx5_mcast_coll_context_t       *ctx;
    ucc_tl_mlx5_mcast_coll_comm_init_spec_t params;
    ucc_tl_mlx5_mcast_p2p_interface_t       p2p;
    int                                     tx;
    struct ibv_cq                          *scq;
    struct ibv_cq                          *rcq;
    ucc_rank_t                              rank;
    ucc_rank_t                              commsize;
    char                                   *grh_buf;
    struct ibv_mr                          *grh_mr;
    uint16_t                                mcast_lid;
    union ibv_gid                           mgid;
    unsigned                                max_inline;
    size_t                                  max_eager;
    int                                     max_per_packet;
    int                                     pending_send;
    int                                     pending_recv;
    struct ibv_mr                          *pp_mr;
    char                                   *pp_buf;
    struct pp_packet                       *pp;
    uint32_t                                psn;
    uint32_t                                last_psn;
    uint32_t                                racks_n;
    uint32_t                                sacks_n;
    uint32_t                                last_acked;
    uint32_t                                naks_n;
    uint32_t                                child_n;
    uint32_t                                parent_n;
    int                                     buf_n;
    struct packet                           p2p_pkt[MAX_COMM_POW2];
    struct packet                           p2p_spkt[MAX_COMM_POW2];
    ucc_list_link_t                         bpool;
    ucc_list_link_t                         pending_q;
    struct mcast_ctx                        mcast;
    int                                     reliable_in_progress;
    int                                     recv_drop_packet_in_progress;
    struct ibv_recv_wr                     *call_rwr;
    struct ibv_sge                         *call_rsgs;
    uint64_t                                timer;
    int                                     stalled;
    int                                     comm_id;
    void                                   *p2p_ctx;
    ucc_base_lib_t                         *lib;
    struct sockaddr_in6                     mcast_addr;
    ucc_rank_t                              parents[MAX_COMM_POW2];
    ucc_rank_t                              children[MAX_COMM_POW2];
    int                                     nack_requests;
    int                                     nacks_counter;
    int                                     n_prep_reliable;
    int                                     n_mcast_reliable;
    int                                     wsize;
    ucc_tl_mlx5_mcast_join_info_t          *group_setup_info;
    ucc_service_coll_req_t                 *group_setup_info_req;
    ucc_status_t                           (*bcast_post) (void*, void*, size_t, ucc_rank_t, ucc_service_coll_req_t**);
    ucc_status_t                           (*bcast_test) (ucc_service_coll_req_t*);
    struct rdma_cm_event                   *event;
    struct pp_packet                       *r_window[1]; // do not add any new variable after here
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

#define PSN_IS_IN_RANGE(_psn, _call, _comm)                                         \
                             (                                                      \
                                ((_psn >= _call->start_psn) &&                      \
                                (_psn < _call->start_psn + _call->num_packets) &&   \
                                (_psn >= _comm->last_acked) &&                      \
                                (_psn < _comm->last_acked + _comm->wsize))          \
                             )

#define PSN_TO_RECV_OFFSET(_psn, _call, _comm)                                      \
                             (                                                      \
                                ((ptrdiff_t)((_psn - _call->start_psn)              \
                                 * (_comm->max_per_packet)))                        \
                             )

#define PSN_TO_RECV_LEN(_psn, _call, _comm)                                         \
                             (                                                      \
                                ((_psn - _call->start_psn + 1) %                    \
                                 _call->num_packets == 0 ? _call->last_pkt_len :    \
                                 _comm->max_per_packet)                             \
                             )

#define PSN_RECEIVED(_psn, _comm)                                                   \
                             (                                                      \
                                (_comm->r_window[(_psn) %                           \
                                 _comm->wsize]->psn == (_psn))                      \
                             )

typedef struct ucc_tl_mlx5_mcast_coll_req {
    ucc_tl_mlx5_mcast_coll_comm_t  *comm;
    size_t                          length; /* bcast buffer size */
    int                             proto;
    struct ibv_mr                  *mr;
    struct ibv_recv_wr             *rwr;
    struct ibv_sge                 *rsgs;
    void                           *rreg;
    char                           *ptr;
    int                             am_root;
    ucc_rank_t                      root;
    void                          **rbufs;
    int                             first_send_psn;
    int                             to_send;
    int                             to_recv;
    ucc_rank_t                      parent;
    uint32_t                        start_psn;
    int                             num_packets;
    int                             last_pkt_len;
    int                             offset;
    ucc_memory_type_t               buf_mem_type;
} ucc_tl_mlx5_mcast_coll_req_t;

typedef struct ucc_tl_mlx5_mcast_oob_p2p_context {
    ucc_context_h base_ctx;
    ucc_team_h    base_team;
    ucc_rank_t    my_team_rank;
    ucc_subset_t  subset;
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
    int                 count  = comm->params.rx_depth - comm->pending_recv;
    int                 i;
    
    if (count <= comm->params.post_recv_thresh) {
        return UCC_OK;
    }

    for (i = 0; i < count - 1; i++) {
        if (NULL == (pp = ucc_tl_mlx5_mcast_buf_get_free(comm))) {
            break;
        }

        rwr[i].wr_id      = ((uint64_t) pp);
        rwr[i].next       = &rwr[i+1];
        sge[2*i + 1].addr = pp->buf;

        ucc_assert((uint64_t)comm->pp <= rwr[i].wr_id
                && ((uint64_t)comm->pp + comm->buf_n * sizeof(struct pp_packet)) > rwr[i].wr_id);
    }
    if (i != 0) {
        rwr[i-1].next = NULL;
        if (ibv_post_recv(comm->mcast.qp, &rwr[0], &bad_wr)) {
            tl_error(comm->lib, "failed to prepost recvs: errno %d", errno);
            return UCC_ERR_NO_RESOURCE;
        }
        comm->pending_recv += i;
    }

    return UCC_OK;
}

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
#endif
