/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#ifndef WORKER_UCC_H_
#define WORKER_UCC_H_

#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>
#include <ucs/datastruct/khash.h>

#include <doca_log.h>
#include <doca_error.h>
#include <doca_urom_plugin.h>

#include "../common/urom_ucc.h"

#define MAX_HOST_DEST_ID INT_MAX /* Maximum destination host id */
#define MIN_THREADS 1		     /* Minimum number of threads per UCC worker */

/* Collective operation check macro */
#define COLL_CHECK(ucc_worker, ctx_id, status)                                 \
    {                                                                          \
        if (ucc_worker->ucc_data[ctx_id].ucc_lib == NULL) {                    \
            DOCA_LOG_ERR("Attempting to perform ucc collective "               \
                        "without initialization");                             \
            status = DOCA_ERROR_NOT_FOUND;                                     \
            goto fail;                                                         \
        }                                                                      \
                                                                               \
        if (ucc_worker->ucc_data[ctx_id].ucc_context == NULL) {                \
            DOCA_LOG_ERR("Attempting to perform ucc collective "               \
                         "without a ucc context");                             \
            status = DOCA_ERROR_NOT_FOUND;                                     \
            goto fail;                                                         \
        }                                                                      \
    }

/* UCC serializing next raw, iter points to the
   offset place and returns the buffer start */
#define urom_ucc_serialize_next_raw(_iter, _type, _offset)                     \
    ({                                                                         \
        _type *_result = (_type *)(*(_iter));                                  \
        *(_iter) = UCS_PTR_BYTE_OFFSET(*(_iter), _offset);                     \
        _result;                                                               \
    })

/* Worker UCC options */
struct worker_ucc_opts {
    uint64_t num_progress_threads;      /* Number of threads */
    uint64_t dpu_worker_binding_stride; /* Each worker thread is bound to this far apart core # from each other */
    uint64_t ppw;		                /* Number of processes per worker */
    uint64_t tpp; 			            /* Threads per host process--create this many duplicate ucc contexts/teams/collectives per single host cmd */
    uint64_t list_size;	                /* Size of progress list */
    uint64_t num_psync;	                /* Number of synchronization/work scratch buffers to allocate for collectives */
};

/* UCC worker queue elements types */
enum ucc_worker_queue_element_type {
    UCC_WORKER_QUEUE_ELEMENT_TYPE_TEAM_CREATE, /* Team element queue type */
    UCC_WORKER_QUEUE_ELEMENT_TYPE_COLLECTIVE,  /* Collective element queue type */
};

/* UROM UCC worker interface */
struct urom_worker_ucc_iface {
    struct urom_plugin_iface super; /* DOCA UROM worker plugin interface */
};

/* UCC data structure */
struct ucc_data {
    ucc_lib_h      ucc_lib;           /* UCC lib handle */
    ucc_lib_attr_t ucc_lib_attr;      /* UCC lib attribute structure */
    ucc_context_h  ucc_context;       /* UCC context */
    ucc_team_h    *ucc_team;          /* Array of UCC team members */
    int64_t        n_teams;           /* Array size */
    long          *pSync;             /* Pointer to synchronization/work scratch buffers */
    uint64_t       psync_offset;      /* Synchronization buffer offset */
    void          *local_work_buffer; /* Local work buffer */
    size_t         len;               /* Buffer length */
    ucp_ep_h       host;              /* The host data endpoint */
};

/* EP map */
KHASH_MAP_INIT_INT64(ep, ucp_ep_h);
/* Memory handles map */
KHASH_MAP_INIT_INT64(memh, ucp_mem_h);
/* Remote key map */
KHASH_MAP_INIT_INT64(rkeys, ucp_rkey_h);

/* UCP data structure */
struct ucc_ucp_data {
    ucp_context_h  ucp_context;    /* UCP context */
    ucp_worker_h   ucp_worker;     /* UCP worker */
    ucp_address_t *worker_address; /* UCP worker address */
    size_t         ucp_addrlen;    /* UCP worker address length */
    khash_t(ep)   *eps;            /* EP hashtable map */
    khash_t(memh) *memh;           /* Memh hashtable map */
    khash_t(rkeys) *rkeys;         /* Rkey hashtable map */
};

/* Context ids map */
KHASH_MAP_INIT_INT64(ctx_id, uint64_t);

struct urom_worker_ucc {
    struct urom_worker_ctx    *super;          /* DOCA base worker context */
    struct ucc_data           *ucc_data;       /* UCC data structure */
    struct ucc_ucp_data        ucp_data;       /* UCP data structure */
    uint64_t                   list_lock;      /* List lock field */
    ucs_list_link_t            completed_reqs; /* List of completed requests */
    struct ucc_queue_element **queue;          /* Elements queue */
    khash_t(ctx_id)           *ids;            /* Ids hashtable map */
    uint64_t                   ctx_id;         /* Context id, incremented with every new dest id */
    uint64_t                   nr_connections; /* Number of connections */
};

/* UCC worker thread args */
struct ctx_thread_args {
    uint64_t                notif_type;   /* Notification type */
    uint64_t                urom_context; /* UROM context */
    int64_t                 start;        /* Start index */
    int64_t                 stride;       /* Number of strides */
    int64_t                 size;         /* The work buffer size */
    int64_t                 myrank;       /* Current thread rank */
    void                   *base_va;      /* Buffer host address */
    size_t                  len;          /* Total buffer length */
    uint64_t                dest_id;      /* Destination id */
    struct urom_worker_ucc *ucc_worker;   /* UCC worker structure */
};

/* UCC collective context structure */
struct coll_ctx {
    union {
        int64_t start;                  /* Collective start for single team */
        int64_t *pids;                  /* Collective team pids */
    };
    int64_t                 stride;     /* Number of strides */
    int64_t                 size;       /* The work buffer size */
    int64_t                 index;      /* Current collective member index */
    struct urom_worker_ucc *ucc_worker; /* UCC worker context */
};

typedef struct ucc_tl_ucp_allreduce_sw_global_work_buf_info {
    void *packed_src_memh;
    void *packed_dst_memh;
} ucc_tl_ucp_allreduce_sw_global_work_buf_info_t;

/* UCC queue element structure */
struct ucc_queue_element {
    enum ucc_worker_queue_element_type              type;                   /* Element type */
    volatile int64_t                                in_use;                 /* If element in use */
    volatile int64_t                                posted;                 /* If element was posted */
    uint64_t                                        dest_id;                /* Element destination id */
    uint64_t                                        ctx_id;                 /* Element context id */
    uint64_t                                        myrank;                 /* Element rank */
    pthread_barrier_t                              *barrier;                /* If not null, call this barrier */
    void                                           *old_dest;               /* Old element destination */
    size_t                                          data_size;              /* Data size */
    ucc_coll_req_h                                  coll_req;               /* UCC collective request */
    struct coll_ctx                                *coll_ctx;               /* UCC worker collective context */
    uint64_t                                        team_id;                /* Team id */
    void                                           *dest_packed_key;        /* Destination data packed key */
    struct urom_worker_notif_desc                  *nd;                     /* Element notification descriptor */
    ucc_worker_key_buf                             *key_duplicate_per_rank; /* per-rank copy of keys */
    ucc_tl_ucp_allreduce_sw_global_work_buf_info_t *gwbi;                   /* gwbi ptr */
};

/* UCC oob allgather request */
struct oob_allgather_req {
    void  *sbuf;         /* Local buffer           */
    void  *rbuf;         /* Remote buffer          */
    size_t msglen;       /* Message length         */
    void  *oob_coll_ctx; /* OOB collective context */
    int    iter;         /* Interation             */
    int    index;        /* Current process index  */
    int   *status;       /* Request status         */
};

/*
 * Execute RMA put operation for target buffer
 *
 * @buffer [in]: target buffer
 * @target [in]: pointer to target
 * @msglen [in]: message length
 * @dest [in]: destination id
 * @myrank [in]: current rank in UCC team
 * @ctx_id [in]: current context id
 * @ucc_worker [in]: UCC worker context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_rma_put(void *buffer,
                         void *target,
                         size_t msglen,
                         uint64_t dest,
                         uint64_t myrank,
                         uint64_t ctx_id,
                         struct urom_worker_ucc *ucc_worker);

/*
 * Execute RMA get operation on target buffer
 *
 * @buffer [in]: target buffer
 * @target [in]: pointer to target
 * @msglen [in]: message length
 * @dest [in]: destination id
 * @myrank [in]: current rank in UCC team
 * @ctx_id [in]: current context id
 * @ucc_worker [in]: UCC worker context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_rma_get(void *buffer,
                         void *target,
                         size_t msglen,
                         uint64_t dest,
                         uint64_t myrank,
                         uint64_t ctx_id,
                         struct urom_worker_ucc *ucc_worker);

/*
 * Execute UCP send operation
 *
 * @msg [in]: send message
 * @len [in]: message length
 * @myrank [in]: current rank in UCC team
 * @dest [in]: destination id
 * @ucc_worker [in]: UCC worker context
 * @req [out]: request result
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_send_nb(void *msg,
                         size_t len,
                         int64_t myrank,
                         int64_t dest,
                         struct urom_worker_ucc *ucc_worker,
                         int *req);

/*
 * Execute UCP recv operation
 *
 * @msg [in]: recv buffer
 * @len [in]: buffer length
 * @dest [in]: destination id
 * @ucc_worker [in]: UCC worker context
 * @req [out]: request result
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_recv_nb(void *msg,
                         size_t len,
                         int64_t dest,
                         struct urom_worker_ucc *ucc_worker,
                         int *req);

/*
 * Execute RMA get host information
 *
 * @buffer [in]: target buffer
 * @target [in]: pointer to target
 * @msglen [in]: message length
 * @ctx_id [in]: context id
 * @packed_key [in]: packed key
 * @ucc_worker [in]: UCC worker context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_rma_get_host(void *buffer,
                              void *target,
                              size_t msglen,
                              uint64_t ctx_id,
                              void *packed_key,
                              struct urom_worker_ucc *ucc_worker);

/*
 * Execute RMA put host information
 *
 * @buffer [in]: target buffer
 * @target [in]: pointer to target
 * @msglen [in]: message length
 * @ctx_id [in]: context id
 * @packed_key [in]: packed key
 * @ucc_worker [in]: UCC worker context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_rma_put_host(void *buffer,
                              void *target,
                              size_t msglen,
                              uint64_t ctx_id,
                              void *packed_key,
                              struct urom_worker_ucc *ucc_worker);

/*
 * UCP endpoint error handling context
 *
 * @arg [in]: user argument
 * @ep [in]: EP handler
 * @ucs_status [in]: UCS status
 */
void urom_ep_err_cb(void *arg, ucp_ep_h ep, ucs_status_t ucs_status);

/*
 * Get DOCA worker plugin interface for UCC plugin.
 * DOCA UROM worker will load the urom_plugin_get_iface symbol to get the UCC interface
 *
 * @iface [out]: Set DOCA UROM plugin interface for UCC
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t urom_plugin_get_iface(struct urom_plugin_iface *iface);

/*
 * Get UCC plugin version, will be used to verify that the host and DPU plugin versions are compatible
 *
 * @version [out]: Set the UCC worker plugin version
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t urom_plugin_get_version(uint64_t *version);

#endif /* WORKER_UCC_H_ */
