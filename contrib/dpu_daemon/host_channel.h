/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef HOST_CHANNEL_H
#define HOST_CHANNEL_H

// #define _DEFAULT_SOURCE
#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <assert.h>

#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

#define IP_STRING_LEN       50
#define PORT_STRING_LEN     8
#define SUCCESS             0
#define ERROR               1
#define DEFAULT_PORT        13337

#define DATA_BUFFER_SIZE     (128*1024*1024)

#define EXCHANGE_LENGTH_TAG 1ull
#define EXCHANGE_RKEY_TAG 2ull
#define EXCHANGE_ADDR_TAG 3ull

extern size_t dpu_ucc_dt_sizes[UCC_DT_USERDEFINED];

typedef struct dpu_req_s {
    int complete;
} dpu_req_t;

/* sync struct type
 * use it for counter, dtype, ar op, length */
typedef struct dpu_sync_s {
    unsigned int        coll_id;
    ucc_datatype_t      dtype;
    ucc_reduction_op_t  op;
    unsigned int        count_total;
    unsigned int        count_in;
    ucc_coll_type_t     coll_type;
} dpu_sync_t;

typedef struct dpu_rkey_s {
    void *rkey_addr;
    size_t rkey_addr_len;
} dpu_rkey_t;

typedef struct dpu_mem_s {
    void *base;
    ucp_mem_h memh;
    dpu_rkey_t rkey;
} dpu_mem_t;

typedef struct dpu_mem_segs_s {
    dpu_mem_t sync;
    dpu_mem_t put;
    dpu_mem_t get;
} dpu_mem_segs_t;

typedef struct dpu_hc_s {
    /* TCP/IP stuff */
    char *hname;
    char *ip;
    int connfd, listenfd;
    uint16_t port;
    /* Local UCX stuff */
    ucp_context_h ucp_ctx;
    ucp_worker_h ucp_worker;
    ucp_worker_attr_t worker_attr;
    union {
        dpu_mem_segs_t mem_segs;
        dpu_mem_t mem_segs_array[3];
    };
    /* Remote UCX stuff */
    ucp_ep_h host_ep;
    uint64_t sync_addr;
    ucp_rkey_h sync_rkey;

    /* bufer size*/
    size_t data_buffer_size;
} dpu_hc_t;

int dpu_hc_init(dpu_hc_t *dpu_hc);
int dpu_hc_accept(dpu_hc_t *hc);
int dpu_hc_reply(dpu_hc_t *hc, unsigned int itt);
int dpu_hc_wait(dpu_hc_t *hc, unsigned int itt);
unsigned int        dpu_hc_get_count_total(dpu_hc_t *hc);
unsigned int        dpu_hc_get_count_in(dpu_hc_t *hc);
ucc_datatype_t      dpu_hc_get_dtype(dpu_hc_t *hc);
ucc_reduction_op_t  dpu_hc_get_op(dpu_hc_t *hc);

size_t dpu_ucc_dt_size(ucc_datatype_t dt);

#endif
