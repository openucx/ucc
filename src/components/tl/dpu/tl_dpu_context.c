/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_dpu.h"
#include "tl_dpu_coll.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <netdb.h>
#include <poll.h>
#include <errno.h>
#include <unistd.h>

static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    ucc_error("error handling callback was invoked with status %d (%s)\n",
                    status, ucs_status_string(status));
}

static int _server_connect(ucc_tl_dpu_context_t *ctx, char *hname, uint16_t port)
{
    int sock = 0, n;
    struct addrinfo *res, *t;
    struct addrinfo hints = { 0 };
    char service[64];

    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    sprintf(service, "%d", port);
    n = getaddrinfo(hname, service, &hints, &res);

    if (n < 0) {
        tl_error(ctx->super.super.lib, "%s:%d: getaddrinfo(): %s for %s:%s\n", __FILE__,__LINE__, gai_strerror(n), hname, service);
        return -1;
    }

    for (t = res; t; t = t->ai_next) {
        sock = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sock >= 0) {
            if (!connect(sock, t->ai_addr, t->ai_addrlen))
                break;
            close(sock);
            sock = -1;
        }
    }

    freeaddrinfo(res);
    return sock;
}

UCC_CLASS_INIT_FUNC(ucc_tl_dpu_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    /* TODO: Need handshake with daemon for detection */
    ucc_tl_dpu_context_config_t *tl_dpu_config =
        ucc_derived_of(config, ucc_tl_dpu_context_config_t);

    ucc_status_t        ucc_status = UCC_OK;
    int sockfd = 0, dpu_found = 0;
    ucp_worker_params_t worker_params;
    ucp_worker_attr_t   worker_attr;
    ucp_params_t        ucp_params;
    ucp_ep_params_t     ep_params;
    ucp_ep_h            ucp_ep;
    ucp_context_h       ucp_context;
    ucp_worker_h        ucp_worker;
    int ret;

    /* Identify DPU */
    char hname[MAX_DPU_HOST_NAME];
    void *rem_worker_addr;
    size_t rem_worker_addr_size;
    

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, tl_dpu_config->super.tl_lib,
                              params->context);
    memcpy(&self->cfg, tl_dpu_config, sizeof(*tl_dpu_config));

    /* Find  DPU based on the host-dpu list */
    gethostname(hname, sizeof(hname) - 1);

    char *h = calloc(1, 256);
    FILE *fp = NULL;

    if (strcmp(tl_dpu_config->host_dpu_list,"") != 0) {

        fp = fopen(tl_dpu_config->host_dpu_list, "r");
        if (fp == NULL) {
            tl_error(self->super.super.lib,
                "Unable to open host_dpu_list \"%s\", disabling dpu team\n", tl_dpu_config->host_dpu_list);
            ucc_status = UCC_ERR_NO_MESSAGE;
        }
        else {
            while (fscanf(fp,"%s", h) != EOF) {
                if (strcmp(h, hname) == 0) {
                    dpu_found = 1;
                    fscanf(fp, "%s", hname);
                    tl_info(self->super.super.lib, "DPU <%s> found!\n", hname);
                    break;
                }
                memset(h, 0, 256);
            }
        }
        if (!dpu_found) {
            ucc_status = UCC_ERR_NO_MESSAGE;
        }
    }
    else {
        tl_error(self->super.super.lib,
            "DPU_ENABLE set, but HOST_LIST not specified. Disabling DPU team!\n");
        ucc_status = UCC_ERR_NO_MESSAGE;
    }
    free(h);

    if (UCC_OK != ucc_status) {
        goto err;
    }

    tl_info(self->super.super.lib, "Connecting to %s", hname);
    sockfd = _server_connect(self, hname, tl_dpu_config->server_port);

    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask      = UCP_PARAM_FIELD_FEATURES |
                                 UCP_PARAM_FIELD_REQUEST_SIZE |
                                 UCP_PARAM_FIELD_REQUEST_INIT |
                                 UCP_PARAM_FIELD_REQUEST_CLEANUP;
    ucp_params.features        = UCP_FEATURE_TAG |
                                 UCP_FEATURE_RMA;
    ucp_params.request_size    = sizeof(ucc_tl_dpu_request_t);
    ucp_params.request_init    = ucc_tl_dpu_req_init;
    ucp_params.request_cleanup = ucc_tl_dpu_req_cleanup;

    ucc_status = ucs_status_to_ucc_status(
                    ucp_init(&ucp_params, NULL, &ucp_context));
    if (ucc_status != UCC_OK) {
        tl_error(self->super.super.lib,
            "failed ucp_init(%s)\n", ucc_status_string(ucc_status));
        goto err;
    }

    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask    = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode   = UCS_THREAD_MODE_SINGLE;

    ucc_status = ucs_status_to_ucc_status(
                    ucp_worker_create(ucp_context, &worker_params, &ucp_worker));
    if (ucc_status != UCC_OK) {
        tl_error(self->super.super.lib,
            "failed ucp_worker_create (%s)\n", ucc_status_string(ucc_status));
        goto err_cleanup_context;
    }

    worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS |
                             UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    worker_attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;
    ucp_worker_query(ucp_worker, &worker_attr);

    ret = send(sockfd, &worker_attr.address_length,
            sizeof(&worker_attr.address_length), 0);
    if (ret < 0) {
        tl_error(self->super.super.lib, "send length failed");
        ucc_status = UCC_ERR_NO_MESSAGE;
        goto err;
    }
    ret = send(sockfd, worker_attr.address, worker_attr.address_length, 0);
    if (ret < 0) {
        tl_error(self->super.super.lib, "send address failed");
        ucc_status = UCC_ERR_NO_MESSAGE;
        goto err;
    }
    ret = recv(sockfd, &rem_worker_addr_size, sizeof(rem_worker_addr_size), MSG_WAITALL);
    if (ret < 0) {
        tl_error(self->super.super.lib, "recv address length failed");
        ucc_status = UCC_ERR_NO_MESSAGE;
        goto err;
    }
    rem_worker_addr = ucc_malloc(rem_worker_addr_size, "rem_worker_addr");
    if (NULL == rem_worker_addr) {
        tl_error(self->super.super.lib, "failed to allocate rem_worker_addr");
        ucc_status = UCC_ERR_NO_MESSAGE;
        goto err;
    }
    ret = recv(sockfd, rem_worker_addr, rem_worker_addr_size, MSG_WAITALL);
    if (ret < 0) {
        tl_error(self->super.super.lib, "recv address failed");
        ucc_status = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ep_params.field_mask       = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS      |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER         |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;

    ep_params.address          = rem_worker_addr;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = err_cb;

    ucc_status = ucs_status_to_ucc_status(
                    ucp_ep_create(ucp_worker, &ep_params, &ucp_ep));
    free(worker_attr.address);
    ucc_free(rem_worker_addr);
    close(sockfd);
    if (ucc_status != UCC_OK) {
        tl_error(self->super.super.lib, "failed to connect to %s (%s)\n",
                       hname, ucc_status_string(ucc_status));
        goto err_cleanup_worker;
    }

    self->ucp_context   = ucp_context;
    self->ucp_worker    = ucp_worker;
    self->ucp_ep        = ucp_ep;

    tl_info(self->super.super.lib, "context created");
    return ucc_status;

err_cleanup_worker:
    ucp_worker_destroy(self->ucp_worker);
err_cleanup_context:
    ucp_cleanup(self->ucp_context);
err:
    return ucc_status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_dpu_context_t)
{
    ucp_request_param_t param;
    ucc_status_t ucc_status;
    void *close_req;

    tl_info(self->super.super.lib, "finalizing tl context: %p", self);

    param.op_attr_mask  = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags         = UCP_EP_CLOSE_FLAG_FORCE;
    close_req           = ucp_ep_close_nbx(self->ucp_ep, &param);
    if (UCS_PTR_IS_PTR(close_req)) {
        do {
            ucp_worker_progress(self->ucp_worker);
            ucc_status = ucs_status_to_ucc_status(
                ucp_request_check_status(close_req));
        } while (ucc_status == UCC_INPROGRESS);
        ucp_request_free (close_req);
    } else if (UCS_PTR_STATUS(close_req) != UCS_OK) {
        tl_error(self->super.super.lib, "failed to close ep %p\n", (void *)self->ucp_ep);
    }
    ucp_worker_destroy(self->ucp_worker);
    ucp_cleanup(self->ucp_context);
}

UCC_CLASS_DEFINE(ucc_tl_dpu_context_t, ucc_tl_context_t);

ucc_status_t ucc_tl_dpu_get_context_attr(const ucc_base_context_t *context,
                                         ucc_base_ctx_attr_t      *attr)
{
    /* TODO */
    return UCC_ERR_NOT_IMPLEMENTED;
}