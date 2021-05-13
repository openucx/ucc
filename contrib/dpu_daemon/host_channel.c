/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "host_channel.h"
#include <unistd.h>
#include <ucc/api/ucc.h>

size_t dpu_ucc_dt_sizes[UCC_DT_USERDEFINED] = {
    [UCC_DT_INT8]    = 1,
    [UCC_DT_UINT8]   = 1,
    [UCC_DT_INT16]   = 2,
    [UCC_DT_UINT16]  = 2,
    [UCC_DT_FLOAT16] = 2,
    [UCC_DT_INT32]   = 4,
    [UCC_DT_UINT32]  = 4,
    [UCC_DT_FLOAT32] = 4,
    [UCC_DT_INT64]   = 8,
    [UCC_DT_UINT64]  = 8,
    [UCC_DT_FLOAT64] = 8,
    [UCC_DT_INT128]  = 16,
    [UCC_DT_UINT128] = 16,
};

size_t dpu_ucc_dt_size(ucc_datatype_t dt)
{
    if (dt < UCC_DT_USERDEFINED) {
        return dpu_ucc_dt_sizes[dt];
    }
    return 0;
}

static int _dpu_host_to_ip(dpu_hc_t *hc)
{
//     printf ("%s\n", __FUNCTION__);
    struct hostent *he;
    struct in_addr **addr_list;
    int i;

    hc->hname = calloc(1, 100 * sizeof(char));
    hc->ip = malloc(100 * sizeof(char));

    int ret = gethostname(hc->hname, 100);
    if (ret) {
        return 1;
    }

    if ( (he = gethostbyname( hc->hname ) ) == NULL)
    {
        // get the host info
        herror("gethostbyname");
        return 1;
    }

    addr_list = (struct in_addr **) he->h_addr_list;
    for(i = 0; addr_list[i] != NULL; i++)
    {
        //Return the first one;
        strcpy(hc->ip , inet_ntoa(*addr_list[i]) );
        return UCC_OK;
    }
    return UCC_ERR_NO_MESSAGE;
}

static int _dpu_listen(dpu_hc_t *hc)
{
    struct sockaddr_in serv_addr;

    if(_dpu_host_to_ip(hc)) {
        return UCC_ERR_NO_MESSAGE;
    }

    hc->port = DEFAULT_PORT;
    /* TODO: if envar(port) - replace */

    /* creates an UN-named socket inside the kernel and returns
     * an integer known as socket descriptor
     * This function takes domain/family as its first argument.
     * For Internet family of IPv4 addresses we use AF_INET
     */
    hc->listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (0 > hc->listenfd) {
        fprintf(stderr, "socket() failed (%s)\n", strerror(errno));
        goto err_ip;
    }
    memset(&serv_addr, 0, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(hc->port);

    /* The call to the function "bind()" assigns the details specified
     * in the structure 『serv_addr' to the socket created in the step above
     */
    if (0 > bind(hc->listenfd, (struct sockaddr*)&serv_addr,
                 sizeof(serv_addr))) {
        fprintf(stderr, "Failed to bind() (%s)\n", strerror(errno));
        goto err_sock;
    }

    /* The call to the function "listen()" with second argument as 10 specifies
     * maximum number of client connections that server will queue for this listening
     * socket.
     */
    if (0 > listen(hc->listenfd, 10)) {
        fprintf(stderr, "listen() failed (%s)\n", strerror(errno));
        goto err_sock;
    }

    return UCC_OK;
err_sock:
    close(hc->listenfd);
err_ip:
    free(hc->ip);
    free(hc->hname);
    return UCC_ERR_NO_MESSAGE;
}

static int _dpu_listen_cleanup(dpu_hc_t *hc)
{
    close(hc->listenfd);
    free(hc->ip);
    free(hc->hname);
}

static void tag_recv_cb (void *request, ucs_status_t status,
                         const ucp_tag_recv_info_t *info, void *user_data)
{
    dpu_req_t *ctx = user_data;
    ctx->complete = 1;
}

static void send_cb(void *request, ucs_status_t status, void *user_data)
{
    dpu_req_t *ctx = user_data;
    ctx->complete = 1;
}

static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    printf ("error handling callback was invoked with status %d (%s)\n",
            status, ucs_status_string(status));
}

static int _dpu_ucx_init(dpu_hc_t *hc)
{
    ucp_params_t ucp_params;
    ucs_status_t status;
    ucp_worker_params_t worker_params;
    int ret = SUCCESS;

//     printf ("%s\n", __FUNCTION__);

    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_TAG |
                          UCP_FEATURE_RMA;
/*
                          UCP_FEATURE_AMO64 |
                          UCP_FEATURE_AMO32;
*/
    status = ucp_init(&ucp_params, NULL, &hc->ucp_ctx);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_init(%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(hc->ucp_ctx, &worker_params, &hc->ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_worker_create (%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err_cleanup;
    }

    hc->worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS |
            UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    hc->worker_attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;
    status = ucp_worker_query (hc->ucp_worker, &hc->worker_attr);
    if (UCS_OK != status) {
        ret = UCC_ERR_NO_MESSAGE;
        goto err_worker;
    }

    return ret;
err_worker:
    ucp_worker_destroy(hc->ucp_worker);
err_cleanup:
    ucp_cleanup(hc->ucp_ctx);
err:
    return ret;
}

static int _dpu_ucx_fini(dpu_hc_t *hc){
    ucp_worker_release_address(hc->ucp_worker, hc->worker_attr.address);
    ucp_worker_destroy(hc->ucp_worker);
    ucp_cleanup(hc->ucp_ctx);
}


static int _dpu_hc_buffer_alloc(dpu_hc_t *hc, dpu_mem_t *mem, size_t size)
{
    ucp_mem_map_params_t mem_params;
    ucp_mem_attr_t mem_attr;
    ucs_status_t status;
    int ret = UCC_OK;

    memset(mem, 0, sizeof(*mem));
    mem->base = calloc(size, sizeof(char));
    memset(&mem_params, 0, sizeof(ucp_mem_map_params_t));

    mem_params.address = mem->base;
    mem_params.length = size;

    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                       UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                       UCP_MEM_MAP_PARAM_FIELD_ADDRESS;

    status = ucp_mem_map(hc->ucp_ctx, &mem_params, &mem->memh);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_mem_map (%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto out;
    }

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                          UCP_MEM_ATTR_FIELD_LENGTH;

    status = ucp_mem_query(mem->memh, &mem_attr);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_mem_query (%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err_map;
    }
    assert(mem_attr.length == size);
    assert(mem_attr.address == mem->base);

    status = ucp_rkey_pack(hc->ucp_ctx, mem->memh,
                           &mem->rkey.rkey_addr,
                           &mem->rkey.rkey_addr_len);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_rkey_pack (%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err_map;
    }
    
    goto out;
err_map:
    ucp_mem_unmap(hc->ucp_ctx, mem->memh);
err_calloc:
    free(mem->base);
out:
    return ret;
}

static int _dpu_hc_buffer_free(dpu_hc_t *hc, dpu_mem_t *mem)
{
    ucp_rkey_buffer_release(mem->rkey.rkey_addr);
    ucp_mem_unmap(hc->ucp_ctx, mem->memh);
    free(mem->base);
}


static size_t _dpu_set_buffer_size(char *_env)
{
    char *env = getenv(_env);
    return env != NULL ? atol(env) : DATA_BUFFER_SIZE;
}
 
int dpu_hc_init(dpu_hc_t *hc)
{
    int ret = UCC_OK;

    memset(hc, 0, sizeof(*hc));

    /* Start listening */
    ret = _dpu_listen(hc);
    if (ret) {
        goto out;
    }
    
    /* init ucx objects */
    ret = _dpu_ucx_init(hc);
    if (ret) {
        goto err_ip;
    }

    /* set buffer size */
    hc->data_buffer_size = _dpu_set_buffer_size("DPU_DATA_BUFFER_SIZE");

    ret = _dpu_hc_buffer_alloc(hc, &hc->mem_segs.put, DATA_BUFFER_SIZE);
    if (ret) {
        goto err_ucx;
    }
    ret = _dpu_hc_buffer_alloc(hc, &hc->mem_segs.get, DATA_BUFFER_SIZE);
    if (ret) {
        goto err_put;
    }
    ret = _dpu_hc_buffer_alloc(hc, &hc->mem_segs.sync, sizeof(dpu_sync_t));
    if (ret) {
        goto err_get;
    }

    goto out;
err_get:
    _dpu_hc_buffer_free(hc, &hc->mem_segs.get);
err_put:
    _dpu_hc_buffer_free(hc, &hc->mem_segs.put);
err_ucx:
    _dpu_ucx_fini(hc);
err_ip:
    _dpu_listen_cleanup(hc);
out:
    return ret;
}

static ucs_status_t _dpu_ep_create (dpu_hc_t *hc, void *rem_worker_addr)
{
    ucs_status_t status;
    ucp_ep_params_t ep_params;

    ep_params.field_mask    = UCP_EP_PARAM_FIELD_FLAGS |
                              UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                              UCP_EP_PARAM_FIELD_ERR_HANDLER |
                              UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode		= UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb    = err_cb;
    ep_params.address = rem_worker_addr;

    status = ucp_ep_create(hc->ucp_worker, &ep_params, &hc->host_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to create an endpoint on the dpu (%s)\n",
                ucs_status_string(status));
        return UCC_ERR_NO_MESSAGE;
    }

    return UCC_OK;
}

static int _dpu_ep_close(dpu_hc_t *hc)
{
    ucp_request_param_t param;
    ucs_status_t status;
    void *close_req;
    int ret = UCC_OK;

    param.op_attr_mask  = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags         = UCP_EP_CLOSE_FLAG_FORCE;
    close_req           = ucp_ep_close_nbx(hc->host_ep, &param);
    if (UCS_PTR_IS_PTR(close_req)) {
        do {
            ucp_worker_progress(hc->ucp_worker);
            status = ucp_request_check_status(close_req);
        } while (status == UCS_INPROGRESS);

        ucp_request_free(close_req);
    } else if (UCS_PTR_STATUS(close_req) != UCS_OK) {
        fprintf(stderr, "failed to close ep %p\n", (void *)hc->host_ep);
        ret = UCC_ERR_NO_MESSAGE;
    }
    return ret;
}


static ucs_status_t _dpu_request_wait(ucp_worker_h ucp_worker, void *request,
                                      dpu_req_t *req_ctx)
{
//     printf ("%s\n", __FUNCTION__);
    ucs_status_t status;

    /* immediate completion */
    if (request == NULL) {
        return UCS_OK;
    }

    if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    }

    while (req_ctx->complete == 0) {
//         printf ("ucp_worker_progress()\n");
//         sleep(1);
        ucp_worker_progress(ucp_worker);
    }
    status = ucp_request_check_status(request);

    ucp_request_free(request);

    return status;
}

static int _dpu_request_finalize (ucp_worker_h ucp_worker, dpu_req_t *request,
                                  dpu_req_t *req_ctx)
{
//     printf ("%s\n", __FUNCTION__);
    ucs_status_t status;
    int ret = SUCCESS;

    status = _dpu_request_wait(ucp_worker, request, req_ctx);
    if (status != UCS_OK) {
        fprintf (stderr, "unable to recv UCX message (%s)\n", ucs_status_string(status));
        return -1;
    }

    /* reset req_ctx */
    req_ctx->complete = 0;

    return ret;
}

static int _dpu_rmem_setup(dpu_hc_t *hc)
{
    int i, ret = SUCCESS;
    ucp_request_param_t param;
    void *request;
    dpu_req_t req_ctx;
    size_t rkeys_total_len = 0, rkey_lens[3];
    uint64_t seg_base_addrs[3];
    char *rkeys = NULL, *rkey_p;

    req_ctx.complete = 0;

    /* XXX */
//     ucp_worker_print_info(hc->ucp_worker, stderr);

    /* recv rkey len & address */    /* recv rkey len & address */
    param.op_attr_mask  = UCP_OP_ATTR_FIELD_CALLBACK |
                          UCP_OP_ATTR_FIELD_USER_DATA;
    param.user_data     = &req_ctx;
    param.cb.recv       = tag_recv_cb;
    param.op_attr_mask  = UCP_OP_ATTR_FIELD_CALLBACK |
                          UCP_OP_ATTR_FIELD_USER_DATA;
    param.user_data     = &req_ctx;
    param.cb.recv       = tag_recv_cb;
    request = ucp_tag_recv_nbx(hc->ucp_worker, &hc->sync_addr, sizeof(uint64_t),
                                      EXCHANGE_ADDR_TAG, (uint64_t)-1, &param);
    ret = _dpu_request_finalize(hc->ucp_worker, request, &req_ctx);
    if (ret) {
        goto err;
    }

    request = ucp_tag_recv_nbx(hc->ucp_worker, &rkey_lens[0], sizeof(size_t),
                                      EXCHANGE_LENGTH_TAG, (uint64_t)-1, &param);
    ret = _dpu_request_finalize(hc->ucp_worker, request, &req_ctx);
    if (ret) {
        goto err;
    }

    rkeys = calloc(1, rkey_lens[0]);
    request = ucp_tag_recv_nbx(hc->ucp_worker, rkeys, rkey_lens[0], EXCHANGE_RKEY_TAG,
                               (uint64_t)-1, &param);

    ret = _dpu_request_finalize(hc->ucp_worker, request, &req_ctx);
    if (ret) {
        goto err;
    }

    ucs_status_t status = ucp_ep_rkey_unpack(hc->host_ep, rkeys, &hc->sync_rkey);
    if (UCS_OK != status) {
        fprintf(stderr, "failed to ucp_ep_rkey_unpack (%s)\n", ucs_status_string(status));
    }
    free(rkeys);

    /* send rkey lens & addresses */
    param.cb.send = send_cb;
   
    /* compute total len */
    for (i = 0; i < 3; i++) {
        rkey_lens[i] = hc->mem_segs_array[i].rkey.rkey_addr_len;
        seg_base_addrs[i] = (uint64_t)hc->mem_segs_array[i].base;
        rkeys_total_len += rkey_lens[i];
//         fprintf (stdout, "rkey_total_len = %lu, rkey_lens[i] = %lu\n",
//                  rkeys_total_len, rkey_lens[i]);
    }

    rkey_p = rkeys = calloc(1, rkeys_total_len);

    /* send rkey_lens */
    request = ucp_tag_send_nbx(hc->host_ep, rkey_lens, 3*sizeof(size_t),
                                     EXCHANGE_LENGTH_TAG, &param);
    ret = _dpu_request_finalize(hc->ucp_worker, request, &req_ctx);
    if (ret) {
        goto err;
    }

    request = ucp_tag_send_nbx(hc->host_ep, seg_base_addrs, 3*sizeof(uint64_t),
                                     EXCHANGE_ADDR_TAG, &param);
    ret = _dpu_request_finalize(hc->ucp_worker, request, &req_ctx);
    if (ret) {
        goto err;
    }

    /* send rkeys */
    for (i = 0; i < 3; i++) {
        memcpy(rkey_p, hc->mem_segs_array[i].rkey.rkey_addr, rkey_lens[i]);
        rkey_p+=rkey_lens[i];
    }

    request = ucp_tag_send_nbx(hc->host_ep, rkeys, rkeys_total_len, EXCHANGE_RKEY_TAG, &param);
    ret = _dpu_request_finalize(hc->ucp_worker, request, &req_ctx);
    if (ret) {
        goto err;
    }

    return SUCCESS;

err:
    printf ("%s ERROR!\n", __FUNCTION__);
    return ret;
}


int dpu_hc_accept(dpu_hc_t *hc)
{
    int ret;
    ucs_status_t status;
    ucp_rkey_h client_rkey_h;
    void *rem_worker_addr;
    size_t rem_worker_addr_len;

    /* In the call to accept(), the server is put to sleep and when for an incoming
         * client request, the three way TCP handshake* is complete, the function accept()
         * wakes up and returns the socket descriptor representing the client socket.
         */
//     fprintf (stderr, "Waiting for connection...\n");
    hc->connfd = accept(hc->listenfd, (struct sockaddr*)NULL, NULL);
    if (-1 == hc->connfd) {
        fprintf(stderr, "Error in accept (%s)!\n", strerror(errno));
    }
//     fprintf (stderr, "Connection established\n");

    ret = send(hc->connfd, &hc->worker_attr.address_length, sizeof(size_t), 0);
    if (-1 == ret) {
        fprintf(stderr, "send worker_address_length failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = send(hc->connfd, hc->worker_attr.address,
               hc->worker_attr.address_length, 0);
    if (-1 == ret) {
        fprintf(stderr, "send worker_address failed!\n");
        fprintf(stderr, "mmap_buffer failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = recv(hc->connfd, &rem_worker_addr_len, sizeof(size_t), MSG_WAITALL);
    if (-1 == ret) {
        fprintf(stderr, "recv address_length failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    rem_worker_addr = calloc(1, rem_worker_addr_len);

    ret = recv(hc->connfd, rem_worker_addr, rem_worker_addr_len, MSG_WAITALL);
    if (-1 == ret) {
        fprintf(stderr, "recv worker address failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    if (ret = _dpu_ep_create(hc, rem_worker_addr)) {
        fprintf(stderr, "dpu_create_ep failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = _dpu_rmem_setup(hc);
    if (ret) {
        fprintf(stderr, "exchange data failed!\n");
        goto err;
    }

    return ret;

err:
    close(hc->connfd);
    return ret;
}

int dpu_hc_wait(dpu_hc_t *hc, unsigned int cntr)
{
    dpu_sync_t *lsync = (dpu_sync_t*)hc->mem_segs.sync.base;
    
    while( lsync->coll_id < cntr) {
        ucp_worker_progress(hc->ucp_worker);
    }
    return 0;
}

ucc_datatype_t dpu_hc_get_dtype(dpu_hc_t *hc)
{
    dpu_sync_t *lsync = (dpu_sync_t*)hc->mem_segs.sync.base;
    return lsync->dtype;
}

ucc_reduction_op_t dpu_hc_get_op(dpu_hc_t *hc)
{
    dpu_sync_t *lsync = (dpu_sync_t*)hc->mem_segs.sync.base;
    return lsync->op;
}

unsigned int dpu_hc_get_count_total(dpu_hc_t *hc)
{
    dpu_sync_t *lsync = (dpu_sync_t*)hc->mem_segs.sync.base;
    return lsync->count_total;
}

unsigned int dpu_hc_get_count_in(dpu_hc_t *hc)
{
    dpu_sync_t *lsync = (dpu_sync_t*)hc->mem_segs.sync.base;
    return lsync->count_in;
}

int dpu_hc_reply(dpu_hc_t *hc, unsigned int cntr)
{
    dpu_sync_t *lsync = (dpu_sync_t*)hc->mem_segs.sync.base;
    uint32_t rsync;
    ucp_request_param_t req_param;
    void *request;
    dpu_req_t req_ctx = { 0 };
    int ret;

    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
            UCP_OP_ATTR_FIELD_DATATYPE |
            UCP_OP_ATTR_FIELD_USER_DATA;
    req_param.datatype     = ucp_dt_make_contig(1);
    req_param.cb.send      = send_cb;
    req_param.user_data     = &req_ctx;

//     static unsigned int cntr = 1;
//     while( lsync->itt < cntr) {
//         ucp_worker_progress(hc->ucp_worker);
//     }

    request = ucp_put_nbx(hc->host_ep, &cntr, sizeof(cntr),
                              hc->sync_addr, hc->sync_rkey,
                              &req_param);
    ret = _dpu_request_finalize(hc->ucp_worker, request, &req_ctx);
    if (ret) {
        return -1;
    }
//     cntr++;
    return 0;
}

#if 0
{
/* Work loop */
/* TEST
 * **** */

free(worker_attr.address);
free(rem_worker_addr);
close(connfd);

ep_close(ucp_worker, dpu_ep);


printf ("END %s\n", __FUNCTION__);

return ret;

err:
return ret;

}
#endif
