/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#include <stdint.h>
#include <stdlib.h>

#include "worker_ucc.h"
#include "../common/urom_ucc.h"

DOCA_LOG_REGISTER(UCC::DOCA_CL : WORKER_UCC_P2P);

void urom_ep_err_cb(void *arg, ucp_ep_h ep, ucs_status_t ucs_status)
{
    (void)arg;
    (void)ep;

    DOCA_LOG_ERR("Endpoint error detected, status: %s",
                 ucs_status_string(ucs_status));
}

/*
 * UCC worker EP lookup function
 *
 * @ucc_worker [in]: UCC worker context
 * @dest [in]: destination id
 * @ep [out]: set UCP endpoint
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t worker_ucc_ep_lookup(struct urom_worker_ucc *ucc_worker,
                                         uint64_t dest, ucp_ep_h *ep)
{
    int             ret;
    khint_t         k;
    void           *addr;
    ucp_ep_h        new_ep;
    doca_error_t    status;
    ucs_status_t    ucs_status;
    ucp_ep_params_t ep_params;

    k = kh_get(ep, ucc_worker->ucp_data.eps, dest);
    if (k != kh_end(ucc_worker->ucp_data.eps)) {
        *ep = kh_value(ucc_worker->ucp_data.eps, k);
        return DOCA_SUCCESS;
    }

    /* Create new EP */
    status = doca_urom_worker_domain_addr_lookup(ucc_worker->super,
                                                 dest, &addr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Id not found in domain:: %#lx", dest);
        return DOCA_ERROR_NOT_FOUND;
    }

    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                            UCP_EP_PARAM_FIELD_ERR_HANDLER |
                               UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_handler.cb = urom_ep_err_cb;
    ep_params.err_handler.arg = NULL;
    ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.address = addr;

    ucs_status = ucp_ep_create(ucc_worker->ucp_data.ucp_worker,
                               &ep_params, &new_ep);
    if (ucs_status != UCS_OK) {
        DOCA_LOG_ERR("ucp_ep_create() returned: %s",
                     ucs_status_string(ucs_status));
        return DOCA_ERROR_INITIALIZATION;
    }

    k = kh_put(ep, ucc_worker->ucp_data.eps, dest, &ret);
    if (ret <= 0) {
        return DOCA_ERROR_DRIVER;
    }
    kh_value(ucc_worker->ucp_data.eps, k) = new_ep;

    *ep = new_ep;
    DOCA_LOG_DBG("Created EP for dest: %#lx", dest);
    return DOCA_SUCCESS;
}

/*
 * UCC worker memh lookup function
 *
 * @ucc_worker [in]: UCC worker context
 * @dest [in]: destination id
 * @memh [out]: set memory handle
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t worker_ucc_memh_lookup(struct urom_worker_ucc *ucc_worker,
                                           uint64_t dest, ucp_mem_h *memh)
{
    ucp_mem_map_params_t mmap_params = {0};
    size_t               memh_len    = 0;
    int                  ret;
    khint_t              k;
    void                *mem_handle;
    ucp_mem_h            memh_id;
    doca_error_t         status;
    ucs_status_t         ucs_status;

    k = kh_get(memh, ucc_worker->ucp_data.memh, dest);
    if (k != kh_end(ucc_worker->ucp_data.memh)) {
        *memh = kh_value(ucc_worker->ucp_data.memh, k);
        return DOCA_SUCCESS;
    }

    /* Lookup memory handle */
    status = doca_urom_worker_domain_memh_lookup(ucc_worker->super, dest, 0,
                                                 &memh_len, &mem_handle);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Id not found in domain:: %#lx", dest);
        return DOCA_ERROR_NOT_FOUND;
    }

    mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    mmap_params.exported_memh_buffer = mem_handle;

    ucs_status = ucp_mem_map(ucc_worker->ucp_data.ucp_context, &mmap_params,
                             &memh_id);
    if (ucs_status != UCS_OK) {
        DOCA_LOG_ERR("Failed to map packed memh %p", mem_handle);
        return DOCA_ERROR_NOT_FOUND;
    }

    k = kh_put(memh, ucc_worker->ucp_data.memh, dest, &ret);
    if (ret <= 0) {
        DOCA_LOG_ERR("Failed to add memh to hashtable map");
        if (ucp_mem_unmap(ucc_worker->ucp_data.ucp_context, memh_id) != UCS_OK) {
            DOCA_LOG_ERR("Failed to unmap memh");
        }
        return DOCA_ERROR_DRIVER;
    }
    kh_value(ucc_worker->ucp_data.memh, k) = memh_id;

    *memh = memh_id;
    DOCA_LOG_DBG("Assigned memh %p for dest: %#lx", memh_id, dest);
    return DOCA_SUCCESS;
}

/*
 * UCC worker memory key lookup function
 *
 * @ucc_worker [in]: UCC worker context
 * @dest [in]: destination id
 * @ep [in]: destination endpoint
 * @va [in]: memory host address
 * @ret_rkey [out]: set remote memory key
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t worker_ucc_key_lookup(struct urom_worker_ucc *ucc_worker,
                                          uint64_t dest,
                                          ucp_ep_h ep,
                                          uint64_t va,
                                          void **ret_rkey)
{
    khint_t      k;
    int          ret;
    void        *packed_key;
    size_t       packed_key_len;
    ucp_rkey_h   rkey;
    doca_error_t status;
    ucs_status_t ucs_status;
    int          seg;

    k = kh_get(rkeys, ucc_worker->ucp_data.rkeys, dest);
    if (k != kh_end(ucc_worker->ucp_data.rkeys)) {
        *ret_rkey = kh_value(ucc_worker->ucp_data.rkeys, k);
        return DOCA_SUCCESS;
    }

    status = doca_urom_worker_domain_seg_lookup(ucc_worker->super, dest,
                                                va, &seg);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Id not found in domain: %#lx", dest);
        return DOCA_ERROR_NOT_FOUND;
    }

    status = doca_urom_worker_domain_mkey_lookup(ucc_worker->super, dest, seg,
                                                 &packed_key_len, &packed_key);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Id not found in domain: %#lx", dest);
        return DOCA_ERROR_NOT_FOUND;
    }

    ucs_status = ucp_ep_rkey_unpack(ep, packed_key, &rkey);
    if (ucs_status != UCS_OK) {
        return DOCA_ERROR_NOT_FOUND;
    }

    k = kh_put(rkeys, ucc_worker->ucp_data.rkeys, dest, &ret);
    if (ret <= 0) {
        return DOCA_ERROR_DRIVER;
    }
    kh_value(ucc_worker->ucp_data.rkeys, k) = rkey;

    *ret_rkey = rkey;
    DOCA_LOG_DBG("Assigned rkey for dest: %#lx", dest);
    return DOCA_SUCCESS;
}

/*
 * UCC send tag completion callback
 *
 * @request [in]: UCP send request
 * @status [in]: send task status
 * @user_data [in]: UCC data
 */
static void send_completion_cb(void *request, ucs_status_t status,
                               void *user_data)
{
    int *req = (int *)user_data;

    if (status != UCS_OK) {
        *req = -1;
    } else {
        *req = 1;
    }

    ucp_request_free(request);
}

/*
 * UCC recv tag completion callback
 *
 * @request [in]: UCP recv request
 * @status [in]: recv task status
 * @info [in]: recv task info
 * @user_data [in]: UCC data
 */
static void recv_completion_cb(void *request, ucs_status_t status,
                               const ucp_tag_recv_info_t *info,
                               void *user_data)
{
    int *req = (int *)user_data;
    (void)info;

    if (status != UCS_OK) {
        *req = -1;
    } else {
        *req = 1;
    }

    ucp_request_free(request);
}

doca_error_t ucc_send_nb(void *msg,
                         size_t len,
                         int64_t myrank,
                         int64_t dest,
                         struct urom_worker_ucc *ucc_worker,
                         int *req)
{
    ucp_ep_h            ep        = NULL;
    ucp_request_param_t req_param = {0};
    doca_error_t        urom_status;
    ucs_status_ptr_t    ucp_status;

    *req                   = 0;
    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE     |
                                UCP_OP_ATTR_FIELD_CALLBACK  |
                                UCP_OP_ATTR_FIELD_USER_DATA;
    req_param.datatype     = ucp_dt_make_contig(len);
    req_param.cb.send      = send_completion_cb;
    req_param.user_data    = (void *)req;

    urom_status = worker_ucc_ep_lookup(ucc_worker, dest, &ep);
    if (urom_status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to send to %ld in UCC oob", dest);
        return DOCA_ERROR_NOT_FOUND;
    }

    /* Process tag send */
    ucp_status = ucp_tag_send_nbx(ep, msg, 1, myrank, &req_param);
    if (ucp_status != UCS_OK) {
        if (UCS_PTR_IS_ERR(ucp_status)) {
            ucp_request_cancel(ucc_worker->ucp_data.ucp_worker, ucp_status);
            ucp_request_free(ucp_status);
            return DOCA_ERROR_NOT_FOUND;
        }
    } else {
        *req = 1;
    }

    return DOCA_SUCCESS;
}

doca_error_t ucc_recv_nb(void *msg, size_t len, int64_t dest, struct urom_worker_ucc *ucc_worker, int *req)
{
    ucp_request_param_t req_param = {};
    ucs_status_ptr_t    ucp_status;

    *req                   = 0;
    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE     |
                                UCP_OP_ATTR_FIELD_CALLBACK  |
                                UCP_OP_ATTR_FIELD_USER_DATA;
    req_param.datatype     = ucp_dt_make_contig(len);
    req_param.cb.recv      = recv_completion_cb;
    req_param.user_data    = (void *)req;

    /* Process tag recv */
    ucp_status = ucp_tag_recv_nbx(ucc_worker->ucp_data.ucp_worker, msg, 1, dest,
                                  0xffff, &req_param);
    if (ucp_status != UCS_OK) {
        if (UCS_PTR_IS_ERR(ucp_status)) {
            ucp_request_cancel(ucc_worker->ucp_data.ucp_worker, ucp_status);
            ucp_request_free(ucp_status);
            return DOCA_ERROR_NOT_FOUND;
        }
    } else {
        *req = 1;
    }
    return DOCA_SUCCESS;
}

doca_error_t ucc_rma_put(void *buffer,
                         void *target,
                         size_t msglen,
                         uint64_t dest,
                         uint64_t myrank,
                         uint64_t ctx_id,
                         struct urom_worker_ucc *ucc_worker)
{
    uint64_t            rva       = (uint64_t)target;
    ucp_request_param_t req_param = {0};
    ucp_mem_h           memh      = NULL;
    ucp_ep_h            ep;
    ucp_rkey_h          rkey;
    doca_error_t        urom_status;
    ucs_status_ptr_t    ucp_status;

    if (dest == MAX_HOST_DEST_ID) {
        ep = ucc_worker->ucc_data[ctx_id].host;
    } else {
        urom_status = worker_ucc_ep_lookup(ucc_worker, dest, &ep);
        if (urom_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to find peer %ld to complete collective",
                         dest);
            return DOCA_ERROR_NOT_FOUND;
        }
    }

    if (dest != MAX_HOST_DEST_ID) {
        urom_status = worker_ucc_memh_lookup(ucc_worker, dest, &memh);
        if (urom_status != DOCA_SUCCESS)
            DOCA_LOG_ERR("Failed to lookup key for peer %ld", dest);
    }

    if (dest == MAX_HOST_DEST_ID) {
        urom_status = worker_ucc_key_lookup(ucc_worker, myrank, ep, rva,
                                            (void **)&rkey);
        if (urom_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to lookup rkey for peer %ld", dest);
            return DOCA_ERROR_NOT_FOUND;
        }
    } else {
        urom_status = worker_ucc_key_lookup(ucc_worker, dest, ep, rva,
                                            (void **)&rkey);
        if (urom_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to lookup rkey for peer %ld", dest);
            return DOCA_ERROR_NOT_FOUND;
        }
    }

    if (memh != NULL) {
        req_param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        req_param.memh = memh;
    }

    ucp_status = ucp_put_nbx(ep, buffer, msglen, rva, rkey, &req_param);
    if (ucp_status != UCS_OK) {
        if (UCS_PTR_IS_ERR(ucp_status)) {
            ucp_request_free(ucp_status);
            return DOCA_ERROR_NOT_FOUND;
        }
        while (ucp_request_check_status(ucp_status) == UCS_INPROGRESS) {
            ucp_worker_progress(ucc_worker->ucp_data.ucp_worker);
        }
        ucp_request_free(ucp_status);
    }
    return DOCA_SUCCESS;
}

doca_error_t ucc_rma_get(void *buffer,
                         void *target,
                         size_t msglen,
                         uint64_t dest,
                         uint64_t myrank,
                         uint64_t ctx_id,
                         struct urom_worker_ucc *ucc_worker)
{
    ucp_ep_h            ep        = NULL;
    ucp_mem_h           memh      = NULL;
    ucp_rkey_h          rkey      = NULL;
    ucp_request_param_t req_param = {0};
    uint64_t            rva       = (uint64_t)target;
    doca_error_t        urom_status;
    ucs_status_ptr_t    ucp_status;

    if (dest == MAX_HOST_DEST_ID) {
        ep = ucc_worker->ucc_data[ctx_id].host;
    } else {
        urom_status = worker_ucc_ep_lookup(ucc_worker, dest, &ep);
        if (urom_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to find peer %ld to complete collective",
                         dest);
            return DOCA_ERROR_NOT_FOUND;
        }
    }

    if (dest != MAX_HOST_DEST_ID) {
        urom_status = worker_ucc_memh_lookup(ucc_worker, dest, &memh);
        if (urom_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to lookup key for peer %ld", dest);
            return DOCA_ERROR_NOT_FOUND;
        }
    }

    if (dest == MAX_HOST_DEST_ID) {
        urom_status = worker_ucc_key_lookup(ucc_worker, myrank, ep, rva,
                                            (void **)&rkey);
        if (urom_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to lookup rkey for peer %ld", dest);
            return DOCA_ERROR_NOT_FOUND;
        }
    } else {
        urom_status = worker_ucc_key_lookup(ucc_worker, dest, ep, rva,
                                            (void **)&rkey);
        if (urom_status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to lookup rkey for peer %ld", dest);
            return DOCA_ERROR_NOT_FOUND;
        }
    }

    if (memh != NULL) {
        req_param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        req_param.memh = memh;
    }

    ucp_status = ucp_get_nbx(ep, buffer, msglen, rva, rkey, &req_param);
    if (ucp_status != UCS_OK) {
        if (UCS_PTR_IS_ERR(ucp_status)) {
            ucp_request_free(ucp_status);
            ucp_rkey_destroy(rkey);
            return DOCA_ERROR_NOT_FOUND;
        }
        while (ucp_request_check_status(ucp_status) == UCS_INPROGRESS) {
            ucp_worker_progress(ucc_worker->ucp_data.ucp_worker);
        }
        ucp_request_free(ucp_status);
    }
    ucp_rkey_destroy(rkey);
    return DOCA_SUCCESS;
}

doca_error_t ucc_rma_get_host(void *buffer,
                             void *target,
                             size_t msglen,
                             uint64_t ctx_id,
                             void *packed_key,
                             struct urom_worker_ucc *ucc_worker)
{
    ucp_ep_h            ep        = NULL;
    ucp_rkey_h          rkey      = NULL;
    uint64_t            rva       = (uint64_t)target;
    ucp_request_param_t req_param = {0};
    ucs_status_t        ucs_status;
    ucs_status_ptr_t    ucp_status;

    if (packed_key == NULL) {
        return DOCA_ERROR_INVALID_VALUE;
    }

    ep = ucc_worker->ucc_data[ctx_id].host;

    ucs_status = ucp_ep_rkey_unpack(ep, packed_key, &rkey);
    if (ucs_status != UCS_OK) {
        DOCA_LOG_ERR("Failed to unpack rkey");
        return DOCA_ERROR_NOT_FOUND;
    }

    ucp_status = ucp_get_nbx(ep, buffer, msglen, rva, rkey, &req_param);
    if (UCS_OK != ucp_status) {
        if (UCS_PTR_IS_ERR(ucp_status)) {
            DOCA_LOG_ERR("Failed to perform ucp_get_nbx(): %s\n",
                         ucs_status_string(UCS_PTR_STATUS(ucp_status)));
            ucp_request_free(ucp_status);
            ucp_rkey_destroy(rkey);
            return DOCA_ERROR_NOT_FOUND;
        }
        while (UCS_INPROGRESS == ucp_request_check_status(ucp_status)) {
            ucp_worker_progress(ucc_worker->ucp_data.ucp_worker);
        }
        if (UCS_PTR_IS_ERR(ucp_status)) {
            DOCA_LOG_ERR("Failed to perform ucp_get_nbx(): %s\n",
                         ucs_status_string(UCS_PTR_STATUS(ucp_status)));
            ucp_request_free(ucp_status);
            ucp_rkey_destroy(rkey);
            return DOCA_ERROR_NOT_FOUND;
        }
        ucp_request_free(ucp_status);
    }
    ucp_rkey_destroy(rkey);
    return DOCA_SUCCESS;
}

doca_error_t ucc_rma_put_host(void *buffer,
                             void *target,
                             size_t msglen,
                             uint64_t ctx_id,
                             void *packed_key,
                             struct urom_worker_ucc *ucc_worker)
{
    ucp_ep_h            ep        = NULL;
    ucp_rkey_h          rkey      = NULL;
    uint64_t            rva       = (uint64_t)target;
    ucp_request_param_t req_param = {0};
    ucs_status_t        ucs_status;
    ucs_status_ptr_t    ucp_status;

    if (packed_key == NULL) {
        return DOCA_ERROR_INVALID_VALUE;
    }

    ep = ucc_worker->ucc_data[ctx_id].host;

    ucs_status = ucp_ep_rkey_unpack(ep, packed_key, &rkey);
    if (ucs_status != UCS_OK) {
        DOCA_LOG_ERR("Failed to unpack rkey");
        return DOCA_ERROR_NOT_FOUND;
    }

    ucp_status = ucp_put_nbx(ep, buffer, msglen, rva, rkey, &req_param);
    if (UCS_OK != ucp_status) {
        if (UCS_PTR_IS_ERR(ucp_status)) {
            DOCA_LOG_ERR("Failed to perform ucp_put_nbx(): %s\n",
                         ucs_status_string(UCS_PTR_STATUS(ucp_status)));
            ucp_request_free(ucp_status);
            ucp_rkey_destroy(rkey);
            return DOCA_ERROR_NOT_FOUND;
        }
        while (UCS_INPROGRESS == ucp_request_check_status(ucp_status)) {
            ucp_worker_progress(ucc_worker->ucp_data.ucp_worker);
        }
        if (UCS_PTR_IS_ERR(ucp_status)) {
            DOCA_LOG_ERR("Failed to perform ucp_put_nbx(): %s\n",
                         ucs_status_string(UCS_PTR_STATUS(ucp_status)));
            ucp_request_free(ucp_status);
            ucp_rkey_destroy(rkey);
            return DOCA_ERROR_NOT_FOUND;
        }
        ucp_request_free(ucp_status);
    }
    ucp_rkey_destroy(rkey);
    return DOCA_SUCCESS;
}
