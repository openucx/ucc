/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp_addr.h"
#include "tl_ucp.h"
enum {
    UCC_TL_UCP_ADDR_EXCHANGE_MAX_ADDRLEN,
    UCC_TL_UCP_ADDR_EXCHANGE_GATHER,
    UCC_TL_UCP_ADDR_EXCHANGE_COMPLETE,
};

ucc_status_t ucc_tl_ucp_addr_exchange_start(ucc_tl_ucp_context_t *ctx,
                                            ucc_team_oob_coll_t oob,
                                            ucc_tl_ucp_addr_storage_t **storage)
{
    ucc_tl_ucp_addr_storage_t *st =
        ucc_malloc(sizeof(*st), "tl_ucp_addr_storage");
    ucc_status_t status;
    if (!st) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for tl_ucp_addr_storage",
                 sizeof(*st));
        return UCC_ERR_NO_MEMORY;
    }
    if ((NULL == ctx->worker_address) &&
        (UCS_OK != ucp_worker_get_address(ctx->ucp_worker, &ctx->worker_address,
                                          &ctx->ucp_addrlen))) {
        tl_error(ctx->super.super.lib, "failed to get ucp worker address");
        status = UCC_ERR_NO_MESSAGE;
        goto cleanup_st;
    }

    st->ctx       = ctx;
    st->oob       = oob;
    st->state     = UCC_TL_UCP_ADDR_EXCHANGE_MAX_ADDRLEN;
    st->addrlens  = ucc_malloc(sizeof(size_t) * oob.participants, "addrlens");
    st->addresses = NULL;
    if (!st->addrlens) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for addrlens array",
                 sizeof(size_t) * oob.participants);
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_st;
    }
    status = oob.allgather(&ctx->ucp_addrlen, st->addrlens, sizeof(size_t),
                           oob.coll_info, &st->oob_req);
    if (UCC_OK != status) {
        tl_error(ctx->super.super.lib, "failed to start oob allgather");
        goto cleanup_addrlens;
    }
    *storage = st;
    return UCC_OK;

cleanup_addrlens:
    free(st->addrlens);
cleanup_st:
    free(st);
    return status;
}

ucc_status_t ucc_tl_ucp_addr_exchange_test(ucc_tl_ucp_addr_storage_t *storage)
{
    ucc_status_t         status;
    void                *my_addr;
    int                  i;
    ucc_team_oob_coll_t *oob = &storage->oob;
    if (storage->state == UCC_TL_UCP_ADDR_EXCHANGE_COMPLETE) {
        return UCC_OK;
    }
    status = oob->req_test(storage->oob_req);
    if (UCC_INPROGRESS == status) {
        return status;
    } else if (UCC_OK != status) {
        oob->req_free(storage->oob_req);
        tl_error(storage->ctx->super.super.lib, "failed during oob req test");
        goto err;
    }
    oob->req_free(storage->oob_req);

    switch (storage->state) {
    case UCC_TL_UCP_ADDR_EXCHANGE_MAX_ADDRLEN:
        storage->max_addrlen = storage->addrlens[0];
        for (i = 0; i < oob->participants; i++) {
            if (storage->addrlens[i] > storage->max_addrlen) {
                storage->max_addrlen = storage->addrlens[i];
            }
        }
        ucc_free(storage->addrlens);
        storage->addrlens = NULL;
        storage->addresses =
            ucc_malloc(storage->max_addrlen * (oob->participants + 1),
                       "tl_ucp_storage_addresses");
        if (!storage->addresses) {
            status = UCC_ERR_NO_MEMORY;
            tl_error(
                storage->ctx->super.super.lib,
                "failed to allocate %zd bytes for tl_ucp storage addresses",
                storage->max_addrlen * (oob->participants + 1));
            goto err;
        }
        my_addr = (void *)((ptrdiff_t)storage->addresses +
                           oob->participants * storage->max_addrlen);
        memcpy(my_addr, storage->ctx->worker_address,
               storage->ctx->ucp_addrlen);
        status =
            oob->allgather(my_addr, storage->addresses, storage->max_addrlen,
                           oob->coll_info, &storage->oob_req);
        if (UCC_OK != status) {
            tl_error(storage->ctx->super.super.lib,
                     "failed to start oob allgather");
            goto err;
        }
        storage->state = UCC_TL_UCP_ADDR_EXCHANGE_GATHER;
        return UCC_INPROGRESS;
    case UCC_TL_UCP_ADDR_EXCHANGE_GATHER:
        storage->state = UCC_TL_UCP_ADDR_EXCHANGE_COMPLETE;
        break;
    }
    return UCC_OK;

err:
    free(storage->addrlens);
    free(storage->addresses);
    free(storage);
    return status;
}

void ucc_tl_ucp_addr_storage_free(ucc_tl_ucp_addr_storage_t *storage)
{
    free(storage->addresses);
    ucc_assert(NULL == storage->addrlens);
    free(storage);
}
