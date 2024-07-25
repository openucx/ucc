/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"
#include "tl_mlx5_pd.h"

typedef struct {
    int          sock, fd;
    uint32_t     pd_handle;
} connection_t;

ucc_status_t do_sendmsg(connection_t *conn)
{
    struct msghdr   msg  = {};
    struct cmsghdr *cmsghdr;
    struct iovec    iov;
    ssize_t         nbytes;
    int *           p;
    char            buf[CMSG_SPACE(sizeof(int))];
    uint32_t        handle;

    handle       = conn->pd_handle;
    iov.iov_base = &handle;
    iov.iov_len  = sizeof(handle);
    memset(buf, 0x0b, sizeof(buf));
    cmsghdr             = (struct cmsghdr *)buf;
    cmsghdr->cmsg_len   = CMSG_LEN(sizeof(int));
    cmsghdr->cmsg_level = SOL_SOCKET;
    cmsghdr->cmsg_type  = SCM_RIGHTS;
    msg.msg_name        = NULL;
    msg.msg_namelen     = 0;
    msg.msg_iov         = &iov;
    msg.msg_iovlen      = 1;
    msg.msg_control     = cmsghdr;
    msg.msg_controllen  = CMSG_LEN(sizeof(int));
    msg.msg_flags       = 0;
    p                   = (int *)CMSG_DATA(cmsghdr);
    *p                  = conn->fd;

    nbytes = sendmsg(conn->sock, &msg, 0);
    if (nbytes == -1) {
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static ucc_status_t do_recvmsg(int sock, int *shared_cmd_fd,
                               uint32_t *shared_pd_handle)
{
    uint32_t        handle;
    struct msghdr   msg;
    struct cmsghdr *cmsghdr;
    struct iovec    iov;
    ssize_t         nbytes;
    int *           p;
    char            buf[CMSG_SPACE(sizeof(int))];

    iov.iov_base = &handle;
    iov.iov_len  = sizeof(handle);

    memset(buf, 0x0d, sizeof(buf));
    cmsghdr             = (struct cmsghdr *)buf;
    cmsghdr->cmsg_len   = CMSG_LEN(sizeof(int));
    cmsghdr->cmsg_level = SOL_SOCKET;
    cmsghdr->cmsg_type  = SCM_RIGHTS;
    msg.msg_name        = NULL;
    msg.msg_namelen     = 0;
    msg.msg_iov         = &iov;
    msg.msg_iovlen      = 1;
    msg.msg_control     = cmsghdr;
    msg.msg_controllen  = CMSG_LEN(sizeof(int));
    msg.msg_flags       = 0;

    nbytes = recvmsg(sock, &msg, 0);
    if (nbytes == -1) {
        return UCC_ERR_NO_MESSAGE;
    }

    p = (int *)CMSG_DATA(cmsghdr);

    *shared_cmd_fd    = *p;
    *shared_pd_handle = handle;

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_socket_init(ucc_tl_mlx5_context_t *ctx,
                                     ucc_rank_t group_size, int *sock_p,
                                     const char *sock_path)
{
    struct sockaddr_storage storage = {};
    struct sockaddr_un *    addr;
    int                     sock;

    sock = socket(PF_LOCAL, SOCK_STREAM, 0);
    if (sock == -1) {
        tl_debug(ctx->super.super.lib,
                 "failed to create server socket errno %d", errno);
        return UCC_ERR_NO_MESSAGE;
    }

    addr             = (struct sockaddr_un *)&storage;
    addr->sun_family = AF_UNIX;
    strncpy(addr->sun_path, sock_path, sizeof(addr->sun_path));
    addr->sun_path[sizeof(addr->sun_path) - 1] = '\0';

    if (bind(sock, (struct sockaddr *)addr, sizeof(struct sockaddr_un)) == -1) {
        tl_debug(ctx->super.super.lib, "failed to bind server socket errno %d",
                 errno);
        goto out;
    }
    if (listen(sock, group_size) == -1) {
        tl_debug(ctx->super.super.lib,
                 "failed to listen to server socket errno %d", errno);
        goto out;
    }
    *sock_p = sock;
    return UCC_OK;

out:
    close(sock);
    return UCC_ERR_NO_MESSAGE;
}

static ucc_status_t client_recv_data(int *shared_cmd_fd,
                                     uint32_t *shared_pd_handle,
                                     const char *sock_path, int *sock_p,
                                     ucc_tl_mlx5_lib_t *lib)
{
    struct sockaddr_storage sockaddr = {};
    ucc_status_t            status   = UCC_OK;
    struct sockaddr_un *    addr;
    int                     sock;

    sock = socket(PF_LOCAL, SOCK_STREAM, 0);
    if (sock == -1) {
        tl_debug(lib, "failed to create client socket errno %d", errno);
        return UCC_ERR_NO_MESSAGE;
    }

    addr             = (struct sockaddr_un *)&sockaddr;
    addr->sun_family = AF_UNIX;
    strncpy(addr->sun_path, sock_path, sizeof(addr->sun_path));
    addr->sun_path[sizeof(addr->sun_path) - 1] = '\0';

    while (connect(sock, (struct sockaddr *)addr, SUN_LEN(addr)) == -1) {
        if (errno != ENOENT) {
            tl_debug(lib, "failed to connect client socket errno %d", errno);
            status = UCC_ERR_NO_MESSAGE;
            goto out;
        }
    }
    if (do_recvmsg(sock, shared_cmd_fd, shared_pd_handle) != UCC_OK) {
        tl_debug(lib, "Failed to recv msg");
        status = UCC_ERR_NO_MESSAGE;
        goto out;
    }

    *sock_p = sock;
    return UCC_OK;

out:
    if (close(sock) == -1) {
        tl_debug(lib, "Failed to close client socket errno %d", errno);
        status = UCC_ERR_NO_MESSAGE;
    }
    return status;
}

static ucc_status_t server_send_data(int command_fd, uint32_t pd_handle,
                                     ucc_rank_t group_size, int sock,
                                     ucc_tl_mlx5_lib_t *lib)
{
    ucc_status_t       status = UCC_OK;
    int                i;
    connection_t       connection[group_size];
    struct sockaddr_un addr;
    socklen_t          addrlen;

    for (i = 0; i < group_size; i++) {
        /* accept incoming connections */
        connection[i].fd        = command_fd;
        connection[i].pd_handle = pd_handle;
        connection[i].sock      = accept(sock, NULL, 0);
        if (connection[i].sock == -1) {
            tl_debug(lib,
                     "failed to accept socket connection request %d,"
                     " errno %d",
                     i, errno);
            goto listen_fail;
        }
        status = do_sendmsg(&connection[i]);
        if (status != UCC_OK) {
            tl_debug(lib, "failed to send cmd_fd");
            goto listen_fail;
        }
    }

    addrlen = sizeof(addr);
    getsockname(sock, (struct sockaddr *)&addr, &addrlen);

    if (remove(addr.sun_path) == -1) {
        tl_debug(lib, "socket file removal failed");
        status = UCC_ERR_NO_MESSAGE;
    }

    return status;

listen_fail:
    if (close(sock) == -1) {
        tl_debug(lib, "failed to close server socket errno %d", errno);
        status = UCC_ERR_NO_MESSAGE;
    }

    return status;
}

ucc_status_t ucc_tl_mlx5_share_ctx_pd(ucc_tl_mlx5_context_t *ctx,
                                      const char *           sock_path,
                                      ucc_rank_t group_size, int is_ctx_owner,
                                      int ctx_owner_sock)
{
    ucc_tl_mlx5_lib_t *lib =
        ucc_derived_of(ctx->super.super.lib, ucc_tl_mlx5_lib_t);
    int          ctx_fd;
    uint32_t     pd_handle;
    ucc_status_t status;

    if (!is_ctx_owner) {
        status =
            client_recv_data(&ctx_fd, &pd_handle, sock_path, &ctx->sock, lib);
        if (UCC_OK != status) {
            tl_debug(lib, "failed to share ctx & pd from client side");
            return status;
        }
        ctx->shared_ctx = ibv_import_device(ctx_fd);
        if (!ctx->shared_ctx) {
            tl_debug(lib, "import context failed");
            return UCC_ERR_NO_MESSAGE;
        }
        ctx->shared_pd = ibv_import_pd(ctx->shared_ctx, pd_handle);
        if (!ctx->shared_pd) {
            tl_debug(lib, "import PD failed");
            if (ibv_close_device(ctx->shared_ctx)) {
                tl_debug(lib, "imported context close failed");
            }
            return UCC_ERR_NO_MESSAGE;
        }
    } else {
        ctx_fd    = ctx->shared_ctx->cmd_fd;
        pd_handle = ctx->shared_pd->handle;
        status = server_send_data(ctx_fd, pd_handle, group_size - 1,
                                  ctx_owner_sock, lib);
        if (UCC_OK != status) {
            tl_debug(lib, "failed to share ctx & pd from server side");
            return status;
        }
    }
    return UCC_OK;
}

static void ucc_tl_mlx5_context_barrier(ucc_context_oob_coll_t *oob,
                                        ucc_context_t *core_ctx,
                                        ucc_base_lib_t *lib)
{
    char        *rbuf;
    char         sbuf;
    void        *req;
    ucc_status_t status;

    if (ucc_unlikely(oob->n_oob_eps < 2)) {
        return;
    }

    rbuf = ucc_malloc(sizeof(char) * oob->n_oob_eps, "tmp_barrier");
    if (!rbuf) {
        tl_debug(lib, "failed to allocate %zd bytes for tmp barrier array",
                 sizeof(char) * oob->n_oob_eps);
        return;
    }
    if (UCC_OK ==
        oob->allgather(&sbuf, rbuf, sizeof(char), oob->coll_info, &req)) {
        ucc_assert(req != NULL);
        while (UCC_OK != (status = oob->req_test(req))) {
            ucc_context_progress(core_ctx);
            if (status < 0) {
                tl_debug(lib, "failed to test oob req");
                break;
            }
        }
        oob->req_free(req);
    }
    ucc_free(rbuf);
}

ucc_status_t ucc_tl_mlx5_remove_shared_ctx_pd(ucc_tl_mlx5_context_t *ctx)
{
    ucc_base_lib_t *lib    = ctx->super.super.lib;
    ucc_status_t    status = UCC_OK;
    int err;

    if (ctx->shared_pd && ctx->is_imported) {
        ibv_unimport_pd(ctx->shared_pd);
    }
    ucc_tl_mlx5_context_barrier(&UCC_TL_CTX_OOB(ctx),
                                ctx->super.super.ucc_context, lib);
    if (ctx->shared_pd && !ctx->is_imported) {
        err = ibv_dealloc_pd(ctx->shared_pd);
        if (err) {
            tl_debug(lib, "failed to dealloc PD, errno %d", err);
            status = UCC_ERR_NO_MESSAGE;
        }
    }

    if (ctx->shared_ctx) {
        if (ibv_close_device(ctx->shared_ctx)) {
            tl_debug(lib, "failed to close ib ctx");
            status = UCC_ERR_NO_MESSAGE;
        }
    }

    return status;
}
