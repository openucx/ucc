/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_helper.h"


static ucc_status_t ucc_tl_get_ipoib_ip(char *ifname, struct sockaddr_storage *addr)
{
    ucc_status_t    ret     = UCC_OK;
    struct ifaddrs *ifaddr  = NULL;
    struct ifaddrs *ifa     = NULL;
    int             is_ipv4 = 0;
    int             family;
    int             n;
    int             is_up;

    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        return UCC_ERR_NO_RESOURCE;
    }

    for (ifa = ifaddr, n = 0; ifa != NULL; ifa=ifa->ifa_next, n++) {
        if (ifa->ifa_addr == NULL) {
            continue;
        }

        family = ifa->ifa_addr->sa_family;
        if (family != AF_INET && family != AF_INET6) {
            continue;
        }

        is_up   = (ifa->ifa_flags & IFF_UP) == IFF_UP;
        is_ipv4 = (family == AF_INET) ? 1 : 0;

        if (is_up && !strncmp(ifa->ifa_name, ifname, strlen(ifname)) ) {
            if (is_ipv4) {
                memcpy((struct sockaddr_in *) addr,
                       (struct sockaddr_in *) ifa->ifa_addr,
                       sizeof(struct sockaddr_in));
            }
            else {
                memcpy((struct sockaddr_in6 *) addr,
                       (struct sockaddr_in6 *) ifa->ifa_addr,
                       sizeof(struct sockaddr_in6));
            }

            ret = UCC_OK;
            break;
        }
    }

    freeifaddrs(ifaddr);

    return ret;
}

static int cmp_files(char *f1, char *f2)
{
    int   answer = 0;
    FILE *fp1;
    FILE *fp2;

    if ((fp1 = fopen(f1, "r")) == NULL) {
        goto out;
    } else if ((fp2 = fopen(f2, "r")) == NULL) {
        goto close;
    }

    int ch1 = getc(fp1);
    int ch2 = getc(fp2);

    while((ch1 != EOF) && (ch2 != EOF) && (ch1 == ch2))
    {
        ch1 = getc(fp1);
        ch2 = getc(fp2) ;
    }

    if (ch1 == ch2) {
        answer = 1;
    }

    fclose(fp2);
close:
    fclose(fp1);
out:
    return answer;
}

#define PREF "/sys/class/net/"
#define SUFF "/device/resource"
#define MAX_STR_LEN 128

static int port_from_file(char *port_file)
{
    char  buf1[MAX_STR_LEN];
    char  buf2[MAX_STR_LEN];
    FILE *fp;
    int   res = -1;
    int   len;

    if ((fp = fopen(port_file, "r")) == NULL) {
        return -1;
    }

    if (fgets(buf1, MAX_STR_LEN - 1, fp) == NULL) {
        goto out;
    }

    len       = strlen(buf1) - 2;
    strncpy(buf2, buf1 + 2, len);
    buf2[len] = 0;
    res       = atoi(buf2);

out:
    fclose(fp);
    return res;
}

static int dev2if(char *dev_name, char *port, struct sockaddr_storage *rdma_src_addr)
{
    ucc_status_t ret     = UCC_OK;
    glob_t       glob_el = {0,};
    char         dev_file [MAX_STR_LEN];
    char         port_file[MAX_STR_LEN];
    char         net_file [MAX_STR_LEN];
    char         if_name  [MAX_STR_LEN];
    char         glob_path[MAX_STR_LEN];
    int          i; 
    char       **p;
    int          len;
    
    sprintf(glob_path, PREF"*");

    sprintf(dev_file, "/sys/class/infiniband/%s"SUFF, dev_name);
    glob(glob_path, 0, 0, &glob_el);
    p = glob_el.gl_pathv;

    if (glob_el.gl_pathc >= 1) {
        for (i = 0; i < glob_el.gl_pathc; i++, p++) {
            sprintf(port_file, "%s/dev_id", *p);
            sprintf(net_file,  "%s"SUFF,    *p);
            if(cmp_files(net_file, dev_file) && port != NULL &&
               port_from_file(port_file) == atoi(port) - 1) {
                len = strlen(net_file) - strlen(PREF) - strlen(SUFF);
                strncpy(if_name, net_file + strlen(PREF), len);
                if_name[len] = 0;
                if (UCC_OK == (ret = ucc_tl_get_ipoib_ip(if_name, rdma_src_addr))) {
                    break;
                }
            }
        }
    }
    globfree(&glob_el);
    return ret;
}

ucc_status_t ucc_tl_probe_ip_over_ib(const char* ib_dev, struct sockaddr_storage *addr)
{
    ucc_status_t            ret;
    char                   *ib      = NULL;
    char                   *ib_name = NULL;
    char                   *port    = NULL;
    struct sockaddr_storage rdma_src_addr;

    if (NULL == ib_dev) {
        ret = UCC_ERR_NO_RESOURCE;
    } else { 
        ib  = strdup(ib_dev);
        ucs_string_split(ib, ":", 2, &ib_name, &port);
        ret = dev2if(ib_name, port, &rdma_src_addr);
    }

    if (UCC_OK == ret && addr) {
        *addr = rdma_src_addr;
    }
    return ret;
}

static ucc_status_t ucc_tl_mlx5_mcast_create_ah(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct ibv_ah_attr ah_attr = {
        .is_global     = 1,
        .grh           = {.sgid_index = 0},
        .dlid          = comm->mcast_lid,
        .sl            = DEF_SL,
        .src_path_bits = DEF_SRC_PATH_BITS,
        .port_num      = comm->ctx->ib_port
    };

    memcpy(ah_attr.grh.dgid.raw, &comm->mgid, sizeof(ah_attr.grh.dgid.raw));

    comm->mcast.ah = ibv_create_ah(comm->ctx->pd, &ah_attr);
    if (!comm->mcast.ah) {
        tl_error(comm->lib, "Failed to create AH");
        return UCC_ERR_NO_RESOURCE;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_fini_mcast_group(ucc_tl_mlx5_mcast_coll_context_t *ctx, ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    char buf[40];

    inet_ntop(AF_INET6, &comm->mcast_addr, buf, 40);

    tl_debug(ctx->lib, "Mcast leave: ctx %p, comm %p, dgid: %s", ctx, comm, buf);

    if (rdma_leave_multicast(ctx->id, (struct sockaddr*)&comm->mcast_addr)) {
        tl_error(comm->lib, "ERROR: mcast tl  rmda_leave_multicast failed");
        return UCC_ERR_NO_RESOURCE;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_clean_mcast_comm(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_tl_mlx5_mcast_context_t *mcast_ctx = ucc_container_of(comm->ctx, ucc_tl_mlx5_mcast_context_t, mcast_context);
    ucc_tl_mlx5_context_t       *mlx5_ctx  = ucc_container_of(mcast_ctx, ucc_tl_mlx5_context_t, mcast);
    ucc_context_h                context   = mlx5_ctx->super.super.ucc_context;
    int ret                                = UCC_OK;

    tl_debug(comm->lib, "Cleaning tl mcast  comm: %p, id %d, mlid %x",
                comm, comm->comm_id, comm->mcast_lid);

    while (UCC_INPROGRESS == ucc_tl_mlx5_mcast_reliable(comm)) {
        ucc_context_progress(context);
    }

    if (comm->mcast.qp) {
        ret = ibv_detach_mcast(comm->mcast.qp, &comm->mgid, comm->mcast_lid);
        if (ret) {
            tl_error(comm->lib, "Couldn't detach QP, ret %d, errno %d", ret, errno);
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->mcast.qp) {
        ret = ibv_destroy_qp(comm->mcast.qp);
        if (ret) {
            tl_error(comm->lib, "Failed to destroy QP %d", ret);
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->rcq) {
        ret = ibv_destroy_cq(comm->rcq);
        if (ret) {
            tl_error(comm->lib, "Couldn't destroy rcq");
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->scq) {
        ret = ibv_destroy_cq(comm->scq);
        if (ret) {
            tl_error(comm->lib, "Couldn't destroy scq");
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->grh_mr) {
        ret = ibv_dereg_mr(comm->grh_mr);
        if (ret) {
            tl_error(comm->lib, "Couldn't destroy grh mr");
            return UCC_ERR_NO_RESOURCE;
        }
    }
    if (comm->grh_buf) {
        ucc_free(comm->grh_buf);
    }

    if (comm->pp) {
        ucc_free(comm->pp);
    }

    if (comm->pp_mr) {
        ret = ibv_dereg_mr(comm->pp_mr);
        if (ret) {
            tl_error(comm->lib, "Couldn't destroy pp mr");
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->pp_buf) {
        ucc_free(comm->pp_buf);
    }

    if (comm->call_rwr) {
        ucc_free(comm->call_rwr);
    }

    if (comm->call_rsgs) {
        ucc_free(comm->call_rsgs);
    }

    if (comm->mcast.ah) {
        ret = ibv_destroy_ah(comm->mcast.ah);
        if (ret) {
            tl_error(comm->lib, "Couldn't destroy ah");
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->mcast_lid) {
        ret = ucc_tl_mlx5_fini_mcast_group(comm->ctx, comm);
        if (ret) {
            tl_error(comm->lib, "Couldn't leave mcast group");
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->ctx->params.print_nack_stats) {
        tl_debug(comm->lib, "comm_id %d, comm_size %d, comm->psn %d, rank %d, "
                    "nacks counter %d, n_mcast_rel %d",
                    comm->comm_id, comm->commsize, comm->psn, comm->rank,
                    comm->nacks_counter, comm->n_mcast_reliable);
    }

    if (comm->p2p_ctx != NULL) {
        ucc_free(comm->p2p_ctx);
    }

    ucc_free(comm);

    return UCC_OK;
}

int ucc_tl_clean_ctx(ucc_tl_mlx5_mcast_coll_context_t *ctx)
{
    tl_debug(ctx->lib, "Cleaning mcast tl  ctx: %p", ctx);

    if (ctx->rcache) {
        ucc_rcache_destroy(ctx->rcache);
    }
    if (ctx->pd) {
        if (ibv_dealloc_pd(ctx->pd)) {
            tl_error(ctx->lib, "ibv_dealloc_pd failed errno %d", errno);
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (rdma_destroy_id(ctx->id)) {
        tl_error(ctx->lib, "rdma_destroy_id failed errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    rdma_destroy_event_channel(ctx->channel);

    ucc_free(ctx);

    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_mcast_join_mcast(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                                 struct sockaddr_in6              *net_addr,
                                                 struct rdma_cm_event            **event,
                                                 int                               is_root)
{
    int  err;
    char buf[40];

    inet_ntop(AF_INET6, net_addr, buf, 40);

    tl_debug(ctx->lib, "joining addr: %s", buf);
    
    if (rdma_join_multicast(ctx->id, (struct sockaddr*)net_addr, NULL)) {
        tl_error(ctx->lib, "rdma_join_multicast failed errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    while (true) {
        if ((err = rdma_get_cm_event(ctx->channel, event)) < 0) {
            if (errno != EINTR) {
                tl_error(ctx->lib, "rdma_get_cm_event failed, errno %d %s", errno, strerror(errno));
                return UCC_ERR_NO_RESOURCE;
            } else {
                continue;
            }
        }
        if ((*event)->event != RDMA_CM_EVENT_MULTICAST_JOIN) {
            tl_error(ctx->lib, "MCAST Failed to join multicast, is_root %d. Unexpected event was "
                    " received: event=%d, str=%s, status=%d",
        is_root, (*event)->event, rdma_event_str((*event)->event), (*event)->status);
            return UCC_ERR_NO_RESOURCE;
        } else {
            break;
        }
    }

    inet_ntop(AF_INET6, (*event)->param.ud.ah_attr.grh.dgid.raw, buf, 40);

    tl_debug(ctx->lib, "is_root %d: joined dgid: %s, mlid 0x%x, sl %d", is_root, buf,
                (*event)->param.ud.ah_attr.dlid, (*event)->param.ud.ah_attr.sl);

    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_mcast_setup_mcast_group(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                                        ucc_tl_mlx5_mcast_coll_comm_t    *comm,
                                                        int                               ctx_id, 
                                                        uint16_t                         *mlid)
{
    int                   status   = 0;
    ucc_status_t          ret      = UCC_OK;
    struct rdma_cm_event  tmp;
    struct rdma_cm_event *event    = NULL;
    struct sockaddr_in6   net_addr = {0,};
    size_t                mgid_s   = sizeof(tmp.param.ud.ah_attr.grh.dgid);
    size_t                mlid_s   = sizeof(tmp.param.ud.ah_attr.dlid);
    void                 *data     = alloca(sizeof(int) + mgid_s + mlid_s);

    if (comm->rank == 0) {
        net_addr.sin6_family   = AF_INET6;
        net_addr.sin6_flowinfo = ctx_id;

        if (UCC_OK == ucc_tl_mlx5_mcast_join_mcast(ctx, &net_addr, &event, true)) {
            memcpy((void*)((ptrdiff_t)data + sizeof(int)),          &event->param.ud.ah_attr.grh.dgid, mgid_s);
            memcpy((void*)((ptrdiff_t)data + sizeof(int) + mgid_s), &event->param.ud.ah_attr.dlid,     mlid_s);
        } else {
            status = -1;
        }

        memcpy(data, &status, sizeof(int));
    }

    ret = ctx->params.bcast(comm->p2p_ctx, data, sizeof(int) + mgid_s + mlid_s, 0);
    if (UCC_OK != ret) {
        goto cleanup;
    }

    status = *(int *) data;
    if (status != 0) {
        ret = UCC_ERR_NO_RESOURCE;
        goto cleanup;
    }

    if (comm->rank != 0) {
        memcpy(&net_addr.sin6_addr, (void*)((ptrdiff_t)data + sizeof(int)), sizeof(struct in6_addr));

        net_addr.sin6_family = AF_INET6;

        if (UCC_OK != ucc_tl_mlx5_mcast_join_mcast(ctx, &net_addr, &event, false)) {
            ret = UCC_ERR_NO_RESOURCE;
            goto cleanup;
        }
    }

    *mlid            = *((uint16_t*)((ptrdiff_t)data + sizeof(int) + mgid_s));
    comm->mcast_addr = net_addr;

    memcpy((void*)&comm->mgid, (void*)((ptrdiff_t)data + sizeof(int)), mgid_s);

cleanup:
    if (event) {
        rdma_ack_cm_event(event);
    }

    return ret;
}

ucc_status_t ucc_tl_setup_mcast(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    uint16_t mlid;

    if (ucc_tl_mlx5_mcast_setup_mcast_group(comm->ctx, comm, comm->comm_id, &mlid) != UCC_OK) {
        return UCC_ERR_NO_RESOURCE;
    }

    comm->mcast_lid = mlid;

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_init_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                        ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct ibv_qp_init_attr qp_init_attr = {0};

    qp_init_attr.qp_type             = IBV_QPT_UD;
    qp_init_attr.send_cq             = comm->scq;
    qp_init_attr.recv_cq             = comm->rcq;
    qp_init_attr.sq_sig_all          = 0;
    qp_init_attr.cap.max_send_wr     = comm->params.sx_depth;
    qp_init_attr.cap.max_recv_wr     = comm->params.rx_depth;
    qp_init_attr.cap.max_inline_data = comm->params.sx_inline;
    qp_init_attr.cap.max_send_sge    = comm->params.sx_sge;
    qp_init_attr.cap.max_recv_sge    = comm->params.rx_sge;

    comm->mcast.qp = ibv_create_qp(ctx->pd, &qp_init_attr);
    if (!comm->mcast.qp) {
        tl_error(ctx->lib, "Failed to create mcast qp, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    comm->max_inline = qp_init_attr.cap.max_inline_data;

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_setup_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                         ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct ibv_port_attr port_attr;
    struct ibv_qp_attr   attr;
    uint16_t             pkey;

    ibv_query_port(ctx->ctx, ctx->ib_port, &port_attr);

    for (ctx->pkey_index = 0; ctx->pkey_index < port_attr.pkey_tbl_len;
         ++ctx->pkey_index) {
        ibv_query_pkey(ctx->ctx, ctx->ib_port, ctx->pkey_index, &pkey);
        if (pkey == DEF_PKEY)
            break;
    }

    if (ctx->pkey_index >= port_attr.pkey_tbl_len) {
        ctx->pkey_index = 0;
        ibv_query_pkey(ctx->ctx, ctx->ib_port, ctx->pkey_index, &pkey);
        if (pkey) {
            tl_debug(ctx->lib, "Cannot find default pkey 0x%04x on port %d, using index 0 pkey:0x%04x",
                       DEF_PKEY, ctx->ib_port, pkey);
        } else {
            tl_error(ctx->lib, "Cannot find valid PKEY");
            return UCC_ERR_NO_RESOURCE;
        }
    }

    attr.qp_state   = IBV_QPS_INIT;
    attr.pkey_index = ctx->pkey_index;
    attr.port_num   = ctx->ib_port;
    attr.qkey       = DEF_QKEY;

    if (ibv_modify_qp(comm->mcast.qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
        tl_error(ctx->lib, "Failed to move mcast qp to INIT, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    if (ibv_attach_mcast(comm->mcast.qp, &comm->mgid, comm->mcast_lid)) {
        tl_error(ctx->lib, "Failed to attach QP to the mcast group, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    /* Ok, now cycle to RTR on everyone */
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(comm->mcast.qp, &attr, IBV_QP_STATE)) {
        tl_error(ctx->lib, "Failed to modify QP to RTR, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn   = DEF_PSN;
    if (ibv_modify_qp(comm->mcast.qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        tl_error(ctx->lib, "Failed to modify QP to RTS, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    /* Create the address handle */
    if (UCC_OK != ucc_tl_mlx5_mcast_create_ah(comm)) {
        tl_error(ctx->lib, "Failed to create adress handle");
        return UCC_ERR_NO_RESOURCE;
    }

    return UCC_OK;
}



