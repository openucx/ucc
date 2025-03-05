/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_helper.h"
#include <glob.h>
#include <net/if.h>
#include <ifaddrs.h>

#define PREF        "/sys/class/net/"
#define SUFF        "/device/resource"
#define MAX_STR_LEN 128

static ucc_status_t ucc_tl_mlx5_get_ipoib_ip(char *ifname, struct sockaddr_storage *addr)
{
    ucc_status_t    status  = UCC_ERR_NO_RESOURCE;
    struct ifaddrs *ifaddr  = NULL;
    struct ifaddrs *ifa     = NULL;
    int             is_ipv4 = 0;
    int             family;
    int             n;
    int             is_up;

    if (getifaddrs(&ifaddr) == -1) {
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
            } else {
                memcpy((struct sockaddr_in6 *) addr,
                       (struct sockaddr_in6 *) ifa->ifa_addr,
                       sizeof(struct sockaddr_in6));
            }

            status = UCC_OK;
            break;
        }
    }

    freeifaddrs(ifaddr);
    return status;
}

static int cmp_files(char *f1, char *f2)
{
    int   answer = 0;
    FILE *fp1;
    FILE *fp2;
    int   ch1;
    int   ch2;

    if ((fp1 = fopen(f1, "r")) == NULL) {
        goto out;
    } else if ((fp2 = fopen(f2, "r")) == NULL) {
        goto close;
    }

    do {
        ch1 = getc(fp1);
        ch2 = getc(fp2);
    } while((ch1 != EOF) && (ch2 != EOF) && (ch1 == ch2));


    if (ch1 == ch2) {
        answer = 1;
    }

    if (fclose(fp2) != 0) {
        return 0;
    }
close:
    if (fclose(fp1) != 0) {
        return 0;
    }
out:
    return answer;
}

static int port_from_file(char *port_file)
{
    int   res = -1;
    char  buf1[MAX_STR_LEN];
    char  buf2[MAX_STR_LEN];
    FILE *fp;
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
    if (fclose(fp) != 0) {
        return -1;
    }
    return res;
}

static ucc_status_t dev2if(char *dev_name, char *port, struct sockaddr_storage
                           *rdma_src_addr)
{
    ucc_status_t status  = UCC_OK;
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
    if (glob(glob_path, 0, 0, &glob_el)) {
        return UCC_ERR_NO_RESOURCE;
    }
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

                status = ucc_tl_mlx5_get_ipoib_ip(if_name, rdma_src_addr);
                if (UCC_OK == status) {
                    break;
                }
            }
        }
    }

    globfree(&glob_el);
    return status;
}

ucc_status_t ucc_tl_mlx5_probe_ip_over_ib(char* ib_dev, struct
                                          sockaddr_storage *addr)
{
    char                   *ib_name = NULL;
    char                   *port    = NULL;
    char                   *ib      = NULL;
    ucc_status_t            status;
    struct sockaddr_storage rdma_src_addr;

    if (ib_dev == NULL) {
        return UCC_ERR_NO_RESOURCE;
    }

    ib = strdup(ib_dev);
    if (!ib) {
        return UCC_ERR_NO_MEMORY;
    }

    ucc_string_split(ib, ":", 2, &ib_name, &port);
    status = dev2if(ib_name, port, &rdma_src_addr);

    if (UCC_OK == status) {
        *addr = rdma_src_addr;
    }
    ucc_free(ib);

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_join_mcast_post(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                               struct sockaddr_in6              *net_addr,
                                               struct mcast_group               *group,
                                               int                               is_root)
{
    char        buf[INET6_ADDRSTRLEN];
    const char *dst;

    dst = inet_ntop(AF_INET6, net_addr, buf, INET6_ADDRSTRLEN);
    if (NULL == dst) {
        tl_warn(ctx->lib, "inet_ntop failed");
        return UCC_ERR_NO_RESOURCE;
    }

    tl_debug(ctx->lib, "joining addr: %s is_root %d group %p", buf, is_root, group);

    if (rdma_join_multicast(ctx->id, (struct sockaddr*)net_addr, (void *)group)) {
        tl_warn(ctx->lib, "rdma_join_multicast failed errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_join_mcast_get_event(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                                    struct rdma_cm_event            **event)
{
    char        buf[INET6_ADDRSTRLEN];
    const char *dst;

    if (rdma_get_cm_event(ctx->channel, event) < 0) {
        if (EINTR != errno) {
            tl_warn(ctx->lib, "rdma_get_cm_event failed, errno %d %s",
                    errno, strerror(errno));
            return UCC_ERR_NO_RESOURCE;
        } else {
            /* need to retry again */
            return UCC_INPROGRESS;
        }
    }

    if (RDMA_CM_EVENT_MULTICAST_JOIN != (*event)->event) {
        tl_warn(ctx->lib, "failed to join multicast, unexpected event was"
                " received: event=%d, str=%s, status=%d",
                 (*event)->event, rdma_event_str((*event)->event),
                 (*event)->status);
        if (rdma_ack_cm_event(*event) < 0) {
            tl_warn(ctx->lib, "rdma_ack_cm_event failed");
        }
        return UCC_ERR_NO_RESOURCE;
    }

    dst = inet_ntop(AF_INET6, (*event)->param.ud.ah_attr.grh.dgid.raw, buf, INET6_ADDRSTRLEN);
    if (NULL == dst) {
        tl_warn(ctx->lib, "inet_ntop failed");
        return UCC_ERR_NO_RESOURCE;
    }

    tl_debug(ctx->lib, "joined dgid: %s, mlid 0x%x, sl %d", buf,
             (*event)->param.ud.ah_attr.dlid, (*event)->param.ud.ah_attr.sl);

    return UCC_OK;

}

ucc_status_t ucc_tl_mlx5_mcast_init_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                        ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    int                      max_inline   = INT_MAX;
    struct ibv_qp_init_attr  qp_init_attr = {0};
    int                      i;
    int                      j;

    qp_init_attr.qp_type             = IBV_QPT_UD;
    qp_init_attr.send_cq             = comm->mcast.scq;   //cq can be shared between multiple QPs
    qp_init_attr.recv_cq             = comm->mcast.rcq;
    qp_init_attr.sq_sig_all          = 0;
    qp_init_attr.cap.max_send_wr     = comm->params.sx_depth;
    qp_init_attr.cap.max_recv_wr     = comm->params.rx_depth;
    qp_init_attr.cap.max_inline_data = comm->params.sx_inline;
    qp_init_attr.cap.max_send_sge    = comm->params.sx_sge;
    qp_init_attr.cap.max_recv_sge    = comm->params.rx_sge;

    for (i = 0; i < comm->mcast_group_count; i++) {
        comm->mcast.groups[i].qp = ibv_create_qp(ctx->pd, &qp_init_attr);
        if (!comm->mcast.groups[i].qp) {
            tl_error(ctx->lib, "Failed to create mcast UD qp index %d, errno %d", i, errno);
            goto error;
        }
        if (qp_init_attr.cap.max_inline_data < max_inline) {
            max_inline = qp_init_attr.cap.max_inline_data;
        }
    }

    if (comm->cuda_mem_enabled) {
        /* max inline send otherwise it segfault during ibv send */
        comm->max_inline = 0;
    } else {
        comm->max_inline = max_inline;
    }

    return UCC_OK;

error:
    for (j = 0; j < i; j++) {
        ibv_destroy_qp(comm->mcast.groups[j].qp);
        comm->mcast.groups[j].qp = NULL;
    }
    return UCC_ERR_NO_RESOURCE;
}

static ucc_status_t ucc_tl_mlx5_mcast_create_ah(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    int i, j, ret;
    struct ibv_ah_attr ah_attr = {
        .is_global     = 1,
        .grh           = {.sgid_index = 0},
        .sl            = DEF_SL,
        .src_path_bits = DEF_SRC_PATH_BITS,
        .port_num      = comm->ctx->ib_port
    };

    for (i = 0; i < comm->mcast_group_count; i ++) {
        ah_attr.dlid  = comm->mcast.groups[i].lid;
        memcpy(ah_attr.grh.dgid.raw, &comm->mcast.groups[i].mgid, sizeof(ah_attr.grh.dgid.raw));

        comm->mcast.groups[i].ah = ibv_create_ah(comm->ctx->pd, &ah_attr);
        if (!comm->mcast.groups[i].ah) {
            tl_error(comm->lib, "failed to create AH index %d", i);
            goto error;
        }
    }

    return UCC_OK;

error:
    for (j = 0; j < i; j++) {
        ret = ibv_destroy_ah(comm->mcast.groups[j].ah);
        if (ret) {
            tl_error(comm->lib, "couldn't destroy ah");
            return UCC_ERR_NO_RESOURCE;
        }
        comm->mcast.groups[j].ah = NULL;
    }
    return UCC_ERR_NO_RESOURCE;
}

ucc_status_t ucc_tl_mlx5_mcast_setup_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                         ucc_tl_mlx5_mcast_coll_comm_t    *comm)
{
    struct ibv_port_attr port_attr;
    struct ibv_qp_attr   attr;
    uint16_t             pkey;
    int                  i;

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
        if (!pkey) {
            tl_warn(ctx->lib, "cannot find valid PKEY");
            return UCC_ERR_NO_RESOURCE;
        }

        tl_debug(ctx->lib, "cannot find default pkey 0x%04x on port %d, using "
                 "index 0 pkey:0x%04x", DEF_PKEY, ctx->ib_port, pkey);
    }

    for (i = 0; i < comm->mcast_group_count; i++) {
        attr.qp_state   = IBV_QPS_INIT;
        attr.pkey_index = ctx->pkey_index;
        attr.port_num   = ctx->ib_port;
        attr.qkey       = DEF_QKEY;

        if (ibv_modify_qp(comm->mcast.groups[i].qp, &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
            tl_error(ctx->lib, "failed to move mcast qp to INIT, errno %d", errno);
            goto error;
        }

        if (ibv_attach_mcast(comm->mcast.groups[i].qp, &comm->mcast.groups[i].mgid,
                             comm->mcast.groups[i].lid)) {
            tl_error(ctx->lib, "failed to attach QP to the mcast group with mcast_lid %d , errno %d",
                     errno, comm->mcast.groups[i].lid);
            goto error;
        }

        attr.qp_state = IBV_QPS_RTR;
        if (ibv_modify_qp(comm->mcast.groups[i].qp, &attr, IBV_QP_STATE)) {
            tl_error(ctx->lib, "failed to modify QP to RTR, errno %d", errno);
            goto error;
        }

        attr.qp_state = IBV_QPS_RTS;
        attr.sq_psn   = DEF_PSN;
        if (ibv_modify_qp(comm->mcast.groups[i].qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
            tl_error(ctx->lib, "failed to modify QP to RTS, errno %d", errno);
            goto error;
        }
    }

    /* create the address handle */
    if (UCC_OK != ucc_tl_mlx5_mcast_create_ah(comm)) {
        tl_warn(ctx->lib, "failed to create adress handle");
        goto error;
    }

    return UCC_OK;

error:
    for (i=0; i < comm->mcast_group_count; i++) {
        ibv_destroy_qp(comm->mcast.groups[i].qp);
        comm->mcast.groups[i].qp = NULL;
    }
    return UCC_ERR_NO_RESOURCE;
}

ucc_status_t ucc_tl_mlx5_mcast_create_rc_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                             ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    int                      i = 0, j = 0;
    struct ibv_srq_init_attr srq_init_attr;
    struct ibv_qp_init_attr  qp_init_attr;

    /* create srq for this RC connection */
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));
    srq_init_attr.attr.max_wr  = comm->params.rx_depth;
    srq_init_attr.attr.max_sge = 2;

    comm->mcast.srq = ibv_create_srq(ctx->pd, &srq_init_attr);
    if (!comm->mcast.srq) {
        tl_error(ctx->lib, "ibv_create_srq() failed");
        return UCC_ERR_NO_RESOURCE;
    }

    comm->mcast.rc_qp = ucc_calloc(1, comm->commsize * sizeof(struct ibv_qp *), "ibv_qp* list");
    if (!comm->mcast.rc_qp) {
        tl_error(ctx->lib, "failed to allocate memory for ibv_qp*");
        goto failed;
    }

    /* create RC qp */
    for (i = 0; i < comm->commsize; i++) {
        memset(&qp_init_attr, 0, sizeof(qp_init_attr));

        qp_init_attr.srq                 = comm->mcast.srq;
        qp_init_attr.qp_type             = IBV_QPT_RC;
        qp_init_attr.send_cq             = comm->mcast.scq;
        qp_init_attr.recv_cq             = comm->mcast.rcq;
        qp_init_attr.sq_sig_all          = 0;
        qp_init_attr.cap.max_send_wr     = comm->params.sx_depth;
        qp_init_attr.cap.max_recv_wr     = 0; // has srq
        qp_init_attr.cap.max_inline_data = 0;
        qp_init_attr.cap.max_send_sge    = comm->params.sx_sge;
        qp_init_attr.cap.max_recv_sge    = comm->params.rx_sge;

        comm->mcast.rc_qp[i] = ibv_create_qp(ctx->pd, &qp_init_attr);
        if (!comm->mcast.rc_qp[i]) {
            tl_error(ctx->lib, "Failed to create mcast RC qp index %d, errno %d", i, errno);
            goto failed;
        }
    }

    return UCC_OK;

failed:
    for (j=0; j<i; j++) {
        if (ibv_destroy_qp(comm->mcast.rc_qp[j])) {
            tl_error(comm->lib, "ibv_destroy_qp failed");
            return UCC_ERR_NO_RESOURCE;
        }
    }
    
    if (ibv_destroy_srq(comm->mcast.srq)) {
        tl_error(comm->lib, "ibv_destroy_srq failed");
        return UCC_ERR_NO_RESOURCE;
    }

    ucc_free(comm->mcast.rc_qp);
    
    return UCC_ERR_NO_RESOURCE;
}

ucc_status_t ucc_tl_mlx5_mcast_modify_rc_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                             ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_rank_t         my_rank = comm->rank;
    struct ibv_qp_attr attr;
    int                i;

    for (i = 0; i < comm->commsize; i++) {
        memset(&attr, 0, sizeof(attr));
         
        attr.qp_state        = IBV_QPS_INIT;
        attr.pkey_index      = 0;
        attr.port_num        = ctx->ib_port;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ  |
                               IBV_ACCESS_REMOTE_ATOMIC;

        if (ibv_modify_qp(comm->mcast.rc_qp[i], &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS)) {
            tl_error(ctx->lib, "Failed to move rc qp to INIT, errno %d", errno);
            return UCC_ERR_NO_RESOURCE;
        }

        memset(&attr, 0, sizeof(attr));

        attr.qp_state              = IBV_QPS_RTR;
        attr.path_mtu              = IBV_MTU_4096;
        attr.dest_qp_num           = comm->one_sided.info[i].rc_qp_num[my_rank];
        attr.rq_psn                = DEF_PSN;
        attr.max_dest_rd_atomic	   = 16;
        attr.min_rnr_timer         = 12;
        attr.ah_attr.is_global     = 0;
        attr.ah_attr.dlid          = comm->mcast.rc_lid[i];
        attr.ah_attr.dlid          = comm->one_sided.info[i].port_lid;
        attr.ah_attr.sl            = DEF_SL;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num      = ctx->ib_port;

        tl_debug(comm->lib, "Connecting to rc qp to rank %d with lid %d qp_num %d port_num %d",
                i, attr.ah_attr.dlid, attr.dest_qp_num, attr.ah_attr.port_num);

        if (ibv_modify_qp(comm->mcast.rc_qp[i], &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN
                          | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
            tl_error(ctx->lib, "Failed to modify rc QP index %d to RTR, errno %d", i, errno);
            return UCC_ERR_NO_RESOURCE;
        }

        memset(&attr, 0, sizeof(attr));

        attr.qp_state      = IBV_QPS_RTS;
        attr.sq_psn        = DEF_PSN;
        attr.timeout       = 14;
        attr.retry_cnt     = 7;
        attr.rnr_retry     = 7; /* infinite */
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(comm->mcast.rc_qp[i], &attr,
                          IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                          IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC)) {
            tl_error(ctx->lib, "Failed to modify rc QP index %i to RTS, errno %d", i, errno);
            return UCC_ERR_NO_RESOURCE;
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_leave_mcast_groups(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                            ucc_tl_mlx5_mcast_coll_comm_t    *comm)
{
    ucc_status_t status = UCC_OK;
    char         buf[INET6_ADDRSTRLEN];
    const char  *dst;
    int          i;

    for (i = 0; i < comm->mcast_group_count; i++) {
        if (comm->mcast.groups[i].mcast_addr.sin6_flowinfo != 0) {
            dst = inet_ntop(AF_INET6, &comm->mcast.groups[i].mcast_addr, buf, INET6_ADDRSTRLEN);
            if (NULL == dst) {
                tl_error(comm->lib, "inet_ntop failed for group %d during mcast leave group", i);
                status = UCC_ERR_NO_RESOURCE;
                continue;
            }

            tl_debug(ctx->lib, "mcast leave: ctx %p, comm %p, dgid: %s group %d", ctx, comm, buf, i);

            if (rdma_leave_multicast(ctx->id, (struct sockaddr*)&comm->mcast.groups[i].mcast_addr)) {
                tl_error(comm->lib, "mcast rmda_leave_multicast failed for group %d", i);
                status = UCC_ERR_NO_RESOURCE;
            }
        }
    }

    return status;
}

ucc_status_t ucc_tl_mlx5_clean_mcast_comm(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_tl_mlx5_mcast_context_t *mcast_ctx = ucc_container_of(comm->ctx, ucc_tl_mlx5_mcast_context_t, mcast_context);
    ucc_tl_mlx5_context_t       *mlx5_ctx  = ucc_container_of(mcast_ctx, ucc_tl_mlx5_context_t, mcast);
    ucc_context_h                context   = mlx5_ctx->super.super.ucc_context;
    int                          ret, i;
    ucc_status_t                 status;

    tl_debug(comm->lib, "cleaning  mcast comm: %p, id %d", comm, comm->comm_id);

    while (UCC_INPROGRESS == (status = ucc_tl_mlx5_mcast_reliable(comm))) {
        ucc_context_progress(context);
    }

    if (UCC_OK != status) {
        tl_error(comm->lib, "failed to clean mcast team: relibality progress status %d",
                 status);
        return status;
    }

    for (i = 0; i < comm->mcast_group_count; i++) {
        if (comm->mcast.groups[i].qp) {
            ret = ibv_detach_mcast(comm->mcast.groups[i].qp, &(comm->mcast.groups[i].mgid), comm->mcast.groups[i].lid);
            if (ret) {
                tl_error(comm->lib, "couldn't detach QP, ret %d, errno %d", ret, errno);
                return UCC_ERR_NO_RESOURCE;
            }

            ret = ibv_destroy_qp(comm->mcast.groups[i].qp);
            if (ret) {
                tl_error(comm->lib, "failed to destroy QP %d", ret);
                return UCC_ERR_NO_RESOURCE;
            }

            comm->mcast.groups[i].qp = NULL;
        }
        if (comm->mcast.groups[i].ah) {
            ret = ibv_destroy_ah(comm->mcast.groups[i].ah);
            if (ret) {
                tl_error(comm->lib, "couldn't destroy ah");
                return UCC_ERR_NO_RESOURCE;
            }
            comm->mcast.groups[i].ah = NULL;
        }
    }

    status = ucc_tl_mlx5_leave_mcast_groups(comm->ctx, comm);
    if (status) {
        tl_error(comm->lib, "couldn't leave mcast group");
        return status;
    }

    if (comm->mcast.rcq) {
        ret = ibv_destroy_cq(comm->mcast.rcq);
        if (ret) {
            tl_error(comm->lib, "couldn't destroy rcq");
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->mcast.scq) {
        ret = ibv_destroy_cq(comm->mcast.scq);
        if (ret) {
            tl_error(comm->lib, "couldn't destroy scq");
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->grh_mr) {
        ret = ibv_dereg_mr(comm->grh_mr);
        if (ret) {
            tl_error(comm->lib, "couldn't destroy grh mr");
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
            tl_error(comm->lib, "couldn't destroy pp mr");
            return UCC_ERR_NO_RESOURCE;
        }
    }

    if (comm->pp_buf) {
        ucc_mc_free(comm->pp_buf_header);
    }

    if (comm->call_rwr) {
        ucc_free(comm->call_rwr);
    }

    if (comm->call_rsgs) {
        ucc_free(comm->call_rsgs);
    }

    if (comm->ctx->params.print_nack_stats) {
        tl_debug(comm->lib, "comm_id %d, comm_size %d, comm->psn %d, rank %d, "
                 "nacks counter %d, n_mcast_rel %d",
                 comm->comm_id, comm->commsize, comm->psn, comm->rank,
                 comm->bcast_comm.nacks_counter, comm->bcast_comm.n_mcast_reliable);
    }

    if (comm->p2p_ctx != NULL) {
        ucc_free(comm->p2p_ctx);
    }

    ucc_free(comm);

    return UCC_OK;
}

