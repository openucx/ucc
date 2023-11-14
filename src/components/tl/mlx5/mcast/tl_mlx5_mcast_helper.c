/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    ucc_status_t            status;
    struct sockaddr_storage rdma_src_addr;

    if (NULL == ib_dev) {
        status = UCC_ERR_NO_RESOURCE;
    } else {
        ucc_string_split(ib_dev, ":", 2, &ib_name, &port);
        status = dev2if(ib_name, port, &rdma_src_addr);
    }

    if (UCC_OK == status) {
        *addr = rdma_src_addr;
    }

    return status;
}

