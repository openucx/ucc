#include "config.h"
#include "ucc_sysinfo_ib.h"
#include "core/ucc_global_opts.h"

#include "utils/ucc_malloc.h"
#include "utils/ucc_proc_info.h"
#include "utils/debug/log_def.h"
#include "utils/ucc_string.h"

#include <infiniband/verbs.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ucc_status_t ucc_sysinfo_ib_init(const ucc_sysinfo_params_t *params)
{
    (void)params;
    return UCC_OK;
}

static ucc_status_t ucc_sysinfo_ib_finalize(void)
{
    return UCC_OK;
}

static int ucc_sysinfo_ib_net_device_matches(const char *entry,
                                             const ucc_nic_info_t *nic)
{
    const char   *port_str;
    size_t        name_len;
    unsigned long port;

    if (!entry || !nic || nic->name[0] == '\0') {
        return 0;
    }

    port_str = strchr(entry, ':');
    if (!port_str) {
        return (strcmp(entry, nic->name) == 0);
    }

    name_len = (size_t)(port_str - entry);
    if (name_len == 0 || name_len >= sizeof(nic->name)) {
        return 0;
    }

    if ((strncmp(entry, nic->name, name_len) != 0) ||
        (nic->name[name_len] != '\0')) {
        return 0;
    }

    port_str++;
    if (*port_str == '\0' || ucc_str_is_number(port_str) != UCC_OK) {
        return 0;
    }

    port = strtoul(port_str, NULL, 10);
    if (port > 0xff) {
        return 0;
    }

    return nic->port == (uint8_t)port;
}

static ucc_status_t ucc_sysinfo_ib_set_visible_devices(
    void *info, int n_info, const ucc_config_names_array_t *devices,
    uint32_t *visible_devices)
{
    ucc_nic_info_t *nics;
    uint32_t        mask;
    int             i, j;

    if (!visible_devices || (n_info > 0 && !info)) {
        return UCC_ERR_INVALID_PARAM;
    }

    if (!devices) {
        return UCC_OK;
    }

    *visible_devices = 0;
    if (n_info <= 0) {
        return UCC_OK;
    }

    for (i = 0; i < devices->count; i++) {
        if (strcmp(devices->names[i], "all") == 0) {
            for (j = 0; j < n_info && j < (int)(sizeof(mask) * 8); j++) {
                *visible_devices |= (1u << j);
            }
            return UCC_OK;
        }
    }

    if (devices->count == 0) {
        for (j = 0; j < n_info && j < (int)(sizeof(mask) * 8); j++) {
            *visible_devices |= (1u << j);
        }
        return UCC_OK;
    }

    nics = (ucc_nic_info_t *)info;
    mask = 0;
    for (i = 0; i < n_info && i < (int)(sizeof(mask) * 8); i++) {
        for (j = 0; j < devices->count; j++) {
            if (ucc_sysinfo_ib_net_device_matches(devices->names[j],
                                                  &nics[i])) {
                mask |= (1u << i);
                break;
            }
        }
    }

    *visible_devices = mask;
    return UCC_OK;
}

static void ucc_sysinfo_ib_fill_pci_info(const char *dev_name,
                                         ucc_nic_info_t *nic)
{
    char         path[PATH_MAX];
    char         line[256];
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int function;
    FILE        *fp;

    nic->pci.host_id  = ucc_local_proc.host_id;
    nic->pci.domain   = 0;
    nic->pci.bus      = 0;
    nic->pci.device   = 0;
    nic->pci.function = 0;

    if (!dev_name) {
        return;
    }

    snprintf(path, sizeof(path),
             "/sys/class/infiniband/%s/device/uevent", dev_name);
    fp = fopen(path, "r");
    if (!fp) {
        return;
    }

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "PCI_SLOT_NAME=", 14) == 0) {
            if (sscanf(line + 14, "%x:%x:%x.%x",
                       &domain, &bus, &device, &function) == 4) {
                nic->pci.domain   = (uint16_t)domain;
                nic->pci.bus      = (uint8_t)bus;
                nic->pci.device   = (uint8_t)device;
                nic->pci.function = (uint8_t)function;
            }
            break;
        }
    }

    fclose(fp);
}

static ucc_status_t ucc_sysinfo_ib_get_info(void **info, int *n_info)
{
    struct ibv_device    **dev_list;
    struct ibv_context    *ctx;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr   port_attr;
    ucc_nic_info_t        *nic_info;
    ucc_status_t           status;
    const char            *dev_name;
    int                    num_devs;
    int                    total_ports;
    int                    i;
    int                    port;
    int                    idx;

    if (info) {
        *info = NULL;
    }

    if (n_info) {
        *n_info = 0;
    }

    dev_list = ibv_get_device_list(&num_devs);
    if (!dev_list) {
        ucc_debug("ibv_get_device_list failed");
        return UCC_ERR_NO_RESOURCE;
    }

    total_ports = 0;
    for (i = 0; i < num_devs; i++) {
        ctx = ibv_open_device(dev_list[i]);
        if (!ctx) {
            ucc_debug("ibv_open_device failed");
            continue;
        }
        if (ibv_query_device(ctx, &dev_attr)) {
            ucc_debug("ibv_query_device failed");
            ibv_close_device(ctx);
            continue;
        }
        for (port = 1; port <= dev_attr.phys_port_cnt; port++) {
            if (ibv_query_port(ctx, port, &port_attr)) {
                continue;
            }
            if (port_attr.state == IBV_PORT_ACTIVE) {
                total_ports++;
            }
        }
        ibv_close_device(ctx);
    }

    if (total_ports == 0) {
        ibv_free_device_list(dev_list);
        return UCC_OK;
    }

    nic_info = (ucc_nic_info_t *)ucc_malloc(
        total_ports * sizeof(*nic_info), "sysinfo_ib_nic_info");
    if (!nic_info) {
        ucc_error("failed to allocate %d bytes for nic info",
                  total_ports * (int)sizeof(*nic_info));
        ibv_free_device_list(dev_list);
        return UCC_ERR_NO_MEMORY;
    }
    memset(nic_info, 0, total_ports * sizeof(*nic_info));

    idx = 0;
    status = UCC_OK;
    for (i = 0; i < num_devs && idx < total_ports; i++) {
        dev_name = ibv_get_device_name(dev_list[i]);
        ctx = ibv_open_device(dev_list[i]);
        if (!ctx) {
            ucc_debug("ibv_open_device failed");
            continue;
        }
        if (ibv_query_device(ctx, &dev_attr)) {
            ucc_debug("ibv_query_device failed");
            ibv_close_device(ctx);
            continue;
        }
        for (port = 1; port <= dev_attr.phys_port_cnt; port++) {
            if (ibv_query_port(ctx, port, &port_attr)) {
                continue;
            }
            if (port_attr.state != IBV_PORT_ACTIVE) {
                continue;
            }

            ucc_sysinfo_ib_fill_pci_info(dev_name, &nic_info[idx]);
            nic_info[idx].port = (uint8_t)port;
            nic_info[idx].guid = (uint64_t)dev_attr.node_guid;
            if (dev_name) {
                snprintf(nic_info[idx].name, sizeof(nic_info[idx].name),
                         "%s", dev_name);
            }
            idx++;
        }
        ibv_close_device(ctx);
    }

    ibv_free_device_list(dev_list);

    if (idx == 0) {
        ucc_free(nic_info);
        return UCC_OK;
    }

    if (info) {
        *info = nic_info;
    } else {
        ucc_free(nic_info);
    }

    if (n_info) {
        *n_info = idx;
    }

    return status;
}

static ucc_config_field_t ucc_sysinfo_ib_config_table[] = {
    {NULL}
};

ucc_sysinfo_ib_t ucc_sysinfo_ib = {
    .super.super.name      = "ib sysinfo",
    .super.type            = UCC_SYSINFO_TYPE_IB,
    .super.ref_cnt         = 0,
    .super.init            = ucc_sysinfo_ib_init,
    .super.finalize        = ucc_sysinfo_ib_finalize,
    .super.ops.get_info    = ucc_sysinfo_ib_get_info,
    .super.ops.set_visible_devices = ucc_sysinfo_ib_set_visible_devices,
    .super.config_table    =
        {
            .name   = "IB sysinfo",
            .prefix = "SYSINFO_IB_",
            .table  = ucc_sysinfo_ib_config_table,
            .size   = sizeof(ucc_sysinfo_config_t),
        },
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_sysinfo_ib.super.config_table,
                                &ucc_config_global_list);
