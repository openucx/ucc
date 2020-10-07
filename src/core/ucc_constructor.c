/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include <ucs/config/parser.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <link.h>
#include <dlfcn.h>
#include <glob.h>
#include <unistd.h>
#include "utils/ucc_log.h"
#include "ucc_lib.h"
#include "utils/ucc_component.h"
#include "ccm/ucc_ccm.h"

struct ucc_static_lib_data ucc_lib_data;

static int
callback(struct dl_phdr_info *info, size_t size, void *data)
{
    char *str;
    if (NULL != (str = strstr(info->dlpi_name, "libucc.so"))) {
        int pos = (int)(str - info->dlpi_name);
        free(ucc_lib_global_config.ccm_path);
        ucc_lib_global_config.ccm_path = (char*)malloc(pos+8);
        strncpy(ucc_lib_global_config.ccm_path, info->dlpi_name, pos);
        ucc_lib_global_config.ccm_path[pos] = '\0';
        strcat(ucc_lib_global_config.ccm_path, "ucc");
    }
    return 0;
}

static void get_default_lib_path()
{
    dl_iterate_phdr(callback, NULL);
}

__attribute__((constructor))
static void ucc_constructor(void)
{
    ucs_status_t status;
    ucc_lib_data.ccm_ifaces = NULL;
    ucc_lib_data.n_ccms_loaded = 0;

    status = ucs_config_parser_fill_opts(&ucc_lib_global_config,
                                         ucc_lib_global_config_table,
                                         "UCC_", NULL, 1);
    if (strlen(ucc_lib_global_config.ccm_path) == 0) {
        get_default_lib_path();
    }
    ucc_info("UCC components path: %s", ucc_lib_global_config.ccm_path);

    if (!ucc_lib_global_config.ccm_path) {
        ucc_error("Failed to get ucc library path. set UCC_CCM_PATH.\n");
        return;
    }
    ucc_components_load("ccm",
                        (ucc_component_iface_t***)&ucc_lib_data.ccm_ifaces,
                        &ucc_lib_data.n_ccms_loaded);
    if (ucc_lib_data.n_ccms_loaded == 0) {
        ucc_error("UCC init: couldn't find any ucc_ccm_<name>.so plugins"
                  " in %s\n", ucc_lib_global_config.ccm_path);
        return;
    }
}

