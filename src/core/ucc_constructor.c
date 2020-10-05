/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include <ucs/config/parser.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <link.h>
#include <dlfcn.h>
#include <glob.h>
#include <unistd.h>
#include "utils/ucc_log.h"
#include "ucc_lib.h"
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

static ucc_status_t ucc_component_load_one(const char *so_path,
                                           ucc_ccm_iface_t **ccm_iface)
{
    char ccm_lib_struct[128];
    void *handle;
    ucc_ccm_iface_t *iface;

    int pos = (int)(strstr(so_path, "ucc_ccm_") - so_path);
    if (pos < 0) {
        return UCC_ERR_NO_MESSAGE;
    }
    strncpy(ccm_lib_struct, so_path+pos, strlen(so_path) - pos - 3);
    ccm_lib_struct[strlen(so_path) - pos - 3] = '\0';
    handle = dlopen(so_path, RTLD_LAZY);
    ucc_debug("Loading library %s\n", so_path);
    if (!handle) {
        ucc_error("Failed to load UCC component library: %s\n. "
                  "Check UCC_CCM_PATH or LD_LIBRARY_PATH\n", so_path);
        goto error;
    }
    iface = (ucc_ccm_iface_t*)dlsym(handle, ccm_lib_struct);
    if (!iface) {
        ucc_error("Failed to get ccm iface %s from %s object\n",
                  ccm_lib_struct, so_path);
        goto iface_error;
    }
    iface->dl_handle = handle;
    (*ccm_iface) = iface;
    return UCC_OK;

iface_error:
    dlclose(handle);
error:
    *ccm_iface = NULL;
    return UCC_ERR_NO_MESSAGE;
}

static void ucc_load_components(void)
{
    const char   *ccm_pattern = "/ucc_ccm_*.so";
    glob_t       globbuf;
    int          i, ccm_array_size;
    char         *pattern;
    ucc_status_t status;

    pattern = (char*)malloc(strlen(ucc_lib_global_config.ccm_path) +
                            strlen(ccm_pattern) + 1);
    strcpy(pattern, ucc_lib_global_config.ccm_path);
    strcat(pattern, ccm_pattern);
    glob(pattern, 0, NULL, &globbuf);
    free(pattern);
    ccm_array_size = 0;
    for(i=0; i<globbuf.gl_pathc; i++) {
        if (ucc_lib_data.n_ccms_loaded == ccm_array_size) {
            ccm_array_size += 8;
            ucc_lib_data.ccm_ifaces = (ucc_ccm_iface_t**)
                realloc(ucc_lib_data.ccm_ifaces, ccm_array_size*sizeof(*ucc_lib_data.ccm_ifaces));
        }
        status = ucc_component_load_one(globbuf.gl_pathv[i],
                                        &ucc_lib_data.ccm_ifaces[ucc_lib_data.n_ccms_loaded]);
        if (status != UCC_OK) {
            continue;
        }
        ucc_lib_data.n_ccms_loaded++;
    }

    if (globbuf.gl_pathc > 0) {
        globfree(&globbuf);
    }
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

    ucc_load_components();
    if (ucc_lib_data.n_ccms_loaded == 0) {
        ucc_error("UCC init: couldn't find any ucc_ccm_<name>.so plugins"
                  " in %s\n", ucc_lib_global_config.ccm_path);
        return;
    }
}

