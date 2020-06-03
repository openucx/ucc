/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#define _GNU_SOURCE

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
#include "ucc_global_opts.h"
#include "ucc_lib.h"
#include "team_lib/ucc_tl.h"

struct ucc_static_lib_data ucc_lib_data;

static int
callback(struct dl_phdr_info *info, size_t size, void *data)
{
    char *str;
    if (NULL != (str = strstr(info->dlpi_name, "libucc.so"))) {
        int pos = (int)(str - info->dlpi_name);
        free(ucc_lib_global_config.team_lib_path);
        ucc_lib_global_config.team_lib_path = (char*)malloc(pos+8);
        strncpy(ucc_lib_global_config.team_lib_path, info->dlpi_name, pos);
        ucc_lib_global_config.team_lib_path[pos] = '\0';
        strcat(ucc_lib_global_config.team_lib_path, "ucc");
    }
    return 0;
}

static void get_default_lib_path()
{
    dl_iterate_phdr(callback, NULL);
}

static ucc_status_t ucc_tl_load(const char *so_path,
                                ucc_tl_iface_t **tl_iface)
{
    char team_lib_struct[128];
    void *handle;
    ucc_tl_iface_t *iface;

    int pos = (int)(strstr(so_path, "ucc_team_lib_") - so_path);
    if (pos < 0) {
        return UCC_ERR_NO_MESSAGE;
    }
    strncpy(team_lib_struct, so_path+pos, strlen(so_path) - pos - 3);
    team_lib_struct[strlen(so_path) - pos - 3] = '\0';
    handle = dlopen(so_path, RTLD_LAZY);
    ucc_debug("Loading library %s\n", so_path);
    if (!handle) {
        ucc_error("Failed to load UCC Team library: %s\n. "
                  "Check UCC_TEAM_LIB_PATH or LD_LIBRARY_PATH\n", so_path);
        *tl_iface = NULL;
        return UCC_ERR_NO_MESSAGE;
    }
    iface = (ucc_tl_iface_t*)dlsym(handle, team_lib_struct);
    iface->dl_handle = handle;
    (*tl_iface) = iface;
    return UCC_OK;
}

static void load_team_lib_plugins(void)
{
    const char    *tl_pattern = "/ucc_team_lib_*.so";
    glob_t        globbuf;
    int           i, tls_array_size;
    char          *pattern;
    ucc_status_t status;

    pattern = (char*)malloc(strlen(ucc_lib_global_config.team_lib_path) +
                            strlen(tl_pattern) + 1);
    strcpy(pattern, ucc_lib_global_config.team_lib_path);
    strcat(pattern, tl_pattern);
    glob(pattern, 0, NULL, &globbuf);
    free(pattern);
    tls_array_size = 0;
    for(i=0; i<globbuf.gl_pathc; i++) {
        if (ucc_lib_data.n_tls_loaded == tls_array_size) {
            tls_array_size += 8;
            ucc_lib_data.tl_ifaces = (ucc_tl_iface_t**)
                realloc(ucc_lib_data.tl_ifaces, tls_array_size*sizeof(*ucc_lib_data.tl_ifaces));
        }
        status = ucc_tl_load(globbuf.gl_pathv[i],
                             &ucc_lib_data.tl_ifaces[ucc_lib_data.n_tls_loaded]);
        if (status != UCC_OK) {
            continue;
        }
        ucc_lib_data.n_tls_loaded++;
    }

    if (globbuf.gl_pathc > 0) {
        globfree(&globbuf);
    }
}

__attribute__((constructor))
static void ucc_constructor(void)
{
    ucs_status_t status;
    ucc_lib_data.tl_ifaces = NULL;
    ucc_lib_data.n_tls_loaded = 0;

    status = ucs_config_parser_fill_opts(&ucc_lib_global_config,
                                         ucc_lib_global_config_table,
                                         "UCC_", NULL, 1);
    if (strlen(ucc_lib_global_config.team_lib_path) == 0) {
        get_default_lib_path();
    }
    ucc_info("UCC team lib path: %s", ucc_lib_global_config.team_lib_path);

    if (!ucc_lib_global_config.team_lib_path) {
        ucc_error("Failed to get ucc library path. set UCC_TEAM_LIB_PATH.\n");
        return;
    }

    load_team_lib_plugins();
    if (ucc_lib_data.n_tls_loaded == 0) {
        ucc_error("UCC init: couldn't find any ucc_team_lib_<name>.so plugins"
                  " in %s\n", ucc_lib_global_config.team_lib_path);
        return;
    }
}

