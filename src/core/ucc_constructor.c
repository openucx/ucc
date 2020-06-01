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

ucc_lib_t ucc_static_lib;

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

static ucc_status_t ucc_team_lib_init(const char *so_path,
                                      ucc_team_lib_t **team_lib)
{
    char team_lib_struct[128];
    void *handle;
    ucc_team_lib_t *lib;

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
        *team_lib = NULL;
        return UCC_ERR_NO_MESSAGE;
    }
    lib = (ucc_team_lib_t*)dlsym(handle, team_lib_struct);
#if 0
    lib->dl_handle = handle;
    if (lib->team_lib_open != NULL) {
        ucc_team_lib_config_t *tl_config = malloc(lib->team_lib_config.size);
        ucs_config_parser_fill_opts(tl_config, lib->team_lib_config.table,
                                    "UCC_", lib->team_lib_config.prefix, 0);
        lib->team_lib_open(lib, tl_config);
        ucs_config_parser_release_opts(tl_config, lib->team_lib_config.table);
        free(tl_config);
    }
#endif
    (*team_lib) = lib;
    return UCC_OK;
}

static void load_team_lib_plugins(ucc_lib_t *lib)
{
    const char    *tl_pattern = "/ucc_team_lib_*.so";
    glob_t        globbuf;
    int           i;
    char          *pattern;
    ucc_status_t status;

    pattern = (char*)malloc(strlen(ucc_lib_global_config.team_lib_path) +
                            strlen(tl_pattern) + 1);
    strcpy(pattern, ucc_lib_global_config.team_lib_path);
    strcat(pattern, tl_pattern);
    glob(pattern, 0, NULL, &globbuf);
    free(pattern);
    for(i=0; i<globbuf.gl_pathc; i++) {
        if (lib->n_libs_opened == lib->libs_array_size) {
            lib->libs_array_size += 8;
            lib->libs = (ucc_team_lib_t**)realloc(lib->libs,
                                                  lib->libs_array_size*sizeof(*lib->libs));
        }
        status = ucc_team_lib_init(globbuf.gl_pathv[i],
                                   &lib->libs[lib->n_libs_opened]);
        if (status != UCC_OK) {
            continue;
        }
        lib->n_libs_opened++;
    }

    if (globbuf.gl_pathc > 0) {
        globfree(&globbuf);
    }
}

__attribute__((constructor))
static void ucc_constructor(void)
{
    ucs_status_t status;

    ucc_lib_t *lib = &ucc_static_lib;
    lib->libs = NULL;
    lib->n_libs_opened = 0;
    lib->libs_array_size = 0;

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

    load_team_lib_plugins(lib);
    if (lib->n_libs_opened == 0) {
        ucc_error("UCC init: couldn't find any ucc_team_lib_<name>.so plugins"
                  " in %s\n", ucc_lib_global_config.team_lib_path);
        return;
    }
}

