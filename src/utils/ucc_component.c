/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <link.h>
#include <dlfcn.h>
#include <glob.h>
#include <unistd.h>
#include <string.h>
#include "ucc_component.h"
#include "ucc_log.h"

static ucc_status_t ucc_component_load_one(const char *so_path, const char *framework_name,
                                           ucc_component_iface_t **c_iface)
{
    char iface_struct[128];
    char pattern[128];
    void *handle;
    ucc_component_iface_t *iface;
    int pos;

    sprintf(pattern, "ucc_%s_", framework_name);
    pos = (int)(strstr(so_path, pattern) - so_path);
    if (pos < 0) {
        return UCC_ERR_NO_MESSAGE;
    }
    strncpy(iface_struct, so_path+pos, strlen(so_path) - pos - 3);
    iface_struct[strlen(so_path) - pos - 3] = '\0';
    handle = dlopen(so_path, RTLD_LAZY);
    ucc_debug("Loading library %s\n", so_path);
    if (!handle) {
        ucc_error("Failed to load UCC component library: %s. "
                  "Check UCC_CCM_PATH or LD_LIBRARY_PATH\n", so_path);
        goto error;
    }
    iface = (ucc_component_iface_t*)dlsym(handle, iface_struct);
    if (!iface) {
        ucc_error("Failed to get iface %s from %s object\n",
                  iface_struct, so_path);
        goto iface_error;
    }
    iface->dl_handle = handle;
    (*c_iface) = iface;
    return UCC_OK;

iface_error:
    dlclose(handle);
error:
    *c_iface = NULL;
    return UCC_ERR_NO_MESSAGE;
}

ucc_status_t ucc_components_load(const char* framework_name,
                                 ucc_component_iface_t ***components, int *n_components)
{
    glob_t                globbuf;
    int                   i, n_loaded, ifaces_array_size;
    char                  *full_pattern;
    ucc_status_t          status;
    ucc_component_iface_t **ifaces = NULL;

    full_pattern = (char*)malloc(strlen(ucc_lib_global_config.ccm_path) +
                                 strlen(framework_name) + 16);
    sprintf(full_pattern, "%s/ucc_%s_*.so", ucc_lib_global_config.ccm_path,
            framework_name);
    glob(full_pattern, 0, NULL, &globbuf);
    free(full_pattern);
    n_loaded          = 0;
    ifaces_array_size = 0;
    for(i=0; i<globbuf.gl_pathc; i++) {
        if (n_loaded == ifaces_array_size) {
            ifaces_array_size += 8;
            ifaces = (ucc_component_iface_t**)
                realloc(ifaces, ifaces_array_size*sizeof(ucc_component_iface_t*));
        }
        status = ucc_component_load_one(globbuf.gl_pathv[i], framework_name,
                                        &ifaces[n_loaded]);
        if (status != UCC_OK) {
            continue;
        }
        n_loaded++;
    }

    if (globbuf.gl_pathc > 0) {
        globfree(&globbuf);
    }
    if (n_loaded) {
        *components   = ifaces;
        *n_components = n_loaded;
    } else {
        *components = NULL;
    }
    return UCC_OK;
}
