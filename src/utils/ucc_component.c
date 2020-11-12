/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "config.h"
#include "ucc_malloc.h"
#include "ucc_component.h"
#include "ucc_log.h"
#include "core/ucc_global_opts.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <link.h>
#include <dlfcn.h>
#include <glob.h>
#include <unistd.h>
#include <assert.h>

#define IFACE_NAME_LEN (UCC_MAX_FRAMEWORK_NAME_LEN + UCC_MAX_COMPONENT_NAME_LEN)

static ucc_status_t ucc_component_load_one(const char *so_path,
                                           const char *framework_pattern,
                                           ucc_component_iface_t **c_iface)
{
    char                  *error, iface_struct[IFACE_NAME_LEN];
    void                  *handle;
    ucc_component_iface_t *iface;
    size_t                 basename_start, iface_struct_name_len;

    basename_start =
        ((ptrdiff_t)strstr(so_path, framework_pattern) - (ptrdiff_t)so_path);
    if (basename_start < 0) {
        return UCC_ERR_NO_MESSAGE;
    }
    /* The name of the iface stract matches the basename of .so component
       object. basename_start - the starting position of the component name
       in the full .so path. The name_len is also decreased by 3 to remove 
       ".so" extension from the name;
     */
    iface_struct_name_len = strlen(so_path) - basename_start - 3;
    ucc_strncpy_safe(iface_struct, so_path + basename_start,
                     iface_struct_name_len + 1);

    handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        ucc_debug("failed to load UCC component library: %s", so_path);
        goto error;
    }
    iface = (ucc_component_iface_t *)dlsym(handle, iface_struct);
    if ((error = dlerror()) != NULL) {
        ucc_error("failed to get iface %s from %s object", iface_struct,
                  so_path);
        goto iface_error;
    }
    if (!iface) {
        ucc_error("iface %s is NULL in %s object", iface_struct, so_path);
        goto iface_error;
    }
    iface->dl_handle = handle;
    (*c_iface)       = iface;
    return UCC_OK;

iface_error:
    dlclose(handle);
error:
    *c_iface = NULL;
    return UCC_ERR_NO_MESSAGE;
}

ucc_status_t ucc_components_load(const char *framework_name,
                                 ucc_component_framework_t *framework)
{
    glob_t globbuf;
    int    i, n_loaded;
    char  *full_pattern, framework_pattern[UCC_MAX_FRAMEWORK_NAME_LEN + 16];
    ucc_status_t            status;
    size_t                  pattern_size;
    ucc_component_iface_t **ifaces = NULL;

    framework->n_components = 0;
    framework->components   = NULL;

    if (strlen(framework_name) == 0 ||
        strlen(framework_name) > UCC_MAX_FRAMEWORK_NAME_LEN) {
        ucc_error("unsupported framework_name length: %s, len %zd",
                  framework_name, strlen(framework_name));
        return UCC_ERR_INVALID_PARAM;
    }
    ucc_snprintf_safe(framework_pattern, sizeof(framework_pattern), "ucc_%s_",
                      framework_name);

    pattern_size =
        strlen(ucc_global_config.component_path) + strlen(framework_name) + 16;
    full_pattern = (char *)ucc_malloc(pattern_size, "full_pattern");
    if (!full_pattern) {
        ucc_error("failed to allocate %zd bytes for full_pattern",
                  pattern_size);
        return UCC_ERR_NO_MEMORY;
    }
    ucc_snprintf_safe(full_pattern, pattern_size, "%s/ucc_%s_*.so",
                      ucc_global_config.component_path, framework_name);
    glob(full_pattern, 0, NULL, &globbuf);
    free(full_pattern);
    n_loaded          = 0;

    dlerror(); /* Clear any existing error */
    ifaces = (ucc_component_iface_t **)ucc_malloc(
        globbuf.gl_pathc * sizeof(ucc_component_iface_t *), "ifaces");
    if (!ifaces) {
        ucc_error("failed to allocate %zd bytes for ifaces",
                  globbuf.gl_pathc * sizeof(ucc_component_iface_t *));
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < globbuf.gl_pathc; i++) {
        status = ucc_component_load_one(globbuf.gl_pathv[i], framework_pattern,
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
        assert(n_loaded <= globbuf.gl_pathc);
        ifaces = ucc_realloc(ifaces, n_loaded * sizeof(ucc_component_iface_t *),
                             "ifaces");
        framework->components   = ifaces;
        framework->n_components = n_loaded;
        return UCC_OK;
    } else {
        if (ifaces) {
            free(ifaces);
        }
        return UCC_ERR_NOT_FOUND;
    }
}

ucc_component_iface_t* ucc_get_component(ucc_component_framework_t *framework,
                                         const char *component_name)
{
    int i;
    for (i = 0; i < framework->n_components; i++) {
        if (0 == strcmp(framework->components[i]->name, component_name)) {
            return framework->components[i];
        }
    }
    return NULL;
}
