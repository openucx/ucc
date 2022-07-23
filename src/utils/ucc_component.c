/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#include "config.h"
#include "ucc_malloc.h"
#include "ucc_component.h"
#include "ucc_log.h"
#include "ucc_math.h"
#include "core/ucc_global_opts.h"
#include "utils/ucc_string.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <link.h>
#include <dlfcn.h>
#include <glob.h>
#include <unistd.h>
#include <assert.h>

#define IFACE_NAME_LEN_MAX                                                     \
    (UCC_MAX_FRAMEWORK_NAME_LEN + UCC_MAX_COMPONENT_NAME_LEN + 32)

static ucc_status_t ucc_component_load_one(const char *so_path,
                                           const char *framework_name,
                                           ucc_component_iface_t **c_iface)
{
    char                   framework_pattern[UCC_MAX_FRAMEWORK_NAME_LEN + 16];
    char                  *error, iface_struct[IFACE_NAME_LEN_MAX];
    void                  *handle;
    ucc_component_iface_t *iface;
    size_t                 basename_start, iface_struct_name_len;

    ucc_snprintf_safe(framework_pattern, sizeof(framework_pattern), "ucc_%s_",
                      framework_name);
    basename_start =
        ((ptrdiff_t)ucc_strstr_last(so_path, framework_pattern) -
         (ptrdiff_t)so_path);
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
        ucc_error("failed to get iface %s from %s object (%s)", iface_struct,
                  so_path, error);
        goto iface_error;
    }
    if (!iface) {
        ucc_error("iface %s is NULL in %s object", iface_struct, so_path);
        goto iface_error;
    }
    iface->dl_handle = handle;
    iface->id        = ucc_str_hash_djb2(iface->name);
    (*c_iface)       = iface;
    return UCC_OK;

iface_error:
    dlclose(handle);
error:
    *c_iface = NULL;
    return UCC_ERR_NO_MESSAGE;
}

#define CHECK_COMPONENT_UNIQ(_framework, _field)                               \
    do {                                                                       \
        ucc_component_iface_t **c = _framework->components;                    \
        int                     i, j;                                          \
        for (i = 0; i < framework->n_components; i++) {                        \
            for (j = i + 1; j < framework->n_components; j++) {                \
                if (c[i]->_field == c[j]->_field) {                            \
                    ucc_error("components %s and %s have the same "            \
                              "default " UCC_PP_MAKE_STRING(_field) " %lu",    \
                              c[i]->name, c[j]->name,                          \
                              (unsigned long)c[i]->_field);                    \
                    return UCC_ERR_INVALID_PARAM;                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

ucc_status_t
ucc_component_check_scores_uniq(ucc_component_framework_t *framework)
{
    CHECK_COMPONENT_UNIQ(framework, score);
    return UCC_OK;
}

static ucc_status_t
ucc_component_check_ids_uniq(ucc_component_framework_t *framework)
{
    CHECK_COMPONENT_UNIQ(framework, id);
    return UCC_OK;
}

ucc_status_t ucc_components_load(const char *framework_name,
                                 ucc_component_framework_t *framework)
{
    glob_t globbuf;
    int    i, n_loaded;
    char  *full_pattern;
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

    pattern_size =
        strlen(ucc_global_config.component_path) + strlen(framework_name) + 16;
    full_pattern = (char *)ucc_malloc(pattern_size, "full_pattern");
    if (!full_pattern) {
        ucc_error("failed to allocate %zd bytes for full_pattern",
                  pattern_size);
        return UCC_ERR_NO_MEMORY;
    }
    ucc_snprintf_safe(full_pattern, pattern_size, "%s/libucc_%s_*.so",
                      ucc_global_config.component_path, framework_name);
    glob(full_pattern, 0, NULL, &globbuf);
    ucc_free(full_pattern);
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
        status = ucc_component_load_one(globbuf.gl_pathv[i], framework_name,
                                        &ifaces[n_loaded]);
        if (status != UCC_OK) {
            continue;
        }
        n_loaded++;
    }

    assert(n_loaded <= globbuf.gl_pathc);
    if (globbuf.gl_pathc > 0) {
        globfree(&globbuf);
    }
    if (!n_loaded) {
        if (ifaces) {
            ucc_free(ifaces);
        }
        return UCC_ERR_NOT_FOUND;
    }

    ifaces = ucc_realloc(ifaces, n_loaded * sizeof(ucc_component_iface_t *),
                         "ifaces");
    framework->components   = ifaces;
    framework->n_components = n_loaded;
    if (UCC_OK != ucc_component_check_ids_uniq(framework)) {
        /* This can only happen when the new component is added
           (potentially as a plugin - black box) and its name hash
           will produce collision with some other component.
           This has nearly 0 probability and must be resolved by
           the component developer via just a rename of a component. */
        ucc_error("all components of a framwork must have uniq name hash");
        status = UCC_ERR_INVALID_PARAM;
        goto err;
    }

    framework->names.names =
        ucc_malloc(sizeof(char *) * n_loaded, "components_names");
    if (!framework->names.names) {
        ucc_error("failed to allocate %zd bytes for components names",
                  sizeof(char *) * n_loaded);
        status = UCC_ERR_NO_MEMORY;
        goto err;
    }
    framework->names.count = n_loaded;
    for (i = 0; i < n_loaded; i++) {
        framework->names.names[i] = strdup(framework->components[i]->name);
    }
    return UCC_OK;
err:
    ucc_free(framework->components);
    return status;
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

char* ucc_get_framework_components_list(ucc_component_framework_t *framework,
                                        const char* delimiter)
{
    char  *list = NULL;
    size_t len  = 0;
    int    i;

    for (i = 0; i < framework->n_components; i++) {
        /* component name + ',' delimiter */
        len += strlen(framework->components[i]->name) + 1;
    }
    if (len) {
        list = ucc_malloc(len, "components_list");
        if (!list) {
            ucc_error("failed to allocate %zd bytes for components_list", len);
            return NULL;
        }
        list[0] = '\0';
        for (i = 0; i < framework->n_components; i++) {
            strncat(list, framework->components[i]->name, len);
            len -= strlen(framework->components[i]->name);
            if (i < framework->n_components - 1) {
                strncat(list, delimiter, len);
                len -= strlen(delimiter);
            }
        }
    }
    return list;
}
