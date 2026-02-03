/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string.h>

#define IFACE_NAME_LEN_MAX                                                     \
    (UCC_MAX_FRAMEWORK_NAME_LEN + UCC_MAX_COMPONENT_NAME_LEN + 32)

static ucc_status_t ucc_components_load_from_path(
    const char *path, const char *framework_name,
    ucc_component_iface_t ***ifaces, int *n_loaded);

static ucc_status_t ucc_component_load_one(const char *so_path,
                                           const char *framework_name,
                                           ucc_component_iface_t **c_iface)
{
    char                   framework_pattern[UCC_MAX_FRAMEWORK_NAME_LEN + 16];
    char                  *error, iface_struct[IFACE_NAME_LEN_MAX];
    void                  *handle;
    ucc_component_iface_t *iface;
    ptrdiff_t              basename_start;
    size_t                 iface_struct_name_len;

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
        error = dlerror();
        ucc_debug("failed to load UCC component library: %s (%s)",
                  so_path, error);
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
    ucc_component_iface_t **ifaces   = NULL;
    int                     n_loaded = 0;
    ucc_status_t            status;
    int                     i;

    framework->n_components = 0;
    framework->components   = NULL;

    if (strlen(framework_name) == 0 ||
        strlen(framework_name) > UCC_MAX_FRAMEWORK_NAME_LEN) {
        ucc_error("unsupported framework_name length: %s, len %zd",
                  framework_name, strlen(framework_name));
        return UCC_ERR_INVALID_PARAM;
    }

    /* Load components from the standard component path */
    status = ucc_components_load_from_path(ucc_global_config.component_path,
                                           framework_name, &ifaces, &n_loaded);
    if (status != UCC_OK) {
        return status;
    }

    /* Set up framework structure */
    framework->components   = ifaces;
    framework->n_components = n_loaded;

    /* Check that all component IDs are unique */
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

    /* Build component names array */
    framework->names.names = ucc_malloc(sizeof(char *) * n_loaded,
                                        "components_names");
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
        list = ucc_malloc(len + 1, "components_list");
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

static ucc_status_t ucc_components_load_from_path(
    const char *path, const char *framework_name,
    ucc_component_iface_t ***ifaces, int *n_loaded)
{
    glob_t                  globbuf;
    int                     i, loaded_count;
    char                   *full_pattern;
    ucc_status_t            status;
    size_t                  pattern_size;
    ucc_component_iface_t **new_ifaces      = NULL;
    ucc_component_iface_t **combined_ifaces = NULL;

    if (!path || strlen(path) == 0) {
        return UCC_ERR_NOT_FOUND;
    }

    pattern_size = strlen(path) + strlen(framework_name) + 16;
    full_pattern = (char *)ucc_malloc(pattern_size, "full_pattern");
    if (!full_pattern) {
        ucc_error(
            "failed to allocate %zd bytes for full_pattern", pattern_size);
        return UCC_ERR_NO_MEMORY;
    }

    ucc_snprintf_safe(
        full_pattern, pattern_size, "%s/libucc_%s_*.so", path, framework_name);
    glob(full_pattern, 0, NULL, &globbuf);
    ucc_free(full_pattern);

    if (globbuf.gl_pathc == 0) {
        globfree(&globbuf);
        return UCC_ERR_NOT_FOUND;
    }

    new_ifaces = (ucc_component_iface_t **)ucc_malloc(
        globbuf.gl_pathc * sizeof(ucc_component_iface_t *), "new_ifaces");
    if (!new_ifaces) {
        ucc_error(
            "failed to allocate %zd bytes for new_ifaces",
            globbuf.gl_pathc * sizeof(ucc_component_iface_t *));
        globfree(&globbuf);
        return UCC_ERR_NO_MEMORY;
    }

    loaded_count = 0;
    dlerror(); /* Clear any existing error */

    for (i = 0; i < globbuf.gl_pathc; i++) {
        status = ucc_component_load_one(
            globbuf.gl_pathv[i], framework_name, &new_ifaces[loaded_count]);
        if (status != UCC_OK) {
            continue;
        }
        loaded_count++;
    }

    globfree(&globbuf);

    if (loaded_count == 0) {
        ucc_free(new_ifaces);
        return UCC_ERR_NOT_FOUND;
    }

    /* Combine with existing ifaces if any */
    if (*ifaces != NULL && *n_loaded > 0) {
        combined_ifaces = (ucc_component_iface_t **)ucc_malloc(
            (*n_loaded + loaded_count) * sizeof(ucc_component_iface_t *),
            "combined_ifaces");
        if (!combined_ifaces) {
            ucc_error("failed to allocate memory for combined_ifaces");
            ucc_free(new_ifaces);
            return UCC_ERR_NO_MEMORY;
        }
        memcpy(
            combined_ifaces,
            *ifaces,
            *n_loaded * sizeof(ucc_component_iface_t *));
        memcpy(
            combined_ifaces + *n_loaded,
            new_ifaces,
            loaded_count * sizeof(ucc_component_iface_t *));
        ucc_free(*ifaces);
        ucc_free(new_ifaces);
        *ifaces = combined_ifaces;
    } else {
        *ifaces = new_ifaces;
    }

    *n_loaded += loaded_count;
    return UCC_OK;
}

ucc_status_t ucc_components_load_user_component(const char *path,
                                                 const char *framework_name,
                                                 ucc_component_framework_t *framework)
{
    ucc_component_iface_t **ifaces = NULL;
    int                     n_loaded = 0;
    int                     i, original_count;
    ucc_status_t            status;

    if (!path || !framework_name || !framework) {
        return UCC_ERR_INVALID_PARAM;
    }

    if (strlen(path) == 0) {
        return UCC_OK;
    }

    ucc_info("loading user %s components from: %s", framework_name, path);

    /* Copy existing components so we can append */
    original_count = framework->n_components;
    if (original_count > 0) {
        ifaces = (ucc_component_iface_t **)ucc_malloc(
            original_count * sizeof(ucc_component_iface_t *), "ifaces");
        if (!ifaces) {
            ucc_error("failed to allocate memory for ifaces");
            return UCC_ERR_NO_MEMORY;
        }
        memcpy(ifaces, framework->components,
               original_count * sizeof(ucc_component_iface_t *));
        n_loaded = original_count;
    }

    status = ucc_components_load_from_path(path, framework_name,
                                           &ifaces, &n_loaded);
    if (status == UCC_ERR_NOT_FOUND) {
        ucc_info("no user %s components found in %s", framework_name, path);
        if (ifaces && n_loaded == original_count) {
            ucc_free(ifaces);
        }
        return UCC_ERR_NOT_FOUND;
    } else if (status != UCC_OK) {
        ucc_free(ifaces);
        return status;
    }

    for (i = original_count; i < n_loaded; i++) {
        ucc_info("loaded user %s component: %s",
                 framework_name, framework->components[i]->name);
    }

    /* Update framework with new component list */
    if (original_count > 0) {
        ucc_free(framework->components);
    }
    framework->components   = ifaces;
    framework->n_components = n_loaded;

    /* Update component names array */
    if (framework->names.names) {
        for (i = 0; i < framework->names.count; i++) {
            free(framework->names.names[i]);
        }
        ucc_free(framework->names.names);
    }

    framework->names.names = ucc_malloc(sizeof(char *) * n_loaded,
                                        "components_names");
    if (!framework->names.names) {
        ucc_error("failed to allocate %zd bytes for components names",
                  sizeof(char *) * n_loaded);
        return UCC_ERR_NO_MEMORY;
    }
    framework->names.count = n_loaded;
    for (i = 0; i < n_loaded; i++) {
        framework->names.names[i] = strdup(framework->components[i]->name);
    }

    ucc_info("loaded %d user %s component(s)", n_loaded - original_count,
             framework_name);
    return UCC_OK;
}
