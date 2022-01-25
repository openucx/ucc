/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_global_opts.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_component.h"
#include "utils/ucc_log.h"
#include "utils/ucc_proc_info.h"
#include "utils/profile/ucc_profile.h"
#include <link.h>
#include <dlfcn.h>

#define UCC_LIB_SO_NAME "libucc.so"
#define UCC_COMPONENT_LIBDIR "ucc"
#define UCC_COMPONENT_LIBDIR_LEN strlen("ucc")

static int callback(struct dl_phdr_info *info, size_t size, void *data)
{
    char  *str;
    char  *component_path;
    size_t len;
    int    pos;
    if ((data != NULL) || (size == 0)) {
        return -1;
    }
    if (NULL != (str = strstr(info->dlpi_name, UCC_LIB_SO_NAME))) {
        pos            = (int)(str - info->dlpi_name);
        len            = pos + UCC_COMPONENT_LIBDIR_LEN + 1;
        component_path = (char *)ucc_malloc(len, "component_path");
        if (!component_path) {
            ucc_error("failed to allocate %zd bytes for component_path", len);
            return -1;
        }
        /* copying up to pos+1 due to ucs_strncpy_safe implementation specifics:
           it'll always write '\0' to the end position of the dest string. */
        ucc_strncpy_safe(component_path, info->dlpi_name, pos + 1);
        len                -= (pos + 1);
        component_path[pos] = '\0';
        strncat(component_path, UCC_COMPONENT_LIBDIR, len);
        ucc_global_config.component_path =
            ucc_global_config.component_path_default = component_path;
    }
    return 0;
}

static void get_default_lib_path()
{
    dl_iterate_phdr(callback, NULL);
}

ucc_status_t ucc_constructor(void)
{
    ucc_global_config_t *cfg = &ucc_global_config;
    ucc_status_t status;
    if (!cfg->initialized) {
        cfg->initialized            = 1;
        cfg->component_path_default = NULL;

        status = ucc_config_parser_fill_opts(
            &ucc_global_config, ucc_global_config_table, "UCC_", NULL, 1);
        if (UCC_OK != status) {
            ucc_error("failed to parse global options");
            return status;
        }
        if (strlen(cfg->component_path) == 0) {
            get_default_lib_path();
        }
        if (!cfg->component_path) {
            ucc_error("failed to get ucc components path");
            return UCC_ERR_NOT_FOUND;
        }
        status = ucc_components_load("cl", &cfg->cl_framework);
        if (UCC_OK != status) {
            ucc_error("no CL components were found in the "
                      "UCC_COMPONENT_PATH: %s",
                      cfg->component_path);
            return status;
        }
        status = ucc_component_check_scores_uniq(&cfg->cl_framework);
        if (UCC_OK != status) {
            ucc_error("CLs must have distinct uniq default scores");
            return status;
        }
        status = ucc_components_load("tl", &cfg->tl_framework);
        if (UCC_OK != status) {
            /* not critical - some CLs may operate w/o use of TL */
            ucc_debug("no TL components were found in the "
                      "UCC_COMPONENT_PATH: %s",
                      cfg->component_path);
        }
        status = ucc_component_check_scores_uniq(&cfg->tl_framework);
        if (UCC_OK != status) {
            ucc_error("TLs must have distinct uniq default scores");
            return status;
        }
        status = ucc_components_load("mc", &cfg->mc_framework);
        if (UCC_OK != status) {
            ucc_error("no memory components were found in the "
                      "UCC_COMPONENT_PATH: %s",
                      cfg->component_path);
            return status;
        }
        status = ucc_components_load("ec", &cfg->ec_framework);
        if (status != UCC_OK) {
            if (status == UCC_ERR_NOT_FOUND) {
                ucc_info("no execution components were found in the "
                         "UCC_COMPONENT_PATH: %s. "
                         "Triggered operations might not work",
                         cfg->component_path);
            } else {
                ucc_error("failed to load execution components %d (%s)",
                           status, ucc_status_string(status));
                return status;
            }
        }

        if (UCC_OK != ucc_local_proc_info_init()) {
            ucc_error("failed to initialize local proc info");
            return status;
        }
#ifdef HAVE_PROFILING
        ucc_profile_init(cfg->profile_mode, cfg->profile_file,
                         cfg->profile_log_size);
#endif
    }
    return UCC_OK;
}

__attribute__((destructor)) static void ucc_destructor(void)
{
    if (ucc_global_config.initialized) {
#ifdef HAVE_PROFILING
        ucc_profile_cleanup();
#endif
        ucc_config_parser_release_opts(&ucc_global_config,
                                       ucc_global_config_table);
    }
}
