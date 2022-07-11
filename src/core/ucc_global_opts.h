/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_GLOBAL_OPTS_H_
#define UCC_GLOBAL_OPTS_H_

#include "config.h"
#include "utils/ucc_component.h"

#include "utils/ucc_parser.h"
#include "utils/ucc_log.h"

typedef struct ucc_global_config {
    /* Log level above which log messages will be printed*/
    ucc_log_component_config_t log_component;
    ucc_component_framework_t  cl_framework;
    ucc_component_framework_t  tl_framework;
    ucc_component_framework_t  mc_framework;
    ucc_component_framework_t  ec_framework;

    /* Coll component libraries path */
    char *component_path;
    char *component_path_default;
    char *install_path;
    int   initialized;
    /* Profiling mode */
    unsigned                   profile_mode;

    /* Profiling output file name */
    char *profile_file;

    /* Limit for profiling log size */
    size_t                     profile_log_size;
    char *                     cfg_filename;
    ucc_file_config_t *        file_cfg;
} ucc_global_config_t;

extern ucc_global_config_t ucc_global_config;
extern ucc_config_field_t  ucc_global_config_table[];

ucc_status_t ucc_constructor(void);
extern ucs_list_link_t ucc_config_global_list;

#endif
