/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_global_opts.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_datastruct.h"
#include "utils/profile/ucc_profile.h"

UCC_LIST_HEAD(ucc_config_global_list);

ucc_global_config_t ucc_global_config = {
    .log_component    = {UCC_LOG_LEVEL_WARN, "UCC"},
    .coll_trace       = {UCC_LOG_LEVEL_WARN, "UCC_COLL"},
    .component_path   = NULL,
    .install_path     = NULL,
    .initialized      = 0,
    .profile_mode     = 0,
    .profile_file     = "",
    .profile_log_size = 0,
    .file_cfg         = 0};

ucc_config_field_t ucc_global_config_table[] = {
    {"LOG_LEVEL", "warn",
     "UCC logging level. Messages with a level higher or equal to the selected "
     "will be printed.\n"
     "Possible values are: fatal, error, warn, info, debug, trace, data, func, "
     "poll.",
     ucc_offsetof(ucc_global_config_t, log_component),
     UCC_CONFIG_TYPE_LOG_COMP},

    {"COLL_TRACE", "warn",
     "UCC collective logging level. Higher level will result in more verbose "
     "collective info. \n "
     "Possible values are: fatal, error, warn, info, debug, trace, data, func, "
     "poll.",
     ucc_offsetof(ucc_global_config_t, coll_trace),
     UCC_CONFIG_TYPE_LOG_COMP
    },

    {"PROFILE_MODE", "",
     "Profile collection modes. If none is specified, profiling is disabled.\n"
     " - log   - Record all timestamps.\n"
     " - accum - Accumulate measurements per location.\n",
     ucc_offsetof(ucc_global_config_t, profile_mode),
     UCC_CONFIG_TYPE_BITMAP(ucs_profile_mode_names)},

    {"PROFILE_FILE", "ucc_%h_%p.prof",
     "File name to dump profiling data to.\n"
     "Substitutions: %h: host, %p: pid, %c: cpu, %t: time, %u: user, %e: "
     "exe.\n",
     ucc_offsetof(ucc_global_config_t, profile_file), UCC_CONFIG_TYPE_STRING},

    {"PROFILE_LOG_SIZE", "4m",
     "Maximal size of profiling log. New records will replace old records.",
     ucc_offsetof(ucc_global_config_t, profile_log_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"CONFIG_FILE", "auto",
     "Location of configuration file.\n"
     "auto - config file is searched in $HOME/ucc.conf first, then, if not "
     "found, in <ucc_install_path>/share/ucc.conf.\n"
     "empty string \"\" - disable use of config file",
     ucc_offsetof(ucc_global_config_t, cfg_filename), UCC_CONFIG_TYPE_STRING},

    {NULL}};
