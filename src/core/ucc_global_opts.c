/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "ucc_global_opts.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_datastruct.h"
#include "utils/profile/ucc_profile.h"

UCC_LIST_HEAD(ucc_config_global_list);

ucc_global_config_t ucc_global_config = {
    .log_component    = {UCC_LOG_LEVEL_WARN, "UCC"},
    .component_path   = "",
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

    {"COMPONENT_PATH", "", "Specifies dynamic components location",
     ucc_offsetof(ucc_global_config_t, component_path), UCC_CONFIG_TYPE_STRING},

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

    {"CONFIG_FILE", "", "Location of configuration file",
     ucc_offsetof(ucc_global_config_t, cfg_filename), UCC_CONFIG_TYPE_STRING},

    {NULL}};

UCC_CONFIG_REGISTER_TABLE(ucc_global_config_table, "UCC global", NULL,
                          ucc_global_config, &ucc_config_global_list)
