/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_global_opts.h"
#include "utils/debug/types.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_datastruct.h"
#include "utils/debug/log.h"
#include "utils/profile/ucc_profile.h"

UCC_LIST_HEAD(ucc_config_global_list);

ucc_global_config_t ucc_global_config = {
    .log_component    = {UCC_LOG_LEVEL_WARN, "UCC", "*"},
    .coll_trace       = {UCC_LOG_LEVEL_WARN, "UCC_COLL", "*"},
    .component_path   = NULL,
    .install_path     = NULL,
    .initialized      = 0,
    .profile_mode     = 0,
    .profile_file     = "",
    .profile_log_size = 0,
    .file_cfg         = 0,
    .log_file         = "",
    .log_file_size    = SIZE_MAX,
    .log_file_rotate  = 0,
    .log_buffer_size  = 1024,
    .log_data_size    = 0,
    .log_print_enable = 0,
    .log_level_trigger = UCC_LOG_LEVEL_FATAL};

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

    {"LOG_FILE", "",
     "If not empty, UCC will print log messages to the specified file instead of stdout.\n"
     "The following substitutions are performed on this string:\n"
     "  %p - Replaced with process ID\n"
     "  %h - Replaced with host name",
     ucc_offsetof(ucc_global_config_t, log_file),
     UCC_CONFIG_TYPE_STRING},

    {"LOG_FILE_SIZE", "inf",
     "The maximal size of log file. The maximal log file size has to be >= LOG_BUFFER.",
     ucc_offsetof(ucc_global_config_t, log_file_size), UCC_CONFIG_TYPE_MEMUNITS},

    {"LOG_FILE_ROTATE", "0",
     "The maximal number of backup log files that could be created to save logs\n"
     "after the previous ones (if any) are completely filled. The value has to be\n"
     "less than the maximal signed integer value.",
     ucc_offsetof(ucc_global_config_t, log_file_rotate), UCC_CONFIG_TYPE_UINT},

    {"LOG_BUFFER", "1024",
     "Buffer size for a single log message.",
     ucc_offsetof(ucc_global_config_t, log_buffer_size), UCC_CONFIG_TYPE_MEMUNITS},

    {"LOG_DATA_SIZE", "0",
     "How much packet payload to print, at most, in data mode.",
     ucc_offsetof(ucc_global_config_t, log_data_size), UCC_CONFIG_TYPE_ULONG},

    {"LOG_PRINT_ENABLE", "n",
     "Enable output of ucc_print(). This option is intended for use by the library developers.",
     ucc_offsetof(ucc_global_config_t, log_print_enable), UCC_CONFIG_TYPE_BOOL},

    {"LOG_LEVEL_TRIGGER", "fatal",
     "Log level to trigger error handling.",
     ucc_offsetof(ucc_global_config_t, log_level_trigger), UCC_CONFIG_TYPE_ENUM(ucc_log_level_names)},

    {"CHECK_ASYMMETRIC_DT", "n",
     "Enable asymmetric datatype checking for rooted collectives\n"
     "(gather, gatherv, scatter, scatterv). Uses allreduce to verify\n"
     "all ranks use the same datatype. Disabled by default for performance.\n"
     "Enable for debugging or when OMPI needs UCC to detect asymmetric\n"
     "datatypes for proper fallback behavior.",
     ucc_offsetof(ucc_global_config_t, check_asymmetric_dt),
     UCC_CONFIG_TYPE_BOOL},

    {NULL}
};
