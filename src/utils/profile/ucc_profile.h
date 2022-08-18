/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PROFILE_H_
#define UCC_PROFILE_H_

#include <ucs/profile/profile_defs.h>
#include "ucc/api/ucc_status.h"

#ifdef HAVE_PROFILING
#  include "ucc_profile_on.h"
#else
#  include "ucc_profile_off.h"
#endif

/**
 * Initialize profiling system.
 *
 * @param [in]  profile_mode  Profiling mode.
 * @param [in]  file_name     Profiling file.
 * @param [in]  max_file_size Limit for profiling log size.
 *
 * @return Status code.
 */
ucc_status_t ucc_profile_init(unsigned profile_mode, const char *file_name,
                              size_t max_file_size);

/**
 * Save and cleanup profiling.
 */
void ucc_profile_cleanup();

#endif
