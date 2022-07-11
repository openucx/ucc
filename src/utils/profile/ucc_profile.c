/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_profile.h"
#include "utils/ucc_compiler_def.h"

ucs_profile_context_t *ucc_profile_ctx;

ucc_status_t ucc_profile_init(unsigned profile_mode, const char *file_name,
                              size_t max_file_size)
{
    ucs_status_t status;

    status = ucs_profile_init(profile_mode, file_name, max_file_size,
                              &ucc_profile_ctx);
    return ucs_status_to_ucc_status(status);
}

void ucc_profile_cleanup()
{
    ucs_profile_cleanup(ucc_profile_ctx);
}
