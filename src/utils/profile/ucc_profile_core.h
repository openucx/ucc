/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifdef HAVE_PROFILING_CORE
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif
#define UCC_CORE_PROFILE_FUNC UCC_PROFILE_FUNC
