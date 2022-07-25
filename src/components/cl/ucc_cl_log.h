/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_CL_LOG_H_
#define UCC_CL_LOG_H_
#include "components/base/ucc_base_iface.h"

#define cl_error(_cl_lib, _fmt, ...) base_error((ucc_base_lib_t*)(_cl_lib), _fmt, ## __VA_ARGS__)
#define cl_warn(_cl_lib, _fmt, ...)  base_warn((ucc_base_lib_t*)(_cl_lib), _fmt, ## __VA_ARGS__)
#define cl_info(_cl_lib, _fmt, ...)  base_info((ucc_base_lib_t*)(_cl_lib), _fmt, ## __VA_ARGS__)
#define cl_debug(_cl_lib, _fmt, ...) base_debug((ucc_base_lib_t*)(_cl_lib), _fmt, ## __VA_ARGS__)
#define cl_trace(_cl_lib, _fmt, ...) base_trace((ucc_base_lib_t*)(_cl_lib), _fmt, ## __VA_ARGS__)

#endif
