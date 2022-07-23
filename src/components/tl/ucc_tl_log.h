/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_LOG_H_
#define UCC_TL_LOG_H_
#include "components/base/ucc_base_iface.h"

#define tl_error(_tl_lib, _fmt, ...) base_error((ucc_base_lib_t*)(_tl_lib), _fmt, ## __VA_ARGS__)
#define tl_warn(_tl_lib, _fmt, ...)  base_warn((ucc_base_lib_t*)(_tl_lib), _fmt, ## __VA_ARGS__)
#define tl_info(_tl_lib, _fmt, ...)  base_info((ucc_base_lib_t*)(_tl_lib), _fmt, ## __VA_ARGS__)
#define tl_debug(_tl_lib, _fmt, ...) base_debug((ucc_base_lib_t*)(_tl_lib), _fmt, ## __VA_ARGS__)
#define tl_trace(_tl_lib, _fmt, ...) base_trace((ucc_base_lib_t*)(_tl_lib), _fmt, ## __VA_ARGS__)

#endif
