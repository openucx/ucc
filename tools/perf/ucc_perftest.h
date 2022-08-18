/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <ucc/api/ucc.h>
#include <iostream>
#include "config.h"
extern "C" {
#include "utils/ucc_malloc.h"
}

#define STR(x) #x
#define UCCCHECK_GOTO(_call, _label, _status)                                  \
    do {                                                                       \
        _status = (_call);                                                     \
        if (UCC_OK != _status) {                                               \
            std::cerr << "UCC perftest error: " << ucc_status_string(_status)  \
                      << " in " << STR(_call) <<  __FILE__ << ":"              \
                      << __LINE__<< "\n";                                      \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#define UCC_MALLOC_CHECK_GOTO(_obj, _label, _status)                           \
    do {                                                                       \
        if (!(_obj)) {                                                         \
            _status = UCC_ERR_NO_MEMORY;                                       \
            std::cerr << "UCC perftest error: " << ucc_status_string(_status)  \
                      << "\n";                                                 \
            goto _label;                                                       \
        }                                                                      \
    } while (0)
