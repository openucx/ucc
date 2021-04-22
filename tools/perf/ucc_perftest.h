/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucc/api/ucc.h>
#include <iostream>

#define STR(x) #x
#define UCCCHECK_GOTO(_call, _label, _status)                                  \
    do {                                                                       \
        ucc_status_t _status = (_call);                                        \
        if (UCC_OK != _status) {                                               \
            std::cerr << "UCC perftest error: " << STR(_call) << "\n";         \
            goto _label;                                                       \
        }                                                                      \
    } while (0)
