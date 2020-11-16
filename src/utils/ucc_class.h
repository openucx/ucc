/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_CLASS_H_
#define UCC_CLASS_H_

#include "config.h"
#include <ucs/type/class.h>
#include "utils/ucc_compiler_def.h"

#define UCC_CLASS_DECLARE         UCS_CLASS_DECLARE
#define UCC_CLASS_DEFINE          UCS_CLASS_DEFINE
#define UCC_CLASS_DELETE          UCS_CLASS_DELETE
#define UCC_CLASS_INIT_FUNC       UCS_CLASS_INIT_FUNC
#define UCC_CLASS_CLEANUP_FUNC    UCS_CLASS_CLEANUP_FUNC

#define UCC_CLASS_NEW(...)                                                     \
    ({                                                                         \
        ucs_status_t _ucs_status = UCS_CLASS_NEW(__VA_ARGS__);                 \
        ucc_status_t _status     = ucs_status_to_ucc_status(_ucs_status);      \
        _status;                                                               \
    })

#define UCC_CLASS_CALL_SUPER_INIT(_superclass, ...)                            \
    {                                                                          \
        {                                                                      \
            ucs_status_t _ucs_status = _UCS_CLASS_INIT_NAME(_superclass)(      \
                &self->super, _myclass->superclass, _init_count,               \
                ##__VA_ARGS__);                                                \
            ucc_status_t _status = ucs_status_to_ucc_status(_ucs_status);      \
            if (_status != UCC_OK) {                                           \
                return _status;                                                \
            }                                                                  \
            if (_myclass->superclass != &_UCS_CLASS_DECL_NAME(void)) {         \
                ++(*_init_count);                                              \
            }                                                                  \
        }                                                                      \
    }

#endif
