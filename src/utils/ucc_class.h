/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_CLASS_H_
#define UCC_CLASS_H_

#include "config.h"
#include <ucs/type/class.h>
#include "utils/ucc_compiler_def.h"

#define UCC_CLASS_DEFINE          UCS_CLASS_DEFINE
#define UCC_CLASS_DELETE          UCS_CLASS_DELETE
#define UCC_CLASS_CLEANUP_FUNC    UCS_CLASS_CLEANUP_FUNC
#define UCC_CLASS_CLEANUP         UCS_CLASS_CLEANUP
#define UCC_CLASS_NEW_FUNC_NAME   UCS_CLASS_NEW_FUNC_NAME

#define UCC_CLASS_INIT_FUNC(_type, ...)                                        \
    ucc_status_t _UCS_CLASS_INIT_NAME(_type)(_type *self,                      \
                                             ucs_class_t *_myclass,            \
                                             int *_init_count, ## __VA_ARGS__) \

#define UCC_CLASS_DECLARE(_type, ...)                                          \
    extern ucs_class_t _UCS_CLASS_DECL_NAME(_type);                            \
    UCC_CLASS_INIT_FUNC(_type, ## __VA_ARGS__);                                \

#define UCC_CLASS_NEW(...)                                                     \
    ({                                                                         \
        ucs_status_t _ucs_status = UCS_CLASS_NEW(__VA_ARGS__);                 \
        ucc_status_t _status     = ucs_status_to_ucc_status(_ucs_status);      \
        _status;                                                               \
    })

#define UCC_CLASS_INIT(...)                                                    \
    ({                                                                         \
        ucs_status_t _ucs_status = UCS_CLASS_INIT(__VA_ARGS__);                \
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

#define UCC_CLASS_DECLARE_NAMED_NEW_FUNC(_name, _argtype, ...)                 \
    ucc_status_t _name(UCS_PP_FOREACH(                                         \
        _UCS_CLASS_INIT_ARG_DEFINE, _,                                         \
        UCS_PP_ZIP((UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__))), (__VA_ARGS__))) \
                           _argtype **obj_p)

#define UCC_CLASS_DEFINE_NAMED_NEW_FUNC(_name, _type, _argtype, ...)           \
    UCC_CLASS_DECLARE_NAMED_NEW_FUNC(_name, _argtype, ##__VA_ARGS__)           \
    {                                                                          \
        ucs_status_t status;                                                   \
        *obj_p = NULL;                                                         \
        status = UCS_CLASS_NEW(                                                \
            _type,                                                             \
            obj_p UCS_PP_FOREACH(_UCS_CLASS_INIT_ARG_PASS, _,                  \
                                 UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__))));   \
        ucs_class_check_new_func_result(status, *obj_p);                       \
        return ucs_status_to_ucc_status(status);                               \
    }

#define UCC_CLASS_DECLARE_NEW_FUNC(_type, _argtype, ...)                       \
    UCC_CLASS_DECLARE_NAMED_NEW_FUNC(UCS_CLASS_NEW_FUNC_NAME(_type), _argtype, \
                                     ##__VA_ARGS__)

#define UCC_CLASS_DEFINE_NEW_FUNC(_type, _argtype, ...)                        \
    UCC_CLASS_DEFINE_NAMED_NEW_FUNC(UCS_CLASS_NEW_FUNC_NAME(_type), _type,     \
                                    _argtype, ##__VA_ARGS__)

#define UCC_CLASS_DECLARE_DELETE_FUNC UCS_CLASS_DECLARE_DELETE_FUNC
#define UCC_CLASS_DEFINE_DELETE_FUNC  UCS_CLASS_DEFINE_DELETE_FUNC
#define UCC_CLASS_DELETE_FUNC_NAME    UCS_CLASS_DELETE_FUNC_NAME
#endif
