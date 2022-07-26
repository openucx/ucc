/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define UCC_CLASS_INIT(_type, _obj, ...) \
    ({ \
        ucs_class_t *_cls = &_UCS_CLASS_DECL_NAME(_type); \
        int _init_counter = 1; \
        ucc_status_t __status; \
        \
        __status = _UCS_CLASS_INIT_NAME(_type)((_type*)(_obj), _cls, \
                                             &_init_counter, ## __VA_ARGS__); \
        if (__status != UCC_OK) { \
            ucs_class_call_cleanup_chain(&_UCS_CLASS_DECL_NAME(_type), \
                                         (_obj), _init_counter); \
        } \
        \
        (__status); \
    })

#define UCC_CLASS_NEW(_type, _obj, ...) \
    _UCC_CLASS_NEW (_type, _obj, ## __VA_ARGS__)

#define _UCC_CLASS_NEW(_type, _obj, ...) \
    ({ \
        ucs_class_t *cls = &_UCS_CLASS_DECL_NAME(_type); \
        ucc_status_t _status; \
        void *obj; \
        \
        obj = ucs_class_malloc(cls); \
        if (obj != NULL) { \
            _status = UCC_CLASS_INIT(_type, obj, ## __VA_ARGS__); \
            if (_status == UCC_OK) { \
                *(_obj) = (typeof(*(_obj)))obj; /* Success - assign pointer */ \
            } else { \
                ucs_class_free(obj); /* Initialization failure */ \
            } \
        } else { \
            _status = UCC_ERR_NO_MEMORY; /* Allocation failure */ \
        } \
        \
        (_status); \
    })

#define UCC_CLASS_CALL_SUPER_INIT(_superclass, ...)                            \
    {                                                                          \
        {                                                                      \
            ucc_status_t _status = _UCS_CLASS_INIT_NAME(_superclass)(      \
                &self->super, _myclass->superclass, _init_count,               \
                ##__VA_ARGS__);                                                \
            if (ucc_unlikely(_status != UCC_OK)) {                                           \
                return _status;                                                \
            }                                                                  \
            if (_myclass->superclass != &_UCS_CLASS_DECL_NAME(void)) {         \
                ++(*_init_count);                                              \
            }                                                                  \
        }                                                                      \
    }

#define UCC_CLASS_CALL_BASE_INIT()                                             \
    {                                                                          \
        {                                                                      \
            if ((_init_count == NULL) || (_myclass == NULL)) {                 \
                return UCC_ERR_INVALID_PARAM;                                  \
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
        ucc_status_t status;                                                   \
        ucs_status_t ucs_status;                                               \
        *obj_p = NULL;                                                         \
        status = UCC_CLASS_NEW(                                                \
            _type,                                                             \
            obj_p UCS_PP_FOREACH(_UCS_CLASS_INIT_ARG_PASS, _,                  \
                                 UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__))));   \
        ucs_status = ucc_status_to_ucs_status(status);                         \
        ucs_class_check_new_func_result(ucs_status, *obj_p);                   \
        return status;                                                         \
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
