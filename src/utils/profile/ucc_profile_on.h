/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PROFILE_ON_H_
#define UCC_PROFILE_ON_H_

#include "core/ucc_global_opts.h"
#include <ucs/profile/profile_on.h>

extern ucs_profile_context_t *ucc_profile_ctx;


#undef UCC_PROFILE_FUNC
#undef UCC_PROFILE_FUNC_VOID
#undef UCC_PROFILE_REQUEST_NEW
#undef UCC_PROFILE_REQUEST_EVENT
#undef UCC_PROFILE_REQUEST_FREE

#ifdef UCS_PROFILE_LOC_ID_DISABLED
/**
 * Create a profiled function. Uses default profile context.
 *
 * Usage:
 *  UCC_PROFILE_FUNC(<retval>, <name>, (a, b), int a, char b)
 *
 * @param _ret_type   Function return type.
 * @param _name       Function name.
 * @param _arglist    List of argument *names* only.
 * @param ...         Argument declarations (with types).
 */
#define UCC_PROFILE_FUNC(_ret_type, _name, _arglist, ...) \
    UCS_PROFILE_CTX_FUNC_ALWAYS(ucc_profile_ctx, _ret_type, _name, _arglist, ## __VA_ARGS__)

/**
 * Create a profiled function whose return type is void. Uses default profile
 * context.
 *
 * Usage:
 *  UCC_PROFILE_FUNC_VOID(<name>, (a, b), int a, char b)
 *
 * @param _name       Function name.
 * @param _arglist    List of argument *names* only.
 * @param ...         Argument declarations (with types).
 */
#define UCC_PROFILE_FUNC_VOID(_name, _arglist, ...) \
    UCS_PROFILE_CTX_FUNC_VOID_ALWAYS(ucc_profile_ctx, _name, _arglist, ## __VA_ARGS__)

/*
 * Profile a new request allocation.
 *
 * @param _req      Request pointer.
 * @param _name     Allocation site name.
 * @param _param32  Custom 32-bit parameter.
 */
#define UCC_PROFILE_REQUEST_NEW(_req, _name, _param32) \
    UCS_PROFILE_CTX_RECORD_ALWAYS(ucc_profile_ctx, UCS_PROFILE_TYPE_REQUEST_NEW, \
                           (_name), (_param32), (uintptr_t)(_req));

/*
 * Profile a request progress event.
 *
 * @param _req      Request pointer.
 * @param _name     Event name.
 * @param _param32  Custom 32-bit parameter.
 */
#define UCC_PROFILE_REQUEST_EVENT(_req, _name, _param32) \
    UCS_PROFILE_CTX_RECORD_ALWAYS(ucc_profile_ctx, UCS_PROFILE_TYPE_REQUEST_EVENT, \
                           (_name), (_param32), (uintptr_t)(_req));

/*
 * Profile a request release.
 *
 * @param _req      Request pointer.
 */
#define UCC_PROFILE_REQUEST_FREE(_req) \
    UCS_PROFILE_CTX_RECORD_ALWAYS(ucc_profile_ctx, UCS_PROFILE_TYPE_REQUEST_FREE, \
                           "", 0, (uintptr_t)(_req));
#else
#define UCC_PROFILE_FUNC(_ret_type, _name, _arglist, ...) \
    _UCS_PROFILE_CTX_FUNC(ucc_profile_ctx, _ret_type, _name, _arglist, ## __VA_ARGS__)

#define UCC_PROFILE_FUNC_VOID(_name, _arglist, ...) \
    _UCS_PROFILE_CTX_FUNC_VOID(ucc_profile_ctx, _name, _arglist, ## __VA_ARGS__)

#define UCC_PROFILE_REQUEST_NEW(_req, _name, _param32) \
    UCS_PROFILE_CTX_RECORD(ucc_profile_ctx, UCS_PROFILE_TYPE_REQUEST_NEW, \
                           (_name), (_param32), (uintptr_t)(_req));

#define UCC_PROFILE_REQUEST_EVENT(_req, _name, _param32) \
    UCS_PROFILE_CTX_RECORD(ucc_profile_ctx, UCS_PROFILE_TYPE_REQUEST_EVENT, \
                           (_name), (_param32), (uintptr_t)(_req));

#define UCC_PROFILE_REQUEST_FREE(_req) \
    UCS_PROFILE_CTX_RECORD(ucc_profile_ctx, UCS_PROFILE_TYPE_REQUEST_FREE, \
                           "", 0, (uintptr_t)(_req));
#endif

#endif
