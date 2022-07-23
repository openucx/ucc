/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PROFILE_OFF_H_
#define UCC_PROFILE_OFF_H_

#undef UCC_PROFILE_FUNC
#undef UCC_PROFILE_REQUEST_NEW
#undef UCC_PROFILE_REQUEST_EVENT
#undef UCC_PROFILE_REQUEST_FREE

#define UCC_PROFILE_FUNC(_ret_type, _name, _arglist, ...)  _ret_type _name(__VA_ARGS__)
#define UCC_PROFILE_FUNC_VOID(_name, _arglist, ...)         void _name(__VA_ARGS__)
#define UCC_PROFILE_REQUEST_NEW(...)                        UCS_EMPTY_STATEMENT
#define UCC_PROFILE_REQUEST_EVENT(...)                      UCS_EMPTY_STATEMENT
#define UCC_PROFILE_REQUEST_FREE(...)                       UCS_EMPTY_STATEMENT

#endif
