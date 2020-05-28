/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_STATUS_H_
#define UCC_STATUS_H_

typedef enum {
    /* Operation completed successfully */
    UCC_OK                         =   0,

    /* Operation is queued and still in progress */
    UCC_IN_PROGRESS                =   1,

    /* Failure codes */
    UCC_ERR_NO_MESSAGE             =  -1,
    UCC_ERR_NO_RESOURCE            =  -2,
    UCC_ERR_NO_MEMORY              =  -4,
    UCC_ERR_INVALID_PARAM          =  -5,
    UCC_ERR_UNREACHABLE            =  -6,
    UCC_ERR_NOT_IMPLEMENTED        =  -8,
    UCC_ERR_MESSAGE_TRUNCATED      =  -9,
    UCC_ERR_NO_PROGRESS            = -10,
    UCC_ERR_BUFFER_TOO_SMALL       = -11,
    UCC_ERR_NO_ELEM                = -12,
    UCC_ERR_UNSUPPORTED            = -22,
    UCC_ERR_LAST                   = -100
} ucc_status_t;
#endif
