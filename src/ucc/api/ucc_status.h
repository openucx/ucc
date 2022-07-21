/**
 * @file ucc_status.h
 * @date 2020
 * @copyright Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_STATUS_H_
#define UCC_STATUS_H_

#ifdef __cplusplus
# define BEGIN_C_DECLS  extern "C" {
# define END_C_DECLS    }
#else
# define BEGIN_C_DECLS
# define END_C_DECLS
#endif

BEGIN_C_DECLS
/**
 * @ingroup UCC_UTILS
 * @brief Status codes for the UCC operations
 *
 * @note In order to evaluate the necessary steps to recover from a certain
 * error, all error codes which can be returned by the external API are grouped
 * by the largest entity permanently effected by the error. Each group ranges
 * between its UCc_ERR_FIRST_<name> and UCC_ERR_LAST_<name> enum values.
 * For example, if a link fails it may be sufficient to destroy (and possibly
 * replace) it, in contrast to an endpoint-level error.
 */

typedef enum {
    /* Operation completed successfully */
    UCC_OK                              =    0,

    UCC_INPROGRESS                      =    1, /*!< Operation is posted and is in progress */

    UCC_OPERATION_INITIALIZED           =    2, /*!< Operation initialized but not posted */

    /* Error status codes */
    UCC_ERR_NOT_SUPPORTED               =   -1,
    UCC_ERR_NOT_IMPLEMENTED             =   -2,
    UCC_ERR_INVALID_PARAM               =   -3,
    UCC_ERR_NO_MEMORY                   =   -4,
    UCC_ERR_NO_RESOURCE                 =   -5,
    UCC_ERR_NO_MESSAGE                  =   -6, /*!< General purpose return code without specific error */
    UCC_ERR_NOT_FOUND                   =   -7,
    UCC_ERR_TIMED_OUT                   =   -8,
    UCC_ERR_IO_ERROR                    =   -9,
    UCC_ERR_UNREACHABLE                 =  -10,
    UCC_ERR_INVALID_ADDR                =  -11,
    UCC_ERR_MESSAGE_TRUNCATED           =  -12,
    UCC_ERR_NO_PROGRESS                 =  -13,
    UCC_ERR_BUFFER_TOO_SMALL            =  -14,
    UCC_ERR_NO_ELEM                     =  -15,
    UCC_ERR_SOME_CONNECTS_FAILED        =  -16,
    UCC_ERR_NO_DEVICE                   =  -17,
    UCC_ERR_BUSY                        =  -18,
    UCC_ERR_CANCELED                    =  -19,
    UCC_ERR_SHMEM_SEGMENT               =  -20,
    UCC_ERR_ALREADY_EXISTS              =  -21,
    UCC_ERR_OUT_OF_RANGE                =  -22,
    UCC_ERR_EXCEEDS_LIMIT               =  -23,
    UCC_ERR_UNSUPPORTED                 =  -24,
    UCC_ERR_REJECTED                    =  -25,
    UCC_ERR_NOT_CONNECTED               =  -26,
    UCC_ERR_CONNECTION_RESET            =  -27,

    UCC_ERR_FIRST_LINK_FAILURE          =  -40,
    UCC_ERR_LAST_LINK_FAILURE           =  -59,
    UCC_ERR_FIRST_ENDPOINT_FAILURE      =  -60,
    UCC_ERR_ENDPOINT_TIMEOUT            =  -80,
    UCC_ERR_LAST_ENDPOINT_FAILURE       =  -89,

    UCC_ERR_LAST                        = -100,
} ucc_status_t;

#define UCC_IS_ENDPOINT_ERROR(_code) \
    (((_code) <= UCC_ERR_FIRST_ENDPOINT_FAILURE) && \
     ((_code) >= UCC_ERR_LAST_ENDPOINT_FAILURE))
/**
 * @ingroup UCC_UTILS
 * @brief Status pointer
 *
 * A pointer can represent one of these values:
 * - NULL / UCS_OK
 * - Error code pointer (UCS_ERR_xx)
 * - Valid pointer
 */
typedef void *ucc_status_ptr_t;

#define UCC_PTR_IS_ERR(_ptr)       (((uintptr_t)(_ptr)) >= ((uintptr_t)UCC_ERR_LAST))
#define UCC_PTR_IS_PTR(_ptr)       (((uintptr_t)(_ptr) - 1) < ((uintptr_t)UCC_ERR_LAST - 1))
#define UCC_PTR_RAW_STATUS(_ptr)   ((ucc_status_t)(intptr_t)(_ptr))
#define UCC_PTR_STATUS(_ptr)       (UCC_PTR_IS_PTR(_ptr) ? UCS_INPROGRESS : UCC_PTR_RAW_STATUS(_ptr))
#define UCC_STATUS_PTR(_status)    ((void*)(intptr_t)(_status))
#define UCC_STATUS_IS_ERR(_status) ((_status) < 0)

/**
 * @ingroup UCC_UTILS
 * @brief Routine to convert status code to string
 */

const char *ucc_status_string(ucc_status_t status);

END_C_DECLS
#endif
