/*
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCC_H_
#define UCC_H_

#include <api/ucc_def.h>
#include <api/ucc_version.h>
#include <api/ucc_status.h>
#include <ucs/config/types.h>
#include <stdio.h>

BEGIN_C_DECLS

/**
 * @defgroup UCC_API Unified Communication Collectives (UCC) API
 * @{
 * This section describes UCC API.
 * @}
 */
/**
 * @ingroup UCC_LIB
 * @brief @todo
 *
 */

enum ucc_lib_params_field {
    UCC_LIB_PARAM_FIELD_REPRODUCIBLE       = UCS_BIT(0),
    UCC_LIB_PARAM_FIELD_THREAD_MODE        = UCS_BIT(1),
    UCC_LIB_PARAM_FIELD_COLL_TYPES         = UCS_BIT(2),
    UCC_LIB_PARAM_FIELD_REDUCTION_OP_TYPES = UCS_BIT(3),
    UCC_LIB_PARAM_FIELD_SYNC_TYPE          = UCS_BIT(4)
};

enum ucc_reproducibility_mode {
    UCC_REPRODUCIBILITY_MODE_REPRODUCIBLE     = UCS_BIT(0),
    UCC_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE = UCS_BIT(1),
};

enum ucc_thread_mode {
    UCC_THREAD_MODE_MULTIPLE = UCS_BIT(0),
    UCC_THREAD_MODE_SINGLE   = UCS_BIT(1),
};


typedef enum {
    UCC_BARRIER = 0,
    UCC_BCAST,
    UCC_ALLREDUCE,
    UCC_REDUCE,
    UCC_ALLTOALL,
    UCC_COLL_LAST
} ucc_coll_type_t;

typedef enum {
    UCC_COLL_CAP_BARRIER    = UCS_BIT(UCC_BARRIER),
    UCC_COLL_CAP_BCAST      = UCS_BIT(UCC_BCAST),
    UCC_COLL_CAP_ALLREDUCE  = UCS_BIT(UCC_ALLREDUCE),
    UCC_COLL_CAP_REDUCE     = UCS_BIT(UCC_REDUCE),
    UCC_COLL_CAP_ALL        = UCS_MASK(UCC_COLL_LAST)
} ucc_coll_cap_t;

/**
 * @ingroup UCC_LIB
 * @brief UCC library initializatoin parameters
 */
typedef struct ucc_lib_params {
    uint64_t field_mask;
    unsigned thread_mode;
    unsigned reproducible;
    unsigned sync_type;
    uint64_t coll_types;
    uint64_t reduction_op_types;
} ucc_lib_params_t;

/**
 * @ingroup UCC_CONFIG
 * @brief Read UCC configuration descriptor
 *
 * The routine fetches the information about UCC configuration from
 * the run-time environment. Then, the fetched descriptor is used for
 * UCC @ref ucc_lib_init "initialization". In addition
 * the application is responsible for @ref ucc_lib_config_release "releasing"
 * the descriptor back to the UCC.
 *
 * @param [in]  env_prefix    If non-NULL, the routine searches for the
 *                            environment variables that start with
 *                            @e \<env_prefix\>_UCC_ prefix.
 *                            Otherwise, the routine searches for the
 *                            environment variables that start with
 *                            @e UCC_ prefix.
 * @param [in]  filename      If non-NULL, read configuration from the file
 *                            defined by @e filename. If the file does not
 *                            exist, it will be ignored and no error reported
 *                            to the application.
 * @param [out] config_p      Pointer to configuration descriptor as defined by
 *                            @ref ucc_lib_config_t "ucc_lib_config_t".
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_lib_config_read(const char *env_prefix, const char *filename,
                                   ucc_lib_config_t **config_p);

/**
 * @ingroup UCC_CONFIG
 * @brief Release configuration descriptor
 *
 * The routine releases the configuration descriptor that was allocated through
 * @ref ucc_config_read "ucc_config_read()" routine.
 *
 * @param [in] config        Configuration descriptor as defined by
 *                            @ref ucc_lib_config_t "ucc_lib_config_t".
 */

void ucc_lib_config_release(ucc_lib_config_t *config);

/**
 * @ingroup UCC_LIB
 * @brief Initialize UCC library.
 *
 * @todo add description
 *
 * @param [in]  params    (Library initialization parameters)
 * @param [in]  config    UCC configuration descriptor allocated through
 *                        @ref ucc_config_read "ucc_config_read()" routine.
 * @param [out] lib       (UCC library handle)
 *
 * @return Error code
 */
ucc_status_t ucc_lib_init(const ucc_lib_params_t *params,
                          const ucc_lib_config_t *config,
                          ucc_lib_h *lib_p);

/**
 * @ingroup UCC_LIB
 * @brief Release UCC library.
 *
 * @todo add description
 *
 * @param [in] lib_p   Handle to @ref ucc_lib_h
 *                     "UCC library".
 */
void ucc_lib_cleanup(ucc_lib_h lib_p);

END_C_DECLS

#endif
