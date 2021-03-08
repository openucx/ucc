/**
 * @file ucc.h
 * @date 2020
 * @copyright Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * @copyright Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_H_
#define UCC_H_

#include <ucc/api/ucc_def.h>
#include <ucc/api/ucc_version.h>
#include <ucc/api/ucc_status.h>
#include <stdio.h>

BEGIN_C_DECLS

/** Unified Collective Communications (UCC) Library Specification
 *
 *  UCC is a collective communication operations API and library that is
 *  flexible, complete, and feature-rich for current and emerging programming
 *  models and runtimes.
 *
 *
 */

/**
  * @defgroup UCC_LIB_INIT_DT Library initialization data-structures
  * @{
  * Library initialization parameters and data-structures
  * @}
  *
  */

/**
  * @defgroup UCC_LIB Library initialization and finalization routines
  * @{
  * Library initialization and finalization routines
  * @}
  */

/**
  * @defgroup UCC_CONTEXT_DT Context abstraction data-structures
  * @{
  *  Data-structures associated with context creation and management routines
  * @}
  */

/**
  * @defgroup UCC_CONTEXT Context abstraction routines
  * @{
  *  Context create and management routines
  * @}
  */

/**
  * @defgroup UCC_TEAM_DT Team abstraction data-structures
  * @{
  *  Data-structures associated with team create and management routines
  * @}
  */

/**
  * @defgroup UCC_TEAM Team abstraction routines
  * @{
  *  Team create and management routines
  * @}
  */

/**
  * @defgroup UCC_COLLECTIVES_DT Collective operations data-structures
  * @{
  *  Data-structures associated with collective operation creation, progress, and
  *  finalize.
  * @}
  */

/**
  * @defgroup UCC_COLLECTIVES Collective Operations
  * @{
  *  Collective operations invocation and progress
  * @}
  */

/**
  * @defgroup UCC_EVENT_DT Events and Triggered operations' datastructures
  * @{
  *  Data-structures associated with event-driven collective execution
  * @}
  */

/**
  * @defgroup UCC_EVENT Events and Triggered Operations
  * @{
  *  Event-driven Collective Execution
  * @}
  */

/**
  * @defgroup UCC_UTILS Utility Operations
  * @{
  *  Helper functions to be used across the library
  * @}
  */

/**
 * *************************************************************
 *                   Library initialization and finalize
 * *************************************************************
 */

/**
 *
 *  @ingroup UCC_LIB_INIT_DT
 *
 *  @brief Enumeration representing the UCC reduction operations
 *
 *  @parblock
 *
 *
 *  Description
 *
 *  @ref ucc_reduction_op_t  represents the UCC reduction operations. It is used by the
 *  library initialization routine @ref ucc_init to request the operations expected by the user.
 *  It is used by the @ref ucc_lib_attr_t to communicate the operations supported by
 *  the library. The user-defined reductions are represented by
 *  UCC_OP_USERDEFINED.
 *
 *  @endparblock
 *
 */
typedef enum {
    UCC_OP_USERDEFINED      = UCC_BIT(0), /*!< User defined reduction operation */
    UCC_OP_SUM              = UCC_BIT(1), /*!< Predefined addition operation */
    UCC_OP_PROD             = UCC_BIT(2),
    UCC_OP_MAX              = UCC_BIT(3),
    UCC_OP_MIN              = UCC_BIT(4),
    UCC_OP_LAND             = UCC_BIT(5),
    UCC_OP_LOR              = UCC_BIT(6),
    UCC_OP_LXOR             = UCC_BIT(7),
    UCC_OP_BAND             = UCC_BIT(8),
    UCC_OP_BOR              = UCC_BIT(9),
    UCC_OP_BXOR             = UCC_BIT(10),
    UCC_OP_MAXLOC           = UCC_BIT(11),
    UCC_OP_MINLOC           = UCC_BIT(12)
} ucc_reduction_op_t;

/**
 *
 *  @ingroup UCC_LIB_INIT_DT
 *
 *  @brief Enumeration representing the collective operations
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_coll_type_t represents the collective operations supported by the
 *  UCC library. Currently, it supports barrier, broadcast, all-reduce, reduce,
 *  alltoall, all-gather, gather, scatter, fan-in and fan-out operations.
 *
 *  @endparblock
 *
 */
typedef enum {
    UCC_COLL_TYPE_ALLGATHER          = UCC_BIT(0),
    UCC_COLL_TYPE_ALLGATHERV         = UCC_BIT(1),
    UCC_COLL_TYPE_ALLREDUCE          = UCC_BIT(2),
    UCC_COLL_TYPE_ALLTOALL           = UCC_BIT(3),
    UCC_COLL_TYPE_ALLTOALLV          = UCC_BIT(4),
    UCC_COLL_TYPE_BARRIER            = UCC_BIT(5),
    UCC_COLL_TYPE_BCAST              = UCC_BIT(6),
    UCC_COLL_TYPE_FANIN              = UCC_BIT(7),
    UCC_COLL_TYPE_FANOUT             = UCC_BIT(8),
    UCC_COLL_TYPE_GATHER             = UCC_BIT(9),
    UCC_COLL_TYPE_GATHERV            = UCC_BIT(10),
    UCC_COLL_TYPE_REDUCE             = UCC_BIT(11),
    UCC_COLL_TYPE_REDUCE_SCATTER     = UCC_BIT(12),
    UCC_COLL_TYPE_REDUCE_SCATTERV    = UCC_BIT(13),
    UCC_COLL_TYPE_SCATTER            = UCC_BIT(14),
    UCC_COLL_TYPE_SCATTERV           = UCC_BIT(15)
} ucc_coll_type_t;

/**
 *
 *  @ingroup UCC_LIB_INIT_DT
 *
 *  @brief Enumeration representing the UCC library's datatype
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_datatype_t represents the datatypes supported by the UCC library’s
 *  collective and reduction operations. The standard operations are signed and
 *  unsigned integers of various sizes, float 16, 32, and 64, and user-defined
 *  datatypes. The UCC_DT_USERDEFINED represents the user-defined datatype. The
 *  UCC_DT_OPAQUE is used to represent the user-defined datatypes for
 *  user-defined reductions. When UCC_DT_OPAQUE is used, the library passes the
 *  data to the user-defined reductions without any modifications.
 *
 *  @endparblock
 *
 */
typedef enum {
    UCC_DT_INT8           = 0,
    UCC_DT_INT16,
    UCC_DT_INT32,
    UCC_DT_INT64,
    UCC_DT_INT128,
    UCC_DT_UINT8,
    UCC_DT_UINT16,
    UCC_DT_UINT32,
    UCC_DT_UINT64,
    UCC_DT_UINT128,
    UCC_DT_FLOAT16,
    UCC_DT_FLOAT32,
    UCC_DT_FLOAT64,
    UCC_DT_USERDEFINED,
    UCC_DT_OPAQUE
} ucc_datatype_t;

/**
 *
 *  @ingroup UCC_LIB_INIT_DT
 *
 *  @brief Enumeration representing the UCC library's thread model
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_thread_mode_t is used to initialize the UCC library’s thread mode.
 *  The UCC library can be configured in three thread modes UCC_THREAD_SINGLE,
 *  UCC_THREAD_FUNNELED, and UCC_LIB_THREAD_MULTIPLE. In the UCC_THREAD_SINGLE
 *  mode, the user program must not be multithreaded. In the UCC_THREAD_FUNNELED
 *  mode, the user program may be multithreaded. However, all UCC interfaces
 *  should be invoked from the same thread. In the UCC_THREAD_MULTIPLE mode, the
 *  user program can be multithreaded and any thread may invoke the UCC
 *  operations.
 *
 *  @endparblock
 *
 */
typedef enum {
    UCC_THREAD_SINGLE       = 0, /*!< Single-threaded library model */
    UCC_THREAD_FUNNELED     = 1, /*!< Funnel thread model */
    UCC_THREAD_MULTIPLE     = 2  /*!< Multithread library model */
} ucc_thread_mode_t;

/**
 *
 *  @ingroup UCC_LIB_INIT_DT
 *
 *  @brief Enumeration representing the collective synchronization model
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_coll_sync_type_t represents the collective synchronization models.
 *  Currently, it supports two synchronization models synchronous and
 *  non-synchronous collective models. In the synchronous collective model, the
 *  collective communication is not started until participants have not entered
 *  the collective operation, and it is not completed until all participants have not
 *  completed the collective. In the non-synchronous collective model, collective
 *  communication can be started as soon as the participant enters the collective
 *  operation and is completed as soon as it completes locally.
 *
 *  @endparblock
 *
 */
typedef enum {
    UCC_NO_SYNC_COLLECTIVES = 0, /*!< Synchornous collectives */
    UCC_SYNC_COLLECTIVES    = 1  /*!< Non-synchronous collectives */
} ucc_coll_sync_type_t;


/**
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief The reduction wrapper provides an interface for the UCC library to invoke
 *         user-defined custom reduction callback
 *
 *  @param [in]  invec          The input elements to be reduced by the user function
 *  @param [in]  inoutvec       The input elements to be reduced and output of the reduction
 *  @param [in]  count          The number of elements of type "dtype" to be reduced
 *  @param [in]  dtype          Datatype specified in the coll_args
 *  @param [in]  custom_op      A pointer to the user defined reduction passed to the coll_args as custom_reduction_op
 *
 *
 *  @parblock
 *
 *  @b Description
 *
 *  This function is called by the UCC library when it needs to perform a non-standard
 *  user-defined reduction operaion during allreduce/reduce collective.
 *
 *  @endparblock
 */
typedef void(*ucc_reduction_wrapper_t)(void *invec, void *inoutvec,
                                       ucc_count_t *count, void *dtype,
                                       void *custom_reduction_op);

/**
 * @brief UCC library initialization parameters
 */

/**
 *
 *  @ingroup UCC_LIB_INIT_DT
 */
enum ucc_lib_params_field{
    UCC_LIB_PARAM_FIELD_THREAD_MODE         = UCC_BIT(0),
    UCC_LIB_PARAM_FIELD_COLL_TYPES          = UCC_BIT(1),
    UCC_LIB_PARAM_FIELD_REDUCTION_TYPES     = UCC_BIT(2),
    UCC_LIB_PARAM_FIELD_SYNC_TYPE           = UCC_BIT(3),
    UCC_LIB_PARAM_FIELD_REDUCTION_WRAPPER   = UCC_BIT(4)
};

/**
 *
 *  @ingroup UCC_LIB_INIT_DT
 */
enum ucc_lib_attr_field{
    UCC_LIB_ATTR_FIELD_THREAD_MODE       = UCC_BIT(0),
    UCC_LIB_ATTR_FIELD_COLL_TYPES        = UCC_BIT(1),
    UCC_LIB_ATTR_FIELD_REDUCTION_TYPES   = UCC_BIT(2),
    UCC_LIB_ATTR_FIELD_SYNC_TYPE         = UCC_BIT(3)
};

/**
 *
 *  @ingroup UCC_LIB_INIT_DT
 *
 *  @brief Structure representing the parameters to customize the library
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_lib_params_t defines the parameters that can be used to customize
 *  the library. The bits in "mask" bit array is defined by @ref
 *  ucc_lib_params_field, which correspond to fields in structure @ref
 *  ucc_lib_params_t. The valid fields of the structure is specified by the
 *  setting the bit to "1" in the bit-array "mask". When bits corresponding to
 *  the fields is not set, the fields are not defined.
 *
 *  @endparblock
 *
 */
typedef struct ucc_lib_params {
    uint64_t                mask;
    ucc_thread_mode_t       thread_mode;
    uint64_t                coll_types;
    uint64_t                reduction_types;
    ucc_coll_sync_type_t    sync_type;
    ucc_reduction_wrapper_t reduction_wrapper;
} ucc_lib_params_t;

/**
 *
 *  @ingroup UCC_LIB_INIT_DT
 *
 *  @brief Structure representing the attributes of the library
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_lib_attr_t defines the attributes of the library. The bits in
 *  "mask" bit array is defined by @ref ucc_lib_attr_field, which correspond to
 *  fields in structure @ref ucc_lib_attr_t. The valid fields of the structure
 *  is specified by the setting the bit to "1" in the bit-array "mask". When
 *  bits corresponding to the fields is not set, the fields are not defined.
 *
 *  @endparblock
 *
 */
typedef struct ucc_lib_attr {
    uint64_t                mask;
    ucc_thread_mode_t       thread_mode;
    uint64_t                coll_types;
    uint64_t                reduction_types;
    ucc_coll_sync_type_t    sync_type;
} ucc_lib_attr_t;


/**
 *  @ingroup UCC_LIB
 *
 *  @brief The @ref ucc_lib_config_read routine provides a method to read library
 *  configuration from the environment and create configuration descriptor.
 *
 *  @param [out] env_prefix    If not NULL, the routine searches for the
 *                             environment variables with the prefix UCC_<env_prefix>. Otherwise, the
 *                             routines search for the environment variables that start with the prefix @
 *                             UCC_.
 *  @param [in]  filename      If not NULL, read configuration values from the
 *                             file defined by @e filename. If the file does not exist, it will be ignored
 *                             and no error will be reported to the user.
 *  @param [out] config        Pointer to configuration descriptor as defined by
 *                             ucc_lib_config_h.
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_lib_config_read allocates the @ref ucc_lib_config_h handle and
 *  fetches the configuration values from the run-time environment. The run-time
 *  environment supported are environment variables or a configuration file.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */

ucc_status_t ucc_lib_config_read(const char *env_prefix, const char *filename,
                                 ucc_lib_config_h *config);

/**
 *  @ingroup UCC_LIB
 *
 *  @brief The @ref ucc_lib_config_release routine releases the configuration descriptor
 *
 *  @param [in] config        Pointer to the configuration descriptor to be released.
 *                            Configuration descriptor as defined by @ref
 *                            ucc_lib_config_h.
 *
 *  @parblock
 *
 *  @b Description
 *
 *  The routine releases the configuration descriptor that was allocated through
 *
 *  @endparblock
 *
 *  @ref ucc_lib_config_read "ucc_lib_config_read()" routine.
 *
 */

void ucc_lib_config_release(ucc_lib_config_h config);

/**
 *  @ingroup UCC_LIB
 *
 *  @brief The @ref ucc_lib_config_print routine prints the configuration information
 *
 *  @param [in]  config        ucc_lib_config_h "Configuration descriptor"
 *                             to print.
 *  @param [in]  stream        Output stream to print the configuration to.
 *  @param [in]  title         Configuration title to print.
 *  @param [in]  print_flags   Flags that control various printing options.
 *
 *  @parblock
 *
 *  @b Description
 *
 *  The routine prints the configuration information that is stored in
 *  ucc_lib_config_h "configuration" descriptor.
 *
 *  @endparblock
 *
 */

void ucc_lib_config_print(const ucc_lib_config_h config, FILE *stream,
                          const char *title, ucc_config_print_flags_t print_flags);

/**
 *  @ingroup UCC_LIB
 *
 *  @brief The @ref ucc_lib_config_modify routine modifies the runtime configuration as described by the
 *  descriptor.
 *
 *  @param [in] config   Pointer to the configuration descriptor to be modified
 *  @param [in] name     Configuration variable to be modified
 *  @param [in] value    Configuration value to set
 *
 *  @parblock
 *
 *  @b Description
 *
 *  The @ref ucc_lib_config_modify routine sets the value of identifier "name"
 *  to "value".
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */

ucc_status_t ucc_lib_config_modify(ucc_lib_config_h config, const char *name,
                                   const char *value);


/**
 *  @ingroup UCC_LIB_INTERNAL
 *
 *  @brief The @ref ucc_init_version is an internal routine that checks
 *  compatibility with a particular UCC API version.
 *  @ref ucc_init should be used to create the UCC library handle.
 */
ucc_status_t ucc_init_version(unsigned api_major_version,
                              unsigned api_minor_version,
                              const ucc_lib_params_t *params,
                              const ucc_lib_config_h config,
                              ucc_lib_h *lib_p);


/**
 *  @ingroup UCC_LIB
 *
 *  @brief The @ref ucc_init initializes the UCC library.
 *
 *
 *  @param [in]  params    user provided parameters to customize the library functionality
 *  @param [in]  config    UCC configuration descriptor allocated through
 *                         @ref ucc_lib_config_read "ucc_config_read()" routine.
 *  @param [out] lib_p     UCC library handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  A local operation to initialize and allocate the resources for the UCC
 *  operations. The parameters passed using the ucc_lib_params_t and
 *  @ref ucc_lib_config_h structures will customize and select the functionality of the
 *  UCC library. The library can be customized for its interaction with the user
 *  threads, types of collective operations, and reductions supported.
 *  On success, the library object will be created and ucc_status_t will return
 *  UCC_OK. On error, the library object will not be created and corresponding
 *  error code as defined by ucc_status_t is returned.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
static inline ucc_status_t ucc_init(const ucc_lib_params_t *params,
                                    const ucc_lib_config_h config,
                                    ucc_lib_h *lib_p)
{
    return ucc_init_version(UCC_API_MAJOR, UCC_API_MINOR, params, config,
                            lib_p);
}


/**
 *  @ingroup UCC_LIB
 *
 *  @brief The @ref ucc_finalize routine finalizes the UCC library.
 *
 *  @param [in] lib_p   Handle to ucc_lib_h
 *                      "UCC library".
 *
 *  @parblock
 *
 *  @b Description
 *
 *  A local operation to release the resources and
 *  cleanup. All participants that invoked @ref ucc_init should call this
 *  routine.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */

ucc_status_t ucc_finalize(ucc_lib_h lib_p);


/**
 *  @ingroup UCC_LIB
 *
 *  @brief The @ref ucc_lib_get_attr routine queries the library attributes.
 *
 *  @param [out]  lib_attr   Library attributes
 *  @param [in]   lib_p      Input library object
 *
 *  @parblock
 *
 *  @b Description
 *
 *  A query operation to get the attributes of the library object. The
 *  attributes are library configured values and reflect the choices made by the
 *  library implementation.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */

ucc_status_t ucc_lib_get_attr(ucc_lib_h lib_p, ucc_lib_attr_t *lib_attr);

/*
 * *************************************************************
 *                   Context Section
 * *************************************************************
 */

/**
 *
 *  @ingroup UCC_CONTEXT_DT
 */
typedef enum {
   UCC_CONTEXT_EXCLUSIVE = 0 ,
   UCC_CONTEXT_SHARED
} ucc_context_type_t;

/**
 *
 *  @ingroup UCC_CONTEXT_DT
 */
enum ucc_context_params_field {
    UCC_CONTEXT_PARAM_FIELD_TYPE                   = UCC_BIT(0),
    UCC_CONTEXT_PARAM_FIELD_COLL_SYNC_TYPE         = UCC_BIT(1),
    UCC_CONTEXT_PARAM_FIELD_COLL_OOB               = UCC_BIT(2),
    UCC_CONTEXT_PARAM_FIELD_ID                     = UCC_BIT(3)
};

/**
 *
 *  @ingroup UCC_CONTEXT_DT
 */
enum ucc_context_attr_field {
    UCC_CONTEXT_ATTR_FIELD_TYPE                   = UCC_BIT(0),
    UCC_CONTEXT_ATTR_FIELD_COLL_SYNC_TYPE         = UCC_BIT(1),
    UCC_CONTEXT_ATTR_FIELD_CONTEXT_ADDR           = UCC_BIT(2),
    UCC_CONTEXT_ATTR_FIELD_CONTEXT_ADDR_LEN       = UCC_BIT(3)
};

/**
 * @ingroup UCC_CONTEXT_DT
 *
 * @brief OOB collective operation for creating the context
 */
typedef struct ucc_context_oob_coll {
    int             (*allgather)(void *src_buf, void *recv_buf, size_t size,
                                 void *allgather_info,  void **request);
    ucc_status_t    (*req_test)(void *request);
    ucc_status_t    (*req_free)(void *request);
    uint32_t        participants;
    void            *coll_info;
}  ucc_context_oob_coll_t;

/**
 *
 *  @ingroup UCC_CONTEXT_DT
 *
 *  @brief Structure representing the parameters to customize the context
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_context_params_t defines the parameters that can be used to
 *  customize the context. The "mask" bit array fields are defined by @ref
 *  ucc_context_params_field. The bits in "mask" bit array is defined by @ref
 *  ucc_context_params_field, which correspond to fields in structure @ref
 *  ucc_context_params_t. The valid fields of the structure is specified by the
 *  setting the bit to "1" in the bit-array "mask". When bits corresponding to
 *  the fields is not set, the fields are not defined.
 *
 *
 *  @endparblock
 *
 */
typedef struct ucc_context_params {
    uint64_t                mask;
    ucc_context_type_t      ctx_type;
    ucc_coll_sync_type_t    sync_type;
    ucc_context_oob_coll_t  oob;
    uint64_t                ctx_id;
} ucc_context_params_t;

/**
 *
 *  @ingroup UCC_CONTEXT_DT
 *
 *  @brief Structure representing context attributes
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_context_attr_t defines the attributes of the context. The bits in
 *  "mask" bit array is defined by @ref ucc_context_attr_field, which correspond to
 *  fields in structure @ref ucc_context_attr_t. The valid fields of the structure
 *  is specified by the setting the bit to "1" in the bit-array "mask". When
 *  bits corresponding to the fields is not set, the fields are not defined.
 *
 *  @endparblock
 *
 */
typedef struct ucc_context_attr {
    uint64_t                mask;
    ucc_context_type_t      ctx_type;
    ucc_coll_sync_type_t    sync_type;
    ucc_context_addr_t      ctx_addr;
    ucc_context_addr_len_t  ctx_addr_len;
} ucc_context_attr_t;

/**
 *  @ingroup UCC_CONTEXT
 *
 *  @brief Routine reads the configuration information for contexts from the
 *  runtime enviornment and creates the configuration descriptor.
 *
 *  @param [in]  lib_handle    Library handle
 *  @param [in]  filename      If not NULL, read configuration values from the
 *                             file defined by @e filename. If the file does not exist,
 *                             it will be ignored and no error will be reported to the user.
 *  @param [out] config        Pointer to configuration descriptor as defined by
 *                             @ref ucc_context_config_h.
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_context_config_read allocates the @ref ucc_lib_config_h handle and
 *  fetches the configuration values from the run-time environment. The run-time
 *  environment supported are environment variables or a configuration file. It uses
 *  the env_prefix from @ref ucc_lib_config_read. If env_prefix is not NULL, the routine
 *  searches for the environment variables with the prefix UCC_<env_prefix>. Otherwise, the
 *  routines search for the environment variables that start with the prefix @ UCC_.
 *
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */

ucc_status_t ucc_context_config_read(ucc_lib_h lib_handle,
                                     const char *filename,
                                     ucc_context_config_h *config);

/**
 *  @ingroup UCC_CONTEXT
 *
 *  @brief The @ref ucc_context_config_release routine releases the configuration descriptor.
 *
 *  @param [in] config        Pointer to the configuration descriptor to be released.
 *                            Configuration descriptor as defined by @ref ucc_context_config_h
 *
 *  @parblock
 *
 *  @b Description
 *
 *  The routine releases the configuration descriptor that was allocated through
 *  @ref ucc_context_config_read "ucc_context_config_read()" routine.
 *
 *  @endparblock
 *
 */

void ucc_context_config_release(ucc_context_config_h config);

/**
 *  @ingroup UCC_CONTEXT
 *
 *  @brief The @ref ucc_context_config_print routine prints the configuration information
 *
 *  @param [in]  config        ucc_context_config_h "Configuration descriptor"
 *                             to print.
 *  @param [in]  stream        Output stream to print the configuration to.
 *  @param [in]  title         Configuration title to print.
 *  @param [in]  print_flags   Flags that control various printing options.
 *
 *  @parblock
 *
 *  @b Description
 *
 *  The routine prints the configuration information that is stored in
 *  ucc_context_config_h "configuration" descriptor.
 *
 *  @endparblock
 *
 */

void ucc_context_config_print(const ucc_context_config_h config, FILE *stream,
                              const char *title, ucc_config_print_flags_t print_flags);

/**
 *  @ingroup UCC_CONTEXT
 *
 *  @brief The @ref ucc_context_config_modify routine modifies the runtime configuration
 *                  of UCC context (optionally for a given CLS)
 *
 *  @param [in] config   Pointer to the configuration descriptor to be modified
 *  @param [in] cls      Comma separated list of CLS or NULL. If NULL then core context config is modified.
 *  @param [in] name     Configuration variable to be modified
 *  @param [in] value    Configuration value to set
 *
 *  @parblock
 *
 *  @b Description
 *
 *  The @ref ucc_context_config_modify routine sets the value of identifier "name"
 *  to "value" for a specified CL.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */

ucc_status_t ucc_context_config_modify(ucc_context_config_h config,
                                       const char *cls, const char *name,
                                       const char *value);

/**
 *  @ingroup UCC_CONTEXT
 *
 *  @brief The @ref ucc_context_create routine creates the context handle.
 *
 *  @param [in]   lib_handle  Library handle
 *  @param [out]  params      Customizations for the communication context
 *  @param [out]  config      Configuration for the communication context to read
 *                            from environment
 *  @param [out]  context     Pointer to the newly created communication context
 *
 *  @parblock
 *
 *  @b Description
 *
 *  The ucc_context_create creates the context and ucc_context_destroy
 *  releases the resources and destroys the context state. The creation of context
 *  does not necessarily indicate its readiness to be used for collective or other
 *  group operations. On success, the context handle will be created and ucc_status_t will return
 *  UCC_OK. On error, the library object will not be created and corresponding
 *  error code as defined by ucc_status_t is returned.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */


ucc_status_t ucc_context_create(ucc_lib_h lib_handle,
                                const ucc_context_params_t *params,
                                const ucc_context_config_h  config,
                                ucc_context_h *context);

/**
 *  @ingroup UCC_CONTEXT
 *
 *  @brief The @ref ucc_context_progress routine progresses the operations
 *  on the context handle.
 *
 *  @param [in]  context  Communication context handle to be progressed
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_context_progress routine progresses the operations on the
 *  content handle. It does not block for lack of resources or communication.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */

ucc_status_t ucc_context_progress(ucc_context_h context);

/**
 *  @ingroup UCC_CONTEXT
 *
 *  @brief The @ref ucc_context_destroy routine frees the context handle.
 *
 *  @param [in]  context  Communication context handle to be released
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_context_destroy routine releases the resources associated
 *  with the handle @e context. All teams associated with the team should be
 *  released before this. It is invalid to associate any team with this handle
 *  after the routine is called.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */

ucc_status_t ucc_context_destroy(ucc_context_h context);

/**
 *  @ingroup UCC_CONTEXT
 *
 *  @brief The routine queries the attributes of the context handle.
 *
 *  @param [in]   context          Communication context
 *  @param [out]  context_attr     Attributes of the communication context
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @brief @ref ucc_context_get_attr routine queries the context handle
 *  attributes described by @ref ucc_context_attr.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */

ucc_status_t ucc_context_get_attr(ucc_context_h context,
                                  ucc_context_attr_t *context_attr);


/*
 * *************************************************************
 *                   Teams Section
 * *************************************************************
 */

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
enum ucc_team_params_field {
    UCC_TEAM_PARAM_FIELD_ORDERING               = UCC_BIT(0),
    UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS      = UCC_BIT(1),
    UCC_TEAM_PARAM_FIELD_EP                     = UCC_BIT(2),
    UCC_TEAM_PARAM_FIELD_EP_LIST                = UCC_BIT(3),
    UCC_TEAM_PARAM_FIELD_EP_RANGE               = UCC_BIT(4),
    UCC_TEAM_PARAM_FIELD_TEAM_SIZE              = UCC_BIT(5),
    UCC_TEAM_PARAM_FIELD_SYNC_TYPE              = UCC_BIT(6),
    UCC_TEAM_PARAM_FIELD_OOB                    = UCC_BIT(7),
    UCC_TEAM_PARAM_FIELD_P2P_CONN               = UCC_BIT(8),
    UCC_TEAM_PARAM_FIELD_MEM_PARAMS             = UCC_BIT(9),
    UCC_TEAM_PARAM_FIELD_EP_MAP                 = UCC_BIT(10)
};

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
enum ucc_team_attr_field {
    UCC_TEAM_ATTR_FIELD_POST_ORDERING          = UCC_BIT(0),
    UCC_TEAM_ATTR_FIELD_OUTSTANDING_CALLS      = UCC_BIT(1),
    UCC_TEAM_ATTR_FIELD_EP                     = UCC_BIT(2),
    UCC_TEAM_ATTR_FIELD_EP_RANGE               = UCC_BIT(3),
    UCC_TEAM_ATTR_FIELD_SYNC_TYPE              = UCC_BIT(4),
    UCC_TEAM_ATTR_FIELD_MEM_PARAMS             = UCC_BIT(5)
};

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
typedef enum  {
    UCC_MEM_CONSTRAINT_SYMMETRIC   = UCC_BIT(0),
    UCC_MEM_CONSTRAINT_PERSISTENT  = UCC_BIT(1),
    UCC_MEM_CONSTRAINT_ALIGN32     = UCC_BIT(2),
    UCC_MEM_CONSTRAINT_ALIGN64     = UCC_BIT(3),
    UCC_MEM_CONSTRAINT_ALIGN128    = UCC_BIT(4),
} ucc_mem_constraints_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
typedef enum {
    UCC_MEM_HINT_REMOTE_ATOMICS    = 0,
    UCC_MEM_HINT_REMOTE_COUNTERS
} ucc_mem_hints_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
typedef struct ucc_mem_map_params {
    void                    *address;
    size_t                  len;
    ucc_mem_hints_t         hints;
    ucc_mem_constraints_t   constraints;
} ucc_mem_map_params_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
typedef struct ucc_team_p2p_conn {
    int             (*conn_info_lookup)(void *conn_ctx, uint64_t ep, ucc_p2p_conn_t **conn_info, void *request);
    int             (*conn_info_release)(ucc_p2p_conn_t *conn_info);
    void            *conn_ctx;
    ucc_status_t    (*req_test)(void *request);
    ucc_status_t    (*req_free)(void *request);
} ucc_team_p2p_conn_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
typedef struct  ucc_team_oob_coll {
    ucc_status_t    (*allgather)(void *src_buf, void *recv_buf, size_t size,
                                 void *allgather_info,  void **request);
    ucc_status_t    (*req_test)(void *request);
    ucc_status_t    (*req_free)(void *request);
    uint32_t         participants;
    void             *coll_info;
}  ucc_team_oob_coll_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
typedef enum {
    UCC_COLLECTIVE_POST_ORDERED     = 0,
    UCC_COLLECTIVE_POST_UNORDERED   = 1
} ucc_post_ordering_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
typedef enum {
    UCC_COLLECTIVE_EP_RANGE_CONTIG      = 0,
    UCC_COLLECTIVE_EP_RANGE_NONCONTIG   = 1
} ucc_ep_range_type_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
struct ucc_ep_map_strided {
    uint64_t    start;
    uint64_t    stride;
};

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
struct ucc_ep_map_array {
    void    *map;
    size_t  elem_size; /*!< 4 if array is int, 8 if e.g. uint64_t */
};

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
struct ucc_ep_map_cb {
    uint64_t   (*cb)(uint64_t ep, void *cb_ctx);
    void       *cb_ctx;
};

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
typedef enum {
    UCC_EP_MAP_FULL     = 1, /*!< The ep range of the team  spans all eps from a context*/
    UCC_EP_MAP_STRIDED  = 2, /*!< The ep range of the team can be described by the 2 values: start, stride.*/
    UCC_EP_MAP_ARRAY    = 3, /*!< The ep range is given as an array of intergers that map the ep in the team to
                                       the team_context rank. */
    UCC_EP_MAP_CB       = 4, /*!< The ep range mapping is defined as callback provided by the UCC user. */
} ucc_ep_map_type_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 */
typedef struct ucc_ep_map_t {
    ucc_ep_map_type_t type;
    uint64_t          ep_num; /*!< number of eps mapped to ctx */
    union {
        struct ucc_ep_map_strided strided;
        struct ucc_ep_map_array   array;
        struct ucc_ep_map_cb      cb;
    };
} ucc_ep_map_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 *
 *  @brief Structure representing the parameters to customize the team
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_team_params_t defines the parameters that can be used to customize
 *  the team. The "mask" bit array fields are defined by @ref
 *  ucc_team_params_field. The bits in "mask" bit array is defined by @ref
 *  ucc_team_params_field, which correspond to fields in structure @ref
 *  ucc_team_params_t. The valid fields of the structure is specified by the
 *  setting the bit to "1" in the bit-array "mask". When bits corresponding to
 *  the fields is not set, the fields are not defined.
 *
 *
 *  @endparblock
 *
 */
typedef struct ucc_team_params {
    uint64_t                mask;
    ucc_post_ordering_t     ordering;
    uint64_t                outstanding_colls;
    uint64_t                ep;
    uint64_t                *ep_list;
    ucc_ep_range_type_t     ep_range;
    uint64_t                team_size;
    ucc_coll_sync_type_t    sync_type;
    ucc_team_oob_coll_t     oob;
    ucc_team_p2p_conn_t     p2p_conn;
    ucc_mem_map_params_t    mem_params;
    ucc_ep_map_t            ep_map;
} ucc_team_params_t;

/**
 *
 *  @ingroup UCC_TEAM_DT
 *
 *  @brief Structure representing the team attributes
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_team_attr_t defines the attributes of the team. The bits in
 *  "mask" bit array is defined by @ref ucc_team_attr_field, which correspond to
 *  fields in structure @ref ucc_team_attr_t. The valid fields of the structure
 *  is specified by the setting the bit to "1" in the bit-array "mask". When
 *  bits corresponding to the fields is not set, the fields are not defined.
 *
 *  @endparblock
 *
 */
typedef struct ucc_team_attr {
    uint64_t               mask;
    ucc_post_ordering_t    ordering;
    uint64_t               outstanding_colls;
    uint64_t               ep;
    ucc_ep_range_type_t    ep_range;
    ucc_coll_sync_type_t   sync_type;
    ucc_mem_map_params_t   mem_params;
} ucc_team_attr_t;


/**
 *  @ingroup UCC_TEAM
 *
 *  @brief The routine is a method to create the team.
 *
 *  @param  [in]  contexts           Communication contexts abstracting the resources
 *  @param  [in]  num_contexts       Number of contexts passed for the create operation
 *  @param  [in]  team_params        User defined configurations for the team
 *  @param  [out] new_team           Team handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_team_create_post is a nonblocking collective operation to create
 *  the team handle. It takes in parameters ucc_context_h and ucc_team_params_t.
 *  The ucc_team_params_t provides user configuration to customize the team and,
 *  ucc_context_h provides the resources for the team and collectives.
 *  The routine returns immediately after posting the operation with the
 *  new team handle. However, the team handle is not ready for posting
 *  the collective operation. ucc_team_create_test operation is used to learn
 *  the status of the new team handle. On error, the team handle will not
 *  be created and corresponding error code as defined by ucc_status_t is
 *  returned.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_team_create_post(ucc_context_h *contexts,
                                  uint32_t num_contexts,
                                  const ucc_team_params_t *team_params,
                                  ucc_team_h *new_team);


/**
 *  @ingroup UCC_TEAM
 *
 *  @brief The routine queries the status of the team creation operation.
 *
 *  @param  [in] team  Team handle to test
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_team_create_test routines tests the status of team handle.
 *  If required it can progress the communication but cannot block on the
 *  communications.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_team_create_test(ucc_team_h team);


/**
 *  @ingroup UCC_TEAM
 *
 *  @brief The team frees the team handle.
 *
 *  @param  [in] team  Destroy previously created team and release all resources
 *                     associated with it.
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_team_destroy is a blocking collective operation to release all
 *  resources associated with the team handle, and destroy the team handle. It is
 *  invalid to post a collective operation after the ucc_team_destroy operation.
 *
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 *
 */
ucc_status_t ucc_team_destroy(ucc_team_h team);

/**
 *  @ingroup UCC_TEAM
 *
 *  @brief The routine returns the attributes of the team.
 *
 *  @param [in]   team        Team handle
 *  @param [out]  team_attr   Attributes of the team
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @brief @ref ucc_team_get_attr routine queries the team handle
 *  attributes. The attributes of the team handle are described by the team
 *  attributes @ref ucc_team_attr_t
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_team_get_attr(ucc_team_h team,
                               ucc_team_attr_t *team_attr);

/**
 *  @ingroup UCC_TEAM
 *
 *  @brief The routine creates a new team from the parent team.
 *
 *  @param [in]    my_ep          Endpoint of the process/thread calling the split operation
 *  @param [in]    parent_team    Parent team handle from which a new team handle is created
 *  @param [in]    included       Variable indicating whether a
 *                                process/thread participates in the newly created team;
 *                                value 1 indicates the participation and value 0 indicates
 *                                otherwise
 *  @param [out]   new_team       Pointer to the new team handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @brief ucc_team_create_from_parent is a nonblocking collective operation,
 *  which creates a new team from the parent team. If a participant intends to
 *  participate in the new team, it passes a TRUE value for the "included"
 *  parameter. Otherwise, it passes FALSE. The routine returns immediately after
 *  the post-operation. To learn the completion of the team create operation, the
 *  ucc_team_create_test operation is used.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_team_create_from_parent(uint64_t my_ep, uint32_t included,
                                         ucc_team_h parent_team,
                                         ucc_team_h *new_team);

/**
 *  @ingroup UCC_TEAM
 *
 *  @brief The routine returns the size of the team.
 *
 *  @param [in]   team      Team handle
 *  @param [out]  size      The size of team as number of endpoints
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_team_get_size routine queries the size of the team. It reflects the
 *  number of unique endpoints in the team.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_team_get_size(ucc_team_h team, uint32_t *size);


/**
 *  @ingroup UCC_TEAM
 *
 *  @brief The routine returns the endpoint of the calling participant.
 *
 *  @param [out]  ep          Endpoint of the participant calling the routine
 *  @param [in]   team        Team handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_team_get_my_ep routine queries and returns the endpoint of the
 *  participant invoking the interface.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_team_get_my_ep(ucc_team_h team, uint64_t *ep);

/**
 *  @ingroup UCC_TEAM
 *
 *  @brief The routine queries all endpoints associated with the team handle.
 *
 *  @param [out]     ep          List of endpoints
 *  @param [out]     num_eps     Number of endpoints
 *  @param [in]      team        Team handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_team_get_all_eps routine queries and returns all endpoints of all
 *  participants in the team.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_team_get_all_eps(ucc_team_h team, uint64_t **ep,
                                  uint64_t *num_eps);


/*
 * *************************************************************
 *                   Collectives Section
 * *************************************************************
 */

/**
 *  @ingroup UCC_COLLECTIVES_DT
 */
typedef enum {
    UCC_COLL_ARGS_FLAG_IN_PLACE             = UCC_BIT(0),
    UCC_COLL_ARGS_FLAG_PERSISTENT           = UCC_BIT(1),
    UCC_COLL_ARGS_FLAG_COUNT_64BIT          = UCC_BIT(2),
    UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT  = UCC_BIT(3)
} ucc_coll_args_flags_t;

/**
 *  @ingroup UCC_COLLECTIVES_DT
 */
typedef enum ucc_memory_type {
    UCC_MEMORY_TYPE_HOST,         /*!< Default system memory */
    UCC_MEMORY_TYPE_CUDA,         /*!< NVIDIA CUDA memory */
    UCC_MEMORY_TYPE_CUDA_MANAGED, /*!< NVIDIA CUDA managed memory */
    UCC_MEMORY_TYPE_ROCM,         /*!< AMD ROCM memory */
    UCC_MEMORY_TYPE_ROCM_MANAGED, /*!< AMD ROCM managed system memory */
    UCC_MEMORY_TYPE_LAST,
    UCC_MEMORY_TYPE_UNKNOWN = UCC_MEMORY_TYPE_LAST
} ucc_memory_type_t;

typedef struct ucc_coll_buffer_info_v {
    void             *buffer;
    ucc_count_t      *counts;
    ucc_aint_t       *displacements;
    ucc_datatype_t    datatype;
    ucc_memory_type_t mem_type;
} ucc_coll_buffer_info_v_t;

typedef struct ucc_coll_buffer_info {
    void             *buffer;
    ucc_count_t       count;
    ucc_datatype_t    datatype;
    ucc_memory_type_t mem_type;
} ucc_coll_buffer_info_t;

/**
 *  @ingroup UCC_COLLECTIVES_DT
 */
typedef enum {
    UCC_ERR_TYPE_LOCAL      = 0,
    UCC_ERR_TYPE_GLOBAL     = 1
} ucc_error_type_t;

/**
 *  @ingroup UCC_COLLECTIVES_DT
 */
enum ucc_coll_args_field {
    UCC_COLL_ARGS_FIELD_FLAGS                           = UCC_BIT(0),
    UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS           = UCC_BIT(1),
    UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS          = UCC_BIT(2),
    UCC_COLL_ARGS_FIELD_TAG                             = UCC_BIT(3),
};

/**
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief Structure representing arguments for the collective operations
 *
 *  @parblock
 *
 *  @b Description
 *  @n @n
 *  @ref ucc_coll_args_t defines the parameters that can be used to customize
 *  the collective operation. The "mask" bit array fields are defined by @ref
 *  ucc_coll_args_field. The bits in "mask" bit array is defined by @ref
 *  ucc_coll_args_field, which correspond to fields in structure @ref
 *  ucc_coll_args_t. The valid fields of the structure are specified by
 *  setting the corresponding bit to "1" in the bit-array "mask".
 *  @n @n
 *  The collective operation is selected by field "coll_type" which must be always
 *  set by user. If allreduce or *  reduce operation is selected, the type of
 *  reduction is selected by the field *  "predefined_reduction_op" or
 *  "custom_reduction_op". For unordered collective
 *  operations, the user-provided "tag" value orders the collective operation.
 *  For rooted collective operations such as reduce, scatter, gather, fan-in, and
 *  fan-out, the "root" field  must be provided by user and specify the participant
 *  endpoint value. The user
 *  can request either "local" or "global" error information using the
 *  "error_type" field.
 *
 *  @n @n
 *  Information about user buffers used for collective operation must be specified
 *  according to the "coll_type".
 *  @endparblock
 *
 */
typedef struct ucc_coll_args {
    uint64_t                        mask;
    ucc_coll_type_t                 coll_type; /*!< Type of collective operation */
    union {
        ucc_coll_buffer_info_t      info;   /*!< Buffer info for the collective */
        ucc_coll_buffer_info_v_t    info_v; /*!< Buffer info for the collective */
    } src;
    union {
        ucc_coll_buffer_info_t      info;   /*!< Buffer info for the collective */
        ucc_coll_buffer_info_v_t    info_v; /*!< Buffer info for the collective */
    } dst;
    struct {
        ucc_reduction_op_t          predefined_op; /*!< Reduction operation, if
                                                        reduce or all-reduce
                                                        operation selected */
        void                       *custom_op; /*!< User defined
                                                    reduction operation */
        void                       *custom_dtype;
    } reduce;
    uint64_t                        flags;
    uint64_t                        root; /*!< Root endpoint for rooted
                                             collectives */
    ucc_error_type_t                error_type; /*!< Error type */
    ucc_coll_id_t                   tag; /*!< Used for ordering collectives */
} ucc_coll_args_t;

/**
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief The routine to initialize a collective operation.
 *
 *  @param [out]   request     Request handle representing the collective operation
 *  @param [in]    coll_args   Collective arguments descriptor
 *  @param [in]    team        Team handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref  ucc_collective_init is a collective initialization operation,
 *  where all participants participate. The user provides all information
 *  required to start and complete the collective operation, which includes the
 *  input and output buffers, operation type, team handle, size, and any other
 *  hints for optimization. On success, the request handle is created and
 *  returned. On error, the request handle is not created and the appropriate
 *  error code is returned. On return, the ownership of buffers is transferred
 *  to the user. If modified, the results of collective operations posted on the
 *  request handle are undefined.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_collective_init(ucc_coll_args_t *coll_args,
                                 ucc_coll_req_h *request, ucc_team_h team);

/**
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief The routine to post a collective operation.
 *
 *  @param [in]     request     Request handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_collective_post routine posts the collective operation. It
 *  does not require synchronization between the participants for the post
 *  operation.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_collective_post(ucc_coll_req_h request);


/**
 *
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief The routine to initialize and post a collective operation.
 *
 *  @param [out]     request     Request handle representing the collective operation
 *  @param [in]      coll_args   Collective arguments descriptor
 *  @param [in]      team        Input Team
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_collective_init_and_post initializes the collective operation
 *  and also posts the operation.
 *
 *  @note: The @ref ucc_collective_init_and_post can be implemented as a
 *  combination of @ref ucc_collective_init and @ref ucc_collective_post
 *  routines.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_collective_init_and_post(ucc_coll_args_t *coll_args,
                                          ucc_coll_req_h *request,
                                          ucc_team_h team);

/**
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief The routine to query the status of the collective operation.
 *
 *  @param [in]  request Request handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_collective_test tests and returns the status of collective
 *  operation.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
static inline ucc_status_t ucc_collective_test(ucc_coll_req_h request)
{
    return request->status;
}

/**
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief The routine to release the collective operation associated with the request object.
 *
 *  @param [in] request - request handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_collective_finalize operation releases all resources
 *  associated with the collective operation represented by the request handle.
 *
 *  @endparblock
 *
 *  @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_collective_finalize(ucc_coll_req_h request);


/**
 * @ingroup UCC_EVENT_DT
 *
 */
typedef enum ucc_event_type {
	UCC_EVENT_COLLECTIVE_POST       = UCC_BIT(0),
	UCC_EVENT_COLLECTIVE_COMPLETE   = UCC_BIT(1),
	UCC_EVENT_COMPUTE_COMPLETE      = UCC_BIT(2),
	UCC_EVENT_OVERFLOW              = UCC_BIT(3)
} ucc_event_type_t;

/**
 * @ingroup UCC_EVENT_DT
 *
 */
typedef enum ucc_ee_type {
    UCC_CUDA_STREAM = UCC_BIT(0),
	UCC_CPU_THREAD  = UCC_BIT(1)
} ucc_ee_type_t;

/**
 * @ingroup UCC_EVENT_DT
 *
 */
typedef struct ucc_event_context {
    size_t              info_size;
    void                *info;
} ucc_event_context_t;

/**
 * @ingroup UCC_EVENT_DT
 *
 */
typedef struct ucc_event {
	ucc_event_type_t        ev_type;
	ucc_event_context_t     *event_context;
	ucc_coll_req_h          req;
} ucc_ev_t;

/**
 * @ingroup UCC_EVENT_DT
 *
 */
typedef struct ucc_ee_params {
     ucc_ee_type_t  ee_type;
     void           *ee_context;
     size_t         ee_context_size;
} ucc_ee_params_t;

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine creates the execution context for collective operations.
 *
 * @param [in] team     team handle
 * @param [in] params   user provided params to customize the execution engine
 * @param [out] ee      execution engine handle
 *
 * @parblock
 *
 * @b Description
 *
 * @endparblock
 *
 * @return Error code as defined by ucc_status_t
 */
 ucc_status_t ucc_ee_create(ucc_team_h team, const ucc_ee_params_t *params, ucc_ee_h *ee);

 /**
 * @ingroup UCC_EVENT
 *
 * @brief The routine destroys the execution context created for collective operations.
 *
 * @param [in] ee   Execution engine handle
 *
 * @parblock
 *
 * @b Description
 *
 * @endparblock
 *
 * @return Error code as defined by ucc_status_t
 */
 ucc_status_t ucc_ee_destroy(ucc_ee_h ee);

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine gets the event from the event queue.
 *
 * @param [in]  ee        execution engine handle
 * @param [out] ev        Event structure fetched from the event queue
 *
 * @parblock
 *
 * @b Description
 *
 * @endparblock
 *
 * @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_ee_get_event(ucc_ee_h ee, ucc_ev_t *ev);

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine acks the events from the event queue.
 *
 * @param [in]  ee      execution engine handle
 * @param [in]  ev      Event to be acked
 *
 * @parblock
 *
 * @b Description
 *
 * @endparblock
 *
 * @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_ee_ack_event(ucc_ee_h ee, ucc_ev_t ev);


/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine to set the event to the tail of the queue.
 *
 * @param [in]  ee        execution engine handle
 * @param [in]  ev        Event structure fetched from the event queue
 *
 * @parblock
 *
 * @b Description
 *
 * @endparblock
 *
 * @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_ee_set_event(ucc_ee_h ee, ucc_ev_t ev);

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine polls the event queue and fetches the event from the head of the queue.
 *
 * @param [in]  ee        execution engine handle
 * @param [out] ev        Event structure fetched from the event queue
 *
 * @parblock
 *
 * @b Description
 *
 * @endparblock
 *
 * @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_ee_poll(ucc_ee_h ee, ucc_ev_t *ev);

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine blocks the calling thread until there is an event on the queue.
 *
 * @param [in]  ee        execution engine handle
 * @param [out] ev        Event structure fetched from the event queue
 *
 * @parblock
 *
 * @b Description
 *
 * @endparblock
 *
 * @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_ee_wait(ucc_ee_h ee, ucc_ev_t *ev);


/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine posts the collective operation on the execution engine, which is
 * launched on the event.
 *
 * @param [in]  ee          execution engine handle
 * @param [in]  ee_event    Event triggering the post operation
 *
 * @parblock
 *
 * @b Description
 *
 * @endparblock
 *
 * @return Error code as defined by ucc_status_t
 */
ucc_status_t ucc_collective_triggered_post(ucc_ee_h ee, ucc_ev_t ee_event);

END_C_DECLS
#endif
