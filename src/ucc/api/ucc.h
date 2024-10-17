/**
 * @file ucc.h
 * @date 2020
 * @copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * @copyright Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
 * @copyright Copyright (C) UChicago Argonne, LLC. 2022.  ALL RIGHTS RESERVED.
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
  * @defgroup UCC_DATATYPE Datatypes data-structures and functions
  * @{
  * Datatypes data-structures and functions
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
  * @defgroup UCC_LIB_INTERNAL Internal library routines
  * @{
  * Internal library routines
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
  * @defgroup UCC_EVENT_DT Events and Triggered operations' data-structures
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
 *  @brief Enumeration representing the collective operations
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_coll_type_t represents the collective operations supported by the
 *  UCC library. The exact set of supported collective operations depends on
 *  UCC build flags, runtime configuration and available communication transports.
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
    UCC_COLL_TYPE_SCATTERV           = UCC_BIT(15),
    UCC_COLL_TYPE_LAST
} ucc_coll_type_t;

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

/**
 *
 *  @ingroup UCC_DATATYPE
 *
 *  @brief Enumeration representing the UCC library's datatype
 *
 *  @parblock
 *
 *  Description
 *
 *  @ref ucc_datatype_t represents the datatypes supported by the UCC library’s
 *  collective and reduction operations. The predefined operations
 *  are signed and unsigned integers of various sizes, float 16, 32, and 64, and
 *  user-defined datatypes. User-defined datatypes are created using
 *  @ref ucc_dt_create_generic interface and can support user-defined reduction
 *  operations. Predefined reduction operations can be used only with
 *  predefined datatypes.
 *
 *  @endparblock
 *
 */
typedef uint64_t ucc_datatype_t;

#define UCC_DT_INT8             UCC_PREDEFINED_DT(0)
#define UCC_DT_INT16            UCC_PREDEFINED_DT(1)
#define UCC_DT_INT32            UCC_PREDEFINED_DT(2)
#define UCC_DT_INT64            UCC_PREDEFINED_DT(3)
#define UCC_DT_INT128           UCC_PREDEFINED_DT(4)
#define UCC_DT_UINT8            UCC_PREDEFINED_DT(5)
#define UCC_DT_UINT16           UCC_PREDEFINED_DT(6)
#define UCC_DT_UINT32           UCC_PREDEFINED_DT(7)
#define UCC_DT_UINT64           UCC_PREDEFINED_DT(8)
#define UCC_DT_UINT128          UCC_PREDEFINED_DT(9)
#define UCC_DT_FLOAT16          UCC_PREDEFINED_DT(10)
#define UCC_DT_FLOAT32          UCC_PREDEFINED_DT(11)
#define UCC_DT_FLOAT64          UCC_PREDEFINED_DT(12)
#define UCC_DT_BFLOAT16         UCC_PREDEFINED_DT(13)
#define UCC_DT_FLOAT128         UCC_PREDEFINED_DT(14)
#define UCC_DT_FLOAT32_COMPLEX  UCC_PREDEFINED_DT(15)
#define UCC_DT_FLOAT64_COMPLEX  UCC_PREDEFINED_DT(16)
#define UCC_DT_FLOAT128_COMPLEX UCC_PREDEFINED_DT(17)
#define UCC_DT_PREDEFINED_LAST  18

/**
 * @ingroup UCC_DATATYPE
 */
enum ucc_generic_dt_ops_field {
    UCC_GENERIC_DT_OPS_FIELD_FLAGS             = UCC_BIT(0),
};

/**
 * @ingroup UCC_DATATYPE
 * @brief Flags that can be specified for generic datatype
 *
 */

typedef enum {
    UCC_GENERIC_DT_OPS_FLAG_CONTIG             = UCC_BIT(0), /*!< If set, the created datatype
                                                                  represents a contiguous memory
                                                                  region with the size specified
                                                                  in @ref ucc_generic_dt_ops.contig_size
                                                                  field of @ref ucc_generic_dt_ops */
    UCC_GENERIC_DT_OPS_FLAG_REDUCE             = UCC_BIT(1), /*!< If set, the created datatype
                                                                  has user-defined reduction
                                                                  operation associated with it.
                                                                  reduce.cb and reduce.ctx fields
                                                                  of @ref ucc_generic_dt_ops must
                                                                  be initialized. Collective operations
                                                                  that involve reduction (allreduce,
                                                                  reduce, reduce_scatter/v) can use
                                                                  user-defined data-types only when
                                                                  this flag is set. */
} ucc_generic_dt_ops_flags_t;

/**
 * @ingroup UCC_DATATYPE
 * @brief Descriptor of user-defined reduction callback
 *
 * This structure is the argument to the reduce.cb callback. It must implement
 * the reduction of n_vectors + 1 data vectors each containing "count" elements.
 * First vector is "src1", other n_vectors have start address
 * v_j = src2 + count * dt_extent * stride * j.
 * The result is stored in dst, so that
 * dst[i] = src1[i] + v0[i] + v1[i] + ... +v_nvectors[i],
 * for i in [0:count), where "+" represents user-defined reduction of 2 elements
 */

typedef struct ucc_reduce_cb_params {
    uint64_t          mask;      /*< for backward compatibility. currently ignored. */
    void             *src1;      /*< input buffer */
    void             *src2;      /*< input buffer: represents n_vectors buffers with
                                   offset "stride" between them */
    void             *dst;       /*< destination buffer */
    size_t            n_vectors; /*< number of vectors from src2 to reduce */
    size_t            count;     /*< number of elements in one vector */
    size_t            stride;    /*< stride in bytes between the vectors in src2 */
    ucc_dt_generic_t *dt;        /*< pointer to user-defined datatype used for
                                     reduction */
    void             *cb_ctx;    /*< user-defined context as defined
                                   by @ref ucc_generic_dt_ops::reduce.cb_ctx */
} ucc_reduce_cb_params_t;
/**
 * @ingroup UCC_DATATYPE
 * @brief UCC generic data type descriptor
 *
 * This structure provides a generic datatype descriptor that is used to create
 * user-defined datatypes.
 */

typedef struct ucc_generic_dt_ops {
    uint64_t mask;
    uint64_t flags;
    size_t   contig_size; /*!< size of the datatype if @ref UCC_GENERIC_DT_OPS_FLAG_CONTIG is set */
    /**
     * @ingroup UCC_DATATYPE
     * @brief Start a packing request.
     *
     * The pointer refers to application defined start-to-pack routine.
     *
     * @param [in]  context        User-defined context.
     * @param [in]  buffer         Buffer to pack.
     * @param [in]  count          Number of elements to pack into the buffer.
     *
     * @return  A custom state that is passed to the subsequent
     *          @ref ucc_generic_dt_ops::pack "pack()" routine.
     */
    void* (*start_pack)(void *context, const void *buffer, size_t count);

    /**
     * @ingroup UCC_DATATYPE
     * @brief Start an unpacking request.
     *
     * The pointer refers to application defined start-to-unpack routine.
     *
     * @param [in]  context        User-defined context.
     * @param [in]  buffer         Buffer to unpack to.
     * @param [in]  count          Number of elements to unpack in the buffer.
     *
     * @return  A custom state that is passed later to the subsequent
     *          @ref ucc_generic_dt_ops::unpack "unpack()" routine.
     */
    void* (*start_unpack)(void *context, void *buffer, size_t count);

    /**
     * @ingroup UCC_DATATYPE
     * @brief Get the total size of packed data.
     *
     * The pointer refers to user defined routine that returns the size of data
     * in a packed format.
     *
     * @param [in]  state          State as returned by
     *                             @ref ucc_generic_dt_ops::start_pack
     *                             "start_pack()" routine.
     *
     * @return  The size of the data in a packed form.
     */
    size_t (*packed_size)(void *state);

    /**
     * @ingroup UCC_DATATYPE
     * @brief Pack data.
     *
     * The pointer refers to application defined pack routine.
     *
     * @param [in]  state          State as returned by
     *                             @ref ucc_generic_dt_ops::start_pack
     *                             "start_pack()" routine.
     * @param [in]  offset         Virtual offset in the output stream.
     * @param [in]  dest           Destination buffer to pack the data.
     * @param [in]  max_length     Maximum length to pack.
     *
     * @return The size of the data that was written to the destination buffer.
     *         Must be less than or equal to @e max_length.
     */
    size_t (*pack) (void *state, size_t offset, void *dest, size_t max_length);

    /**
     * @ingroup UCC_DATATYPE
     * @brief Unpack data.
     *
     * The pointer refers to application defined unpack routine.
     *
     * @param [in]  state          State as returned by
     *                             @ref ucc_generic_dt_ops::start_unpack
     *                             "start_unpack()" routine.
     * @param [in]  offset         Virtual offset in the input stream.
     * @param [in]  src            Source to unpack the data from.
     * @param [in]  length         Length to unpack.
     *
     * @return UCC_OK or an error if unpacking failed.
     */
    ucc_status_t (*unpack)(void *state, size_t offset, const void *src, size_t length);

    /**
     * @ingroup UCC_DATATYPE
     * @brief Finish packing/unpacking.
     *
     * The pointer refers to application defined finish routine.
     *
     * @param [in]  state          State as returned by
     *                             @ref ucc_generic_dt_ops::start_pack
     *                             "start_pack()"
     *                             and
     *                             @ref ucc_generic_dt_ops::start_unpack
     *                             "start_unpack()"
     *                             routines.
     */
    void (*finish)(void *state);

    /**
     * @ingroup UCC_DATATYPE
     * @brief User-defined reduction callback
     *
     * The pointer refers to user-defined reduction routine.
     *
     * @param [in]  params reduction descriptor
     */

    struct {
        ucc_status_t (*cb)(const ucc_reduce_cb_params_t *params);
        void *cb_ctx;
    } reduce;
} ucc_generic_dt_ops_t;


/**
 * @ingroup UCC_DATATYPE
 * @brief Create a generic datatype.
 *
 * This routine creates a generic datatype object. The generic datatype is
 * described by the @a ops @ref ucc_generic_dt_ops_t "object" which provides
 * a table of routines defining the operations for generic datatype manipulation.
 * Typically, generic datatypes are used for integration with datatype engines
 * provided with MPI implementations (MPICH, Open MPI, etc). The application
 * is responsible for releasing the @a datatype_p  object using
 * @ref ucc_dt_destroy "ucc_dt_destroy()" routine.
 *
 * @param [in]  ops          Generic datatype function table as defined by
 *                           @ref ucc_generic_dt_ops_t .
 * @param [in]  context      Application defined context passed to this
 *                           routine.  The context is passed as a parameter
 *                           to the routines in the @a ops table.
 * @param [out] datatype_p   A pointer to datatype object.
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_dt_create_generic(const ucc_generic_dt_ops_t *ops, void *context,
                                   ucc_datatype_t *datatype_p);

/**
 * @ingroup UCC_DATATYPE
 * @brief Destroy generic datatype
 */
void ucc_dt_destroy(ucc_datatype_t datatype);

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
 *  the library.
 *
 *  @endparblock
 *
 */
typedef enum {
    UCC_OP_SUM,
    UCC_OP_PROD,
    UCC_OP_MAX,
    UCC_OP_MIN,
    UCC_OP_LAND,
    UCC_OP_LOR,
    UCC_OP_LXOR,
    UCC_OP_BAND,
    UCC_OP_BOR,
    UCC_OP_BXOR,
    UCC_OP_MAXLOC,
    UCC_OP_MINLOC,
    UCC_OP_AVG,
    UCC_OP_LAST
} ucc_reduction_op_t;

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
 *  UCC_THREAD_FUNNELED, and UCC_THREAD_MULTIPLE. In the UCC_THREAD_SINGLE
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
    UCC_NO_SYNC_COLLECTIVES = 0, /*!< Non-synchronous collectives */
    UCC_SYNC_COLLECTIVES    = 1  /*!< Synchronous collectives */
} ucc_coll_sync_type_t;


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
    UCC_LIB_PARAM_FIELD_SYNC_TYPE           = UCC_BIT(3)
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
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_lib_config_modify(ucc_lib_config_h config, const char *name,
                                   const char *value);

/**
 * @ingroup UCC_LIB
 * @brief Get UCC library version.
 *
 * This routine returns the UCC library version.
 *
 * @param [out] major_version       Filled with library major version.
 * @param [out] minor_version       Filled with library minor version.
 * @param [out] release_number      Filled with library release number.
 */
void ucc_get_version(unsigned *major_version, unsigned *minor_version,
                     unsigned *release_number);

/**
 * @ingroup UCC_LIB
 * @brief Get UCC library version as a string.
 *
 * This routine returns the UCC library version as a string which consists of:
 * "major.minor.release".
 */
const char *ucc_get_version_string(void);


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
 *  @param [in]  params    User provided parameters to customize the library functionality
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
 *  error code as defined by @ref ucc_status_t is returned.
 *
 *  @endparblock
 *
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @return Error code as defined by @ref ucc_status_t
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
    UCC_CONTEXT_PARAM_FIELD_TYPE              = UCC_BIT(0),
    UCC_CONTEXT_PARAM_FIELD_SYNC_TYPE         = UCC_BIT(1),
    UCC_CONTEXT_PARAM_FIELD_OOB               = UCC_BIT(2),
    UCC_CONTEXT_PARAM_FIELD_ID                = UCC_BIT(3),
    UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS        = UCC_BIT(4)
};

/**
 *
 *  @ingroup UCC_CONTEXT_DT
 */
enum ucc_context_attr_field {
    UCC_CONTEXT_ATTR_FIELD_TYPE               = UCC_BIT(0),
    UCC_CONTEXT_ATTR_FIELD_SYNC_TYPE          = UCC_BIT(1),
    UCC_CONTEXT_ATTR_FIELD_CTX_ADDR           = UCC_BIT(2),
    UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN       = UCC_BIT(3),
    UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE   = UCC_BIT(4)
};

/**
 * @ingroup UCC_CONTEXT_DT
 *
 * @brief OOB collective operation for creating the context
 */
typedef struct ucc_oob_coll {
    ucc_status_t    (*allgather)(void *src_buf, void *recv_buf, size_t size,
                                 void *allgather_info,  void **request);
    ucc_status_t    (*req_test)(void *request);
    ucc_status_t    (*req_free)(void *request);
    void            *coll_info;
    uint32_t        n_oob_eps; /*!< Number of endpoints participating in the
                                    oob operation (e.g., number of processes
                                    representing a ucc team) */
    uint32_t        oob_ep; /*!< Integer value that represents the position
                                 of the calling processes in the given oob op:
                                 the data specified by "src_buf" will be placed
                                 at the offset "oob_ep*size" in the "recv_buf".
                                 oob_ep must be uniq at every calling process
                                 and should be in the range [0:n_oob_eps). */

}  ucc_oob_coll_t;

typedef ucc_oob_coll_t ucc_context_oob_coll_t;
typedef ucc_oob_coll_t ucc_team_oob_coll_t;

/**
 *
 *  @ingroup UCC_CONTEXT_DT
 */
typedef struct ucc_mem_map {
    void *   address; /*!< the address of a buffer to be attached to a UCC context */
    size_t   len;     /*!< the length of the buffer */
} ucc_mem_map_t;

/**
 *
 * @ingroup UCC_CONTEXT_DT
 */
typedef struct ucc_mem_map_params {
    ucc_mem_map_t *segments;   /*!< array of ucc_mem_map elements */
    uint64_t       n_segments; /*!< the number of ucc_mem_map elements */
} ucc_mem_map_params_t;

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
    ucc_context_type_t      type;
    ucc_coll_sync_type_t    sync_type;
    ucc_context_oob_coll_t  oob;
    uint64_t                ctx_id;
    ucc_mem_map_params_t    mem_params;
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
    ucc_context_type_t      type;
    ucc_coll_sync_type_t    sync_type;
    ucc_context_addr_h      ctx_addr;
    ucc_context_addr_len_t  ctx_addr_len;
    uint64_t                global_work_buffer_size;
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
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @param [in] config    Pointer to the configuration descriptor to be modified
 *  @param [in] component CL/TL component (e.g. "tl/ucp" or "cl/basic") or NULL.
                          If NULL then core context config is modified.
 *  @param [in] name      Configuration variable to be modified
 *  @param [in] value     Configuration value to set
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
 *  @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_context_config_modify(ucc_context_config_h config,
                                       const char *component, const char *name,
                                       const char *value);

/**
 *  @ingroup UCC_CONTEXT
 *
 *  @brief The @ref ucc_context_create routine creates the context handle.
 *
 *  @param [in]   lib_handle  Library handle
 *  @param [in]   params      Customizations for the communication context
 *  @param [in]   config      Configuration for the communication context to read
 *                            from environment
 *  @param [out]  context     Pointer to the newly created communication context
 *
 *  @parblock
 *
 *  @b Description
 *
 *  The @ref ucc_context_create creates the context and @ref ucc_context_destroy
 *  releases the resources and destroys the context state. The creation of
 *  context does not necessarily indicate its readiness to be used for
 *  collective or other group operations. On success, the context handle will be
 *  created and ucc_status_t will return UCC_OK. On error, the context object
 *  will not be created and corresponding error code as defined by
 *  @ref ucc_status_t is returned.
 *
 *  @endparblock
 *
 *  @return Error code as defined by @ref ucc_status_t
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
 *
 *  @parblock
 *
 *  @b Description
 *
 *  The @ref ucc_context_progress routine progresses the operations on the
 *  content handle. It does not block for lack of resources or communication.
 *
 *  @endparblock
 *
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @return Error code as defined by @ref ucc_status_t
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
    UCC_TEAM_PARAM_FIELD_EP_MAP                 = UCC_BIT(10),
    UCC_TEAM_PARAM_FIELD_ID                     = UCC_BIT(11),
    UCC_TEAM_PARAM_FIELD_FLAGS                  = UCC_BIT(12)
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
    UCC_TEAM_ATTR_FIELD_MEM_PARAMS             = UCC_BIT(5),
    UCC_TEAM_ATTR_FIELD_SIZE                   = UCC_BIT(6),
    UCC_TEAM_ATTR_FIELD_EPS                    = UCC_BIT(7)
};

/**
 *
 * @ingroup UCC_TEAM_DT
 */
enum ucc_team_flags {
    UCC_TEAM_FLAG_COLL_WORK_BUFFER             = UCC_BIT(0) /*< If set, this indicates
                                                                the user will provide
                                                                a scratchpad buffer for
                                                                use in one-sided
                                                                collectives. Otherwise,
                                                                an internal buffer will
                                                                used. */
};

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
typedef enum {

    /**
      * When set to this value, the collective participants shall post the operation
      * in the same order.
      */
    UCC_COLLECTIVE_POST_ORDERED             = 0,

    /**
      * When set to this value, the collective participants shall post the operation
      * in any order.
      */
    UCC_COLLECTIVE_POST_UNORDERED           = 1,

    /**
      * When set to this value, the collective participants shall initialize the operation
      * in the same order.
      */
    UCC_COLLECTIVE_INIT_ORDERED             = 2,

    /**
      * When set to this value, the collective participants shall initialize the operation
      * in any order.
      */
    UCC_COLLECTIVE_INIT_UNORDERED           = 3,

    /**
      * When set to this value, the collective participants shall initialize and
      * post the operation in the same order.
      */
    UCC_COLLECTIVE_INIT_AND_POST_ORDERED    = 4,

    /**
      * When set to this value, the collective participants shall initialize and
      * post the operation in any order.
      */
    UCC_COLLECTIVE_INIT_AND_POST_UNORDERED  = 5
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
    int64_t     stride;
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
    UCC_EP_MAP_FULL     = 1, /*!< The ep range of the team spans all eps from a context. */
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
 *
 *  @endparblock
 *
 */
typedef struct ucc_team_params {

    uint64_t                mask;
    uint64_t                flags;
    /** @ref ucc_team_params.ordering is set to one the values defined by @ref
      *  ucc_post_ordering_t
      */
    ucc_post_ordering_t     ordering;

    /** @ref ucc_team_params.outstanding_colls represents the number of outstanding non-blocking
      * calls the user expects to post to the team. If the user posts more non-blocking
      * calls than set, the behavior is undefined. If not set, there is no limit on
      * the number of outstanding calls to be posted.
      */
    uint64_t                outstanding_colls;

    /** @ref ucc_team_params.ep The endpoint is a non-negative unique integer identifying the
      * participant in the collective. If ep is not set, and @ref ucc_team_params.oob is not set, the
      * library generates the ep. The generated ep can be queried using the @ref
      * ucc_team_get_attr interface.
      */
    uint64_t                ep;

    /** @ref ucc_team_params.ep_list The endpoint list provides the list of eps participating to
     *  create the team.
     */
    uint64_t                *ep_list;

    /** @ref ucc_team_params.ep_range can be either contiguous or not
     *  contiguous. It is a hint to the library.
     */
    ucc_ep_range_type_t     ep_range;

    /** @ref ucc_team_params.team_size The team size is the number of participants in the team. If
      * @ref ucc_team_params.oob is provided, the team size and @ref
      * ucc_oob_coll.n_oob_eps should be the same.
      */
    uint64_t                team_size;

    /**
      * @ref ucc_team_params.sync_type The options for sync_type are provided by @ref
      * ucc_coll_sync_type_t
      */
    ucc_coll_sync_type_t    sync_type;

   /** @ref ucc_team_params.oob The signature of the function is defined by @ref
     * ucc_oob_coll_t
     * . The oob is used for exchanging information between the team
     * participants during team creation. The user is
     * responsible for implementing the oob operation. The relation between @ref
     * ucc_team_params.ep and @ref ucc_oob_coll.oob_ep is defined as below:
     *
     * - When both are not provided. The library is responsible for generating the ep,
     * which can be then queried via the @ref ucc_team_get_attr interface. This
     * requires, however, ucc_params_t ep_map to be set and context created by
     * @ref ucc_oob_coll. The behavior is undefined, when neither @ref
     * ucc_team_params.ep or @ref ucc_team_params.ep_map, or @ref
     * ucc_team_params.oob is not set.
     *
     * - When @ref ucc_team_params.ep is provided and @ref ucc_team_params.oob is
     * not provided. The “ep” is the unique integer for the participant.
     *
     * - When @ref ucc_oob_coll.oob_ep is provided and @ref ucc_team_params.ep
     * is not provided. The “ep” will be equivalent to @ref ucc_oob_coll.oob_ep.
     *
     * - When both are provided, the @ref ucc_oob_coll.oob_ep and @ref
     * ucc_team_params_t.ep should be same. Otherwise, it
     * is undefined.
     */
    ucc_team_oob_coll_t     oob;

    /** @ref ucc_team_params.p2p_conn is a callback function for the gathering
      * the point-to-point communication information.
      */
    ucc_team_p2p_conn_t     p2p_conn;

    /** @ref ucc_team_params.mem_params provides an ability to attach a buffer
      * to the team. This can be used as input/output or control buffer for the
      * team. Typically, it can be useful for one-sided collective
      * implementation.
      */
    ucc_mem_map_params_t    mem_params;

    /** @ref ucc_team_params.ep_map provides a mapping between @ref
      * ucc_oob_coll.oob_ep used by
      * the team and @ref ucc_oob_coll.oob_ep
      * used by the context. The mapping options are defined by @ref
      * ucc_ep_map_t. The definition is valid only when context is created with
      * an @ref ucc_oob_coll.
      */
    ucc_ep_map_t            ep_map;

    /** @ref ucc_team_params.id
      * The team id is a unique integer identifying the team that is active. The
      * integer is unique within the process and not the job .i.e., any two active
      * non-overlapping teams can have the same id. This semantic helps to avoid a
      * global information exchange .i.e, the processes or threads not
      * participating in the particular, need not participate in the team
      * creation. If not provided, the team id is created internally. For the MPI
      * programming model, this can be inherited from the MPI communicator id.
      */
    uint64_t                id;
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
    uint32_t               size;
    uint64_t              *eps;
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
 *  the team handle. Overlapping of multiple ucc\_team\_create\_post operations
 *  are invalid. The post takes in parameters ucc_context_h and ucc_team_params_t.
 *  The ucc_team_params_t provides user configuration to customize the team and,
 *  ucc_context_h provides the resources for the team and collectives.
 *  The routine returns immediately after posting the operation with the
 *  new team handle. However, the team handle is not ready for posting
 *  the collective operation. ucc_team_create_test operation is used to learn
 *  the status of the new team handle. On error, the team handle will not
 *  be created and corresponding error code as defined by @ref ucc_status_t is
 *  returned.
 *
 *  @endparblock
 *
 *  @return Error code as defined by @ref ucc_status_t
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
 *  communications. On error, the team handle becomes invalid, user is responsible
 *  to call ucc_team_destroy to destroy team and free allocated resources.
 *
 *  @endparblock
 *
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @ref ucc_team_destroy is a nonblocking collective operation to release all
 *  resources associated with the team handle, and destroy the team handle. It is
 *  invalid to post a collective operation after the ucc_team_destroy operation.
 *  It is invalid to call @ref ucc_team_destroy operation while @ref
 *  ucc_team_create_post is in progress. It is the user's responsibility to ensure
 *  there is one outstanding @ref ucc_team_create_post or @ref ucc_team_destroy
 *  operation is in progress.
 *
 *
 *  @endparblock
 *
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_team_create_from_parent(uint64_t my_ep, uint32_t included,
                                         ucc_team_h parent_team,
                                         ucc_team_h *new_team);

/*
 * *************************************************************
 *                   Collectives Section
 * *************************************************************
 */

/**
 *  @ingroup UCC_COLLECTIVES_DT
 */
typedef enum {
    UCC_COLL_ARGS_FLAG_IN_PLACE             = UCC_BIT(0), /*!< If set, the
                                                            output buffer is
                                                            identical to the
                                                            input buffer.*/
    UCC_COLL_ARGS_FLAG_PERSISTENT           = UCC_BIT(1), /*!< If set, the
                                                            collective is
                                                            considered
                                                            persistent.
                                                            Only, the
                                                            persistent
                                                            collective
                                                            can be called
                                                            multiple times with
                                                            the same request.
                                                           */
    UCC_COLL_ARGS_FLAG_COUNT_64BIT          = UCC_BIT(2), /*!< If set, the count
                                                            is 64bit, otherwise,
                                                            it is 32 bit. */
    UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT  = UCC_BIT(3), /*!< If set, the
                                                            displacement is
                                                            64bit, otherwise, it
                                                            is 32 bit. */
    UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER    = UCC_BIT(4), /*!< If set, the src
                                                            buffer is considered
                                                            contiguous.
                                                            Particularly, useful
                                                            for alltoallv
                                                            operation.*/
    UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER    = UCC_BIT(5), /*!<If set, the dst
                                                            buffer is considered
                                                            contiguous.
                                                            Particularly, useful
                                                            for alltoallv
                                                            operation. */
    UCC_COLL_ARGS_FLAG_TIMEOUT              = UCC_BIT(6), /*!<If set and the elapsed
                                                            time after @ref ucc_collective_post
                                                            (or @ref ucc_collective_triggered_post)
                                                            is greater than @ref ucc_coll_args_t.timeout,
                                                            the library returns UCC_ERR_TIMED_OUT
                                                            on the calling thread.
                                                            Note, the status is not guaranteed
                                                            to be global on all the processes
                                                            participating in the collective.*/
    UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS   = UCC_BIT(7)  /*!< If set, both src
                                                            and dst buffers
                                                            reside in a memory
                                                            mapped region.
                                                            Useful for one-sided
                                                            collectives. */
} ucc_coll_args_flags_t;

/**
 *  @ingroup UCC_COLLECTIVES_DT
 */
typedef enum {
    UCC_COLL_ARGS_HINT_OPTIMIZE_OVERLAP_CPU  = UCC_BIT(24), /*!< When the flag is
                                                              set, the user
                                                              prefers the library
                                                              to choose an
                                                              algorithm
                                                              implementation
                                                              optimized for the
                                                              best overlap of CPU
                                                              resources. */
    UCC_COLL_ARGS_HINT_OPTIMIZE_OVERLAP_GPU  = UCC_BIT(25), /*!< When the flag is
                                                              set, the user
                                                              prefers the library
                                                              to choose an
                                                              algorithm
                                                              implementation
                                                              optimized for the
                                                              best overlap of GPU
                                                              resources. */
    UCC_COLL_ARGS_HINT_OPTIMIZE_LATENCY     = UCC_BIT(26), /*!<  When the flag is
                                                           set, the user prefers
                                                           the library to choose
                                                           an algorithm
                                                           implementation
                                                           optimized for the
                                                           latency. */

    UCC_COLL_ARGS_HINT_CONTIG_SRC_BUFFER    = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER,
                                                /*!< When the flag is set, the source
                                                 * buffer is contiguous. */
    UCC_COLL_ARGS_HINT_CONTIG_DST_BUFFER    = UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER
                                                /*!< When the flag is set, the
                                                 * destination buffer is
                                                 * contiguous. */
} ucc_coll_args_hints_t;

/**
 *  @ingroup UCC_COLLECTIVES_DT
 */
typedef struct ucc_coll_buffer_info_v {
    void             *buffer; /*!< Starting address of the send/recv buffer */
    ucc_count_t      *counts; /*!< Array of counts of type @ref ucc_count_t
                                describing the total number of elements */
    ucc_aint_t       *displacements; /*!< Displacement array of team size and
                                       type @ref ucc_aint_t. Entry i specifies
                                       the displacement relative to the start
                                       address for the incoming data(
                                       outgoing data) for the team member i. For
                                       send buffer the data is fetched from this
                                       displacement and for receive buffer the
                                       incoming data is placed at this
                                       displacement. */
    ucc_datatype_t    datatype; /*!< Datatype of each buffer element */
    ucc_memory_type_t mem_type; /*!< Memory type of buffer as defined by @ref
                                  ucc_memory_type */
} ucc_coll_buffer_info_v_t;

/**
 *  @ingroup UCC_COLLECTIVES_DT
 */
typedef struct ucc_coll_buffer_info {
    void             *buffer;   /*!< Starting address of the send/recv buffer */
    ucc_count_t       count;    /*!< Total number of elements in the buffer */
    ucc_datatype_t    datatype; /*!< Datatype of each buffer element */
    ucc_memory_type_t mem_type; /*!< Memory type of buffer as defined by @ref
                                  ucc_memory_type */
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
    UCC_COLL_ARGS_FIELD_TAG                             = UCC_BIT(1),
    UCC_COLL_ARGS_FIELD_CB                              = UCC_BIT(2),
    UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER              = UCC_BIT(3),
    UCC_COLL_ARGS_FIELD_ACTIVE_SET                      = UCC_BIT(4)
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
    ucc_reduction_op_t              op; /*!< Predefined reduction operation, if
                                             reduce, allreduce, reduce_scatter
                                             operation is selected.
                                             The field is only specified for collectives
                                             that use pre-defined datatypes */
    uint64_t                        flags; /*!< Provide flags and hints for the
                                             collective operations */
    uint64_t                        root; /*!< Root endpoint for rooted
                                             collectives */
    ucc_error_type_t                error_type; /*!< Error type */
    ucc_coll_id_t                   tag; /*!< Used for ordering collectives */
    void                           *global_work_buffer; /*!< User allocated scratchpad
                                                             buffer for one-sided
                                                             collectives. The buffer
                                                             provided should be at least
                                                             the size returned by @ref
                                                             ucc_context_get_attr with
                                                             the field mask -
                                                             UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE
                                                             set to 1. The buffer must be initialized
                                                             to 0. */
    ucc_coll_callback_t             cb;
    double                          timeout; /*!< Timeout in seconds */
    struct {
        uint64_t start;
        int64_t  stride;
        uint64_t size;
    } active_set;
} ucc_coll_args_t;

/**
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief The routine to initialize a collective operation.
 *
 *  @param [in]    coll_args   Collective arguments descriptor
 *  @param [out]   request     Request handle representing the collective operation
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
 *  @return Error code as defined by @ref ucc_status_t
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
 *  operation. On error, request handle becomes invalid, user is responsible
 *  to call ucc_collective_finalize to free allocated resources.
 *
 *  @endparblock
 *
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @return Error code as defined by @ref ucc_status_t
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
 *  operation. On error, request handle becomes invalid, user is responsible
 *  to call ucc_collective_finalize to free allocated resources.
 *
 *  @endparblock
 *
 *  @return Error code as defined by @ref ucc_status_t
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
 *  @param [in] request - Request handle
 *
 *  @parblock
 *
 *  @b Description
 *
 *  @ref ucc_collective_finalize operation releases all resources
 *  associated with the collective operation represented by the request handle.
 *  In UCC_THREAD_MULTIPLE mode, the user is responsible for ensuring that
 *  @ref ucc_collective_finalize is called after the status is UCC_OK and after
 *  completing the execution of any callback registered with @ref ucc_coll_args_t.
 *
 *  @endparblock
 *
 *  @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_collective_finalize(ucc_coll_req_h request);

/**
 * @ingroup UCC_EVENT_DT
 *
 */
typedef enum ucc_event_type {
    UCC_EVENT_COLLECTIVE_POST     = UCC_BIT(0),
    UCC_EVENT_COLLECTIVE_COMPLETE = UCC_BIT(1),
    UCC_EVENT_COMPUTE_COMPLETE    = UCC_BIT(2),
    UCC_EVENT_OVERFLOW            = UCC_BIT(3)
} ucc_event_type_t;

/**
 * @ingroup UCC_EVENT_DT
 *
 */
typedef enum ucc_ee_type {
    UCC_EE_FIRST = 0,
    UCC_EE_CUDA_STREAM = UCC_EE_FIRST,
    UCC_EE_CPU_THREAD,
    UCC_EE_ROCM_STREAM,
    UCC_EE_LAST,
    UCC_EE_UNKNOWN = UCC_EE_LAST
} ucc_ee_type_t;

/**
 * @ingroup UCC_EVENT_DT
 *
 */
typedef struct ucc_event {
    ucc_event_type_t ev_type;
    void *           ev_context;
    size_t           ev_context_size;
    ucc_coll_req_h   req;
} ucc_ev_t;

/**
 * @ingroup UCC_EVENT_DT
 *
 */
typedef struct ucc_ee_params {
    ucc_ee_type_t ee_type;
    void *        ee_context;
    size_t        ee_context_size;
} ucc_ee_params_t;

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine creates the execution context for collective operations.
 *
 * @param [in]  team    Team handle
 * @param [in]  params  User provided params to customize the execution engine
 * @param [out] ee      Execution engine handle
 *
 * @parblock
 *
 * @b Description
 *
 * @ref ucc_ee_create creates the execution engine. It enables event-driven
 * collective execution. @ref ucc_ee_params_t allows the execution engine to be
 * configured to abstract either GPU and CPU threads. The execution engine is
 * created and coupled with the team. There can be many execution engines
 * coupled to the team. However, attaching the same execution engine to multiple
 * teams is not allowed. The execution engine is created after the team is
 * created and destroyed before the team is destroyed. It is the user's
 * responsibility to destroy the execution engines before the team. If the team
 * is destroyed before the execution engine is destroyed, the result is
 * undefined.
 *
 * @endparblock
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_ee_create(ucc_team_h team, const ucc_ee_params_t *params,
                           ucc_ee_h *ee);

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
 * @ref ucc_ee_destroy releases the resources attached with the
 * execution engine and destroys the execution engine. All events and triggered
 * operations related to this ee are invalid after the destroy operation. To
 * avoid race between the creation and destroying the execution engine, for a
 * given ee, the @ref ucc_ee_create and @ref ucc_ee_destroy must be invoked from
 * the same thread.
 *
 * @endparblock
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_ee_destroy(ucc_ee_h ee);

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine gets the event from the event queue.
 *
 * @param [in]  ee        Execution engine handle
 * @param [out] ev        Event structure fetched from the event queue
 *
 * @parblock
 *
 * @b Description
 *
 * @ref ucc_ee_get_event fetches the events from the execution engine. If there
 * are no events posted on the ee, it returns immediately without waiting for
 * events. All events must be acknowledged using the @ref ucc_ee_ack_event
 * interface. The event acknowledged is destroyed by the library. An event
 * fetched with @ref ucc_ee_get_event but not acknowledged might consume
 * resources in the library.
 *
 * @endparblock
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_ee_get_event(ucc_ee_h ee, ucc_ev_t **ev);

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine acks the events from the event queue.
 *
 * @param [in]  ee      Execution engine handle
 * @param [in]  ev      Event to be acked
 *
 * @parblock
 *
 * @b Description
 *
 * An event acknowledged by the user using @ref ucc_ee_ack_event is destroyed by
 * the library. Any triggered operations on the event should be completed before
 * calling this interface. The behavior is undefined if the user acknowledges
 * the event while waiting on the event or triggering operations on the event.
 *
 * @endparblock
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_ee_ack_event(ucc_ee_h ee, ucc_ev_t *ev);

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine to set the event to the tail of the queue.
 *
 * @param [in]  ee        Execution engine handle
 * @param [in]  ev        Event structure fetched from the event queue
 *
 * @parblock
 *
 * @b Description
 *
 * @ref ucc_ee_set_event sets the event on the execution engine. If the
 * operations are waiting on the event when the user sets the event, the
 * operations are launched. The events created by the user need to be destroyed
 * by the user.
 *
 * @endparblock
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_ee_set_event(ucc_ee_h ee, ucc_ev_t *ev);

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine blocks the calling thread until there is an event on the queue.
 *
 * @param [in]  ee        Execution engine handle
 * @param [out] ev        Event structure fetched from the event queue
 *
 * @parblock
 *
 * @b Description
 *
 * The user thread invoking the @ref ucc_ee_wait interface is blocked until an
 * event is posted to the execution engine.
 *
 * @endparblock
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_ee_wait(ucc_ee_h ee, ucc_ev_t *ev);

/**
 * @ingroup UCC_EVENT
 *
 * @brief The routine posts the collective operation on the execution engine, which is
 * launched on the event.
 *
 * @param [in]  ee          Execution engine handle
 * @param [in]  ee_event    Event triggering the post operation
 *
 * @parblock
 *
 * @b Description
 *
 * @ref ucc_collective_triggered_post allow the users to schedule a collective
 * operation that executes in the future when an event occurs on the execution
 * engine. On error, request handle associated with event becomes invalid,
 * user is responsible to call ucc_collective_finalize to free allocated resources.
 *
 * @endparblock
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_collective_triggered_post(ucc_ee_h ee, ucc_ev_t *ee_event);

END_C_DECLS
#endif
