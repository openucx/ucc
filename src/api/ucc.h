/**
 * @file ucc.h
 * @date 2020
 * @copyright Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_H_
#define UCC_H_

#include <api/ucc_def.h>
#include <api/ucc_version.h>
#include <api/ucc_status.h>
#include <stdio.h>
#include <stdbool.h>

/**
 * *************************************************************
 *                   Library initialization and finalize 
 * *************************************************************
 */

typedef enum {
    UCC_THREAD_MODE       = UCC_BIT(0),
    UCC_COLL_TYPES        = UCC_BIT(1),
    UCC_REDUCTION_TYPES   = UCC_BIT(2),
    UCC_SYNC_TYPE         = UCC_BIT(3) 
} ucc_lib_params_mask_t;

typedef enum {
    UCC_OP_USERDEFINED      = UCC_BIT(0),
    UCC_OP_SUM              = UCC_BIT(1),
    UCC_OP_PROD             = UCC_BIT(2),
    UCC_OP_MAX              = UCC_BIT(3),
    UCC_OP_MIN              = UCC_BIT(4),
    UCC_OP_AND              = UCC_BIT(5),
    UCC_OP_OR               = UCC_BIT(6),
    UCC_OP_XOR              = UCC_BIT(7),
    UCC_OP_LAND             = UCC_BIT(8),
    UCC_OP_LOR              = UCC_BIT(9),
    UCC_OP_LXOR             = UCC_BIT(10),
    UCC_OP_BAND             = UCC_BIT(11),
    UCC_OP_BOR              = UCC_BIT(12),
    UCC_OP_BXOR             = UCC_BIT(13),
    UCC_OP_MAXLOC           = UCC_BIT(14),
    UCC_OP_MINLOC           = UCC_BIT(15)
} ucc_reduction_op_t;

typedef enum {
    UCC_BARRIER            = UCC_BIT(0),
    UCC_BCAST              = UCC_BIT(1),
    UCC_ALLREDUCE          = UCC_BIT(2),
    UCC_REDUCE             = UCC_BIT(3),
    UCC_ALLTOALL           = UCC_BIT(4),
    UCC_ALLGATHER          = UCC_BIT(5),
    UCC_GATHER             = UCC_BIT(6),
    UCC_SCATTER            = UCC_BIT(7),
    UCC_MCAST              = UCC_BIT(8),
    UCC_FANIN              = UCC_BIT(9),
    UCC_FANOUT             = UCC_BIT(10),
    UCC_COLL_LAST          = UCC_BIT(11)
} ucc_coll_type_t;

typedef enum {
    UCC_THREAD_SINGLE       = UCC_BIT(0),
    UCC_THREAD_FUNNELED     = UCC_BIT(1),
    UCC_THREAD_MULTIPLE     = UCC_BIT(2)
} ucc_thread_mode_t;

typedef enum {
	UCC_NO_SYNC_COLLECTIVES = 0,
	UCC_SYNC_COLLECTIVES = 1
} ucc_coll_sync_type_t;

/**
 * @brief UCC library initializatoin parameters
 */

typedef struct ucc_lib_params {
    uint64_t                mask;
    ucc_thread_mode_t       thread_mode;
    ucc_coll_type_t         coll_types;
    ucc_reduction_op_t      reduction_types;
    ucc_coll_sync_type_t    sync_type;
} ucc_lib_params_t;


typedef struct ucc_lib_attribs {
    uint64_t                mask;
    ucc_thread_mode_t       thread_mode;
    ucc_coll_type_t         coll_types;
    ucc_reduction_op_t      reduction_types;
    ucc_coll_sync_type_t    sync_type;
} ucc_lib_attribs_t;


/**
 * @ingroup UCC_LIB
 * @brief @ref ucc_config_read allocates the @ref ucc_lib_config_t structure and
 * fetches the configuration values from the run-time environment. The run-time
 * environment supported are environment variables or a configuration file.
 *
 *   @param [out] config_p      Pointer to configuration descriptor as defined by
 *                              @ref ucc_lib_config_t.
 *   @param [out] env_prefix    If not NULL, the routine searches for the
 *                              environment variables with the prefix UCC_<env_prefix>. Otherwise, the
 *                              routines search for the environment variables that start with the prefix @
 *                              UCC_.
 *   @param [in]  filename      If not NULL, read configuration values from the
 *                              file defined by @e filename. If the file does not exist, it will be ignored
 *                              and no error will be reported to the user.
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_lib_config_read(const char *env_prefix, const char *filename,
                                 ucc_lib_config_t **config_p);

/**
 * @ingroup UCC_LIB
 * @brief Release configuration descriptor
 *
 * The routine releases the configuration descriptor that was allocated through
 * @ref ucc_lib_config_read "ucc_lib_config_read()" routine.
 *
 * @param [in] config        Pointer to the configuration descriptor to be released. 
 *                           Configuration descriptor as defined by
 */

void ucc_lib_config_release(ucc_lib_config_t *config);

/**
 * @ingroup UCC_LIB
 * @brief Print configuration information
 *
 * The routine prints the configuration information that is stored in
 * @ref ucc_lib_config_h "configuration" descriptor.
 *
 * @param [in]  config        @ref ucc_lib_config_h "Configuration descriptor"
 *                            to print.
 * @param [in]  stream        Output stream to print the configuration to.
 * @param [in]  title         Configuration title to print.
 * @param [in]  print_flags   Flags that control various printing options.
 */

void ucc_lib_config_print(const ucc_lib_config_t *config, FILE *stream,
                           const char *title, ucc_config_print_flags_t print_flags);

/**
 * @ingroup UCC_LIB
 * @brief Modify the configuration descriptor
 *
 * @param [in] config   Pointer to the configuration descriptor to be modified
 * @param [in] name     Configuration variable to be modified
 * @param [in] value    Configuration value to set
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_lib_config_modify(ucc_lib_config_t *config, const char *name,
                                    const char *value);


/**
 * @ingroup UCC_LIB
 *
 * @brief A local operation to initialize and allocate the resources for the UCC
 * operations. The parameters passed using the ucc_lib_params_t and
 * ucc_lib_config structures will customize and select the functionality of the
 * UCC library. The library can be customized for its interaction with the user
 * threads, types of collective operations, and reductions supported.
 * On success, the library object will be created and ucc_status_t will return
 * UCC_OK. On error, the library object will not be created and corresponding
 * error code as defined by ucc_status_t is returned.

 * @param [in]  params    user provided parameters to customize the library functionality
 * @param [in]  config    UCC configuration descriptor allocated through
 *                        @ref ucc_config_read "ucc_config_read()" routine.
 * @param [out] lib_p     UCC library handle
 *
 * @return Error code as defined by @ref ucc_status_t
 *
 */

ucc_status_t ucc_init(const ucc_lib_params_t *params,
                      const ucc_lib_config_t *config,
                      ucc_lib_h *lib_p);

/**
 * @ingroup UCC_LIB
 * @brief @ref ucc_finalize is a local operation to release the resources and
 * cleanup. All participants that invoked @ref ucc_init should call this
 * routine.
 *
 * @param [in] lib_p   Handle to @ref ucc_lib_h
 *                     "UCC library".
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_finalize(ucc_lib_h lib_p);


/**
 * @ingroup UCC_LIB
 * @brief A query operation to get the attributes of the library object. The
 * attributes are library configured values and reflect the choices made by the
 * library implementation.
 *
 * @param [out]  lib_atrib  Library attributes
 * @param [in]   lib_p      Input library object
 * 
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_lib_get_attribs(ucc_lib_h lib_p, ucc_lib_attribs_t *lib_atrib);

/**
 * *************************************************************
 *                   Context Section 
 * *************************************************************
 */



typedef enum {
   UCC_CONTEXT_EXCLUSIVE = 0 ,
   UCC_CONTEXT_SHARED
} ucc_context_type_t;

enum ucc_context_attribs_field {
    UCC_CONTEXT_ATTRIBS_TYPE                   = UCC_BIT(0),
    UCC_CONTEXT_ATTRIBS_COLL_SYNC_TYPE         = UCC_BIT(1),
    UCC_CONTEXT_ATTRIBS_COLL_OOB               = UCC_BIT(2)
};

enum ucc_context_params_field {
    UCC_CONTEXT_PARAMS_TYPE                   = UCC_BIT(0),
    UCC_CONTEXT_PARAMS_COLL_SYNC_TYPE         = UCC_BIT(1),
    UCC_CONTEXT_PARAMS_COLL_OOB               = UCC_BIT(2)
};

typedef struct ucc_context_params {
    uint64_t                mask;
    ucc_context_type_t      ctx_type;
    ucc_coll_sync_type_t    sync_type;
    ucc_context_oob_coll_t  oob_func;
} ucc_context_params_t;


typedef struct ucc_context_attribs {
    uint64_t                mask;
    ucc_context_type_t      ctx_type;
    ucc_coll_sync_type_t    sync_type;
} ucc_context_attribs_t;


ucc_status_t ucc_context_config_read(ucc_lib_h lib, const char *env_prefix,
                                     const char *filename, ucc_context_config_t
                                     **config);

void ucc_context_config_release(ucc_context_config_t *config);


/**
 * @ingroup UCC_CONTEXT
 *
 * @brief The ucc_context_create creates the context and ucc_context_destroy
 * releases the resources and destroys the context state. The creation of context
 * does not necessarily indicate its readiness to be used for collective or other
 * group operations.
 * On success, the context handle will be created and ucc_status_t will return
 * UCC_OK. On error, the library object will not be created and corresponding
 * error code as defined by ucc_status_t is returned.
 *
 * @param [in]   lib_handle  Library handle
 * @param [out]  params      Customizations for the communication context
 * @param [out]  config      Configuration for the communication context to read
 *                           from environment
 * @param [out]  context     Pointer to the newly created communication context
 * 
 * @return Error code as defined by @ref ucc_status_t
*/


ucc_status_t ucc_context_create(ucc_lib_h lib,
                                  const ucc_context_params_t *params,
                                  const ucc_context_config_t *config,
                                  ucc_context_h *context);

/**
 * @ingroup UCC_CONTEXT
 *
 * @brief @ref ucc_context_progress routine progresses the operations on the
 * content handle. It does not block for lack of resources or communication. 
 *
 * @param [in]  context  Communication context handle to be progressed
 * 
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_context_progress(ucc_context_h context);

/**
 * @ingroup UCC_CONTEXT
 * @brief @ref ucc_context_destroy routine releases the resources associated
 * with the handle @e context. All teams associated with the team should be
 * released before this. It is invalid to associate any team with this handle
 * after the routine is called.
 * 
 * @param [in]  context  Communication context handle to be released
 * 
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_context_destroy(ucc_context_h context);

/**
 * @ingroup UCC_CONTEXT
 * @brief @ref ucc_context_get_attribs routine queries the context handle
 * attributes. The attributes of the context handle are described by the context
 * attributes @ref ucc_context_attrib_t
 *
 *  @param [in]   context          Communication context
 *  @param [out]  context_attrib   Attributes of the communication context
 * 
 *  @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_context_get_attribs(ucc_context_h context, 
                                     ucc_context_attribs_t *context_atrib);


/**
 * *************************************************************
 *                   Teams Section 
 * *************************************************************
 */

enum ucc_team_params_field {
    UCC_PARAMS_POST_ORDERING          = UCC_BIT(0),
    UCC_PARAMS_OUTSTANDING_CALLS      = UCC_BIT(1),
    UCC_PARAMS_EP                     = UCC_BIT(2),
    UCC_PARAMS_EP_TYPE                = UCC_BIT(3),
    UCC_PARAMS_SYNC_TYPE              = UCC_BIT(4),
    UCC_PARAMS_OOB                    = UCC_BIT(5),
    UCC_PARAMS_MEM_PARAMS             = UCC_BIT(6)
};

enum ucc_team_attribs_field {
    UCC_ATTRIBS_POST_ORDERING          = UCC_BIT(0),
    UCC_ATTRIBS_OUTSTANDING_CALLS      = UCC_BIT(1),
    UCC_ATTRIBS_EP                     = UCC_BIT(2),
    UCC_ATTRIBS_EP_TYPE                = UCC_BIT(3),
    UCC_ATTRIBS_SYNC_TYPE              = UCC_BIT(4),
    UCC_ATTRIBS_OOB                    = UCC_BIT(5),
    UCC_MEM_PARAMS                     = UCC_BIT(6)
};

typedef enum  {
    UCC_MEM_SYMMETRIC   = UCC_BIT(0),
    UCC_MEM_PERSISTENT  = UCC_BIT(1),
    UCC_MEM_ALIGN32     = UCC_BIT(2),
    UCC_MEM_ALIGN64     = UCC_BIT(3),
    UCC_MEM_ALIGN128    = UCC_BIT(4),
} ucc_mem_constraints_t;

typedef enum {
	UCC_REMOTE_ATOMICS,
	UCC_REMOTE_COUNTERS
} ucc_mem_hints_t;

typedef struct ucc_mem_map_params {
    void                    *address;
    size_t                  len;
    ucc_mem_hints_t         hints;
    ucc_mem_constraints_t   constraints;
} ucc_mem_map_params_t;

typedef struct  ucc_team_oob_coll {
    int             (*allgather)(void *src_buf, void *recv_buf, size_t size,
                                 void *allgather_info,  void **request);
    ucc_status_t    (*req_test)(void *request);
    ucc_status_t    (*req_free)(void *request);
    uint32_t 	     participants;
    void             *coll_info;
}  ucc_team_oob_coll_t;

typedef enum {
    UCC_COLLECTIVE_POST_ORDERED = 0,
    UCC_COLLECTIVE_POST_UNORDERED = 1
} ucc_post_ordering_t;

typedef enum {
    UCC_COLLECTIVE_EP_RANGE_CONTIG = 0, 
    UCC_COLLECTIVE_EP_RANGE_NONCONTIG = 1
} ucc_ep_range_type_t;

typedef struct ucc_team_params {
    uint64_t                mask;
    ucc_post_ordering_t     ordering;
    uint64_t                outstanding_colls;
    uint64_t                ep;
    ucc_ep_range_type_t     ep_range;
    ucc_coll_sync_type_t    sync_type;
    ucc_team_oob_coll_t     oob_collective;
    ucc_mem_map_params_t    mem_params;
} ucc_team_params_t;

typedef struct ucc_team_attribs {
    uint64_t               mask;
    ucc_post_ordering_t    ordering;
    uint64_t               outstanding_colls;
    uint64_t               ep;
    ucc_ep_range_type_t    ep_range;
    ucc_coll_sync_type_t   sync_type;
    ucc_mem_map_params_t   mem_params;
} ucc_team_attribs_t;


/**
 * @ingroup UCC_TEAM
 *
 * @brief ucc_team_create_post is a nonblocking collective operation to create
 * the team handle. It takes in parameters ucc_context_h, num_handles,
 * ucc_team_params_t and returns a ucc_team_handle_h. The ucc_team_params_t
 * provides user configuration to customize the team. The routine returns
 * immediately after posting the operation with the new team handle. However,
 * the team handle is not ready for posting the collective operation.
 * ucc_team_create_test operation is used to learn the status of the new team
 * handle. On error, the team handle will not be created and corresponding error
 * code as defined by ucc_status_t is returned.
 *
 * @param  [in] contexts     Communication context abstracting the resources
 * @param  [in] num_contexs  Number of context provided as input
 * @param  [in] params       User defined configurations for the team
 * @param  [out] ucc_team    Team handle created
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_team_create_post(
              ucc_context_h context,
              ucc_team_params_t team_params,
              ucc_team_h *new_team);


/**
 *  @ingroup UCC_TEAM
 *
 *  @brief @ref ucc_team_create_test routines tests the status of team handle.
 *  If required it can progress the communication but cannot block on the
 *  communications.
 *
 *  @param  [in] ucc_team  Team handle to test
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_team_create_test();


/**
 * @ingroup UCC_TEAM
 *
 * @brief ucc_team_destroy is a blocking collective operation to release all
 * resources associated with the team handle, and destroy the team handle. It is
 * invalid to post a collective operation after the ucc_team_destroy operation.
 *
 * @param  [in] team  Destroy previously created team and release all resources
 *                     associated with it.
 *
 * @return Error code as defined by @ref ucc_status_t
 *
 */

ucc_status_t ucc_team_destroy(ucc_team_h team);

/**
 * @ingroup UCC_TEAM
 * @brief @ref ucc_team_get_attribs routine queries the team handle
 * attributes. The attributes of the team handle are described by the team
 * attributes @ref ucc_team_attrib_t
 *
 *  @param [in]   ucc_team       Team handle 
 *  @param [out]  team_attribs   Attributes of the team
 *
 *  @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_team_get_attribs(ucc_team_h ucc_team, ucc_team_attribs_t
                                  *team_atribs);

/**
 * @ingroup UCC_TEAM
 *
 * @brief ucc_team_create_from_parent is a nonblocking collective operation,
 * which creates a new team from the parent team. If a participant intends to
 * participate in the new team, it passes a TRUE value for the “included”
 * parameter. Otherwise, it passes FALSE. The routine returns immediately after
 * the post-operation. To learn the completion of the team create operation, the
 * ucc_team_create_test operation is used. 

 * @param [in]     my_ep         Endpoint of the process/thread calling the split operation
 * @param [in]    parent_team    Parent team handle from which a new team handle is created
 * @param [in]    included       Boolean variable indicating whether the
 *                               process/thread participates in the newly created team
 * @param [out]    new_ucc_team   Pointer to the new team handle
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_team_create_from_parent(uint64_t my_ep, bool included,
                                             ucc_team_h parent_team, 
                                             ucc_team_h *new_ucc_team);

/**
 * @ingroup UCC_TEAM
 *
 * @ref ucc_team_get_size routine queries the size of the team. It reflects the
 * number of unique endpoints in the team.
 *
 * @param [in]   ucc_team  Team handle
 * @param [out]  size      The size of team as number of endpoints
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_team_get_size(ucc_team_h ucc_team, uint32_t *size);


/**
 * @ingroup UCC_TEAM
 *
 * @ref ucc_team_get_my_ep routine queries and returns the endpoint of the
 * participant invoking the interface.
 *
 * @param [out]  ep          Endpoint of the participant calling the routine
 * @param [in]   ucc_team    Team handle
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_team_get_my_ep(ucc_team_h ucc_team, uint64_t *ep);

/**
 * @ingroup UCC_TEAM
 *
 * @ref ucc_team_my_ep routine queries and returns all endpoints of all
 * participants in the team.
 *
 * @param [out]     ep          List of endpoints
 * @param [out]     num_eps     Number of endpoints
 * @param [in]      ucc_team    Team handle
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_team_get_all_eps(ucc_team_h ucc_team, uint64_t **ep, uint64_t
                                  *num_eps);


/**
 * *************************************************************
 *                   Collectives Section 
 * *************************************************************
 */
typedef enum {
    UCC_DT_INT8 = 0,
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

typedef struct ucc_coll_buffer_info {
    uint64_t        mask;
    void            *src_buffer;
    ucc_count_t     *src_counts;
    ucc_aint_t      *src_displacements;
    void            *dst_buffer;
    ucc_count_t     *dst_counts;
    ucc_aint_t      *dst_displacements;
    ucc_datatype_t  src_datatype;
    ucc_datatype_t  dst_datatype;
    uint64_t        flags;
} ucc_coll_buffer_info_t;

typedef struct ucc_reduction_info {
    ucc_datatype_t      dt;
    ucc_reduction_op_t  op;
    size_t              count;
} ucc_reduction_info_t;

typedef enum {
    UCC_ERR_TYPE_LOCAL=0,
    UCC_ERR_TYPE_GLOBAL=1
} ucc_error_type_t;

typedef uint16_t ucc_coll_id_t ;

typedef struct ucc_coll_ext_op_args {
    ucc_coll_buffer_info_t      buffer_info;
    ucc_reduction_info_t        reduction_info;
    ucc_error_type_t            error_type;
    ucc_coll_id_t               tag;
    uint64_t                    root;
} ucc_coll_ext_op_args_t;

typedef struct ucc_coll_op_args {
    uint64_t                    mask;
    ucc_coll_type_t             coll_type;
    ucc_coll_ext_op_args_t      ext_args;
} ucc_coll_op_args_t;

/**
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief @ref  ucc_collective_init is a collective initialization operation,
 *  where all participants participate. The user provides all information
 *  required to start and complete the collective operation, which includes the
 *  input and output buffers, operation type, team handle, size, and any other
 *  hints for optimization. On success, the request handle is created and
 *  returned. On error, the request handle is not created and the appropriate
 *  error code is returned. On return, the ownership of buffers is transferred
 *  to the user. If modified, the results of collective operations posted on the
 *  request handle are undefined.
 *
 *   @param [out]   request     Request handle representing the collective operation
 *   @param [in]    coll_args   Collective arguments descriptor
 *   @param [in]    ucc_team    Team handle
 *
 *   @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_collective_init(ucc_coll_op_args_t *coll_args,
                                 ucc_coll_req_h *request, ucc_team_h team);

/**
 *  @ingroup UCC_COLLECTIVES
 *
 *  @brief @ref ucc_collective_post routine posts the collective operation. It
 *  does not require synchronization between the participants for the post
 *  operation.
 *
 *  @param [in]     request     Request handle
 *
 *  @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_collective_post(ucc_coll_req_h request);


/**
 *
 * @ingroup UCC_COLLECTIVES
 *
 * @brief @ref ucc_collective_init_and_post initializes the collective operation
 * and also posts the operation.
 *
 * @note: The @ref ucc_collecitve_init_and_post can be implemented as a
 * combination of @ref ucc_collective_init and @ref ucc_collective_post
 * routines.
 *
 * @param [out]     request     Request handle representing the collective operation
 * @param [in]      coll_args   Collective arguments descriptor
 * @param [in]      ucc_team    Input Team
 *
 * @return Error code as defined by @ref ucc_status_t
 */

ucc_status_t ucc_collective_init_and_post(ucc_coll_op_args_t *coll_args,
                                          ucc_coll_req_h *request,
                                          ucc_team_h team);

/**
 * @ingroup UCC_COLLECTIVES
 *
 * @brief @ucc_collective_test tests and returns the status of collective
 * operation.
 *
 * @param [in]  request Request handle
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_collective_test(ucc_coll_req_h request);

/**
 * @ingroup UCC_COLLECTIVES
 *
 * @brief @ref ucc_collective_finalize operation releases all resources
 * associated with the collective operation represented by the request handle.
 *
 * @param [in] request - request handle
 *
 * @return Error code as defined by @ref ucc_status_t
 */
ucc_status_t ucc_collective_finalize(ucc_coll_req_h request);

#endif
