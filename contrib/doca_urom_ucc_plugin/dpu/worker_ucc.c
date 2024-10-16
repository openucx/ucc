/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>

#include <signal.h>
#include <unistd.h>

#include <ucs/arch/atomic.h>

#include "worker_ucc.h"
#include "../common/urom_ucc.h"

DOCA_LOG_REGISTER(UROM::WORKER::UCC);

static uint64_t           plugin_version = 0x01;   /* UCC plugin DPU version */
static volatile uint64_t *queue_front;             /* Front queue node */
static volatile uint64_t *queue_tail;              /* Tail queue node */
static volatile uint64_t *queue_size;              /* Queue size */
static int                ucc_component_enabled;   /* Shared between worker threads */
static pthread_t          context_progress_thread; /* UCC progress thread context */
static uint64_t           queue_lock = 0;          /* Threads queue lock */
static pthread_t         *progress_thread = NULL;  /* Progress threads array */

/* UCC opts structure */
struct worker_ucc_opts worker_ucc_opts = {
    .num_progress_threads      = 1,
    .ppw                       = 32,
    .tpp                       = 1,
    .list_size                 = 64,
    .num_psync                 = 128,
    .dpu_worker_binding_stride = 1,
};

/* Progress thread arguments structure */
struct thread_args {
    uint64_t                thread_id;  /* Progress thread id */
    struct urom_worker_ucc *ucc_worker; /* UCC worker context */
};

/* Determine number of cores by counting the number of lines containing
   "processor" in /proc/cpuinfo */
int get_ncores()
{
    static int core_count = 0;
    int        count = 0;
    FILE      *fptr;
    char       str[100];
    char      *pos;
    int        index;

    // just read the file once and return the stored value on subsequent calls
    if (core_count != 0) {
        return core_count;
    }

    fptr = fopen("/proc/cpuinfo", "rb");

    if (fptr == NULL) {
        printf("Failed to open /proc/cpuinfo\n");
        exit(EXIT_FAILURE);
    }

    while ((fgets(str, 100, fptr)) != NULL) {
        index = 0;
        while ((pos = strstr(str + index, "processor")) != NULL) {
            index = (pos - str) + 1;
            count++;
        }
    }

    fclose(fptr);
    core_count = count;
    return count;
}

void dpu_thread_set_affinity_specific_core(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    if (core_id >=0 && core_id < get_ncores()) {
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    } else {
        printf("bad core id: %d\n", core_id);
        exit(-1);
    }
}

void dpu_thread_set_affinity(int thread_id)
{
    int       coreid      = thread_id;
    int       do_stride   = worker_ucc_opts.dpu_worker_binding_stride;
    int       num_threads = worker_ucc_opts.num_progress_threads;
    int       num_cores;
    int       stride;
    cpu_set_t cpuset;

    num_cores = get_ncores();
    stride    = num_cores / num_threads;

    CPU_ZERO(&cpuset);

    if(do_stride) {
        stride = do_stride;
        if (num_threads % 2 != 0) {
            stride = 1;
        }
        coreid *= stride;
    }

    if (coreid >=0 && coreid < num_cores) {
        CPU_SET(coreid, &cpuset);
        pthread_setaffinity_np(progress_thread[thread_id],
                               sizeof(cpuset), &cpuset);
    }
}

/*
 * Find available queue element
 *
 * @ctx_id [in]: UCC context id
 * @ucc_worker [in]: UCC command descriptor
 * @ret_qe [out]: set available queue element
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t find_qe_slot(uint64_t ctx_id,
                                 struct urom_worker_ucc *ucc_worker,
                                 struct ucc_queue_element **ret_qe)
{
    int      thread_id = ctx_id % worker_ucc_opts.num_progress_threads;
    uint64_t next      = (queue_tail[thread_id] + 1) % worker_ucc_opts.list_size;
    int      curr      = queue_tail[thread_id];

    if (next == queue_front[thread_id]) {
        *ret_qe = NULL;
        return DOCA_ERROR_FULL;
    }

    *ret_qe = &ucc_worker->queue[thread_id][curr];
    if ((*ret_qe)->in_use != 0) {
        *ret_qe = NULL;
        return DOCA_ERROR_BAD_STATE;
    }
    queue_tail[thread_id] = next;
    return DOCA_SUCCESS;
}

/*
 * Open UCC worker plugin
 *
 * @ctx [in]: DOCA UROM worker context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t urom_worker_ucc_open(struct urom_worker_ctx *ctx)
{
    uint64_t                i, j;
    doca_error_t            result;
    ucs_status_t            status;
    ucp_params_t            ucp_params;
    ucp_config_t           *ucp_config;
    ucp_worker_params_t     worker_params;
    struct urom_worker_ucc *ucc_worker;

    if (ctx == NULL) {
        return DOCA_ERROR_INVALID_VALUE;
    }

    ucc_worker = calloc(1, sizeof(*ucc_worker));
    if (ucc_worker == NULL) {
        DOCA_LOG_ERR("Failed to allocate UCC worker context");
        return DOCA_ERROR_NO_MEMORY;
    }

    if (worker_ucc_opts.num_progress_threads < MIN_THREADS) {
        worker_ucc_opts.num_progress_threads = MIN_THREADS;
        DOCA_LOG_WARN("Number of threads for UCC Offload "
                      "must be 1 or more, set to 1");
    }

    ucc_worker->ctx_id = 0;
    ucc_worker->nr_connections = 0;
    ucc_worker->ucc_data = calloc(worker_ucc_opts.ppw * worker_ucc_opts.tpp,
                                  sizeof(struct ucc_data));
    if (ucc_worker->ucc_data == NULL) {
        DOCA_LOG_ERR("Failed to allocate UCC worker context");
        result = DOCA_ERROR_NO_MEMORY;
        goto ucc_free;
    }

    ucc_worker->queue = (struct ucc_queue_element **)
                            malloc(sizeof(struct ucc_queue_element *) *
                                   worker_ucc_opts.num_progress_threads);
    if (ucc_worker->queue == NULL) {
        DOCA_LOG_ERR("Failed to allocate UCC elements queue");
        result = DOCA_ERROR_NO_MEMORY;
        goto ucc_data_free;
    }
    for (i = 0; i < worker_ucc_opts.num_progress_threads; i++) {
        ucc_worker->queue[i] = calloc(worker_ucc_opts.list_size,
                                      sizeof(struct ucc_queue_element));
        if (ucc_worker->queue[i] == NULL) {
            DOCA_LOG_ERR("Failed to allocate queue elements");
            result = DOCA_ERROR_NO_MEMORY;
            goto queue_free;
        }
    }

    queue_front = (volatile uint64_t *)
                calloc(worker_ucc_opts.num_progress_threads, sizeof(uint64_t));
    if (queue_front == NULL) {
        result = DOCA_ERROR_NO_MEMORY;
        goto queue_free;
    }

    queue_tail = (volatile uint64_t *)
                calloc(worker_ucc_opts.num_progress_threads, sizeof(uint64_t));
    if (queue_tail == NULL) {
        result = DOCA_ERROR_NO_MEMORY;
        goto queue_front_free;
    }

    queue_size = (volatile uint64_t *)
                calloc(worker_ucc_opts.num_progress_threads, sizeof(uint64_t));
    if (queue_size == NULL) {
        result = DOCA_ERROR_NO_MEMORY;
        goto queue_tail_free;
    }

    status = ucp_config_read(NULL, NULL, &ucp_config);
    if (status != UCS_OK) {
        DOCA_LOG_ERR("Failed to read UCP config");
        goto queue_size_free;
    }

    status = ucp_config_modify(ucp_config, "PROTO_ENABLE", "y");
    if (status != UCS_OK) {
        DOCA_LOG_ERR("Failed to read UCP config");
        ucp_config_release(ucp_config);
        goto queue_size_free;
    }

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_TAG   | UCP_FEATURE_RMA |
                            UCP_FEATURE_AMO64 | UCP_FEATURE_EXPORTED_MEMH;
    status = ucp_init(&ucp_params, ucp_config,
                      &ucc_worker->ucp_data.ucp_context);
    ucp_config_release(ucp_config);
    if (status != UCS_OK) {
        DOCA_LOG_ERR("Failed to initialized UCP");
        result = DOCA_ERROR_DRIVER;
        goto queue_size_free;
    }

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    status = ucp_worker_create(ucc_worker->ucp_data.ucp_context,
                               &worker_params,
                               &ucc_worker->ucp_data.ucp_worker);
    if (status != UCS_OK) {
        DOCA_LOG_ERR("Unable to create ucp worker");
        result = DOCA_ERROR_DRIVER;
        goto ucp_cleanup;
    }

    ucc_worker->ucp_data.eps = kh_init(ep);
    if (ucc_worker->ucp_data.eps == NULL) {
        DOCA_LOG_ERR("Failed to init EP hashtable map");
        result = DOCA_ERROR_DRIVER;
        goto worker_destroy;
    }

    ucc_worker->ucp_data.memh = kh_init(memh);
    if (ucc_worker->ucp_data.memh == NULL) {
        DOCA_LOG_ERR("Failed to init memh hashtable map");
        result = DOCA_ERROR_DRIVER;
        goto eps_destroy;
    }

    ucc_worker->ucp_data.rkeys = kh_init(rkeys);
    if (ucc_worker->ucp_data.rkeys == NULL) {
        DOCA_LOG_ERR("Failed to init rkeys hashtable map");
        result = DOCA_ERROR_DRIVER;
        goto memh_destroy;
    }

    ucc_worker->ids = kh_init(ctx_id);
    if (ucc_worker->ids == NULL) {
        DOCA_LOG_ERR("Failed to init ids hashtable map");
        result = DOCA_ERROR_DRIVER;
        goto rkeys_destroy;
    }

    ucc_worker->super = ctx;
    ucc_worker->list_lock = 0;
    ucc_component_enabled = 1;
    ucs_list_head_init(&ucc_worker->completed_reqs);

    ctx->plugin_ctx = ucc_worker;
    DOCA_LOG_INFO("UCC worker open flow is done");
    return DOCA_SUCCESS;

rkeys_destroy:
    kh_destroy(rkeys, ucc_worker->ucp_data.rkeys);
memh_destroy:
    kh_destroy(memh, ucc_worker->ucp_data.memh);
eps_destroy:
    kh_destroy(ep, ucc_worker->ucp_data.eps);
worker_destroy:
    ucp_worker_destroy(ucc_worker->ucp_data.ucp_worker);
ucp_cleanup:
    ucp_cleanup(ucc_worker->ucp_data.ucp_context);
queue_size_free:
    free((void *)queue_size);
queue_tail_free:
    free((void *)queue_tail);
queue_front_free:
    free((void *)queue_front);
queue_free:
    for (j = 0; j < i; j++)
        free(ucc_worker->queue[j]);
    free(ucc_worker->queue);
ucc_data_free:
    free(ucc_worker->ucc_data);
ucc_free:
    free(ucc_worker);
    return result;
}

static void ucc_worker_join_and_free_threads()
{
    uint64_t i;
    if (progress_thread) {
        for (i = 0; i < worker_ucc_opts.num_progress_threads; i++) {
            pthread_join(progress_thread[i], NULL);
        }
        free(progress_thread);
        progress_thread = NULL;
    }
}

/*
 * Close UCC worker plugin
 *
 * @worker_ctx [in]: DOCA UROM worker context
 */
static void urom_worker_ucc_close(struct urom_worker_ctx *worker_ctx)
{
    struct urom_worker_ucc *ucc_worker = worker_ctx->plugin_ctx;
    uint64_t i;

    if (worker_ctx == NULL)
        return;

    ucc_component_enabled = 0;

    ucc_worker_join_and_free_threads();

    /* Destroy hash tables */
    kh_destroy(rkeys,  ucc_worker->ucp_data.rkeys);
    kh_destroy(memh,   ucc_worker->ucp_data.memh);
    kh_destroy(ep,     ucc_worker->ucp_data.eps);
    kh_destroy(ctx_id, ucc_worker->ids);

    /* UCP cleanup */
    ucp_worker_destroy(ucc_worker->ucp_data.ucp_worker);
    ucp_cleanup(ucc_worker->ucp_data.ucp_context);

    /* UCC worker resources destroy */
    free((void *)queue_size);
    free((void *)queue_tail);
    free((void *)queue_front);
    free(ucc_worker->ucc_data);

    /* Queue elements destroy */
    for (i = 0; i < worker_ucc_opts.num_progress_threads; i++)
        free(ucc_worker->queue[i]);

    free(ucc_worker->queue);

    /* UCC worker destroy */
    free(ucc_worker);
}

/*
 * Unpacking UCC worker command
 *
 * @packed_cmd [in]: packed worker command
 * @packed_cmd_len [in]: packed worker command length
 * @cmd [out]: set unpacked UROM worker command
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t urom_worker_ucc_cmd_unpack(void  *packed_cmd,
                                               size_t packed_cmd_len,
                                               struct urom_worker_cmd **cmd)
{
    uint64_t                    extended_mem = 0;
    void                       *ptr;
    int                         is_count_64;
    int                         is_disp_64;
    size_t                      team_size;
    size_t                      disp_pack_size;
    size_t                      count_pack_size;
    ucc_coll_args_t            *coll_args;
    struct urom_worker_ucc_cmd *ucc_cmd;

    if (packed_cmd_len < sizeof(struct urom_worker_ucc_cmd)) {
        DOCA_LOG_INFO("Invalid packed command length");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *cmd = packed_cmd;
    ptr = packed_cmd +
            ucs_offsetof(struct urom_worker_cmd, plugin_cmd) +
            sizeof(struct urom_worker_ucc_cmd);
    ucc_cmd = (struct urom_worker_ucc_cmd *)(*cmd)->plugin_cmd;

    switch (ucc_cmd->cmd_type) {
    case UROM_WORKER_CMD_UCC_LIB_CREATE:
        ucc_cmd->lib_create_cmd.params = ptr;
        extended_mem += sizeof(ucc_lib_params_t);
        break;
    case UROM_WORKER_CMD_UCC_COLL:
        coll_args = ptr;
        ucc_cmd->coll_cmd.coll_args = ptr;
        ptr += sizeof(ucc_coll_args_t);
        extended_mem += sizeof(ucc_coll_args_t);
        if (ucc_cmd->coll_cmd.work_buffer_size > 0) {
            ucc_cmd->coll_cmd.work_buffer = ptr;
            ptr += ucc_cmd->coll_cmd.work_buffer_size;
            extended_mem += ucc_cmd->coll_cmd.work_buffer_size;
        }
        if (coll_args->coll_type == UCC_COLL_TYPE_ALLTOALLV       ||
            coll_args->coll_type == UCC_COLL_TYPE_ALLGATHERV      ||
            coll_args->coll_type == UCC_COLL_TYPE_GATHERV         ||
            coll_args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV ||
            coll_args->coll_type == UCC_COLL_TYPE_SCATTERV) {

            team_size = ucc_cmd->coll_cmd.team_size;
            is_count_64 =
                ((coll_args->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
                (coll_args->flags & UCC_COLL_ARGS_FLAG_COUNT_64BIT));
            is_disp_64 =
                ((coll_args->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
                (coll_args->flags & UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT));

            count_pack_size = ((is_count_64) ?
                                sizeof(uint64_t) :
                                sizeof(uint32_t)) * team_size;
            disp_pack_size  = ((is_disp_64)  ?
                                sizeof(uint64_t) :
                                sizeof(uint32_t)) * team_size;

            coll_args->src.info_v.counts = ptr;
            ptr += count_pack_size;
            extended_mem += count_pack_size;
            coll_args->dst.info_v.counts = ptr;
            ptr += count_pack_size;
            extended_mem += count_pack_size;

            coll_args->src.info_v.displacements = ptr;
            ptr += disp_pack_size;
            extended_mem += disp_pack_size;
            coll_args->dst.info_v.displacements = ptr;
            ptr += disp_pack_size;
            extended_mem += disp_pack_size;
        }
        break;

    case UROM_WORKER_CMD_UCC_CREATE_PASSIVE_DATA_CHANNEL:
        ucc_cmd->pass_dc_create_cmd.ucp_addr = ptr;
        extended_mem += ucc_cmd->pass_dc_create_cmd.addr_len;
        break;

    default:
        DOCA_LOG_ERR("Invalid UCC cmd: %u", ucc_cmd->cmd_type);
        break;
    }

    if ((*cmd)->len != extended_mem + sizeof(struct urom_worker_ucc_cmd)) {
        DOCA_LOG_ERR("Invalid UCC command length");
        return DOCA_ERROR_INVALID_VALUE;
    }

    return DOCA_SUCCESS;
}

/*
 * UCC worker safe push notification function
 *
 * @ucc_worker [in]: UCC worker context
 * @nd [in]: UROM worker notification descriptor
 */
static void
ucc_worker_safe_push_notification(struct urom_worker_ucc        *ucc_worker,
                                  struct urom_worker_notif_desc *nd)
{
    uint64_t lvalue = 0;

    lvalue = ucs_atomic_cswap64(&ucc_worker->list_lock, 0, 1);
    while (lvalue != 0)
        lvalue = ucs_atomic_cswap64(&ucc_worker->list_lock, 0, 1);

    ucs_list_add_tail(&ucc_worker->completed_reqs, &nd->entry);

    lvalue = ucs_atomic_cswap64(&ucc_worker->list_lock, 1, 0);
}

/*
 * UCC worker host destination remove
 *
 * @ucc_worker [in]: UCC worker context
 * @dest_id [in]: Host client dest id
 */
static void worker_ucc_dest_remove(struct urom_worker_ucc *ucc_worker,
                                   uint64_t dest_id)
{
    khint_t k;

    k = kh_get(ctx_id, ucc_worker->ids, dest_id);
    if (k == kh_end(ucc_worker->ids)) {
        DOCA_LOG_ERR("Destination id - %lu does not exist", dest_id);
        return;
    }
    kh_del(ctx_id, ucc_worker->ids, k);
    ucc_worker->ctx_id--;
}

/*
 * UCC worker host destinations lookup function
 *
 * @ucc_worker [in]: UCC worker context
 * @dest_id [in]: Host client dest id
 * @ctx_id [out]: Host client context id
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t worker_ucc_dest_lookup(struct urom_worker_ucc *ucc_worker,
                                           uint64_t                dest_id,
                                           uint64_t               *ctx_id)
{
    int     ret;
    khint_t k;

    k = kh_get(ctx_id, ucc_worker->ids, dest_id);
    if (k != kh_end(ucc_worker->ids)) {
        *ctx_id = kh_value(ucc_worker->ids, k);
        return DOCA_SUCCESS;
    }

    *ctx_id = ucc_worker->ctx_id;
    k = kh_put(ctx_id, ucc_worker->ids, dest_id, &ret);
    if (ret < 0) {
        DOCA_LOG_ERR("Failed to put new context id");
        return DOCA_ERROR_DRIVER;
    }

    ucc_worker->ctx_id++;

    kh_value(ucc_worker->ids, k) = *ctx_id;
    DOCA_LOG_DBG("UCC worker added connection %ld", *ctx_id);
    return DOCA_SUCCESS;
}

/*
 * Handle UCC library create command
 *
 * @ucc_worker [in]: UCC worker context
 * @cmd_desc [in]: UCC command descriptor
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_lib_create(struct urom_worker_ucc      *ucc_worker,
                           struct urom_worker_cmd_desc *cmd_desc)
{
    struct urom_worker_cmd        *cmd     = (struct urom_worker_cmd *)
                                                &cmd_desc->worker_cmd;
    struct urom_worker_ucc_cmd    *ucc_cmd = (struct urom_worker_ucc_cmd *)
                                                cmd->plugin_cmd;
    uint64_t                       ctx_id;
    uint64_t                       i;
    doca_error_t                   result;
    ucc_status_t                   ucc_status;
    ucc_lib_config_h               lib_config;
    ucc_lib_params_t              *lib_params;
    struct urom_worker_notify     *notif;
    struct urom_worker_notif_desc *nd;
    struct urom_worker_notify_ucc *ucc_notif;

    /* Prepare notification */
    nd = calloc(1, sizeof(*nd) + sizeof(*ucc_notif));
    if (nd == NULL)
        return DOCA_ERROR_NO_MEMORY;

    nd->dest_id = cmd_desc->dest_id;

    notif = (struct urom_worker_notify *)&nd->worker_notif;
    notif->type = cmd->type;
    notif->urom_context = cmd->urom_context;
    notif->len = sizeof(*ucc_notif);
    notif->status = DOCA_SUCCESS;

    ucc_notif = (struct urom_worker_notify_ucc *)notif->plugin_notif;
    ucc_notif->notify_type = UROM_WORKER_NOTIFY_UCC_LIB_CREATE_COMPLETE;
    ucc_notif->dpu_worker_id = ucc_cmd->dpu_worker_id;

    lib_params = ucc_cmd->lib_create_cmd.params;
    lib_params->mask |= UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params->thread_mode = UCC_THREAD_MULTIPLE;

    result = worker_ucc_dest_lookup(ucc_worker, cmd_desc->dest_id, &ctx_id);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to lookup command destination");
        goto fail;
    }

    ucc_worker->nr_connections++;

    if (ucc_worker->nr_connections > worker_ucc_opts.ppw) {
        DOCA_LOG_ERR("Too many processes connected to a single worker");
        result = DOCA_ERROR_FULL;
        goto dest_remove;
    }

    if (UCC_OK != ucc_lib_config_read(NULL, NULL, &lib_config)) {
        DOCA_LOG_ERR("Failed to read UCC lib config");
        result = DOCA_ERROR_DRIVER;
        goto reduce_conn;
    }

    for (i = 0; i < worker_ucc_opts.tpp; i++) {
        ucc_status = ucc_init(lib_params, lib_config,
                &ucc_worker->ucc_data[ctx_id*worker_ucc_opts.tpp + i].ucc_lib);
        if (ucc_status != UCC_OK) {
            DOCA_LOG_ERR("Failed to init UCC lib");
            result = DOCA_ERROR_DRIVER;
            goto reduce_conn;
        }
    }
    ucc_lib_config_release(lib_config);

    DOCA_LOG_DBG("Created UCC lib successfully");
    notif->status = DOCA_SUCCESS;
    ucc_worker_safe_push_notification(ucc_worker, nd);
    return notif->status;

reduce_conn:
    ucc_worker->nr_connections--;
dest_remove:
    worker_ucc_dest_remove(ucc_worker, cmd_desc->dest_id);
fail:
    DOCA_LOG_ERR("Failed to create UCC lib");
    notif->status = result;
    ucc_worker_safe_push_notification(ucc_worker, nd);
    return result;
}

/*
 * UCC library destroy
 *
 * @ucc_worker [in]: UCC worker context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t ucc_worker_lib_destroy(struct urom_worker_ucc *ucc_worker)
{
    doca_error_t result = DOCA_SUCCESS;
    uint64_t     j, k;
    int64_t      i;
    ucc_status_t status;

    ucc_component_enabled = 0;

    ucc_worker_join_and_free_threads();

    for (j = 0; j < ucc_worker->nr_connections; j++) {
        for (k = 0; k < worker_ucc_opts.tpp; k++) {
            struct ucc_data *ucc_ptr =
                &ucc_worker->ucc_data[j*worker_ucc_opts.tpp + k];
            for (i = 0; i < ucc_ptr->n_teams; i++) {
                if (!ucc_ptr->ucc_team[i]) {
                    continue;
                }
                status = ucc_team_destroy(ucc_ptr->ucc_team[i]);
                if (status != UCC_OK) {
                    DOCA_LOG_ERR("Failed to destroy UCC team of "
                                 "data index %lu and team index %ld", j, i);
                    result = DOCA_ERROR_DRIVER;
                }
                free(ucc_ptr->pSync);
            }
            if (ucc_ptr->ucc_context) {
                status = ucc_context_destroy(ucc_ptr->ucc_context);
                if (status != UCC_OK) {
                    DOCA_LOG_ERR("Failed to destroy UCC context of "
                                 "UCC data index %lu", j);
                    result = DOCA_ERROR_DRIVER;
                }
                ucc_ptr->ucc_context = NULL;
            }
            if (ucc_ptr->ucc_lib) {
                status = ucc_finalize(ucc_ptr->ucc_lib);
                if (status != UCC_OK) {
                    DOCA_LOG_ERR("Failed to finalize UCC lib "
                                 "of UCC data index %lu", j);
                    result = DOCA_ERROR_DRIVER;
                }
            }
        }
    }

    return result;
}

/*
 * Handle UCC library destroy command
 *
 * @ucc_worker [in]: UCC worker context
 * @cmd_desc [in]: UCC command descriptor
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_lib_destroy(struct urom_worker_ucc      *ucc_worker,
                            struct urom_worker_cmd_desc *cmd_desc)
{
    struct urom_worker_cmd        *cmd     = (struct urom_worker_cmd *)
                                                &cmd_desc->worker_cmd;
    struct urom_worker_ucc_cmd    *ucc_cmd = (struct urom_worker_ucc_cmd *)
                                                cmd->plugin_cmd;
    struct urom_worker_notify     *notif;
    struct urom_worker_notif_desc *nd;
    struct urom_worker_notify_ucc *ucc_notif;

    /* Prepare notification */
    nd = calloc(1, sizeof(*nd) + sizeof(*ucc_notif));
    if (nd == NULL)
        return DOCA_ERROR_NO_MEMORY;

    nd->dest_id = cmd_desc->dest_id;

    notif = (struct urom_worker_notify *)&nd->worker_notif;
    notif->type = cmd->type;
    notif->urom_context = cmd->urom_context;
    notif->len = sizeof(*ucc_notif);

    ucc_notif = (struct urom_worker_notify_ucc *)notif->plugin_notif;
    ucc_notif->notify_type = UROM_WORKER_NOTIFY_UCC_LIB_DESTROY_COMPLETE;
    ucc_notif->dpu_worker_id = ucc_cmd->dpu_worker_id;

    notif->status = ucc_worker_lib_destroy(ucc_worker);
    ucc_worker_safe_push_notification(ucc_worker, nd);
    return notif->status;
}

/*
 * Thread progress handles queue collective element
 *
 * @qe [in]: UCC thread queue element
 * @ucc_worker [in]: UCC worker context
 * @thread_id [in]: UCC thread id
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
handle_progress_thread_coll_element(struct ucc_queue_element *qe,
                                    struct urom_worker_ucc   *ucc_worker,
                                    int                       thread_id)
{
    int64_t                        lvalue     = 0;
    ucc_status_t                   ucc_status = UCC_OK;
    doca_error_t                   status     = DOCA_SUCCESS;
    ucc_status_t                   tmp_status;
    struct ucc_queue_element      *qe_back;
    struct urom_worker_notify_ucc *ucc_notif;

    if (!qe->posted) {
        ucc_status = ucc_collective_post(qe->coll_req);
        if (UCC_OK != ucc_status) {
            DOCA_LOG_ERR("Failed to post UCC collective: %s",
                         ucc_status_string(ucc_status));
            status = DOCA_ERROR_DRIVER;
            goto exit;
        }
        qe->posted = 1;
    }

    ucc_status = ucc_collective_test(qe->coll_req);
    if (ucc_status == UCC_INPROGRESS) {
        ucc_context_progress(ucc_worker->ucc_data[qe->ctx_id].ucc_context);
        lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
        while (lvalue != 0)
            lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
        status = find_qe_slot(qe->ctx_id, ucc_worker, &qe_back);
        lvalue = ucs_atomic_cswap64(&queue_lock, 1, 0);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to find queue slot for team creation");
            ucc_status = UCC_ERR_NO_RESOURCE;
            goto exit;
        }
        *qe_back = *qe;
        qe->in_use = 0;
        queue_front[thread_id] = (queue_front[thread_id] + 1)
                                    % worker_ucc_opts.list_size;
        return DOCA_ERROR_IN_PROGRESS;
    } else if (ucc_status == UCC_OK) {
        if (qe->barrier) {
            pthread_barrier_wait(qe->barrier);
            if (qe->nd != NULL) {
                pthread_barrier_destroy(qe->barrier);
                free(qe->barrier);
                qe->barrier = NULL;
            }
        }
        if (qe->key_duplicate_per_rank) {
            free(qe->key_duplicate_per_rank);
            qe->key_duplicate_per_rank = NULL;
        }
        if (qe->old_dest) {
            DOCA_LOG_DBG("Putting data back to host %p with size %lu",
                         qe->old_dest, qe->data_size);
            if (qe->dest_packed_key != NULL) {
                status = ucc_rma_put_host(
                            ucc_worker->ucc_data[qe->ctx_id].local_work_buffer
                                + qe->data_size,
                            qe->old_dest,
                            qe->data_size,
                            qe->ctx_id,
                            qe->dest_packed_key,
                            ucc_worker);
                if (status != DOCA_SUCCESS) {
                    DOCA_LOG_ERR("Failed to find queue slot for team creation");
                    goto exit;
                }
            } else {
                status = ucc_rma_put(
                            ucc_worker->ucc_data[qe->ctx_id].local_work_buffer
                                + qe->data_size,
                            qe->old_dest,
                            qe->data_size,
                            MAX_HOST_DEST_ID,
                            qe->myrank,
                            qe->ctx_id,
                            ucc_worker);
                if (status != DOCA_SUCCESS) {
                    DOCA_LOG_ERR("Failed to find queue slot for team creation");
                    goto exit;
                }
            }
        }
        if (qe->gwbi != NULL && qe->nd != NULL) {
            free(qe->gwbi);
        }
    } else {
        DOCA_LOG_ERR("ucc_collective_test() returned failure (%d)", ucc_status);
        status = DOCA_ERROR_DRIVER;
        goto exit;
    }

    status = DOCA_SUCCESS;
    tmp_status = ucc_collective_test(qe->coll_req);
    if (tmp_status != UCC_OK) {
        ucc_status = (ucc_status == UCC_OK) ? tmp_status : ucc_status;
        status = DOCA_ERROR_DRIVER;
    }
    tmp_status = ucc_collective_finalize(qe->coll_req);
    if (tmp_status != UCC_OK) {
        ucc_status = (ucc_status == UCC_OK) ? tmp_status : ucc_status;
        status = DOCA_ERROR_DRIVER;
    }

exit:
    if (qe->nd != NULL) {
        ucc_notif = (struct urom_worker_notify_ucc *)
                        qe->nd->worker_notif.plugin_notif;
        ucc_notif->coll_nqe.status = ucc_status;
        qe->nd->worker_notif.status = status;
        ucc_worker_safe_push_notification(ucc_worker, qe->nd);
    }
    queue_front[thread_id] = (queue_front[thread_id] + 1)
                                % worker_ucc_opts.list_size;
    ucs_atomic_add64(&queue_size[thread_id], -1);
    qe->in_use = 0;
    return status;
}

/*
 * Thread progress handles queue team element
 *
 * @qe [in]: UCC thread queue element
 * @ucc_worker [in]: UCC worker context
 * @thread_id [in]: UCC thread id
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
handle_progress_thread_team_element(struct ucc_queue_element *qe,
                                    struct urom_worker_ucc   *ucc_worker,
                                    int                       thread_id)
{
    struct urom_worker_notify_ucc *ucc_notif  = NULL;
    int64_t                        lvalue     = 0;
    ucc_status_t                   ucc_status = UCC_OK;
    doca_error_t                   status     = DOCA_SUCCESS;
    struct ucc_queue_element      *qe_back;

    if(qe->nd != NULL) {
        ucc_notif = (struct urom_worker_notify_ucc *)
                        qe->nd->worker_notif.plugin_notif;
    }

    ucc_status = ucc_team_create_test(
                    ucc_worker->ucc_data[qe->ctx_id].ucc_team[qe->team_id]);
    if (ucc_status == UCC_INPROGRESS) {
        ucc_status = ucc_context_progress(
                        ucc_worker->ucc_data[qe->ctx_id].ucc_context);
        lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
        while (lvalue != 0)
            lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
        status = find_qe_slot(qe->ctx_id, ucc_worker, &qe_back);
        lvalue = ucs_atomic_cswap64(&queue_lock, 1, 0);
        if (status != DOCA_SUCCESS)
            goto exit;
        *qe_back = *qe;
        queue_front[thread_id] = (queue_front[thread_id] + 1)
                                    % worker_ucc_opts.list_size;
        qe->in_use = 0;
        return DOCA_ERROR_IN_PROGRESS;
    } else if (ucc_status != UCC_OK) {
        DOCA_LOG_ERR("UCC team create test failed (%d) on team %ld for ctx %ld",
                 ucc_status,
                 qe->team_id,
                 qe->ctx_id);
        if (ucc_notif)
            ucc_notif->team_create_nqe.team = NULL;
        status = DOCA_ERROR_DRIVER;
    } else {
        if (qe->barrier) {
            pthread_barrier_wait(qe->barrier);
            if (qe->nd != NULL) {
                pthread_barrier_destroy(qe->barrier);
                free(qe->barrier);
                qe->barrier = NULL;
            }
        }
        DOCA_LOG_INFO("Finished team creation (%ld:%ld)",
                      qe->ctx_id, qe->team_id);
        if (ucc_notif) {
            ucc_notif->team_create_nqe.team =
                ucc_worker->ucc_data[qe->ctx_id].ucc_team[qe->team_id];
        }
        status = DOCA_SUCCESS;
    }

exit:
    free(qe->coll_ctx);
    if (qe->nd != NULL) {
        qe->nd->worker_notif.status = status;
        ucc_worker_safe_push_notification(ucc_worker, qe->nd);
    }
    queue_front[thread_id] = (queue_front[thread_id] + 1)
                                % worker_ucc_opts.list_size;
    ucs_atomic_add64(&queue_size[thread_id], -1);
    qe->in_use = 0;
    return status;
}

/*
 * Progress context thread main function
 *
 * @arg [in]: UCC worker arg
 * @return: NULL (dummy return because of pthread requirement)
 */
static void *urom_worker_ucc_progress_thread(void *arg)
{
    struct thread_args       *targs      = (struct thread_args *)arg;
    int                       thread_id  = targs->thread_id;
    struct urom_worker_ucc   *ucc_worker = targs->ucc_worker;
    doca_error_t              status     = DOCA_SUCCESS;
    int                       i;
    int                       front;
    int                       size;
    struct ucc_queue_element *qe;

    while (ucc_component_enabled) {
        size = queue_size[thread_id];
        for (i = 0; i < size; i++) {
            front = queue_front[thread_id];
            qe = &ucc_worker->queue[thread_id][front];
            if (qe->in_use != 1) {
                DOCA_LOG_WARN("Found queue element in " 
                              "queue and marked not in use");
                continue;
            }
            if (qe->type == UCC_WORKER_QUEUE_ELEMENT_TYPE_TEAM_CREATE) {
                status = handle_progress_thread_team_element(qe,
                                                             ucc_worker,
                                                             thread_id);
                if (status == DOCA_ERROR_IN_PROGRESS)
                    continue;

                if (status != DOCA_SUCCESS)
                    goto exit;
            } else if (qe->type == UCC_WORKER_QUEUE_ELEMENT_TYPE_COLLECTIVE) {
                status = handle_progress_thread_coll_element(qe,
                                                             ucc_worker,
                                                             thread_id);
                if (status == DOCA_ERROR_IN_PROGRESS)
                    continue;

                if (status != DOCA_SUCCESS)
                    goto exit;
            } else
                DOCA_LOG_ERR("Unknown queue element type");
        }
        sched_yield();
    }
exit:
    pthread_exit(NULL);
}

/*
 * UCC OOB allgather free
 *
 * @req [in]: allgather request data
 * @return: UCC_OK on success and UCC_ERR otherwise
 */
static ucc_status_t urom_worker_ucc_oob_allgather_free(void *req)
{
    free(req);
    return UCC_OK;
}

/*
 * UCC oob allgather function
 *
 * @sbuf [in]: local buffer to send to other processes
 * @rbuf [in]: global buffer to includes other processes source buffer
 * @msglen [in]: source buffer length
 * @oob_coll_ctx [in]: collection info
 * @req [out]: set allgather request data
 * @return: UCC_OK on success and UCC_ERR otherwise
 */
static ucc_status_t urom_worker_ucc_oob_allgather(void  *sbuf,
                                                  void  *rbuf,
                                                  size_t msglen,
                                                  void  *oob_coll_ctx,
                                                  void **req)
{
    struct coll_ctx          *ctx = (struct coll_ctx *) oob_coll_ctx;
    char                     *recv_buf;
    int                       index;
    int                       size;
    int                       i;
    struct oob_allgather_req *oob_req;

    size  = ctx->size;
    index = ctx->index;

    oob_req = malloc(sizeof(*oob_req));
    if (oob_req == NULL) {
        DOCA_LOG_ERR("Failed to allocate OOB UCC request");
        return UCC_ERR_NO_MEMORY;
    }

    oob_req->sbuf         = sbuf;
    oob_req->rbuf         = rbuf;
    oob_req->msglen       = msglen;
    oob_req->oob_coll_ctx = oob_coll_ctx;
    oob_req->iter         = 0;

    oob_req->status = calloc(ctx->size * 2, sizeof(int));
    *req            = oob_req;

    for (i = 0; i < size; i++) {
        recv_buf = (char *)rbuf + i * msglen;
        ucc_recv_nb(recv_buf, msglen, i, ctx->ucc_worker, &oob_req->status[i]);
    }

    for (i = 0; i < size; i++) {
        ucc_send_nb(sbuf, msglen, index, i, ctx->ucc_worker,
                    &oob_req->status[i + size]);
    }

    return UCC_OK;
}

/*
 * UCC oob allgather test function
 *
 * @req [in]: UCC allgather request
 * @return: UCC_OK on success and UCC_ERR otherwise
 */
static ucc_status_t urom_worker_ucc_oob_allgather_test(void *req)
{
    int                       nr_probes = 5;
    struct coll_ctx          *ctx;
    struct oob_allgather_req *oob_req;
    int                       i;
    int                       probe_count;
    int                       nr_done;
    int                       size;

    oob_req = (struct oob_allgather_req *)req;
    ctx = (struct coll_ctx *)oob_req->oob_coll_ctx;
    size = ctx->size;

    for (probe_count = 0; probe_count < nr_probes; probe_count++) {
        nr_done = 0;
        for (i = 0; i < size * 2; i++) {
            if (oob_req->status[i] != 1 &&
                ctx->ucc_worker->ucp_data.ucp_worker != NULL) {
                ucp_worker_progress(ctx->ucc_worker->ucp_data.ucp_worker);
            } else {
                ++nr_done;
            }
        }
        if (nr_done == size * 2)
            return UCC_OK;
    }

    return UCC_INPROGRESS;
}

/*
 * Handle UCC context creation of progress threads
 *
 * @arg [in]: UCC worker context argument
 * @return: NULL (dummy return because of pthread requirement)
 */
static void *urom_worker_ucc_ctx_progress_thread(void *arg)
{
    struct ctx_thread_args        *args       = (struct ctx_thread_args *)arg;
    ucc_mem_map_t                **maps       = NULL;
    size_t                         len        = args->len;
    int64_t                        size       = args->size;
    int64_t                        start      = args->start;
    int64_t                        stride     = args->stride;
    int64_t                        myrank     = args->myrank;
    uint64_t                       dest_id    = args->dest_id;
    struct urom_worker_ucc        *ucc_worker = args->ucc_worker;
    ucc_context_params_t           ctx_params = {0};
    struct urom_worker_notif_desc *nd;
    struct urom_worker_notify     *notif;
    struct urom_worker_notify_ucc *ucc_notif;
    struct thread_args            *targs;
    uint64_t                       n_threads, i, j;
    int                            ret;
    uint64_t                       ctx_id;
    char                           str_buf[256];
    ucc_status_t                   ucc_status;
    doca_error_t                   status;
    struct coll_ctx              **coll_ctx;
    ucc_context_config_h           ctx_config;

    nd = calloc(1, sizeof(*nd) + sizeof(*ucc_notif));
    if (nd == NULL) {
        status = DOCA_ERROR_NO_MEMORY;
        goto exit;
    }

    nd->dest_id         = args->dest_id;
    notif               = (struct urom_worker_notify *)&nd->worker_notif;
    notif->type         = args->notif_type;
    notif->len          = sizeof(*ucc_notif);
    notif->urom_context = args->urom_context;

    ucc_notif                = (struct urom_worker_notify_ucc *)
                                    notif->plugin_notif;
    ucc_notif->notify_type   = UROM_WORKER_NOTIFY_UCC_CONTEXT_CREATE_COMPLETE;
    ucc_notif->dpu_worker_id = args->myrank;

    status = worker_ucc_dest_lookup(ucc_worker, dest_id, &ctx_id);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to lookup command destination");
        goto fail;
    }

    maps     = (ucc_mem_map_t **) 
                calloc(worker_ucc_opts.tpp, sizeof(ucc_mem_map_t *));
    coll_ctx = (struct coll_ctx **)
                calloc(worker_ucc_opts.tpp, sizeof(struct coll_ctx *));

    for (i = 0; i < worker_ucc_opts.tpp; i++) {
        uint64_t thread_ctx_id = ctx_id * worker_ucc_opts.tpp + i;

        if (ucc_worker->ucc_data[thread_ctx_id].ucc_lib == NULL) {
            DOCA_LOG_ERR("Attempting to create UCC context "
                         "without first initializing a UCC lib");
            status = DOCA_ERROR_BAD_STATE;
            goto fail;
        }

        if (ucc_context_config_read(ucc_worker->ucc_data[thread_ctx_id].ucc_lib,
                                    NULL, &ctx_config) != UCC_OK) {
            DOCA_LOG_ERR("Failed to read UCC context config");
            status = DOCA_ERROR_DRIVER;
            goto fail;
        }

        /* Set to sliding window */
        if (UCC_OK != ucc_context_config_modify(
                        ctx_config,
                        "tl/ucp", "TUNE",
                        "allreduce:0-inf:@sliding_window")) {
            DOCA_LOG_ERR("Failed to modify TL_UCP_TUNE UCC lib config");
            status = DOCA_ERROR_DRIVER;
            goto cfg_release;
        }

        /* Set estimated num of eps */
        sprintf(str_buf, "%ld", size);
        ucc_status = ucc_context_config_modify(ctx_config, NULL,
                                               "ESTIMATED_NUM_EPS", str_buf);
        if (ucc_status != UCC_OK) {
            DOCA_LOG_ERR("UCC context config modify "
                         "failed for estimated_num_eps");
            status = DOCA_ERROR_DRIVER;
            goto cfg_release;
        }

        ucc_worker->ucc_data[thread_ctx_id].local_work_buffer =
            calloc(1, len * 2);
        if (ucc_worker->ucc_data[thread_ctx_id].local_work_buffer == NULL) {
            DOCA_LOG_ERR("Failed to allocate local work buffer");
            status = DOCA_ERROR_NO_MEMORY;
            goto cfg_release;
        }

        ucc_worker->ucc_data[thread_ctx_id].pSync =
            calloc(worker_ucc_opts.num_psync, sizeof(long));
        if (ucc_worker->ucc_data[thread_ctx_id].pSync == NULL) {
            DOCA_LOG_ERR("Failed to pSync array");
            status = DOCA_ERROR_NO_MEMORY;
            goto buf_free;
        }
        ucc_worker->ucc_data[thread_ctx_id].len = len * 2;

        maps[i] = (ucc_mem_map_t *)calloc(3, sizeof(ucc_mem_map_t));
        if (maps[i] == NULL) {
            DOCA_LOG_ERR("Failed to allocate UCC memory map array");
            status = DOCA_ERROR_NO_MEMORY;
            goto psync_free;
        }

        maps[i][0].address =
            ucc_worker->ucc_data[thread_ctx_id].local_work_buffer;
        maps[i][0].len = len * 2;
        maps[i][1].address = ucc_worker->ucc_data[thread_ctx_id].pSync;
        maps[i][1].len = worker_ucc_opts.num_psync * sizeof(long);

        coll_ctx[i] = (struct coll_ctx *)malloc(sizeof(struct coll_ctx));
        if (coll_ctx[i] == NULL) {
            DOCA_LOG_ERR("Failed to allocate UCC worker coll context");
            status = DOCA_ERROR_NO_MEMORY;
            goto maps_free;
        }

        if (stride <= 0)  {/* This is an array of ids */
            coll_ctx[i]->pids   = (int64_t *)start;
        } else {
            coll_ctx[i]->start  = start;
        }

        coll_ctx[i]->stride     = stride;
        coll_ctx[i]->size       = size;
        coll_ctx[i]->index      = myrank;
        coll_ctx[i]->ucc_worker = ucc_worker;

        ctx_params.mask                  = UCC_CONTEXT_PARAM_FIELD_OOB |
                                            UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
        ctx_params.oob.allgather         = urom_worker_ucc_oob_allgather;
        ctx_params.oob.req_test          = urom_worker_ucc_oob_allgather_test;
        ctx_params.oob.req_free          = urom_worker_ucc_oob_allgather_free;
        ctx_params.oob.coll_info         = (void *)coll_ctx[i];
        ctx_params.oob.n_oob_eps         = size;
        ctx_params.oob.oob_ep            = myrank;
        ctx_params.mem_params.segments   = maps[i];
        ctx_params.mem_params.n_segments = 2;

        ucc_status = ucc_context_create(
                        ucc_worker->ucc_data[thread_ctx_id].ucc_lib,
                        &ctx_params,
                        ctx_config,
                        &ucc_worker->ucc_data[thread_ctx_id].ucc_context);
        if (ucc_status != UCC_OK) {
            DOCA_LOG_ERR("Failed to create UCC context");
            status = DOCA_ERROR_DRIVER;
            goto coll_free;
        }
        ucc_context_config_release(ctx_config);
    }

    if (ctx_id == 0) {
        n_threads = worker_ucc_opts.num_progress_threads;
        targs = calloc(n_threads, sizeof(*targs));
        if (targs == NULL) {
            DOCA_LOG_ERR("Failed to create threads args");
            status = DOCA_ERROR_NO_MEMORY;
            goto context_destroy;
        }
        progress_thread = calloc(n_threads, sizeof(*progress_thread));
        if (progress_thread == NULL) {
            DOCA_LOG_ERR("Failed to create threads args");
            status = DOCA_ERROR_NO_MEMORY;
            goto targs_free;
        }

        DOCA_LOG_DBG("Creating [%ld] progress %lu threads", myrank, n_threads);
        for (i = 0; i < n_threads; i++) {
            targs[i].thread_id = i;
            targs[i].ucc_worker = ucc_worker;
            ret = pthread_create(&progress_thread[i],
                         NULL,
                         urom_worker_ucc_progress_thread,
                         (void *)&targs[i]);
            if (ret != 0) {
                DOCA_LOG_ERR("Failed to create progress thread");
                status = DOCA_ERROR_IO_FAILED;
                goto threads_free;
            }
        }
    }

    status = DOCA_SUCCESS;
    ucc_notif->context_create_nqe.context =
        ucc_worker->ucc_data[ctx_id].ucc_context;
    DOCA_LOG_DBG("UCC context created, ctx_id %lu, context %p",
                 ctx_id, ucc_worker->ucc_data[ctx_id].ucc_context);
    goto exit;

threads_free:
    for (j = 0; j < i; j++) {
        pthread_cancel(progress_thread[j]);
    }
    free(progress_thread);
targs_free:
    free(targs);
context_destroy:
    for(i = 0; i < worker_ucc_opts.tpp; i++) {
        if(ucc_worker->ucc_data[ctx_id*worker_ucc_opts.tpp + i].ucc_context) {
            ucc_context_destroy(
            ucc_worker->ucc_data[ctx_id*worker_ucc_opts.tpp + i].ucc_context);
        }
    }
coll_free:
    for(i = 0; i < worker_ucc_opts.tpp; i++) {
        if(coll_ctx[i]) {
            free(coll_ctx[i]);
        }
    }
    free(coll_ctx);
maps_free:
    for(i = 0; i < worker_ucc_opts.tpp; i++) {
        if(maps[i]) {
            free(maps[i]);
        }
    }
    free(maps);
psync_free:
    for(i = 0; i < worker_ucc_opts.tpp; i++) {
        if(ucc_worker->ucc_data[ctx_id*worker_ucc_opts.tpp + i].pSync) {
            free(ucc_worker->ucc_data[ctx_id*worker_ucc_opts.tpp + i].pSync);
        }
    }
buf_free:
    for(i = 0; i < worker_ucc_opts.tpp; i++) {
        if(ucc_worker->ucc_data[ctx_id*worker_ucc_opts.tpp + i].
                local_work_buffer) {
            free(ucc_worker->ucc_data[ctx_id*worker_ucc_opts.tpp + i].
                    local_work_buffer);
        }
    }
cfg_release:
    ucc_context_config_release(ctx_config);
fail:
exit:
    nd->worker_notif.status = status;
    ucc_worker_safe_push_notification(ucc_worker, nd);
    free(args);
    pthread_exit(NULL);
}

/*
 * Handle UCC context create command
 *
 * @ucc_worker [in]: UCC worker context
 * @cmd_desc [in]: UCC command descriptor
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_context_create(struct urom_worker_ucc *ucc_worker,
                               struct urom_worker_cmd_desc *cmd_desc)
{
    struct urom_worker_cmd        *cmd     = (struct urom_worker_cmd *)
                                                &cmd_desc->worker_cmd;
    struct urom_worker_ucc_cmd    *ucc_cmd = (struct urom_worker_ucc_cmd *)
                                                cmd->plugin_cmd;
    struct urom_worker_notif_desc *nd;
    struct urom_worker_notify_ucc *ucc_notif;
    int                            ret;
    struct ctx_thread_args        *args;

    args = calloc(1, sizeof(*args));
    if (args == NULL) {
        return DOCA_ERROR_NO_MEMORY;
    }

    args->notif_type   = cmd->type;
    args->urom_context = cmd->urom_context;
    args->start        = ucc_cmd->context_create_cmd.start;
    args->stride       = ucc_cmd->context_create_cmd.stride;
    args->size         = ucc_cmd->context_create_cmd.size;
    args->myrank       = ucc_cmd->dpu_worker_id;
    args->base_va      = ucc_cmd->context_create_cmd.base_va;
    args->len          = ucc_cmd->context_create_cmd.len;
    args->dest_id      = cmd_desc->dest_id;
    args->ucc_worker   = ucc_worker;

    ret = pthread_create(&context_progress_thread, NULL,
                         urom_worker_ucc_ctx_progress_thread, (void *)args);
    if (ret != 0) {
        nd = calloc(1, sizeof(*nd) + sizeof(*ucc_notif));
        if (nd == NULL) {
            return DOCA_ERROR_NO_MEMORY;
        }

        nd->dest_id = cmd_desc->dest_id;
        nd->worker_notif.status = DOCA_ERROR_IO_FAILED;
        nd->worker_notif.type = cmd->type;
        nd->worker_notif.len = sizeof(*ucc_notif);
        nd->worker_notif.urom_context = cmd->urom_context;
        ucc_notif = (struct urom_worker_notify_ucc *)
                        nd->worker_notif.plugin_notif;
        ucc_notif->dpu_worker_id = ucc_cmd->dpu_worker_id;
        ucc_notif->notify_type = UROM_WORKER_NOTIFY_UCC_CONTEXT_CREATE_COMPLETE;
        ucc_worker_safe_push_notification(ucc_worker, nd);
        return DOCA_ERROR_IO_FAILED;
    }

    return DOCA_SUCCESS;
}

/*
 * Handle UCC context destroy command
 *
 * @ucc_worker [in]: UCC worker context
 * @cmd_desc [in]: UCC command descriptor
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_context_destroy(struct urom_worker_ucc *ucc_worker,
                                struct urom_worker_cmd_desc *cmd_desc)
{
    struct urom_worker_cmd        *cmd     = (struct urom_worker_cmd *)
                                            &cmd_desc->worker_cmd;
    struct urom_worker_ucc_cmd    *ucc_cmd = (struct urom_worker_ucc_cmd *)
                                            cmd->plugin_cmd;
    uint64_t                       ctx_id, i;
    doca_error_t                   status;
    struct urom_worker_notify     *notif;
    struct urom_worker_notif_desc *nd;
    struct urom_worker_notify_ucc *ucc_notif;

    /* Prepare notification */
    nd = calloc(1, sizeof(*nd) + sizeof(*ucc_notif));
    if (nd == NULL) {
        return DOCA_ERROR_NO_MEMORY;
    }

    nd->dest_id         = cmd_desc->dest_id;
    notif               = (struct urom_worker_notify *)&nd->worker_notif;
    notif->type         = cmd->type;
    notif->urom_context = cmd->urom_context;
    notif->len          = sizeof(*ucc_notif);

    ucc_notif                = (struct urom_worker_notify_ucc *)
                                notif->plugin_notif;
    ucc_notif->notify_type   = UROM_WORKER_NOTIFY_UCC_CONTEXT_DESTROY_COMPLETE;
    ucc_notif->dpu_worker_id = ucc_cmd->dpu_worker_id;

    status = worker_ucc_dest_lookup(ucc_worker, cmd_desc->dest_id, &ctx_id);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to lookup command destination");
        goto exit;
    }

    for (i = 0; i < worker_ucc_opts.tpp; i++) {
        uint64_t thread_ctx_id = ctx_id * worker_ucc_opts.tpp + i;

        if (ucc_worker->ucc_data[thread_ctx_id].ucc_context) {
            if (ucc_context_destroy(
                    ucc_worker->ucc_data[thread_ctx_id].ucc_context) != UCC_OK) {
                DOCA_LOG_ERR("Failed to destroy UCC context");
                status = DOCA_ERROR_DRIVER;
                goto exit;
            }
            ucc_worker->ucc_data[thread_ctx_id].ucc_context = NULL;
        }
    }

    status = DOCA_SUCCESS;
exit:
    notif->status = status;
    ucc_worker_safe_push_notification(ucc_worker, nd);
    return status;
}

/*
 * Handle UCC team command
 *
 * @ucc_worker [in]: UCC worker context
 * @cmd_desc [in]: UCC command descriptor
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_team_create(struct urom_worker_ucc *ucc_worker,
                            struct urom_worker_cmd_desc *cmd_desc)
{
    struct urom_worker_cmd        *cmd       = (struct urom_worker_cmd *)
                                                    &cmd_desc->worker_cmd;
    struct urom_worker_ucc_cmd    *ucc_cmd   = (struct urom_worker_ucc_cmd *)
                                                    cmd->plugin_cmd;
    size_t                         curr_team = 0;
    struct urom_worker_notif_desc *nd;
    struct urom_worker_notify_ucc *ucc_notif;
    uint64_t                       ctx_id, i;
    ucc_ep_map_t                   map;
    doca_error_t                   status;
    ucc_status_t                   ucc_status;
    struct coll_ctx               *coll_ctx;
    struct ucc_queue_element      *qe;
    ucc_team_params_t              team_params;
    struct urom_worker_notify     *notif;
    pthread_barrier_t             *barrier;
    uint64_t                       lvalue;

    /* Prepare notification */
    nd = calloc(1, sizeof(*nd) + sizeof(*ucc_notif));
    if (nd == NULL) {
        return DOCA_ERROR_NO_MEMORY;
    }

    nd->dest_id              = cmd_desc->dest_id;
    notif                    = (struct urom_worker_notify *)&nd->worker_notif;
    notif->type              = cmd->type;
    notif->urom_context      = cmd->urom_context;
    notif->len               = sizeof(*ucc_notif);
    ucc_notif                = (struct urom_worker_notify_ucc *)
                                    notif->plugin_notif;
    ucc_notif->notify_type   = UROM_WORKER_NOTIFY_UCC_TEAM_CREATE_COMPLETE;
    ucc_notif->dpu_worker_id = ucc_cmd->dpu_worker_id;

    status = worker_ucc_dest_lookup(ucc_worker, cmd_desc->dest_id, &ctx_id);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to lookup command destination");
        goto exit;
    }

    barrier = malloc(sizeof(pthread_barrier_t));
    pthread_barrier_init(barrier, NULL, worker_ucc_opts.tpp);

    for (i = 0; i < worker_ucc_opts.tpp; i++) {
        uint64_t thread_ctx_id = ctx_id * worker_ucc_opts.tpp + i;

        curr_team = ucc_worker->ucc_data[thread_ctx_id].n_teams;
        if (ucc_worker->ucc_data[thread_ctx_id].ucc_context == NULL ||
            ucc_cmd->team_create_cmd.context_h !=
                ucc_worker->ucc_data[ctx_id].ucc_context) {
            DOCA_LOG_ERR("Attempting to create UCC "
                         "team over non-existent context");
            status = DOCA_ERROR_INVALID_VALUE;
            goto exit;
        }

        if (ucc_cmd->team_create_cmd.stride <= 0) {
            map.type = UCC_EP_MAP_ARRAY;
            map.ep_num = ucc_cmd->team_create_cmd.size;
            map.array.map = (void *)ucc_cmd->team_create_cmd.start;
            map.array.elem_size = 8;
        } else {
            map.type = UCC_EP_MAP_STRIDED;
            map.ep_num = ucc_cmd->team_create_cmd.size;
            map.strided.start = ucc_cmd->team_create_cmd.start;
            map.strided.stride = ucc_cmd->team_create_cmd.stride;
        }

        team_params.mask      = UCC_TEAM_PARAM_FIELD_EP            |
                                    UCC_TEAM_PARAM_FIELD_TEAM_SIZE |
                                    UCC_TEAM_PARAM_FIELD_EP_MAP    |
                                    UCC_TEAM_PARAM_FIELD_EP_RANGE;
        team_params.ep        = ucc_cmd->dpu_worker_id;
        team_params.ep_map    = map;
        team_params.ep_range  = UCC_COLLECTIVE_EP_RANGE_CONTIG;
        team_params.team_size = ucc_cmd->team_create_cmd.size;

        coll_ctx = (struct coll_ctx *) malloc(sizeof(*coll_ctx));
        if (coll_ctx == NULL) {
            DOCA_LOG_ERR("Failed to allocate collective context");
            status = DOCA_ERROR_NO_MEMORY;
            goto exit;
        }

        coll_ctx->start = ucc_cmd->team_create_cmd.start;
        coll_ctx->stride = ucc_cmd->team_create_cmd.stride;
        coll_ctx->size = ucc_cmd->team_create_cmd.size;
        coll_ctx->index = ucc_cmd->dpu_worker_id;
        coll_ctx->ucc_worker = ucc_worker;

        if (ucc_worker->ucc_data[thread_ctx_id].ucc_team == NULL) {
            ucc_worker->ucc_data[thread_ctx_id].ucc_team =
                malloc(sizeof(ucc_worker->ucc_data[thread_ctx_id].ucc_team));
            if (ucc_worker->ucc_data[thread_ctx_id].ucc_team == NULL) {
                status = DOCA_ERROR_NO_MEMORY;
                goto coll_free;
            }
        }

        ucc_status = ucc_team_create_post(
                        &ucc_worker->ucc_data[thread_ctx_id].ucc_context,
                        1,
                        &team_params,
                        &ucc_worker->ucc_data[thread_ctx_id].ucc_team[curr_team]);

        if (ucc_status != UCC_OK) {
            DOCA_LOG_ERR("ucc_team_create_post() failed");
            status = DOCA_ERROR_DRIVER;
            goto team_free;
        }
        ucc_worker->ucc_data[thread_ctx_id].n_teams++;

        lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
        while (lvalue != 0) {
            lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
        }
        status = find_qe_slot(thread_ctx_id, ucc_worker, &qe);
        lvalue = ucs_atomic_cswap64(&queue_lock, 1, 0);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to find queue slot for team creation");
            goto team_free;
        }

        qe->type     = UCC_WORKER_QUEUE_ELEMENT_TYPE_TEAM_CREATE;
        qe->coll_ctx = coll_ctx;
        qe->dest_id  = cmd_desc->dest_id;
        qe->ctx_id   = thread_ctx_id;
        qe->team_id  = curr_team;
        qe->myrank   = ucc_cmd->dpu_worker_id;
        qe->in_use   = 1;
        qe->barrier  = barrier;
        if (i == 0) {
            qe->nd = nd;
        } else {
            qe->nd = NULL;
        }
        ucs_atomic_add64(
            &queue_size[thread_ctx_id % worker_ucc_opts.num_progress_threads],
            1);

        continue;

team_free:
        free(ucc_worker->ucc_data[thread_ctx_id].ucc_team);
coll_free:
        free(coll_ctx);
        goto exit;
    }

    return DOCA_SUCCESS;

exit:
    notif->status = status;
    ucc_worker_safe_push_notification(ucc_worker, nd);
    return status;
}

size_t urom_worker_get_dt_size(ucc_datatype_t dt)
{
    size_t size_mod = 8;
    switch (dt) {
    case UCC_DT_INT8:
    case UCC_DT_UINT8:
        size_mod = sizeof(char);
        break;
    case UCC_DT_INT32:
    case UCC_DT_UINT32:
    case UCC_DT_FLOAT32:
        size_mod = sizeof(int);
        break;
    case UCC_DT_INT64:
    case UCC_DT_UINT64:
    case UCC_DT_FLOAT64:
        size_mod = sizeof(uint64_t);
        break;
    case UCC_DT_INT128:
    case UCC_DT_UINT128:
    case UCC_DT_FLOAT128:
        size_mod = sizeof(__int128_t);
        break;
    default:
        break;
    }
    return size_mod;
}


static doca_error_t post_nthreads_colls(
    uint64_t ctx_id, struct urom_worker_ucc *ucc_worker,
    ucc_coll_args_t *coll_args,
    ucc_team_h ucc_team, uint64_t myrank, int in_place,
    ucc_tl_ucp_allreduce_sw_global_work_buf_info_t *gwbi,
    struct urom_worker_notif_desc *nd,
    struct urom_worker_cmd_desc *cmd_desc,
    struct urom_worker_notify *notif,
    ucc_worker_key_buf *key_duplicate_per_rank)
{
    doca_error_t              status           = DOCA_SUCCESS;
    pthread_barrier_t        *barrier          = NULL;
    int64_t 				  team_idx         = 0;
    size_t                    threads          = worker_ucc_opts.tpp;
    size_t                    src_count        = coll_args->src.info.count;
    size_t                    dst_count        = coll_args->dst.info.count;
    size_t                    src_thread_count = src_count / threads;
    size_t                    dst_thread_count = dst_count / threads;
    size_t                    src_thread_size  = src_thread_count *
                                                    urom_worker_get_dt_size(
                                                    coll_args->src.info.datatype);
    size_t                    dst_thread_size  = dst_thread_count *
                                                    urom_worker_get_dt_size(
                                                    coll_args->dst.info.datatype);
    void                     *src_buf          = coll_args->src.info.buffer;
    void                     *dst_buf          = coll_args->dst.info.buffer;
    ucc_coll_req_h            coll_req;
    struct ucc_queue_element *qe;
    ucc_status_t              ucc_status;
    size_t                    i;
    uint64_t 				  lvalue;
    int64_t 				  j;

    coll_args->mask  |= UCC_COLL_ARGS_FIELD_FLAGS |
                            UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER;
    coll_args->flags |= UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS;

    coll_args->global_work_buffer = gwbi;

    barrier = malloc(sizeof(pthread_barrier_t));
    pthread_barrier_init(barrier, NULL, worker_ucc_opts.tpp);

    for(i = 0; i < threads; i++) {
        uint64_t thread_ctx_id = ctx_id*worker_ucc_opts.tpp + i;

        gwbi = malloc(sizeof(ucc_tl_ucp_allreduce_sw_global_work_buf_info_t));
        if (gwbi == NULL) {
            DOCA_LOG_ERR("Failed to initialize UCC collective: "
                         "Couldnt malloc global work buffer");
            status = DOCA_ERROR_DRIVER;
            goto fail;
        }

        gwbi->packed_src_memh = key_duplicate_per_rank[i].rkeys;
        gwbi->packed_dst_memh = key_duplicate_per_rank[i].rkeys +
                                key_duplicate_per_rank[i].src_len;

        coll_args->global_work_buffer = gwbi;

        if(!in_place) {
            coll_args->src.info.count = src_thread_count;
        }
        coll_args->dst.info.count = dst_thread_count;

        if(!in_place) {
            coll_args->src.info.buffer = src_buf + i * src_thread_size;
        }
        coll_args->dst.info.buffer = dst_buf + i * dst_thread_size;

        if (i == threads - 1) {
            if(!in_place) {
                coll_args->src.info.count += src_count % threads;
            }
            coll_args->dst.info.count += dst_count % threads;
        }

        if (i == 0) {
            // the threads made these teams at the same time, so their index is the same in their arrays
            // TODO: is there a better way to associate these teams with each other? maybe use a map?
            for (j = 0; j < ucc_worker->ucc_data[thread_ctx_id].n_teams; j++) {
                if (ucc_worker->ucc_data[thread_ctx_id].ucc_team[j] == ucc_team) {
                    team_idx = j;
                    break;
                }
            }
        }

        ucc_status = ucc_collective_init(coll_args, &coll_req,
                                         ucc_worker->ucc_data[thread_ctx_id].
                                            ucc_team[team_idx]);
        if (UCC_OK != ucc_status) {
            DOCA_LOG_ERR("Failed to initialize UCC collective: %s",
                         ucc_status_string(ucc_status));
            status = DOCA_ERROR_DRIVER;
            goto fail;
        }

        if (thread_ctx_id >= worker_ucc_opts.num_progress_threads) {
            DOCA_LOG_ERR("Warning--possible deadlock: multiple threads posting"
                         "to the same queue, and the qe is going to barrier. "
                         "Ensure tpp < num progress threads to avoid this\n");
        }

        lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
        while (lvalue != 0)
            lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
        status = find_qe_slot(thread_ctx_id, ucc_worker, &qe);
        lvalue = ucs_atomic_cswap64(&queue_lock, 1, 0);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to find queue slot for team creation");
            goto req_destroy;
        }

        qe->type                   = UCC_WORKER_QUEUE_ELEMENT_TYPE_COLLECTIVE;
        qe->coll_req               = coll_req;
        qe->myrank                 = myrank;
        qe->dest_id                = cmd_desc->dest_id;
        qe->old_dest               = NULL;
        qe->data_size              = 0;
        qe->gwbi                   = gwbi;
        qe->dest_packed_key        = NULL;
        qe->ctx_id                 = thread_ctx_id;
        qe->in_use                 = 1;
        qe->posted                 = 0;
        qe->barrier                = barrier;
        qe->key_duplicate_per_rank = key_duplicate_per_rank;

        if (i == 0) {
            qe->nd = nd;
        } else {
            qe->nd = NULL;
        }

        ucs_atomic_add64(
            &queue_size[thread_ctx_id % worker_ucc_opts.num_progress_threads],
            1);
    }

    return DOCA_SUCCESS;

req_destroy:
    ucc_collective_finalize(coll_req);
fail:
    notif->status = status;
    ucc_worker_safe_push_notification(ucc_worker, nd);
    return status;
}


/*
 * Handle UCC collective init command
 *
 * @ucc_worker [in]: UCC worker context
 * @cmd_desc [in]: UCC command descriptor
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_coll_init(struct urom_worker_ucc *ucc_worker,
                          struct urom_worker_cmd_desc *cmd_desc)
{
    size_t                                          size       = 0;
    size_t                                          size_mod   = 8;
    void                                           *old_dest   = NULL;
    void                                           *packed_key = NULL;
    struct urom_worker_cmd                         *cmd        =
        (struct urom_worker_cmd *)&cmd_desc->worker_cmd;
    struct urom_worker_ucc_cmd                     *ucc_cmd    =
        (struct urom_worker_ucc_cmd *)cmd->plugin_cmd;
    ucc_tl_ucp_allreduce_sw_global_work_buf_info_t *gwbi       = NULL;
    int                                             in_place   = 0;
    ucc_worker_key_buf                             *key_duplicate_per_rank;
    ucc_worker_key_buf                             *keys;
    uint64_t                                        ctx_id, myrank, lvalue, i;
    ucc_team_h                                      team;
    void                                           *work_buffer;
    doca_error_t                                    status;
    ucc_coll_req_h                                  coll_req;
    ucc_status_t                                    ucc_status;
    ucc_coll_args_t                                *coll_args;
    struct ucc_queue_element                       *qe;
    struct urom_worker_notify                      *notif;
    struct urom_worker_notif_desc                  *nd;
    struct urom_worker_notify_ucc                  *ucc_notif;

    /* Prepare notification */
    nd = calloc(1, sizeof(*nd) + sizeof(*ucc_notif));
    if (nd == NULL) {
        return DOCA_ERROR_NO_MEMORY;
    }

    nd->dest_id              = cmd_desc->dest_id;

    notif                    = (struct urom_worker_notify *)
                                &nd->worker_notif;
    notif->type              = cmd->type;
    notif->urom_context      = cmd->urom_context;
    notif->len               = sizeof(*ucc_notif);

    ucc_notif                = (struct urom_worker_notify_ucc *)
                                notif->plugin_notif;
    ucc_notif->notify_type   = UROM_WORKER_NOTIFY_UCC_COLLECTIVE_COMPLETE;
    ucc_notif->dpu_worker_id = ucc_cmd->dpu_worker_id;

    if (ucc_cmd->coll_cmd.team == NULL) {
        DOCA_LOG_ERR("Attempting to perform UCC collective without a UCC team");
        status = DOCA_ERROR_INVALID_VALUE;
        goto fail;
    }

    status = worker_ucc_dest_lookup(ucc_worker, cmd_desc->dest_id, &ctx_id);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to lookup command destination");
        goto fail;
    }

    if (ucc_cmd->coll_cmd.work_buffer_size > 0 &&
            ucc_cmd->coll_cmd.work_buffer) {
        work_buffer = ucc_cmd->coll_cmd.work_buffer;
    } else {
        work_buffer = NULL;
    }

    team      = ucc_cmd->coll_cmd.team;
    coll_args = ucc_cmd->coll_cmd.coll_args;
    myrank    = ucc_cmd->dpu_worker_id;

    COLL_CHECK(ucc_worker, ctx_id, status);

    if ( (coll_args->mask & UCC_COLL_ARGS_FIELD_FLAGS  ) &&
         (coll_args->flags & UCC_COLL_ARGS_FLAG_IN_PLACE) ) {
        in_place = 1;
    }

    if (coll_args->mask & UCC_COLL_ARGS_FIELD_CB)
        /* Cannot support callbacks to host data.. just won't work */
        coll_args->mask = coll_args->mask & (~UCC_COLL_ARGS_FIELD_CB);

    if (coll_args->coll_type == UCC_COLL_TYPE_ALLTOALL ||
        coll_args->coll_type == UCC_COLL_TYPE_ALLTOALLV) {
        if (!ucc_cmd->coll_cmd.use_xgvmi) {
            size_mod = urom_worker_get_dt_size(coll_args->src.info.datatype);
            size = coll_args->src.info.count * size_mod;
            if (coll_args->mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER) {
                /* Perform get based on passed information */
                keys = work_buffer;
                status = ucc_rma_get_host(
                            ucc_worker->ucc_data[ctx_id].local_work_buffer,
                            coll_args->src.info.buffer,
                            size,
                            ctx_id,
                            keys->rkeys,
                            ucc_worker);
                if (status != DOCA_SUCCESS) {
                    DOCA_LOG_ERR("UCC component unable to obtain source buffer");
                    goto fail;
                }
                packed_key = keys->rkeys + keys->src_len;
            } else {
                /* Perform get based on domain information */
                status = ucc_rma_get(
                            ucc_worker->ucc_data[ctx_id].local_work_buffer,
                            coll_args->src.info.buffer,
                            size,
                            MAX_HOST_DEST_ID,
                            myrank,
                            ctx_id,
                            ucc_worker);
                if (status != DOCA_SUCCESS) {
                    DOCA_LOG_ERR("UCC component unable to obtain source buffer");
                    goto fail;
                }
            }
            coll_args->src.info.buffer =
                ucc_worker->ucc_data[ctx_id].local_work_buffer;
            old_dest = coll_args->dst.info.buffer;
            coll_args->dst.info.buffer =
                ucc_worker->ucc_data[ctx_id].local_work_buffer + size;
        }
        if (!(coll_args->mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER) ||
            !work_buffer) {
            coll_args->mask  |= UCC_COLL_ARGS_FIELD_FLAGS              |
                                UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER;
            coll_args->flags |= UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS;
            coll_args->global_work_buffer =
                ucc_worker->ucc_data[ctx_id].pSync +
                (ucc_worker->ucc_data[ctx_id].psync_offset %
                    worker_ucc_opts.num_psync);
            ucc_worker->ucc_data[ctx_id].psync_offset++;
        } else {
            if (work_buffer != NULL) {
                coll_args->global_work_buffer = work_buffer;
            }
        }
    } else if (coll_args->coll_type == UCC_COLL_TYPE_ALLREDUCE ||
               coll_args->coll_type == UCC_COLL_TYPE_ALLGATHER) {
        if (!ucc_cmd->coll_cmd.use_xgvmi) {
            DOCA_LOG_ERR("Failed to initialize UCC collective:"
                         "Allreduce must use xgvmi");
            status = DOCA_ERROR_DRIVER;
            goto fail;
        }
        if (!(coll_args->mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER) ||
            !work_buffer) {
            DOCA_LOG_ERR("Failed to initialize UCC collective:"
                         "Allreduce must use global work buffer");
            status = DOCA_ERROR_DRIVER;
            goto fail;
        }

        keys = work_buffer;

        gwbi = malloc(sizeof(ucc_tl_ucp_allreduce_sw_global_work_buf_info_t));
        if (gwbi == NULL) {
            DOCA_LOG_ERR("Failed to initialize UCC collective: "
                         "Couldnt malloc global work buffer");
            status = DOCA_ERROR_DRIVER;
            goto fail;
        }

        gwbi->packed_src_memh = keys->rkeys;
        gwbi->packed_dst_memh = keys->rkeys + keys->src_len;

        key_duplicate_per_rank = malloc(sizeof(ucc_worker_key_buf) *
                                        worker_ucc_opts.tpp);
        if (key_duplicate_per_rank == NULL) {
            printf("couldnt malloc key_duplicate_per_rank\n");
        }
        for (i = 0; i < worker_ucc_opts.tpp; i++) {
            memcpy(key_duplicate_per_rank[i].rkeys, keys->rkeys,
                    keys->src_len + keys->dst_len);
            key_duplicate_per_rank[i].src_len = keys->src_len;
            key_duplicate_per_rank[i].dst_len = keys->dst_len;
        }

        status = post_nthreads_colls(
            ctx_id, ucc_worker, coll_args,
            team, myrank, in_place,
            gwbi, nd, cmd_desc, notif, key_duplicate_per_rank);

        return status;
    }

    ucc_status = ucc_collective_init(coll_args, &coll_req, team);
    if (UCC_OK != ucc_status) {
        DOCA_LOG_ERR("Failed to initialize UCC collective: %s",
                     ucc_status_string(ucc_status));
        status = DOCA_ERROR_DRIVER;
        goto fail;
    }

    ucc_status = ucc_collective_post(coll_req);
    if (UCC_OK != ucc_status) {
        DOCA_LOG_ERR("Failed to post UCC collective: %s",
                     ucc_status_string(ucc_status));
        status = DOCA_ERROR_DRIVER;
        goto req_destroy;
    }

    lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
    while (lvalue != 0) {
        lvalue = ucs_atomic_cswap64(&queue_lock, 0, 1);
    }
    status = find_qe_slot(ctx_id, ucc_worker, &qe);
    lvalue = ucs_atomic_cswap64(&queue_lock, 1, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to find queue slot for team creation");
        goto req_destroy;
    }

    qe->type     = UCC_WORKER_QUEUE_ELEMENT_TYPE_COLLECTIVE;
    qe->coll_req = coll_req;
    qe->myrank   = myrank;
    qe->dest_id  = cmd_desc->dest_id;
    if (!ucc_cmd->coll_cmd.use_xgvmi) {
        DOCA_LOG_DBG("Setting old dest to %p", old_dest);
        qe->old_dest  = old_dest;
        qe->data_size = size;
    } else {
        qe->old_dest  = NULL;
        qe->data_size = 0;
    }
    qe->gwbi            = gwbi;
    qe->dest_packed_key = packed_key;
    qe->ctx_id          = ctx_id;
    qe->in_use          = 1;
    qe->posted          = 1;
    qe->barrier         = NULL;
    qe->nd              = nd;
    ucs_atomic_add64(&queue_size[ctx_id % worker_ucc_opts.num_progress_threads],
                     1);
    return DOCA_SUCCESS;
req_destroy:
    ucc_collective_finalize(coll_req);
fail:
    notif->status = status;
    ucc_worker_safe_push_notification(ucc_worker, nd);
    return status;
}

/*
 * Handle UCC passive data channel create command
 *
 * @ucc_worker [in]: UCC worker context
 * @cmd_desc [in]: UCC command descriptor
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_pass_dc_create(struct urom_worker_ucc *ucc_worker,
                               struct urom_worker_cmd_desc *cmd_desc)
{
    struct urom_worker_cmd        *cmd     = (struct urom_worker_cmd *)
                                                &cmd_desc->worker_cmd;
    struct urom_worker_ucc_cmd    *ucc_cmd = (struct urom_worker_ucc_cmd *)
                                                cmd->plugin_cmd;
    uint64_t                       ctx_id;
    ucp_ep_h                       new_ep;
    doca_error_t                   status;
    ucs_status_t                   ucs_status;
    ucp_ep_params_t                ep_params;
    struct urom_worker_notify     *notif;
    struct urom_worker_notif_desc *nd;
    struct urom_worker_notify_ucc *ucc_notif;

    /* Prepare notification */
    nd = calloc(1, sizeof(*nd) + sizeof(*ucc_notif));
    if (nd == NULL)
        return DOCA_ERROR_NO_MEMORY;

    nd->dest_id = cmd_desc->dest_id;

    notif                    = (struct urom_worker_notify *)&nd->worker_notif;
    notif->type              = cmd->type;
    notif->urom_context      = cmd->urom_context;
    notif->len               = sizeof(*ucc_notif);
    ucc_notif                = (struct urom_worker_notify_ucc *)
                                notif->plugin_notif;
    ucc_notif->notify_type   =
        UROM_WORKER_NOTIFY_UCC_PASSIVE_DATA_CHANNEL_COMPLETE;
    ucc_notif->dpu_worker_id = ucc_cmd->dpu_worker_id;

    status = worker_ucc_dest_lookup(ucc_worker, cmd_desc->dest_id, &ctx_id);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to lookup command destination");
        goto fail;
    }

    if (ucc_worker->ucc_data[ctx_id].host == NULL) {
        ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS     |
                                    UCP_EP_PARAM_FIELD_ERR_HANDLER        |
                                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        ep_params.err_handler.cb  = urom_ep_err_cb;
        ep_params.err_handler.arg = NULL;
        ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
        ep_params.address         = ucc_cmd->pass_dc_create_cmd.ucp_addr;

        ucs_status = ucp_ep_create(ucc_worker->ucp_data.ucp_worker,
                                    &ep_params, &new_ep);
        if (ucs_status != UCS_OK) {
            DOCA_LOG_ERR("ucp_ep_create() returned: %s",
                            ucs_status_string(ucs_status));
            status = DOCA_ERROR_DRIVER;
            goto fail;
        }

        ucc_worker->ucc_data[ctx_id].host = new_ep;
        DOCA_LOG_DBG("Created passive data channel for host for rank %lu",
                     ucc_cmd->dpu_worker_id);
    } else {
        DOCA_LOG_DBG("Passive data channel already created");
    }
    status = DOCA_SUCCESS;
fail:
    notif->status = status;
    ucc_worker_safe_push_notification(ucc_worker, nd);
    return status;
}

/*
 * Handle UROM UCC worker commands function
 *
 * @ctx [in]: DOCA UROM worker context
 * @cmd_list [in]: command descriptor list to handle
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_worker_cmd(struct urom_worker_ctx *ctx,
                           ucs_list_link_t *cmd_list)
{
    doca_error_t                 status     = DOCA_SUCCESS;
    struct urom_worker_ucc      *ucc_worker = (struct urom_worker_ucc *)
                                                ctx->plugin_ctx;
    struct urom_worker_ucc_cmd  *ucc_cmd;
    struct urom_worker_cmd_desc *cmd_desc;
    struct urom_worker_cmd      *cmd;

    while (!ucs_list_is_empty(cmd_list)) {
        cmd_desc = ucs_list_extract_head(cmd_list,
                                         struct urom_worker_cmd_desc, entry);
        status = urom_worker_ucc_cmd_unpack(&cmd_desc->worker_cmd,
                                            cmd_desc->worker_cmd.len, &cmd);
        if (status != DOCA_SUCCESS) {
            free(cmd_desc);
            return status;
        }
        ucc_cmd = (struct urom_worker_ucc_cmd *)cmd->plugin_cmd;
        switch (ucc_cmd->cmd_type) {
        case UROM_WORKER_CMD_UCC_LIB_CREATE:
            status = urom_worker_ucc_lib_create(ucc_worker, cmd_desc);
            break;
        case UROM_WORKER_CMD_UCC_LIB_DESTROY:
            status = urom_worker_ucc_lib_destroy(ucc_worker, cmd_desc);
            break;
        case UROM_WORKER_CMD_UCC_CONTEXT_CREATE:
            status = urom_worker_ucc_context_create(ucc_worker, cmd_desc);
            break;
        case UROM_WORKER_CMD_UCC_CONTEXT_DESTROY:
            status = urom_worker_ucc_context_destroy(ucc_worker, cmd_desc);
            break;
        case UROM_WORKER_CMD_UCC_TEAM_CREATE:
            status = urom_worker_ucc_team_create(ucc_worker, cmd_desc);
            break;
        case UROM_WORKER_CMD_UCC_COLL:
            status = urom_worker_ucc_coll_init(ucc_worker, cmd_desc);
            break;
        case UROM_WORKER_CMD_UCC_CREATE_PASSIVE_DATA_CHANNEL:
            status = urom_worker_ucc_pass_dc_create(ucc_worker, cmd_desc);
            break;
        default:
            DOCA_LOG_INFO("Invalid UCC command type: %u", ucc_cmd->cmd_type);
            status = DOCA_ERROR_INVALID_VALUE;
            break;
        }
        free(cmd_desc);
        if (status != DOCA_SUCCESS) {
            return status;
        }
    }

    return status;
}

/*
 * Get UCC worker address
 *
 * UROM worker calls the function twice, first one to get address length and second one to get address data
 *
 * @worker_ctx [in]: DOCA UROM worker context
 * @addr [out]: set worker address
 * @addr_len [out]: set worker address length
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_addr(struct urom_worker_ctx *worker_ctx,
                     void *addr, uint64_t *addr_len)
{
    struct urom_worker_ucc *ucc_worker = (struct urom_worker_ucc *)
                                            worker_ctx->plugin_ctx;
    ucs_status_t status;

    if (ucc_worker->ucp_data.worker_address == NULL) {
        status = ucp_worker_get_address(ucc_worker->ucp_data.ucp_worker,
                        &ucc_worker->ucp_data.worker_address,
                        &ucc_worker->ucp_data.ucp_addrlen);
        if (status != UCS_OK) {
            DOCA_LOG_ERR("Failed to get ucp worker address");
            return DOCA_ERROR_INITIALIZATION;
        }
    }

    if (*addr_len < ucc_worker->ucp_data.ucp_addrlen) {
        /* Return required buffer size on error */
        *addr_len = ucc_worker->ucp_data.ucp_addrlen;
        return DOCA_ERROR_INVALID_VALUE;
    }

    *addr_len = ucc_worker->ucp_data.ucp_addrlen;
    memcpy(addr, ucc_worker->ucp_data.worker_address, *addr_len);
    return DOCA_SUCCESS;
}

/*
 * Check UCC worker tasks progress to get notifications
 *
 * @ctx [in]: DOCA UROM worker context
 * @notif_list [out]: set notification descriptors for completed tasks
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_progress(struct urom_worker_ctx *ctx,
                         ucs_list_link_t *notif_list)
{
    uint64_t                       lvalue     = 0;
    struct urom_worker_ucc        *ucc_worker = (struct urom_worker_ucc *)
                                                    ctx->plugin_ctx;
    struct urom_worker_notif_desc *nd;

    if (ucs_list_is_empty(&ucc_worker->completed_reqs)) {
        return DOCA_ERROR_EMPTY;
    }

    if (ucc_component_enabled) {
        lvalue = ucs_atomic_cswap64(&ucc_worker->list_lock, 0, 1);
        while (lvalue != 0) {
            lvalue = ucs_atomic_cswap64(&ucc_worker->list_lock, 0, 1);
        }
    }

    while (!ucs_list_is_empty(&ucc_worker->completed_reqs)) {
        nd = ucs_list_extract_head(&ucc_worker->completed_reqs,
                                   struct urom_worker_notif_desc, entry);
        ucs_list_add_tail(notif_list, &nd->entry);
    }

    if (ucc_component_enabled) {
        lvalue = ucs_atomic_cswap64(&ucc_worker->list_lock, 1, 0);
    }

    return DOCA_SUCCESS;
}

/*
 * Packing UCC notification
 *
 * @notif [in]: UCC notification to pack
 * @packed_notif_len [in/out]: set packed notification command buffer size
 * @packed_notif [out]: set packed notification command buffer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
urom_worker_ucc_notif_pack(struct urom_worker_notify *notif,
                           size_t *packed_notif_len,
                           void *packed_notif)
{
    void *pack_tail = packed_notif;
    int   pack_len;
    void *pack_head;

    /* Pack base command */
    pack_len  = ucs_offsetof(struct urom_worker_notify, plugin_notif)
                                + sizeof(struct urom_worker_notify_ucc);
    pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);

    memcpy(pack_head, notif, pack_len);
    *packed_notif_len = pack_len;

    return DOCA_SUCCESS;
}

/* Define UROM UCC plugin interface, set plugin functions */
static struct urom_worker_ucc_iface urom_worker_ucc = {
    .super.open       = urom_worker_ucc_open,
    .super.close      = urom_worker_ucc_close,
    .super.addr       = urom_worker_ucc_addr,
    .super.worker_cmd = urom_worker_ucc_worker_cmd,
    .super.progress   = urom_worker_ucc_progress,
    .super.notif_pack = urom_worker_ucc_notif_pack,
};

doca_error_t urom_plugin_get_iface(struct urom_plugin_iface *iface)
{
    if (iface == NULL) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    DOCA_STRUCT_CTOR(urom_worker_ucc.super);
    *iface = urom_worker_ucc.super;
    return DOCA_SUCCESS;
}

doca_error_t urom_plugin_get_version(uint64_t *version)
{
    if (version == NULL) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    *version = plugin_version;
    return DOCA_SUCCESS;
}
