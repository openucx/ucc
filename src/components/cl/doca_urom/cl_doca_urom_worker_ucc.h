/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
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

#ifndef UCC_CL_DOCA_UROM_WORKER_UCC_H_
#define UCC_CL_DOCA_UROM_WORKER_UCC_H_

#include <ucc/api/ucc.h>

#include <doca_log.h>
#include <doca_error.h>

#include <doca_urom.h>

#include "urom_ucc.h"

struct export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void         *packed_memh;
    size_t        packed_memh_len;
    void         *packed_key;
    size_t        packed_key_len;
    uint64_t      memh_id;
};

/* UCC context create result */
struct ucc_cl_doca_urom_context_create_result {
    void *context; /* Pointer to UCC context */
};

/* UCC team create result */
struct ucc_cl_doca_urom_team_create_result {
    void *team; /* Pointer to UCC team */
};

/* UCC collective result */
struct ucc_cl_doca_urom_collective_result {
    ucc_status_t status; /* UCC collective status */
};

/* UCC passive data channel result */
struct ucc_cl_doca_urom_pass_dc_result {
    ucc_status_t status; /* UCC data channel status */
};

/* UCC task result structure */
struct ucc_cl_doca_urom_result {
    doca_error_t result;	/* Task result */
    uint64_t dpu_worker_id; /* DPU worker id */
    union {
        struct ucc_cl_doca_urom_context_create_result context_create; /* Context create result */
        struct ucc_cl_doca_urom_team_create_result    team_create;    /* Team create result */
        struct ucc_cl_doca_urom_collective_result     collective;     /* Collective result */
        struct ucc_cl_doca_urom_pass_dc_result        pass_dc;        /* Passive data channel result */
    };
};

void ucc_cl_doca_urom_collective_finished(
		doca_error_t result, union doca_data cookie,
		uint64_t dpu_worker_id, ucc_status_t status);

ucc_status_t ucc_cl_doca_urom_buffer_export_ucc(
		ucp_context_h ucp_context, void *buf,
		size_t len, struct export_buf *ebuf);

/*
 * UCC team create callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 * @team [in]: pointer to UCC team
 */
void ucc_cl_doca_urom_team_create_finished(
		doca_error_t result, union doca_data cookie,
		uint64_t dpu_worker_id, void *team);

/*
 * UCC lib create callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 */
void ucc_cl_doca_urom_lib_create_finished(
		doca_error_t result, union doca_data cookie, uint64_t dpu_worker_id);

/*
 * UCC passive data channel callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 * @status [in]: channel creation status
 */
void ucc_cl_doca_urom_pss_dc_finished(
		doca_error_t result, union doca_data cookie,
		uint64_t dpu_worker_id, ucc_status_t status);

/*
 * UCC lib destroy callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 */
void ucc_cl_doca_urom_lib_destroy_finished(
		doca_error_t result, union doca_data cookie, uint64_t dpu_worker_id);

/*
 * UCC context create callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 * @context [in]: pointer to UCC context
 */
void ucc_cl_doca_urom_ctx_create_finished(
		doca_error_t result, union doca_data cookie,
		uint64_t dpu_worker_id, void *context);

/*
 * UCC lib create task callback function, will be called once the task is finished
 *
 * @result [in]: task status
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 */
typedef void (*ucc_cl_doca_urom_lib_create_finished_cb)(
	doca_error_t result, union doca_data cookie, uint64_t dpu_worker_id);

/*
 * UCC lib destroy task callback function, will be called once the task is finished
 *
 * @result [in]: task status
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 */
typedef void (*ucc_cl_doca_urom_lib_destroy_finished_cb)(
	doca_error_t result, union doca_data cookie, uint64_t dpu_worker_id);

/*
 * UCC context create task callback function, will be called once the task is finished
 *
 * @result [in]: task status
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @context [in]: pointer to UCC context
 */
typedef void (*ucc_cl_doca_urom_ctx_create_finished_cb)(
	doca_error_t result, union doca_data cookie,
	uint64_t dpu_worker_id, void *context);

/*
 * UCC context destroy task callback function, will be called once the task is finished
 *
 * @result [in]: task status
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 */
typedef void (*ucc_cl_doca_urom_ctx_destroy_finished_cb)(
	doca_error_t result, union doca_data cookie, uint64_t dpu_worker_id);

/*
 * UCC team create task callback function, will be called once the task is finished
 *
 * @result [in]: task status
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @team [in]: pointer to UCC team
 */
typedef void (*ucc_cl_doca_urom_team_create_finished_cb)(
	doca_error_t result, union doca_data cookie,
	uint64_t dpu_worker_id, void *team);

/*
 * UCC collective task callback function, will be called once the task is finished
 *
 * @result [in]: task status
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @status [in]: UCC status
 */
typedef void (*ucc_cl_doca_urom_collective_finished_cb)(
	doca_error_t result, union doca_data cookie,
	uint64_t dpu_worker_id, ucc_status_t status);

/*
 * UCC passive data channel task callback function, will be called once the task is finished
 *
 * @result [in]: task status
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @status [in]: UCC status
 */
typedef void (*ucc_cl_doca_urom_pd_channel_finished_cb)(
	doca_error_t result, union doca_data cookie,
	uint64_t dpu_worker_id, ucc_status_t status);

/*
 * Create UCC library task
 *
 * @worker_ctx [in]: DOCA UROM worker context
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @params [in]: UCC team parameters
 * @cb [in]: program callback to call once the task is finished
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_task_lib_create(
		struct doca_urom_worker *worker_ctx, union doca_data cookie,
		uint64_t dpu_worker_id, void *params, 
		ucc_cl_doca_urom_lib_create_finished_cb cb);

/*
 * Create UCC library destroy task
 *
 * @worker_ctx [in]: DOCA UROM worker context
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @cb [in]: program callback to call once the task is finished
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_task_lib_destroy(
		struct doca_urom_worker *worker_ctx, union doca_data cookie,
		uint64_t dpu_worker_id,
		ucc_cl_doca_urom_lib_destroy_finished_cb cb);

/*
 * Create UCC context task
 *
 * @worker_ctx [in]: DOCA UROM worker context
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @start [in]: the started index
 * @array [in]: array of indexes, set stride to <= 0 if array is used
 * @stride [in]: number of strides
 * @size [in]: collective context world size
 * @base_va [in]: shared buffer address
 * @len [in]: buffer length
 * @cb [in]: program callback to call once the task is finished
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_task_ctx_create(
		struct doca_urom_worker *worker_ctx, union doca_data cookie, 
		uint64_t dpu_worker_id, int64_t start, int64_t *array,
		int64_t stride, int64_t size, void *base_va, uint64_t len,
		ucc_cl_doca_urom_ctx_create_finished_cb cb);

/*
 * Create UCC context destroy task
 *
 * @worker_ctx [in]: DOCA UROM worker context
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @context [in]: pointer of UCC context
 * @cb [in]: program callback to call once the task is finished
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_task_ctx_destroy(
		struct doca_urom_worker *worker_ctx, union doca_data cookie,
		uint64_t dpu_worker_id, void *context,
		ucc_cl_doca_urom_ctx_destroy_finished_cb cb);

/*
 * Create UCC team task
 *
 * @worker_ctx [in]: DOCA UROM worker context
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @start [in]: team start index
 * @stride [in]: number of strides
 * @size [in]: stride size
 * @context [in]: UCC context
 * @cb [in]: program callback to call once the task is finished
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_task_team_create(
		struct doca_urom_worker *worker_ctx, union doca_data cookie,
		uint64_t dpu_worker_id, int64_t start, int64_t stride, int64_t size,
		void *context, ucc_cl_doca_urom_team_create_finished_cb cb);

/*
 * Create UCC collective task
 *
 * @worker_ctx [in]: DOCA UROM worker context
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @coll_args [in]: collective arguments
 * @team [in]: UCC team
 * @use_xgvmi [in]: if operation uses XGVMI
 * @work_buffer [in]: work buffer
 * @work_buffer_size [in]: buffer size
 * @team_size [in]: team size
 * @cb [in]: program callback to call once the task is finished
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_task_collective(
		struct doca_urom_worker *worker_ctx, union doca_data cookie,
		uint64_t dpu_worker_id, void *coll_args, void *team, int use_xgvmi,
		void *work_buffer, size_t work_buffer_size, size_t team_size,
		ucc_cl_doca_urom_collective_finished_cb cb);

/*
 * Create UCC passive data channel task
 *
 * @worker_ctx [in]: DOCA UROM worker context
 * @cookie [in]: user cookie
 * @dpu_worker_id [in]: UCC DPU worker id
 * @ucp_addr [in]: UCP worker address on host
 * @addr_len [in]: UCP worker address length
 * @cb [in]: program callback to call once the task is finished
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_task_pd_channel(
		struct doca_urom_worker *worker_ctx, union doca_data cookie, 
		uint64_t dpu_worker_id, void *ucp_addr, size_t addr_len,
		ucc_cl_doca_urom_pd_channel_finished_cb cb);

/*
 * This method inits UCC plugin.
 *
 * @plugin_id [in]: UROM plugin ID
 * @version [in]: plugin version on DPU side
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_save_plugin_id(
		uint64_t plugin_id, uint64_t version);

#endif /* UCC_CL_DOCA_UROM_WORKER_UCC_H_ */
