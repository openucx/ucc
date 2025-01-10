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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <doca_argp.h>
#include <doca_ctx.h>
#include <doca_log.h>

#include "cl_doca_urom_common.h"

DOCA_LOG_REGISTER(UCC::DOCA_CL : UROM_COMMON);

doca_error_t ucc_cl_doca_urom_start_urom_service(
		struct doca_pe *pe, struct doca_dev *dev, uint64_t nb_workers,
		struct doca_urom_service **service)
{
	enum doca_ctx_states      state;
	struct doca_urom_service *inst;
	doca_error_t              result, tmp_result;

	/* Create service context */
	result = doca_urom_service_create(&inst);
	if (result != DOCA_SUCCESS)
		return result;

	result = doca_pe_connect_ctx(pe, doca_urom_service_as_ctx(inst));
	if (result != DOCA_SUCCESS)
		goto service_cleanup;

	result = doca_urom_service_set_max_workers(inst, nb_workers);
	if (result != DOCA_SUCCESS)
		goto service_cleanup;

	result = doca_urom_service_set_dev(inst, dev);
	if (result != DOCA_SUCCESS)
		goto service_cleanup;

	result = doca_ctx_start(doca_urom_service_as_ctx(inst));
	if (result != DOCA_SUCCESS)
		goto service_cleanup;

	result = doca_ctx_get_state(doca_urom_service_as_ctx(inst), &state);
	if (result != DOCA_SUCCESS || state != DOCA_CTX_STATE_RUNNING)
		goto service_stop;

	*service = inst;
	return DOCA_SUCCESS;

service_stop:
	tmp_result = doca_ctx_stop(doca_urom_service_as_ctx(inst));
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("failed to stop UROM service");
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}

service_cleanup:
	tmp_result = doca_urom_service_destroy(inst);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("failed to destroy UROM service");
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
	return result;
}

doca_error_t ucc_cl_doca_urom_start_urom_worker(
		struct doca_pe *pe, struct doca_urom_service *service,
		uint64_t worker_id, uint32_t *gid, uint64_t nb_tasks,
		doca_cpu_set_t *cpuset, char **env, size_t env_count, uint64_t plugins,
		struct doca_urom_worker **worker)
{
	enum doca_ctx_states     state;
	struct doca_urom_worker *inst;
	doca_error_t             result, tmp_result;

	result = doca_urom_worker_create(&inst);
	if (result != DOCA_SUCCESS)
		return result;

	result = doca_urom_worker_set_service(inst, service);
	if (result != DOCA_SUCCESS)
		goto worker_cleanup;

	result = doca_pe_connect_ctx(pe, doca_urom_worker_as_ctx(inst));
	if (result != DOCA_SUCCESS)
		goto worker_cleanup;

	result = doca_urom_worker_set_id(inst, worker_id);
	if (result != DOCA_SUCCESS)
		goto worker_cleanup;

	if (gid != NULL) {
		result = doca_urom_worker_set_gid(inst, *gid);
		if (result != DOCA_SUCCESS)
			goto worker_cleanup;
	}

	if (env != NULL) {
		result = doca_urom_worker_set_env(inst, env, env_count);
		if (result != DOCA_SUCCESS)
			goto worker_cleanup;
	}

	result = doca_urom_worker_set_max_inflight_tasks(inst, nb_tasks);
	if (result != DOCA_SUCCESS)
		goto worker_cleanup;

	result = doca_urom_worker_set_plugins(inst, plugins);
	if (result != DOCA_SUCCESS)
		goto worker_cleanup;

	if (cpuset != NULL) {
		result = doca_urom_worker_set_cpuset(inst, *cpuset);
		if (result != DOCA_SUCCESS)
			goto worker_cleanup;
	}

	result = doca_ctx_start(doca_urom_worker_as_ctx(inst));
	if (result != DOCA_ERROR_IN_PROGRESS)
		goto worker_cleanup;

	result = doca_ctx_get_state(doca_urom_worker_as_ctx(inst), &state);
	if (result != DOCA_SUCCESS)
		goto worker_stop;

	if (state != DOCA_CTX_STATE_STARTING) {
		result = DOCA_ERROR_BAD_STATE;
		goto worker_stop;
	}

	*worker = inst;
	return DOCA_SUCCESS;

worker_stop:
	tmp_result = doca_ctx_stop(doca_urom_worker_as_ctx(inst));
	if (tmp_result != DOCA_SUCCESS && tmp_result != DOCA_ERROR_IN_PROGRESS) {
		DOCA_LOG_ERR("failed to request stop UROM worker");
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}

	do {
		doca_pe_progress(pe);
		doca_ctx_get_state(doca_urom_worker_as_ctx(inst), &state);
	} while (state != DOCA_CTX_STATE_IDLE);

worker_cleanup:
	tmp_result = doca_urom_worker_destroy(inst);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("failed to destroy UROM worker");
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}

	return result;
}

doca_error_t ucc_cl_doca_urom_start_urom_domain(
		struct doca_pe *pe, struct doca_urom_domain_oob_coll *oob, 
		uint64_t *worker_ids, struct doca_urom_worker **workers,
		size_t nb_workers, struct ucc_cl_doca_urom_domain_buffer_attrs *buffers,
		size_t nb_buffers, struct doca_urom_domain **domain)
{
	struct doca_urom_domain *inst;
	enum doca_ctx_states     state;
	doca_error_t             result, tmp_result;
	size_t                   i;

	result = doca_urom_domain_create(&inst);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("failed to create domain");
		return result;
	}

	result = doca_pe_connect_ctx(pe, doca_urom_domain_as_ctx(inst));
	if (result != DOCA_SUCCESS)
		goto domain_destroy;

	result = doca_urom_domain_set_oob(inst, oob);
	if (result != DOCA_SUCCESS)
		goto domain_destroy;

	result = doca_urom_domain_set_workers(inst, worker_ids, workers, nb_workers);
	if (result != DOCA_SUCCESS)
		goto domain_destroy;

    /* The buffers in the domain are used for gets/puts from the host without
       XGVMI. Also, the domain is used for the OOB exchange given to the DPU-
       side UCC instance */
	if (nb_workers != 0 && buffers != NULL) {
		result = doca_urom_domain_set_buffers_count(inst, nb_buffers);
		if (result != DOCA_SUCCESS)
			goto domain_destroy;

		for (i = 0; i < nb_buffers; i++) {
			result = doca_urom_domain_add_buffer(inst,
							     buffers[i].buffer,
							     buffers[i].buf_len,
							     buffers[i].memh,
							     buffers[i].memh_len,
							     buffers[i].mkey,
							     buffers[i].mkey_len);
			if (result != DOCA_SUCCESS)
				goto domain_destroy;
		}
	}

	result = doca_ctx_start(doca_urom_domain_as_ctx(inst));
	if (result != DOCA_ERROR_IN_PROGRESS)
		goto domain_stop;

	result = doca_ctx_get_state(doca_urom_domain_as_ctx(inst), &state);
	if (result != DOCA_SUCCESS)
		goto domain_stop;

	if (state != DOCA_CTX_STATE_STARTING) {
		result = DOCA_ERROR_BAD_STATE;
		goto domain_stop;
	}

	*domain = inst;
	return DOCA_SUCCESS;

domain_stop:
	tmp_result = doca_ctx_stop(doca_urom_domain_as_ctx(inst));
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("failed to stop UROM domain");
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}

domain_destroy:
	tmp_result = doca_urom_domain_destroy(inst);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("failed to destroy UROM domain");
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
	return result;
}


doca_error_t ucc_cl_doca_urom_open_doca_device_with_ibdev_name(
		const uint8_t *value, size_t val_size,
		ucc_cl_doca_urom_tasks_check func, struct doca_dev **retval)
{
	char                  buf[DOCA_DEVINFO_IBDEV_NAME_SIZE]      = {};
	char                  val_copy[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {};
	struct doca_devinfo **dev_list;
	uint32_t              nb_devs;
	int                   res;
	size_t                i;

	/* Set default return value */
	*retval = NULL;

	/* Setup */
	if (val_size > DOCA_DEVINFO_IBDEV_NAME_SIZE) {
		DOCA_LOG_ERR("Value size too large. failed to locate device");
		return DOCA_ERROR_INVALID_VALUE;
	}
	memcpy(val_copy, value, val_size);

	res = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("failed to load doca devices list: %s", doca_error_get_descr(res));
		return res;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		res = doca_devinfo_get_ibdev_name(dev_list[i], buf, DOCA_DEVINFO_IBDEV_NAME_SIZE);
		if (res == DOCA_SUCCESS && strncmp(buf, val_copy, val_size) == 0) {
			/* If any special capabilities are needed */
			if (func != NULL && func(dev_list[i]) != DOCA_SUCCESS)
				continue;

			/* if device can be opened */
			res = doca_dev_open(dev_list[i], retval);
			if (res == DOCA_SUCCESS) {
				doca_devinfo_destroy_list(dev_list);
				return res;
			}
		}
	}

	DOCA_LOG_WARN("Matching device not found");
	res = DOCA_ERROR_NOT_FOUND;

	doca_devinfo_destroy_list(dev_list);
	return res;
}
