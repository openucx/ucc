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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef UCC_CL_DOCA_UROM_COMMON_H_
#define UCC_CL_DOCA_UROM_COMMON_H_

#include <doca_dev.h>
#include <doca_urom.h>
#include <doca_pe.h>
#include <doca_error.h>

/* Function to check if a given device is capable of executing some task */
typedef doca_error_t (*ucc_cl_doca_urom_tasks_check)(struct doca_devinfo *);

/*
 * Struct contains domain shared buffer details
 */
struct ucc_cl_doca_urom_domain_buffer_attrs {
	void *buffer;	 /* Buffer address */
	size_t buf_len;	 /* Buffer length */
	void *memh;	 /* Buffer packed memory handle */
	size_t memh_len; /* Buffer packed memory handle length */
	void *mkey;	 /* Buffer packed memory key */
	size_t mkey_len; /* Buffer packed memory key length*/
};

/*
 * Open a DOCA device according to a given IB device name
 *
 * @value [in]: IB device name
 * @val_size [in]: input length, in bytes
 * @func [in]: pointer to a function that checks if the device have some task capabilities (Ignored if set to NULL)
 * @retval [out]: pointer to doca_dev struct, NULL if not found
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_open_doca_device_with_ibdev_name(
				const uint8_t *value,
				size_t val_size,
				ucc_cl_doca_urom_tasks_check func,
				struct doca_dev **retval);

/*
 * Start UROM service context
 *
 * @pe [in]: Progress engine
 * @dev [in]: service DOCA device
 * @nb_workers [in]: number of workers
 * @service [out]: service context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_start_urom_service(struct doca_pe *pe,
				struct doca_dev *dev,
				uint64_t nb_workers,
				struct doca_urom_service **service);

/*
 * Start UROM worker context
 *
 * @pe [in]: Progress engine
 * @service [in]: service context
 * @worker_id [in]: Worker id
 * @gid [in]: worker group id (optional attribute)
 * @nb_tasks [in]: number of tasks
 * @cpuset [in]: worker CPU affinity to set
 * @env [in]: worker environment variables array
 * @env_count [in]: worker environment variables array size
 * @plugins [in]: worker plugins
 * @worker [out]: set worker context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_start_urom_worker(struct doca_pe *pe,
				struct doca_urom_service *service,
				uint64_t worker_id,
				uint32_t *gid,
				uint64_t nb_tasks,
				doca_cpu_set_t *cpuset,
				char **env,
				size_t env_count,
				uint64_t plugins,
				struct doca_urom_worker **worker);

/*
 * Start UROM domain context
 *
 * @pe [in]: Progress engine
 * @oob [in]: OOB allgather operations
 * @worker_ids [in]: workers ids participate in domain
 * @workers [in]: workers participate in domain
 * @nb_workers [in]: number of workers in domain
 * @buffers [in]: shared buffers
 * @nb_buffers [out]: number of shared buffers
 * @domain [out]: domain context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ucc_cl_doca_urom_start_urom_domain(struct doca_pe *pe,
				struct doca_urom_domain_oob_coll *oob,
				uint64_t *worker_ids,
				struct doca_urom_worker **workers,
				size_t nb_workers,
				struct ucc_cl_doca_urom_domain_buffer_attrs *buffers,
				size_t nb_buffers,
				struct doca_urom_domain **domain);

#endif /* UCC_CL_DOCA_UROM_COMMON_H_ */
