/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_SYS_H_
#define UCC_SYS_H_

#include "ucc/api/ucc_status.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_log.h"
#include <stddef.h>
#include <unistd.h>
#include <assert.h>
#include <sys/syscall.h>

ucc_status_t ucc_sysv_alloc(size_t *size, void **addr, int *shm_id);

ucc_status_t ucc_sysv_free(void *addr);

const char* ucc_sys_get_lib_path();

ucc_status_t ucc_sys_dirname(const char *path, char **out);

ucc_status_t ucc_sys_path_join(const char *path1, const char *path2,
                               char **out);

size_t ucc_get_page_size();

/**
 * Open an output stream according to user configuration:
 *   - file:<name> - file name, %p, %h, %c are substituted.
 *   - stdout
 *   - stderr
 *
 * @param [in]  config_str     The file name or name of the output stream
 *                             (stdout/stderr).
 * @param [in]  err_log_level  Logging level that should be used for printing
 *                             errors.
 * @param [out] p_fstream      Pointer that is filled with the stream handle.
 *                             User is responsible to close the stream handle then.
 * @param [out] p_need_close   Pointer to the variable that is set to whether
 *                             fclose() should be called to release resources (1)
 *                             or not (0).
 * @param [out] p_next_token   Pointer that is set to remainder of @config_str.
 * @param [out] p_filename     Pointer to the variable that is filled with the
 *                             resulted name of the log file (if it is not NULL).
 *                             Caller is responsible to release memory then.
 *
 * @return UCC_OK if successful, or error code otherwise.
 */
 ucc_status_t
 ucc_open_output_stream(const char *config_str, ucc_log_level_t err_log_level,
                        FILE **p_fstream, int *p_need_close,
                        const char **p_next_token, char **p_filename);

/**
 * @return Path to the main executable.
 */
 const char *ucc_get_exe();

 /**
 * @return Host name.
 */
const char *ucc_get_host_name();


/**
 * @return user name.
 */
const char *ucc_get_user_name();

/**
 * Get the first processor number we are bound to.
 */
int ucc_get_first_cpu();

/**
 * Get current thread (LWP) id.
 */
pid_t ucc_get_tid(void);

#endif
