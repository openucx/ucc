/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "ucc_sys.h"
#include "ucc_math.h"
#include "ucc_log.h"
#include <sys/shm.h>
#include <unistd.h>
#include <errno.h>
#include <dlfcn.h>
#include <libgen.h>
#include "ucc_string.h"

ucc_status_t ucc_sysv_alloc(size_t *size, void **addr, int *shm_id)
{
    size_t alloc_size;
    void *ptr;
    int ret;

    alloc_size = ucc_align_up(*size, getpagesize());

    *shm_id = shmget(IPC_PRIVATE, alloc_size, IPC_CREAT | 0666);
    if (*shm_id < 0) {
        ucc_error("failed to shmget with IPC_PRIVATE, size %zd, IPC_CREAT "
                  "errno: %d(%s)", alloc_size, errno, strerror(errno));
        return UCC_ERR_NO_RESOURCE;
    }

    ptr = shmat(*shm_id, NULL, 0);
    /* Remove segment, the attachment keeps a reference to the mapping */
    /* FIXME having additional attaches to a removed segment is not portable
    * behavior */
    ret = shmctl(*shm_id, IPC_RMID, NULL);
    if (ret != 0) {
        ucc_warn("shmctl(IPC_RMID, shmid=%d) errno: %d(%s)", *shm_id, errno,
                 strerror(errno));
    }

    if (ptr == (void*)-1) {
        ucc_error("failed to shmat errno: %d(%s)", errno, strerror(errno));
        if (errno == ENOMEM) {
            return UCC_ERR_NO_MEMORY;
        } else {
            return UCC_ERR_NO_MESSAGE;
        }
    }
    *size = alloc_size;
    *addr = ptr;

    return UCC_OK;
}

ucc_status_t ucc_sysv_free(void *addr)
{
    int ret;

    ret = shmdt(addr);
    if (ret) {
        ucc_warn("failed to detach shm at %p, errno: %d(%s)", addr, errno,
                 strerror(errno));
        return UCC_ERR_INVALID_PARAM;
    }

    return UCC_OK;
}

/**
 * @return Regular page size on the system.
 */
size_t ucc_get_page_size()
{
    static long page_size = 0;
    if (page_size == 0) {
        page_size = sysconf(_SC_PAGESIZE);
        if (page_size < 0) {
            page_size = 4096;
            ucc_debug("_SC_PAGESIZE undefined, setting default value to %ld",
                      page_size);
        }
    }

    return page_size;
}

static ucc_status_t ucc_sys_get_lib_info(Dl_info *dl_info)
{
    int ret;

    (void)dlerror();
    ret = dladdr(ucc_sys_get_lib_info, dl_info);
    if (ret == 0) {
        return UCC_ERR_NO_MESSAGE;
    }

    return UCC_OK;
}


const char* ucc_sys_get_lib_path()
{
    ucc_status_t status;
    Dl_info      dl_info;

    status = ucc_sys_get_lib_info(&dl_info);
    if (status != UCC_OK) {
        return NULL;
    }

    return dl_info.dli_fname;
}

ucc_status_t ucc_sys_dirname(const char *path, char **out)
{
    char *path_dup = strdup(path);
    char *dirname_path;

    if (!path_dup) {
        return UCC_ERR_NO_MEMORY;
    }
    dirname_path = strdup(dirname(path_dup));
    free(path_dup);
    *out = dirname_path;
    return UCC_OK;
}

ucc_status_t ucc_sys_path_join(const char *path1, const char *path2, char **out)
{
    const char *strs[3] = {path1, "/", path2};

    return ucc_str_concat_n(strs, 3, out);
}

ucc_status_t
ucc_open_output_stream(const char *config_str, ucc_log_level_t err_log_level,
                       FILE **p_fstream, int *p_need_close,
                       const char **p_next_token, char **p_filename)
{
    FILE *output_stream;
    char filename[256];
    char *template;
    const char *p;
    size_t len;

    *p_next_token   = config_str;
    if (p_filename != NULL) {
        *p_filename = NULL;
    }

    len = strcspn(config_str, ":");
    if (!strncmp(config_str, "stdout", len)) {
        *p_fstream    = stdout;
        *p_need_close = 0;
        *p_next_token = config_str + len;
    } else if (!strncmp(config_str, "stderr", len)) {
        *p_fstream    = stderr;
        *p_need_close = 0;
        *p_next_token = config_str + len;
    } else {
        if (!strncmp(config_str, "file:", 5)) {
            p = config_str + 5;
        } else {
            p = config_str;
        }

        len = strcspn(p, ":");
        template = strndup(p, len);
        ucc_fill_filename_template(template, filename, sizeof(filename));
        free(template);

        output_stream = fopen(filename, "w");
        if (output_stream == NULL) {
            ucc_log(err_log_level, "failed to open '%s' for writing: %m",
                    filename);
            return UCC_ERR_IO_ERROR;
        }

        if (p_filename != NULL) {
            *p_filename = ucc_strdup(filename, "filename");
            if (*p_filename == NULL) {
                ucc_log(err_log_level, "failed to allocate filename for '%s'",
                        filename);
                fclose(output_stream);
                return UCC_ERR_NO_MEMORY;
            }
        }

        *p_fstream    = output_stream;
        *p_need_close = 1;
        *p_next_token = p + len;
    }

    return UCC_OK;
}

int ucc_get_first_cpu()
{
    //TODO: fix, not implemented
    return 0;
}

const char *ucc_get_user_name()
{
    static char username[256] = {0};

    if (*username == 0) {
        getlogin_r(username, sizeof(username));
    }
    return username;
}

const char *ucc_get_host_name()
{
    static char hostname[HOST_NAME_MAX] = {0};

    if (*hostname == 0) {
        gethostname(hostname, sizeof(hostname));
        strtok(hostname, ".");
    }
    return hostname;
}

pid_t ucc_get_tid(void)
{
#ifdef SYS_gettid
    return syscall(SYS_gettid);
#elif defined(HAVE_SYS_THR_H)
    long id;

    thr_self(&id);
    return (id);
#else
#error "Port me"
#endif
}

const char *ucc_get_exe()
{
    static char exe[1024];
    int ret;

    ret = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (ret < 0) {
        exe[0] = '\0';
    } else {
        exe[ret] = '\0';
    }

    return exe;
}
