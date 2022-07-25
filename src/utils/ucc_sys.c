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
