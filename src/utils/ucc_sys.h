/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
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

ucc_status_t ucc_sysv_alloc(size_t *size, void **addr, int *shm_id);

ucc_status_t ucc_sysv_free(void *addr);

size_t ucc_get_page_size();

#endif
=======
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_QUEUE_H_
#define UCC_QUEUE_H_

#include <unistd.h>
#include <assert.h>
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_log.h"

/*
 * If a certain system constant (name) is undefined on the underlying system the
 * sysconf routine returns -1.  ucs_sysconf return the negative value
 * a user and the user is responsible to define default value or abort.
 *
 * If an error occurs sysconf modified errno and ucs_sysconf aborts.
 *
 * Otherwise, a non-negative values is returned.
 */
static long ucc_sysconf(int name)
{
    long rc;
    errno = 0;

    rc = sysconf(name);
    ucc_assert(errno == 0); //was ucs_assert_always - needed in ucc?

    return rc;
}

static size_t ucc_get_page_size()
{
    static long page_size = 0;
    if (page_size == 0) {
        page_size = ucs_sysconf(_SC_PAGESIZE);
        if (page_size < 0) {
            page_size = 4096;
            ucc_debug("_SC_PAGESIZE undefined, setting default value to %ld",
                      page_size);
        }
    }

    return page_size;
}
