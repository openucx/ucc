/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucc_sys.h"
#include <errno.h>
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

/**
 * @return Regular page size on the system.
 */
size_t ucc_get_page_size()
{
    static long page_size = 0;
    if (page_size == 0) {
        page_size = ucc_sysconf(_SC_PAGESIZE);
        if (page_size < 0) {
            page_size = 4096;
            ucc_debug("_SC_PAGESIZE undefined, setting default value to %ld",
                      page_size);
        }
    }

    return page_size;
}
