#
# Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
#

tl_shm_enabled=n
CHECK_TLS_REQUIRED(["shm"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    AC_MSG_RESULT([SHM support: yes])
    tl_modules="${tl_modules}:shm"
    tl_shm_enabled=y
], [])

AM_CONDITIONAL([TL_SHM_ENABLED], [test "$tl_shm_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/shm/Makefile])
