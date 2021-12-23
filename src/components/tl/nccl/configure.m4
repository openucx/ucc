#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

tl_nccl_enabled=n
CHECK_TLS_REQUIRED(["nccl"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    CHECK_NCCL
    AC_MSG_RESULT([NCCL support: $nccl_happy])
    if test $nccl_happy = "yes"; then
        tl_modules="${tl_modules}:nccl"
        tl_nccl_enabled=y
    fi
], [])

AM_CONDITIONAL([TL_NCCL_ENABLED], [test "$tl_nccl_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/nccl/Makefile])
