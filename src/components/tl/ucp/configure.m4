#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

tl_ucp_enabled=n
CHECK_TLS_REQUIRED(["ucp"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    if test $ucx_happy = "yes"; then
        tl_modules="${tl_modules}:ucp"
        tl_ucp_enabled=y
    fi
], [])

AM_CONDITIONAL([TL_UCP_ENABLED], [test "$tl_ucp_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/ucp/Makefile])
