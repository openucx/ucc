#
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

tl_ucp_enabled=n
CHECK_TLS_REQUIRED(["ucp"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    if test $ucx_happy = "yes"; then
        tl_modules="${tl_modules}:ucp"
        tl_ucp_enabled=y
        CHECK_NEED_TL_PROFILING(["tl_ucp"])
        AS_IF([test "$TL_PROFILING_REQUIRED" = "y"],
              [
                AC_DEFINE([HAVE_PROFILING_TL_UCP], [1], [Enable profiling for TL UCP])
                prof_modules="${prof_modules}:tl_ucp"
              ], [])
    fi
], [])

AM_CONDITIONAL([TL_UCP_ENABLED], [test "$tl_ucp_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/ucp/Makefile])
