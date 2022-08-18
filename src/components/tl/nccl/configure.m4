#
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        CHECK_NEED_TL_PROFILING(["tl_nccl"])
        AS_IF([test "$TL_PROFILING_REQUIRED" = "y"],
              [
                AC_DEFINE([HAVE_PROFILING_TL_NCCL], [1], [Enable profiling for TL NCCL])
                prof_modules="${prof_modules}:tl_nccl"
              ], [])
    fi
], [])

AM_CONDITIONAL([TL_NCCL_ENABLED], [test "$tl_nccl_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/nccl/Makefile])
