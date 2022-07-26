#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
#

tl_self_enabled=n
CHECK_TLS_REQUIRED(["self"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    tl_modules="${tl_modules}:self"
    tl_self_enabled=y
    CHECK_NEED_TL_PROFILING(["tl_self"])
    AS_IF([test "$TL_PROFILING_REQUIRED" = "y"],
          [
            AC_DEFINE([HAVE_PROFILING_TL_SELF], [1], [Enable profiling for TL SELF])
            prof_modules="${prof_modules}:tl_self"
          ], [])
], [])

AM_CONDITIONAL([TL_SELF_ENABLED], [test "$tl_self_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/self/Makefile])
