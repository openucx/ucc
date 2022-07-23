#
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

tl_sharp_enabled=n
CHECK_TLS_REQUIRED(["sharp"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    CHECK_SHARP
    AC_MSG_RESULT([SHARP support: $sharp_happy])
    if test $sharp_happy = "yes"; then
       tl_modules="${tl_modules}:sharp"
       tl_sharp_enabled=y
       CHECK_NEED_TL_PROFILING(["tl_sharp"])
       AS_IF([test "$TL_PROFILING_REQUIRED" = "y"],
             [
               AC_DEFINE([HAVE_PROFILING_TL_SHARP], [1], [Enable profiling for TL SHARP])
               prof_modules="${prof_modules}:tl_sharp"
             ], [])
    fi
], [])

AM_CONDITIONAL([TL_SHARP_ENABLED], [test "$tl_sharp_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/sharp/Makefile])
