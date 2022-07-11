#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
#

tl_rccl_enabled=n
CHECK_TLS_REQUIRED(["rccl"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    CHECK_RCCL
    AC_MSG_RESULT([RCCL support: $rccl_happy])
    if test $rccl_happy = "yes"; then
        tl_modules="${tl_modules}:rccl"
        tl_rccl_enabled=y
    fi
], [])

AM_CONDITIONAL([TL_RCCL_ENABLED], [test "$tl_rccl_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/rccl/Makefile])
