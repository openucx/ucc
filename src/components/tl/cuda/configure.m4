#
# Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
#

tl_cuda_enabled=n
CHECK_TLS_REQUIRED(["cuda"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    if test $cuda_happy = "yes" -a $nvml_happy = "yes"; then
       tl_modules="${tl_modules}:cuda"
       tl_cuda_enabled=y
    fi
], [])

AM_CONDITIONAL([TL_CUDA_ENABLED], [test "$tl_cuda_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/cuda/Makefile])
