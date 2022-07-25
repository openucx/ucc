#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

tl_cuda_enabled=n
CHECK_TLS_REQUIRED(["cuda"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    if test $cuda_happy = "yes" -a $nvml_happy = "yes"; then
       tl_modules="${tl_modules}:cuda"
       tl_cuda_enabled=y
        CHECK_NEED_TL_PROFILING(["tl_cuda"])
        AS_IF([test "$TL_PROFILING_REQUIRED" = "y"],
              [
                AC_DEFINE([HAVE_PROFILING_TL_CUDA], [1], [Enable profiling for TL CUDA])
                prof_modules="${prof_modules}:tl_cuda"
              ], [])
    fi
], [])

AM_CONDITIONAL([TL_CUDA_ENABLED], [test "$tl_cuda_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/cuda/Makefile])
