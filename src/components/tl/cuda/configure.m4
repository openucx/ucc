#
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Check for NVLS support
AS_IF([test "$tl_cuda_enabled" = "y" -a "$nvls_happy" = "yes"],
[
    AC_DEFINE([HAVE_TL_CUDA_NVLS], [1], [Enable NVLS support in TL CUDA])
    AC_MSG_RESULT([TL CUDA NVLS support: enabled])
],
[
    AC_MSG_RESULT([TL CUDA NVLS support: disabled])
])

AM_CONDITIONAL([TL_CUDA_ENABLED], [test "$tl_cuda_enabled" = "y"])
AM_CONDITIONAL([TL_CUDA_NVLS_ENABLED], [test "$tl_cuda_enabled" = "y" -a "$nvls_happy" = "yes"])
AC_CONFIG_FILES([src/components/tl/cuda/Makefile])
