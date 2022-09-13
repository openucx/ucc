#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

tl_mlx5_enabled=n
CHECK_TLS_REQUIRED(["mlx5"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    mlx5_happy=no
    if test "x$mlx5dv_happy" = "xyes" -a "x$have_mlx5dv_wr_raw_wqe" = "xyes"; then
       mlx5_happy=yes
    fi
    AC_MSG_RESULT([MLX5 support: $mlx5_happy])
    if test $mlx5_happy = "yes"; then
        tl_modules="${tl_modules}:mlx5"
        tl_mlx5_enabled=y
        CHECK_NEED_TL_PROFILING(["tl_mlx5"])
        AS_IF([test "$TL_PROFILING_REQUIRED" = "y"],
              [
                AC_DEFINE([HAVE_PROFILING_TL_MLX5], [1], [Enable profiling for TL MLX5])
                prof_modules="${prof_modules}:tl_mlx5"
              ], [])
    fi
], [])

AM_CONDITIONAL([TL_MLX5_ENABLED], [test "$tl_mlx5_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/mlx5/Makefile])
