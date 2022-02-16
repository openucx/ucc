#
# Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
#

tl_mlx5_enabled=n
CHECK_TLS_REQUIRED(["mlx5"])
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
    CHECK_MLX5DV
    mlx5_happy=no
    if test "x$mlx5dv_happy" = "xyes" -a "x$have_mlx5dv_wr_raw_wqe" = "xyes"; then
       mlx5_happy=yes
    fi
    AC_MSG_RESULT([MLX5 support: $mlx5_happy])
    if test $mlx5_happy = "yes"; then
        tl_modules="${tl_modules}:mlx5"
        tl_mlx5_enabled=y
    fi
], [])

AM_CONDITIONAL([TL_MLX5_ENABLED], [test "$tl_mlx5_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/mlx5/Makefile])
