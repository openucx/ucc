#
# Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_MLX5DV],[
CHECK_IBVERBS
AS_IF([test "x$mlx5dv_checked" != "xyes" -a "x$ibverbs_happy" = "xyes"],[
    mlx5dv_happy="no"

    AC_ARG_WITH([mlx5dv],
            [AS_HELP_STRING([--with-mlx5dv=(DIR)], [Enable the use of MLX5DV (default is guess).])],
            [], [with_mlx5dv=guess])

    AS_IF([test "x$with_mlx5dv" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test ! -z "$with_mlx5dv" -a "x$with_mlx5dv" != "xyes" -a "x$with_mlx5dv" != "xguess"],
        [
            AS_IF([test ! -d $with_mlx5dv],
                  [AC_MSG_ERROR([Provided "--with-mlx5dv=${with_mlx5dv}" location does not exist])], [])
            check_mlx5dv_dir="$with_mlx5dv"
            check_mlx5dv_libdir="$with_mlx5dv/lib"
            CPPFLAGS="-I$with_mlx5dv/include $save_CPPFLAGS"
            LDFLAGS="-L$check_mlx5dv_libdir $save_LDFLAGS"
        ])

        AS_IF([test ! -z "$with_mlx5dv_libdir" -a "x$with_mlx5dv_libdir" != "xyes"],
        [
            check_mlx5dv_libdir="$with_mlx5dv_libdir"
            LDFLAGS="-L$check_mlx5dv_libdir $save_LDFLAGS"
        ])

        AC_CHECK_HEADER([infiniband/mlx5dv.h],
        [
            mlx5dv_happy="yes"
            AC_CHECK_DECLS([mlx5dv_wr_raw_wqe],
                [
                    have_mlx5dv_wr_raw_wqe=yes
                ], [], [[#include <infiniband/mlx5dv.h>]])
        ],
        [
            AC_MSG_WARN([ibmlx5dv header files not found])
            mlx5dv_happy=no
        ])


        AS_IF([test "x$mlx5dv_happy" = "xyes"],
        [
            AS_IF([test "x$check_mlx5dv_dir" != "x"],
            [
                AC_MSG_RESULT([MLX5DV dir: $check_mlx5dv_dir])
                AC_SUBST(MLX5DV_CPPFLAGS, "-I$check_mlx5dv_dir/include/")
            ])

            AS_IF([test "x$check_mlx5dv_libdir" != "x"],
            [
                AC_SUBST(MLX5DV_LDFLAGS, "-L$check_mlx5dv_libdir")
            ])

            AC_SUBST(MLX5DV_LIBADD, "-lmlx5dv")
        ],
        [
            AS_IF([test "x$with_mlx5dv" != "xguess"],
            [
                AC_MSG_ERROR([MLX5DV support is requested but MLX5DV packages cannot be found $CPPFLAGS $LDFLAGS])
            ],
            [
                AC_MSG_WARN([MLX5DV not found])
            ])
        ])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [
        AC_MSG_WARN([MLX5DV was explicitly disabled])
    ])

    mlx5dv_checked=yes
])
])
