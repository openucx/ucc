#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_IBVERBS],[
AS_IF([test "x$ibverbs_checked" != "xyes"],[
    ibverbs_happy="no"

    AC_ARG_WITH([ibverbs],
            [AS_HELP_STRING([--with-ibverbs=(DIR)], [Enable the use of IBVERBS (default is guess).])],
            [], [with_ibverbs=guess])

    AS_IF([test "x$with_ibverbs" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test ! -z "$with_ibverbs" -a "x$with_ibverbs" != "xyes" -a "x$with_ibverbs" != "xguess"],
        [
            AS_IF([test ! -d $with_ibverbs],
                  [AC_MSG_ERROR([Provided "--with-ibverbs=${with_ibverbs}" location does not exist])], [])
            check_ibverbs_dir="$with_ibverbs"
            check_ibverbs_libdir="$with_ibverbs/lib"
            CPPFLAGS="-I$with_ibverbs/include $save_CPPFLAGS"
            LDFLAGS="-L$check_ibverbs_libdir $save_LDFLAGS"
        ])

        AS_IF([test ! -z "$with_ibverbs_libdir" -a "x$with_ibverbs_libdir" != "xyes"],
        [
            check_ibverbs_libdir="$with_ibverbs_libdir"
            LDFLAGS="-L$check_ibverbs_libdir $save_LDFLAGS"
        ])
        AC_CHECK_HEADER([infiniband/verbs.h],
        [
            AC_CHECK_LIB([ibverbs], [ibv_get_device_list],
            [
                ibverbs_happy="yes"
            ],
            [
                ibverbs_happy="no"
            ])
        ],
        [
            AC_MSG_WARN([ibibverbs header files not found])
            ibverbs_happy=no
        ])


        AS_IF([test "x$ibverbs_happy" = "xyes"],
        [
            AS_IF([test "x$check_ibverbs_dir" != "x"],
            [
                AC_MSG_RESULT([IBVERBS dir: $check_ibverbs_dir])
                AC_SUBST(IBVERBS_CPPFLAGS, "-I$check_ibverbs_dir/include/")
            ])

            AS_IF([test "x$check_ibverbs_libdir" != "x"],
            [
                AC_SUBST(IBVERBS_LDFLAGS, "-L$check_ibverbs_libdir")
            ])

            AC_SUBST(IBVERBS_LIBADD, "-libverbs")
            AC_DEFINE([HAVE_IBVERBS], [1], [ibverbs support])

            AC_CHECK_HEADER([infiniband/mlx5dv.h],
            [
                AC_CHECK_LIB([mlx5], [mlx5dv_query_device],
                             [
                                 mlx5dv_happy="yes"
                                 AC_SUBST(MLX5DV_LIBADD, "-lmlx5")
                                 AC_DEFINE([HAVE_MLX5DV], [1], [mlx5dv support])
                             ],
                             [mlx5dv_happy=no], [-libverbs])

                AS_IF([test "x$mlx5dv_happy" = "xyes"],
                [
                    AC_CHECK_DECLS([mlx5dv_wr_raw_wqe],
                    [
                        have_mlx5dv_wr_raw_wqe=yes
                    ], [], [[#include <infiniband/mlx5dv.h>]])
                ], [])
            ],
            [
                AC_MSG_WARN([ibmlx5dv header files not found])
                mlx5dv_happy=no
            ])
        ],
        [
            AS_IF([test "x$with_ibverbs" != "xguess"],
            [
                AC_MSG_ERROR([IBVERBS support is requested but IBVERBS packages cannot be found $CPPFLAGS $LDFLAGS])
            ],
            [
                AC_MSG_WARN([IBVERBS not found])
            ])
        ])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [
        AC_MSG_WARN([IBVERBS was explicitly disabled])
    ])

    ibverbs_checked=yes
    AM_CONDITIONAL([HAVE_IBVERBS], [test "x$ibverbs_happy" != xno])
    AM_CONDITIONAL([HAVE_MLX5DV], [test "x$mlx5dv_happy" != xno])
])
])
