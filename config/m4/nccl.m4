#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_NCCL],[
AS_IF([test "x$nccl_checked" != "xyes"],[
    nccl_happy="no"

    AC_ARG_WITH([nccl],
            [AS_HELP_STRING([--with-nccl=(DIR)], [Enable the use of NCCL (default is guess).])],
            [], [with_nccl=guess])

    AS_IF([test "x$with_nccl" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test ! -z "$with_nccl" -a "x$with_nccl" != "xyes" -a "x$with_nccl" != "xguess"],
        [
            AS_IF([test ! -d $with_nccl],
                  [AC_MSG_ERROR([Provided "--with-nccl=${with_nccl}" location does not exist])], [])
            check_nccl_dir="$with_nccl"
            check_nccl_libdir="$with_nccl/lib"
            CPPFLAGS="-I$with_nccl/include $save_CPPFLAGS"
            LDFLAGS="-L$check_nccl_libdir $save_LDFLAGS"
        ])

        AS_IF([test ! -z "$with_nccl_libdir" -a "x$with_nccl_libdir" != "xyes"],
        [
            check_nccl_libdir="$with_nccl_libdir"
            LDFLAGS="-L$check_nccl_libdir $save_LDFLAGS"
        ])

        AS_IF([test "x$cuda_happy" = "xyes"],
        [
            CPPFLAGS="$CUDA_CPPFLAGS $CPPFLAGS"
            LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"
            AC_CHECK_HEADERS([nccl.h],
            [
                AC_CHECK_LIB([nccl], [ncclCommInitRank],
                [
                    nccl_happy="yes"
                ],
                [
                    nccl_happy="no"
                ], [-lcuda])
            ],
            [
                nccl_happy="no"
            ])
        ],
        [
            nccl_happy="no"
        ])

        AS_IF([test "x$nccl_happy" = "xyes"],
        [
            AS_IF([test "x$check_nccl_dir" != "x"],
            [
                AC_MSG_RESULT([NCCL dir: $check_nccl_dir])
                AC_SUBST(NCCL_CPPFLAGS, "-I$check_nccl_dir/include/")
            ])

            AS_IF([test "x$check_nccl_libdir" != "x"],
            [
                AC_SUBST(NCCL_LDFLAGS, "-L$check_nccl_libdir")
            ])

            AC_SUBST(NCCL_LIBADD, "-lnccl")
        ],
        [
            AS_IF([test "x$with_nccl" != "xguess"],
            [
                AC_MSG_ERROR([NCCL support is requested but NCCL packages cannot be found $CPPFLAGS $LDFLAGS])
            ],
            [
                AC_MSG_WARN([NCCL not found])
            ])
        ])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [
        AC_MSG_WARN([NCCL was explicitly disabled])
    ])

    nccl_checked=yes
])
])
