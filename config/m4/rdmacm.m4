#
# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_RDMACM],[
AS_IF([test "x$rdmacm_checked" != "xyes"],[
    rdmacm_happy="no"

    AC_ARG_WITH([rdmacm],
            [AS_HELP_STRING([--with-rdmacm=(DIR)], [Enable the use of rdmacm (default is guess).])],
            [], [with_rdmacm=guess])

    AS_IF([test "x$with_rdmacm" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test ! -z "$with_rdmacm" -a "x$with_rdmacm" != "xyes" -a "x$with_rdmacm" != "xguess"],
        [
            AS_IF([test ! -d $with_rdmacm],
                  [AC_MSG_ERROR([Provided "--with-rdmacm=${with_rdmacm}" location does not exist])], [])
            check_rdmacm_dir="$with_rdmacm"

        ],
        [
            check_rdmacm_dir="/usr"
        ]
        )

        AS_IF([test -d "$check_rdmacm_dir/lib64"],[libsuff="64"],[libsuff=""])

        check_rdmacm_libdir="$check_rdmacm_dir/lib$libsuff"
        CPPFLAGS="-I$check_rdmacm_dir/include $save_CPPFLAGS"
        LDFLAGS="-L$check_rdmacm_libdir $save_LDFLAGS"

        AC_CHECK_HEADER([$check_rdmacm_dir/include/rdma/rdma_cma.h],
        [
            AC_CHECK_LIB([rdmacm], [rdma_establish],
            [
                rdmacm_happy="yes"
            ],
            [
                rdmacm_happy="no"
            ])
        ],
        [
            AC_MSG_WARN([rdmacm header files not found])
            rdmacm_happy=no
        ])


        AS_IF([test "x$rdmacm_happy" = "xyes"],
        [
            AS_IF([test "x$check_rdmacm_dir" != "x"],
            [
                AC_MSG_RESULT([rdmacm dir: $check_rdmacm_dir])
                AC_SUBST(RDMACM_CPPFLAGS, "-I$check_rdmacm_dir/include/")
            ])

            AS_IF([test "x$check_rdmacm_libdir" != "x"],
            [
                AC_SUBST(RDMACM_LDFLAGS, "-L$check_rdmacm_libdir")
            ])

            AC_SUBST(RDMACM_LIBADD, "-lrdmacm")
            AC_DEFINE([HAVE_RDMACM], [1], [rdmacm support])
        ],
        [
            AS_IF([test "x$with_rdmacm" != "xguess"],
            [
                AC_MSG_ERROR([rdmacm support is requested but rdmacm packages cannot be found $CPPFLAGS $LDFLAGS])
            ],
            [
                AC_MSG_WARN([rdmacm not found])
            ])
        ])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [
        AC_MSG_WARN([rdmacm was explicitly disabled])
    ])

    rdmacm_checked=yes
    AM_CONDITIONAL([HAVE_RDMACM], [test "x$rdmacm_happy" != xno])
])
])
