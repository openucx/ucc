#
# Copyright (c) 2001-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_UCX],[
UCX_MIN_REQUIRED_MAJOR=1
UCX_MIN_REQUIRED_MINOR=11
AS_IF([test "x$ucx_checked" != "xyes"],[
    ucx_happy="no"

    AC_ARG_WITH([ucx],
            [AS_HELP_STRING([--with-ucx=(DIR)], [Enable the use of UCX (default is guess).])],
            [], [with_ucx=guess])

    AS_IF([test "x$with_ucx" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test ! -z "$with_ucx" -a "x$with_ucx" != "xyes" -a "x$with_ucx" != "xguess"],
        [
            AS_IF([test ! -d $with_ucx],
                  [AC_MSG_ERROR([Provided "--with-ucx=${with_ucx}" location does not exist])], [])
            check_ucx_dir="$with_ucx"
            check_ucx_libdir="$with_ucx/lib"
            CPPFLAGS="-I$with_ucx/include $save_CPPFLAGS"
            LDFLAGS="-L$check_ucx_libdir $save_LDFLAGS"
        ])

        AS_IF([test "x$check_ucx_dir" = "x" -a "x$HPCX_UCX_DIR" != "x"],
        [
            check_ucx_dir="$HPCX_UCX_DIR"
            check_ucx_libdir="$HPCX_UCX_DIR/lib"
            CPPFLAGS="-I$check_ucx_dir/include $save_CPPFLAGS"
            LDFLAGS="-L$check_ucx_libdir $save_LDFLAGS"
        ])

        AS_IF([test ! -z "$with_ucx_libdir" -a "x$with_ucx_libdir" != "xyes"],
        [
            check_ucx_libdir="$with_ucx_libdir"
            LDFLAGS="-L$check_ucx_libdir $save_LDFLAGS"
        ])

        AC_CHECK_HEADERS([ucp/api/ucp.h],
        [
            AC_CHECK_LIB([ucp], [ucp_tag_send_nb],
            [
                ucx_happy="yes"
            ],
            [
                ucx_happy="no"
            ], [-luct -lucm -lucs])
        ],
        [
            ucx_happy="no"
        ])

        AC_CHECK_HEADERS([ucs/sys/uid.h],
        [
            AC_CHECK_LIB([ucs], [ucs_get_system_id],
            [
                AC_DEFINE([HAVE_UCS_GET_SYSTEM_ID], 1, [Enable use of ucs unique machine identifier])
            ],
            [],[-luct -lucm -lucp])
        ],
        [])

        AS_IF([test "x$ucx_happy" = "xyes"],
        [
            AS_IF([test "x$check_ucx_dir" != "x"],
            [
                AC_MSG_RESULT([UCX dir: $check_ucx_dir])
                AC_SUBST(UCX_CPPFLAGS, "-I$check_ucx_dir/include/")
                AC_SUBST(UCS_CPPFLAGS, "-I$check_ucx_dir/include/")
                ucx_major=$(cat $check_ucx_dir/include/ucp/api/ucp_version.h | grep -Po "UCP_API_MAJOR\s+\K\d+")
                ucx_minor=$(cat $check_ucx_dir/include/ucp/api/ucp_version.h | grep -Po "UCP_API_MINOR\s+\K\d+")
                AC_MSG_RESULT([Detected UCX version: ${ucx_major}.${ucx_minor}])
                AS_IF([test $ucx_major -eq 1 && test $ucx_minor -lt ${UCX_MIN_REQUIRED_MINOR}],
                [
                   AC_MSG_ERROR([Required UCX version: ${UCX_MIN_REQUIRED_MAJOR}.${UCX_MIN_REQUIRED_MINOR}])
                   ucx_happy=no
                ], [])
            ])

            AS_IF([test "x$check_ucx_libdir" != "x"],
            [
                AC_SUBST(UCX_LDFLAGS, "-L$check_ucx_libdir")
                AC_SUBST(UCS_LDFLAGS, "-L$check_ucx_libdir")
                AC_SUBST(UCS_LIBDIR, $check_ucx_libdir)
            ])

            AC_SUBST(UCX_LIBADD, "-lucp -lucm")
            AC_SUBST(UCS_LIBADD, "-lucs")

            AC_CHECK_MEMBER(ucs_mpool_params_t.ops,
                [AC_DEFINE([UCS_HAVE_MPOOL_PARAMS], [1], [params interface for ucs_mpool_init])],
                [],
                [#include <ucs/datastruct/mpool.h>])
        ],
        [
            AS_IF([test "x$with_ucx" != "xguess"],
            [
                AC_MSG_ERROR([UCX support is requested but UCX packages cannot be found])
            ],
            [
                AC_MSG_WARN([UCX not found])
            ])
        ])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [
        AC_MSG_WARN([UCX was explicitly disabled])
    ])

    ucx_checked=yes
    AM_CONDITIONAL([HAVE_UCX], [test "x$ucx_happy" != xno])
])
])
