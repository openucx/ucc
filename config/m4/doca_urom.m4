#
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_DOCA_UROM],[
AS_IF([test "x$doca_urom_checked" != "xyes"],[
    doca_urom_happy="no"
    AC_ARG_WITH([doca_urom],
            [AS_HELP_STRING([--with-doca_urom=(DIR)], [Enable the use of DOCA_UROM (default is guess).])],
            [], [with_doca_urom=guess])
    AS_IF([test "x$with_doca_urom" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_LDFLAGS="$LDFLAGS"
        AS_IF([test ! -z "$with_doca_urom" -a "x$with_doca_urom" != "xyes" -a "x$with_doca_urom" != "xguess"],
        [
            AS_IF([test ! -d $with_doca_urom],
                  [AC_MSG_ERROR([Provided "--with-doca_urom=${with_doca_urom}" location does not exist])])
            check_doca_urom_dir="$with_doca_urom"
            check_doca_urom_libdir="$with_doca_urom/lib64"
            CPPFLAGS="-I$with_doca_urom/include $UCS_CPPFLAGS $save_CPPFLAGS"
            LDFLAGS="-L$check_doca_urom_libdir $save_LDFLAGS"
        ])
        AS_IF([test ! -z "$with_doca_urom_libdir" -a "x$with_doca_urom_libdir" != "xyes"],
        [
            check_doca_urom_libdir="$with_doca_urom_libdir"
            LDFLAGS="-L$check_doca_urom_libdir $save_LDFLAGS"
        ])
        AC_CHECK_HEADERS([doca_urom.h],
        [
            AC_CHECK_LIB([doca_urom], [doca_urom_service_create],
            [
                doca_urom_happy="yes"
            ],
            [
                echo "CPPFLAGS: $CPPFLAGS"
                doca_urom_happy="no"
            ], [-ldoca_common -ldoca_argp -ldoca_urom])
        ],
        [
            doca_urom_happy="no"
        ])
        AS_IF([test "x$doca_urom_happy" = "xyes"],
        [
            AS_IF([test "x$check_doca_urom_dir" != "x"],
            [
                AC_MSG_RESULT([DOCA_UROM dir: $check_doca_urom_dir])
                AC_SUBST(DOCA_UROM_CPPFLAGS, "-I$check_doca_urom_dir/include/ $doca_urom_old_headers")
            ])
            AS_IF([test "x$check_doca_urom_libdir" != "x"],
            [
                AC_SUBST(DOCA_UROM_LDFLAGS, "-L$check_doca_urom_libdir")
            ])
            AC_SUBST(DOCA_UROM_LIBADD, "-ldoca_common -ldoca_argp -ldoca_urom")
            AC_DEFINE([HAVE_DOCA_UROM], 1, [Enable DOCA_UROM support])
        ],
        [
            AS_IF([test "x$with_doca_urom" != "xguess"],
            [
                AC_MSG_ERROR([DOCA_UROM support is requested but DOCA_UROM packages cannot be found! $CPPFLAGS $LDFLAGS])
            ],
            [
                AC_MSG_WARN([DOCA_UROM not found])
            ])
        ])
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"
    ],
    [
        AC_MSG_WARN([DOCA_UROM was explicitly disabled])
    ])
    doca_urom_checked=yes
    AM_CONDITIONAL([HAVE_DOCA_UROM], [test "x$doca_urom_happy" != xno])
])])
