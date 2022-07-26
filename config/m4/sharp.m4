#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_SHARP],[

AS_IF([test "x$sharp_checked" != "xyes"],[

sharp_happy="no"

AC_ARG_WITH([sharp],
            [AS_HELP_STRING([--with-sharp=(DIR)], [Enable the use of SHARP (default is guess).])],
            [], [with_sharp=guess])

AS_IF([test "x$with_sharp" != "xno"],
    [save_CPPFLAGS="$CPPFLAGS"
     save_CFLAGS="$CFLAGS"
     save_LDFLAGS="$LDFLAGS"

     AS_IF([test ! -z "$with_sharp" -a "x$with_sharp" != "xyes" -a "x$with_sharp" != "xguess"],
            [
            check_sharp_dir="$with_sharp"
            AS_IF([test -d "$with_sharp/lib64"],[libsuff="64"],[libsuff=""])
            check_sharp_libdir="$with_sharp/lib$libsuff"
            CPPFLAGS="-I$with_sharp/include $save_CPPFLAGS"
            LDFLAGS="-L$check_sharp_libdir $save_LDFLAGS"
            ])
        AS_IF([test ! -z "$with_sharp_libdir" -a "x$with_sharp_libdir" != "xyes"],
            [check_sharp_libdir="$with_sharp_libdir"
            LDFLAGS="-L$check_sharp_libdir $save_LDFLAGS"])

        AC_CHECK_HEADERS([sharp/api/sharp_coll.h],
            [AC_CHECK_LIB([sharp_coll] , [sharp_coll_init],
                           [sharp_happy="yes"],
                           [AC_MSG_WARN([SHARP is not detected. Disable.])
                            sharp_happy="no"])
            ], [sharp_happy="no"])


        AS_IF([test "x$sharp_happy" = "xyes"],
            [
                AC_SUBST(SHARP_CPPFLAGS, "-I$check_sharp_dir/include/ ")
                AC_SUBST(SHARP_LDFLAGS, "-lsharp_coll -L$check_sharp_dir/lib")
            ],
            [
                AS_IF([test "x$with_sharp" != "xguess"],
                    [AC_MSG_ERROR([SHARP support is requested but SHARP packages cannot be found])],
                    [AC_MSG_WARN([SHARP not found])])
            ])
        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [AC_MSG_WARN([SHARP was explicitly disabled])])

sharp_checked=yes
])
])
