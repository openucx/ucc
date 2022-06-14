#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
# Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_RCCL],[
AS_IF([test "x$rccl_checked" != "xyes"],[
    rccl_happy="no"

    AC_ARG_WITH([rccl],
            [AS_HELP_STRING([--with-rccl=(DIR)], [Enable the use of RCCL (default is guess).])],
            [], [with_rccl=guess])

    AS_IF([test "x$with_rccl" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test ! -z "$with_rccl" -a "x$with_rccl" != "xyes" -a "x$with_rccl" != "xguess"],
        [
            AS_IF([test ! -d $with_rccl],
                  [AC_MSG_ERROR([Provided "--with-rccl=${with_rccl}" location does not exist])], [])])
            check_rccl_dir="$with_rccl"
            check_rccl_libdir="$with_rccl/lib"
            CPPFLAGS="-I$with_rccl/include $save_CPPFLAGS"
            LDFLAGS="-L$check_rccl_libdir $save_LDFLAGS"
        ])

        AS_IF([test ! -z "$with_rccl_libdir" -a "x$with_rccl_libdir" != "xyes"],
        [
            check_rccl_libdir="$with_rccl_libdir"
            LDFLAGS="-L$check_rccl_libdir $save_LDFLAGS"
        ])

        AS_IF([test "x$rocm_happy" = "xyes"],
        [
            CPPFLAGS="$HIP_CPPFLAGS $CPPFLAGS"
            LDFLAGS="$ROCM_LDFLAGS $LDFLAGS"
            AC_CHECK_HEADER([rccl/rccl.h],
            [
                AC_CHECK_LIB([rccl], [ncclCommInitRank],
                [
                    rccl_happy="yes"
                ],
                [
                    rccl_happy="no"
                ])
            ],
            [
                AC_CHECK_HEADER([rccl.h],
                [
                    AC_CHECK_LIB([rccl], [ncclCommInitRank],
 	    	    [
			rccl_happy="yes"
		        rccl_old_headers="-DRCCL_OLD_HEADERS"
                    ],
                    [
			rccl_happy="no"
                    ])
                ]),
            ])
        ],
        [
            rccl_happy="no"
        ])

        AS_IF([test "x$rccl_happy" = "xyes"],
        [
            AS_IF([test "x$check_rccl_dir" != "x"],
            [
                AC_MSG_RESULT([RCCL dir: $check_rccl_dir])
                AC_SUBST(RCCL_CPPFLAGS, "-I$check_rccl_dir/include/ $rccl_old_headers")
            ])

            AS_IF([test "x$check_rccl_libdir" != "x"],
            [
                AC_SUBST(RCCL_LDFLAGS, "-L$check_rccl_libdir")
            ])

            AC_SUBST(RCCL_LIBADD, "-lrccl")
        ],
        [
            AS_IF([test "x$with_rccl" != "xguess"],
            [
                AC_MSG_ERROR([RCCL support is requested but RCCL packages cannot be found! $CPPFLAGS $LDFLAGS])
            ],
            [
                AC_MSG_WARN([RCCL not found])
            ])
        ])

        CFLAGS="$save_CFLAGS -D__HIP_PLATFORM_AMD__"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"
    ],
    [
        AC_MSG_WARN([RCCL was explicitly disabled])
    ])

    rccl_checked=yes
])
