#
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_NVLS],[
AS_IF([test "x$nvls_checked" != "xyes"],[
    nvls_happy="no"

    # Use "check" as default to auto-detect NVLS support
    # "yes" = user explicitly requested --with-nvls (error if unavailable)
    # "no"  = user explicitly requested --without-nvls (skip entirely)
    # "check" = auto-detect (enable if available, silently disable if not)
    AC_ARG_WITH([nvls],
            [AS_HELP_STRING([--with-nvls], [Enable NVLS (NVLINK SHARP) support (default is auto-detect).])],
            [], [with_nvls=check])

    AS_IF([test "x$with_nvls" = "xno"],
    [
        AC_MSG_RESULT([NVLS was explicitly disabled])
    ],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test "x$cuda_happy" = "xyes"],
        [
            # Check for CUDA 12.0+ which supports NVLS
            AS_IF([test $CUDA_MAJOR_VERSION -ge 12],
            [
                nvls_happy="yes"
                AC_DEFINE([HAVE_NVLS], [1], [Enable NVLS support])
            ],
            [
                nvls_happy="no"
                AS_IF([test "x$with_nvls" = "xyes"],
                [
                    AC_MSG_ERROR([NVLS support is requested but NVLS cannot be enabled. Requires CUDA 12.0 or later (detected: $CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION).])
                ],
                [
                    AC_MSG_WARN([NVLS requires CUDA 12.0 or later, but CUDA $CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION was detected])
                ])
            ])
        ],
        [
            nvls_happy="no"
            AS_IF([test "x$with_nvls" = "xyes"],
            [
                AC_MSG_ERROR([NVLS support is requested but CUDA is not available. NVLS requires CUDA 12.0 or later.])
            ])
        ])

        AS_IF([test "x$nvls_happy" = "xyes"],
        [
            AC_MSG_RESULT([NVLS support: enabled])
        ])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"
    ])

    nvls_checked=yes
])
])
