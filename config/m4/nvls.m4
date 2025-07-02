#
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_NVLS],[
AS_IF([test "x$nvls_checked" != "xyes"],[
    nvls_happy="no"

    AC_ARG_WITH([nvls],
            [AS_HELP_STRING([--with-nvls], [Enable NVLS (NVLINK SHARP) support (default is no).])],
            [], [with_nvls=no])

    AS_IF([test "x$with_nvls" != "xno"],
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
                AC_MSG_WARN([NVLS requires CUDA 12.0 or later, but CUDA $CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION was detected])
            ])
        ],
        [
            nvls_happy="no"
        ])

        AS_IF([test "x$nvls_happy" = "xyes"],
        [
            AC_MSG_RESULT([NVLS support: enabled])
        ],
        [
            AS_IF([test "x$with_nvls" = "xyes"],
            [
                AC_MSG_ERROR([NVLS support is requested but NVLS cannot be enabled. Requires CUDA 12.0 or later (detected: $CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION).])
            ],
            [
                AC_MSG_WARN([NVLS not available - requires CUDA 12.0+ with NVLS support])
                nvls_happy="no"
            ])
        ])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [
        AC_MSG_WARN([NVLS was explicitly disabled])
    ])

    nvls_checked=yes
])
])
