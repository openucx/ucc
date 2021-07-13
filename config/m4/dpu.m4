#
# Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_DPU],[
AS_IF([test "x$dpu_checked" != "xyes"],[
    dpu_happy="no"
    
    AC_ARG_WITH([dpu],
            AC_HELP_STRING([--with-dpu=yes/no], [Enable/Disable DPU team]),
            [AS_IF([test "x$with_dpu" != "xno"], dpu_happy="yes", dpu_happy="no")],
            [dpu_happy="no"])
    AM_CONDITIONAL([HAVE_DPU], [test "x$dpu_happy" != xno])
    AC_MSG_RESULT([DPU support: $dpu_happy])
])
])