#
# Copyright (c) 2001-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_TL_COLL_PLUGINS],[
    AC_ARG_WITH([tlcp],
            [AS_HELP_STRING([--with-tlcp=(cp1,cp2)], [Enable build of TL collectives plugins])],
            [TLCP_REQUIRED=${with_tlcp}], [TLCP_REQUIRED=no])

    AS_IF([test "x$with_tlcp" != "xno"],
    [
        AM_CONDITIONAL([BUILD_TL_COLL_PLUGINS], [true])
        m4_include([config/m4/tl_coll_plugins_list.m4])
    ],
    [
        AC_MSG_RESULT([TL coll plugins are disabled])
        AM_CONDITIONAL([BUILD_TL_COLL_PLUGINS], [false])
    ])
])

AC_DEFUN([CHECK_TLCP_REQUIRED], [
    tlcp_name=$1
    CHECKED_TLCP_REQUIRED=n
    AS_IF([ test "$TLCP_REQUIRED" = "all" || test "$TLCP_REQUIRED" = "yes" ],
          [CHECKED_TLCP_REQUIRED=y],
    [
        for t in $(echo ${TLCP_REQUIRED} | tr "," " "); do
            AS_IF([ test "$t" == "$tlcp_name" ], [CHECKED_TLCP_REQUIRED=y], [])
        done
    ])
])
