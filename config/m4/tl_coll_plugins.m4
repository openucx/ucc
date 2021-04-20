#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_TL_COLL_PLUGINS],[
    AC_ARG_WITH([tlcp],
            [AS_HELP_STRING([--with-tlcp=(tl1,tl2)], [Enable build of TL collectives plugins])],
            [], [with_tlcp=no])

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

