#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([ENABLE_MODULE_PROFILING],
[
    AS_IF([test "x$with_profiling" = xall],
        [
            prof_modules=":core:mc:tl_ucp"
            AC_DEFINE([HAVE_PROFILING_CORE], [1], [Enable profiling for CORE])
            AC_DEFINE([HAVE_PROFILING_TL_UCP], [1], [Enable profiling for TL UCP])
            AC_DEFINE([HAVE_PROFILING_MC], [1], [Enable profiling for MC])
        ],
        [
            case $1 in
            *core*)
                prof_modules="${prof_modules}:core"
                AC_DEFINE([HAVE_PROFILING_CORE], [1], [Enable profiling for CORE])
                ;;
            esac
            case $1 in
            *mc*)
                prof_modules="${prof_modules}:mc"
                AC_DEFINE([HAVE_PROFILING_MC], [1], [Enable profiling for MC])
                ;;
            esac
            case $1 in
            *tl_ucp*)
                prof_modules="${prof_modules}:tl_ucp"
                AC_DEFINE([HAVE_PROFILING_TL_UCP], [1], [Enable profiling for TL UCP])
                ;;
            esac
        ])
])

#
# Enables profiling support.
#
AC_ARG_ENABLE([profiling],
    AS_HELP_STRING([--enable-profiling], [Enable profiling support, default: NO]),
    [:],
    [enable_profiling=no])
AC_ARG_WITH([profiling],
    AS_HELP_STRING([--with-profiling], [Enable profiling for particular UCC components]),
    [:],
    [with_profiling=all])
AS_IF([test "x$enable_profiling" = xyes],
    [AS_MESSAGE([enabling profiling])
    AC_DEFINE([HAVE_PROFILING], [1], [Enable profiling])
    HAVE_PROFILING=yes
    ENABLE_MODULE_PROFILING(${with_profiling})]
    [:]
)
AM_CONDITIONAL([HAVE_PROFILING],[test "x$HAVE_PROFILING" = "xyes"])

#
# Enables logging levels above INFO for debug build
#
AC_ARG_ENABLE([debug],
    AS_HELP_STRING([--enable-debug], [Enable extra debugging code (default is NO).]),
    [:],
    [enable_debug=no])

AS_IF([test "x$enable_debug" = xyes],
    [AS_MESSAGE([debug build])
    AC_DEFINE([ENABLE_DEBUG], [1], [Enable debugging code])
    CFLAGS="$CFLAGS -O0 -g3"
    CXXFLAGS="$CXXFLAGS -O0 -g3"
    AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_POLL], [Highest log level])],
    [CFLAGS="$CFLAGS -O3 -g -DNDEBUG"
    CXXFLAGS="$CXXFLAGS -O3 -g -DNDEBUG"
    AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_DEBUG], [Highest log level])
    ])
