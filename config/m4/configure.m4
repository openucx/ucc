#
# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([ENABLE_MODULE_PROFILING],
[
    AS_IF([test "x$with_profiling" = xall],
        [
            prof_modules=":core:mc:cl_hier"
            AC_DEFINE([HAVE_PROFILING_CORE], [1], [Enable profiling for CORE])
            AC_DEFINE([HAVE_PROFILING_CL_HIER], [1], [Enable profiling for CL HIER])
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
            *cl_hier*)
                prof_modules="${prof_modules}:cl_hier"
                AC_DEFINE([HAVE_PROFILING_CL_HIER], [1], [Enable profiling for CL HIER])
                ;;
            esac
        ])
])

AC_DEFUN([CHECK_NEED_TL_PROFILING], [
    tl_name=$1
    TL_PROFILING_REQUIRED=n
    AS_IF([ test x"$with_profiling" = "xall" ], [TL_PROFILING_REQUIRED=y],
    [
       for t in $(echo ${with_profiling} | tr ":" " "); do
           AS_IF([ test "$t" == "$tl_name" ], [TL_PROFILING_REQUIRED=y], [])
       done
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

# use -g3 if compiler supports it, otherwise just -g
    CHECK_COMPILER_FLAG([-g3], [-g3],
                        [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                        [CFLAGS="$CFLAGS -O0 -g3"
                         CXXFLAGS="$CXXFLAGS -O0 -g3"],
                        [CFLAGS="$CFLAGS -O0 -g"
                         CXXFLAGS="$CXXFLAGS -O0 -g"])
    AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_POLL], [Highest log level])],
    [CFLAGS="$CFLAGS -O3 -g -DNDEBUG"
    CXXFLAGS="$CXXFLAGS -O3 -g -DNDEBUG"
    AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_DEBUG], [Highest log level])
    ])

#
# Enables additional checks
#
AC_ARG_ENABLE([assert],
    AS_HELP_STRING([--enable-assert], [Enable extra correctness checks (default is NO).]),
    [AC_DEFINE([UCC_ENABLE_ASSERT], [1], [Enable asserts])],
    [enable_assert=no])
