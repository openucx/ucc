#
# Copyright (C) Advanced Micro Devices, Inc. 2016 - 2023. ALL RIGHTS RESERVED.
# Copyright (c) 2001-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

ROCM_ARCH_NATIVE="--offload-arch=native"
ROCM_ARCH908="--offload-arch=gfx908"
ROCM_ARCH90A="--offload-arch=gfx90a"
ROCM_ARCH94="--offload-arch=gfx942"
ROCM_ARCH95="--offload-arch=gfx950"
ROCM_ARCH10="--offload-arch=gfx1030"
ROCM_ARCH11="--offload-arch=gfx1100 \
--offload-arch=gfx1101 \
--offload-arch=gfx1102"
ROCM_ARCH12="--offload-arch=gfx1200 \
--offload-arch=gfx1201"

# ROCM_PARSE_FLAGS(ARG, VAR_LIBS, VAR_LDFLAGS, VAR_CPPFLAGS)
# ----------------------------------------------------------
# Parse whitespace-separated ARG into appropriate LIBS, LDFLAGS, and
# CPPFLAGS variables.
AC_DEFUN([ROCM_PARSE_FLAGS],
[for arg in $$1 ; do
    AS_CASE([$arg],
        [yes],               [],
        [no],                [],
        [-l*|*.a|*.so],      [$2="$$2 $arg"],
        [-L*|-WL*|-Wl*],     [$3="$$3 $arg"],
        [-I*],               [$4="$$4 $arg"],
        [*lib|*lib/|*lib64|*lib64/],[AS_IF([test -d $arg], [$3="$$3 -L$arg"],
                                 [AC_MSG_WARN([$arg of $1 not parsed])])],
        [*include|*include/],[AS_IF([test -d $arg], [$4="$$4 -I$arg"],
                                 [AC_MSG_WARN([$arg of $1 not parsed])])],
        [AC_MSG_WARN([$arg of $1 not parsed])])
done])

# ROCM_BUILD_FLAGS(ARG, VAR_LIBS, VAR_LDFLAGS, VAR_CPPFLAGS, VAR_ROOT)
# ----------------------------------------------------------
# Parse value of ARG into appropriate LIBS, LDFLAGS, and
# CPPFLAGS variables.
AC_DEFUN([ROCM_BUILD_FLAGS],
    $4="-I$1/include/hsa -I$1/include"
    $3="-L$1/lib -L$1/lib64 -L$1/hsa/lib"
    $2="-lhsa-runtime64 -lhsakmt"
    $5="$1"
)

# HIP_BUILD_FLAGS(ARG, VAR_LIBS, VAR_LDFLAGS, VAR_CPPFLAGS)
# ----------------------------------------------------------
# Parse value of ARG into appropriate LIBS, LDFLAGS, and
# CPPFLAGS variables.
AC_DEFUN([HIP_BUILD_FLAGS],
    $4="-D__HIP_PLATFORM_AMD__ -I$1/include/hip -I$1/include -I$1/llvm/include"
    $3="-L$1/lib -L$1/llvm/lib"
    $2="-lamdhip64"
)

# CHECK_ROCM_VERSION(HIP_VERSION_MAJOR, ROCM_VERSION_CONDITION)
# ----------------------------------------------------------
# Checks ROCm version and marks condition as 1 (TRUE) or 0 (FALSE)
AC_DEFUN([CHECK_ROCM_VERSION], [
AC_COMPILE_IFELSE(
[AC_LANG_PROGRAM([[#include <${with_rocm}/include/hip/hip_version.h>
    ]], [[
#if HIP_VERSION_MAJOR >= $1
return 0;
#else
intr make+compilation_fail()
#endif
    ]])],
    [$2=1],
    [$2=0])
])

#
# Check for ROCm  support
#
AC_DEFUN([CHECK_ROCM],[

AS_IF([test "x$rocm_checked" != "xyes"],[

AC_ARG_WITH([rocm],
    [AS_HELP_STRING([--with-rocm=(DIR)],
        [Enable the use of ROCm (default is autodetect).])],
    [],
    [with_rocm=guess])
AC_ARG_WITH([rocm-arch],
    [AS_HELP_STRING([--with-rocm-arch=arch-code],
        [Defines target GPU architecture,
            see rocm documentation for valid --offload-arch options for details
            'all-arch-no-native' for all default architectures but not native])],
        [], [with_rocm_arch=all])
rocm_happy=no
hip_happy=no
AS_IF([test "x$with_rocm" != "xno"],
    [AS_CASE(["x$with_rocm"],
        [x|xguess|xyes],
            [AC_MSG_NOTICE([ROCm path was not specified. Guessing ...])
             with_rocm="/opt/rocm"
             ROCM_BUILD_FLAGS([$with_rocm],
                          [ROCM_LIBS], [ROCM_LDFLAGS], [ROCM_CPPFLAGS], [ROCM_ROOT])],
        [x/*],
            [AC_MSG_NOTICE([ROCm path given as $with_rocm ...])
             ROCM_BUILD_FLAGS([$with_rocm],
                          [ROCM_LIBS], [ROCM_LDFLAGS], [ROCM_CPPFLAGS], [ROCM_ROOT])],
        [AC_MSG_NOTICE([ROCm flags given ...])
         ROCM_PARSE_FLAGS([$with_rocm],
                          [ROCM_LIBS], [ROCM_LDFLAGS], [ROCM_CPPFLAGS])])

    SAVE_CPPFLAGS="$CPPFLAGS"
    SAVE_LDFLAGS="$LDFLAGS"
    SAVE_LIBS="$LIBS"

    CPPFLAGS="$ROCM_CPPFLAGS $CPPFLAGS"
    LDFLAGS="$ROCM_LDFLAGS $LDFLAGS"
    LIBS="$ROCM_LIBS $LIBS"

    rocm_happy=yes
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_HEADERS([hsa.h], [rocm_happy=yes], [rocm_happy=no])])
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_HEADERS([hsa_ext_amd.h], [rocm_happy=yes], [rocm_happy=no])])
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_LIB([hsa-runtime64], [hsa_init], [rocm_happy=yes], [rocm_happy=no])])

    AS_IF([test "x$rocm_happy" = "xyes"],
          [AC_DEFINE([HAVE_ROCM], 1, [Enable ROCM support])
           AC_SUBST([ROCM_CPPFLAGS])
           AC_SUBST([ROCM_LDFLAGS])
           AC_SUBST([ROCM_LIBS])
           AC_SUBST([ROCM_ROOT])],
          [AC_MSG_WARN([ROCm not found])])


    # Check whether we run on ROCm 6.0 or higher
    CHECK_ROCM_VERSION(6, ROCM_VERSION_60_OR_GREATER)
    AC_MSG_CHECKING([if ROCm version is 6.0 or above])

    AS_IF([test "x$rocm_happy" = "xyes"],
        [AS_IF([test "x$with_rocm_arch" = "xall"],
          [ROCM_ARCH="${ROCM_ARCH908} ${ROCM_ARCH90A} ${ROCM_ARCH94} ${ROCM_ARCH95} ${ROCM_ARCH10} ${ROCM_ARCH11} ${ROCM_ARCH12} ${ROCM_ARCH_NATIVE}"],
        [AS_IF([test "x$with_rocm_arch" = "xall-arch-no-native"],
          [ROCM_ARCH="${ROCM_ARCH908} ${ROCM_ARCH90A} ${ROCM_ARCH94} ${ROCM_ARCH95} ${ROCM_ARCH10} ${ROCM_ARCH11} ${ROCM_ARCH12}"],
        [ROCM_ARCH="$with_rocm_arch"])])
        AS_IF([test "$ROCM_VERSION_60_OR_GREATER" = "1"],
          AC_SUBST([ROCM_ARCH], ["$ROCM_ARCH"]),
          AC_SUBST([ROCM_ARCH], [""]))])
    CPPFLAGS="$SAVE_CPPFLAGS"
    LDFLAGS="$SAVE_LDFLAGS"
    LIBS="$SAVE_LIBS"

    HIP_BUILD_FLAGS([$with_rocm], [HIP_LIBS], [HIP_LDFLAGS], [HIP_CPPFLAGS])

    if test "$ROCM_VERSION_60_OR_GREATER" = "1" ; then
        AC_MSG_RESULT([yes])
    else
        AC_MSG_RESULT([no])
        # Check whether we run on ROCm 5.0-5.7
        CHECK_ROCM_VERSION(5, ROCM_VERSION_50_OR_GREATER)
        AC_MSG_CHECKING([if ROCm version is 5.0 - 5.7])
        if test "$ROCM_VERSION_50_OR_GREATER" = "1" ; then
            AC_MSG_RESULT([yes])
        else
            AC_MSG_RESULT([no])
            HIP_CPPFLAGS="${HIP_CPPFLAGS} -I${with_rocm}/hip/include"
            HIP_LDFLAGS="${HIP_LDFLAGS} -L${with_rocm}/hip/lib"
        fi
    fi

    CPPFLAGS="$HIP_CPPFLAGS $CPPFLAGS"
    LDFLAGS="$HIP_LDFLAGS $LDFLAGS"
    LIBS="$HIP_LIBS $LIBS"

    hip_happy=no
    AC_CHECK_LIB([hip_hcc], [hipFree], [AC_MSG_WARN([Please install ROCm-3.7.0 or above])], [hip_happy=yes])
    AS_IF([test "x$hip_happy" = xyes],
          [AC_CHECK_HEADERS([hip/hip_runtime.h], [hip_happy=yes], [hip_happy=no])])
    AS_IF([test "x$hip_happy" = xyes],
          [AC_CHECK_LIB([amdhip64], [hipFree], [hip_happy=yes], [hip_happy=no])])
    AS_IF([test "x$hip_happy" = xyes], [HIP_CXXFLAGS="--std=gnu++11"], [])

    CPPFLAGS="$SAVE_CPPFLAGS"
    LDFLAGS="$SAVE_LDFLAGS"
    LIBS="$SAVE_LIBS"

    if test "$ROCM_VERSION_60_OR_GREATER" = "1" ; then
        AC_MSG_NOTICE([using amdclang as ROCm version is 6.0 or above])
        AS_IF([test "x$hip_happy" = "xyes"],
              [AC_PATH_PROG([HIPCC], [amdclang], [notfound], [$PATH:$with_rocm/bin])])
             AS_IF([test "$HIPCC" = "notfound"], [hip_happy="no"])
    else
        AC_MSG_NOTICE([using hipcc as ROCm version is 3.7.0 to ROCm 5.7.1])
        AS_IF([test "x$hip_happy" = "xyes"],
              [AC_PATH_PROG([HIPCC], [hipcc], [notfound], [$PATH:$with_rocm/bin])])
             AS_IF([test "$HIPCC" = "notfound"], [hip_happy="no"])
    fi

    AS_IF([test "x$hip_happy" = "xyes"],
          [AC_DEFINE([HAVE_HIP], 1, [Enable HIP support])
           AC_SUBST([HIPCC])
           AC_SUBST([HIP_CPPFLAGS])
           AC_SUBST([HIP_CXXFLAGS])
           AC_SUBST([HIP_LDFLAGS])
           AC_SUBST([HIP_LIBS])],
          [AC_MSG_WARN([HIP Runtime not found])])

    ],
    [AC_MSG_WARN([ROCm was explicitly disabled])]
)



rocm_checked=yes
AM_CONDITIONAL([HAVE_ROCM], [test "x$rocm_happy" != xno])
AM_CONDITIONAL([HAVE_HIP], [test "x$hip_happy" != xno])

])

])
