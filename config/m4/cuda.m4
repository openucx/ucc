#
# Copyright (c) 2001-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

CUDA_MIN_REQUIRED_MAJOR=11
CUDA_MIN_REQUIRED_MINOR=0

ARCH6="-gencode=arch=compute_50,code=sm_50"
ARCH8="-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_61,code=compute_61"
ARCH9="-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_70,code=compute_70"
ARCH10="-gencode=arch=compute_75,code=sm_75"
ARCH11="-gencode=arch=compute_80,code=sm_80 \
-gencode=arch=compute_80,code=compute_80"

AC_DEFUN([CHECK_CUDA],[
AS_IF([test "x$cuda_checked" != "xyes"],
   [
    AC_ARG_WITH([cuda],
                [AS_HELP_STRING([--with-cuda=(DIR)], [Enable the use of CUDA (default is guess).])],
                [], [with_cuda=guess])
    AC_ARG_WITH([nvcc-gencode],
                [AS_HELP_STRING([--with-nvcc-gencode=arch,code],
                                [Defines target GPU architecture,
                                 see nvcc -gencode option for details])],
                [], [with_nvcc_gencode=default])
    AS_IF([test "x$with_cuda" = "xno"],
        [
         cuda_happy=no
         nvml_happy=no
        ],
        [
         save_CPPFLAGS="$CPPFLAGS"
         save_LDFLAGS="$LDFLAGS"
         save_LIBS="$LIBS"
         CUDA_CPPFLAGS=""
         CUDA_LDFLAGS=""
         CUDA_LIBS=""
         AS_IF([test ! -z "$with_cuda" -a "x$with_cuda" != "xyes" -a "x$with_cuda" != "xguess"],
               [check_cuda_dir="$with_cuda"
                AS_IF([test -d "$with_cuda/lib64"], [libsuff="64"], [libsuff=""])
                check_cuda_libdir="$with_cuda/lib$libsuff"
                CUDA_CPPFLAGS="-I$with_cuda/include"
                CUDA_LDFLAGS="-L$check_cuda_libdir -L$check_cuda_libdir/stubs"])
         AS_IF([test ! -z "$with_cuda_libdir" -a "x$with_cuda_libdir" != "xyes"],
               [check_cuda_libdir="$with_cuda_libdir"
                CUDA_LDFLAGS="-L$check_cuda_libdir -L$check_cuda_libdir/stubs"])
         CPPFLAGS="$CPPFLAGS $CUDA_CPPFLAGS"
         LDFLAGS="$LDFLAGS $CUDA_LDFLAGS"
         # Check cuda header files
         AC_CHECK_HEADERS([cuda.h cuda_runtime.h],
                          [cuda_happy="yes"], [cuda_happy="no"])
         # Check cuda libraries
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([cuda], [cuDeviceGetUuid],
                             [CUDA_LIBS="$CUDA_LIBS -lcuda"], [cuda_happy="no"])])
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([cudart], [cudaGetDeviceCount],
                             [CUDA_LIBS="$CUDA_LIBS -lcudart"], [cuda_happy="no"])])

        # Check nvml header files
        AC_CHECK_HEADERS([nvml.h],
                         [nvml_happy="yes"],
                         [AS_IF([test "x$with_cuda" != "xguess"],
                                [AC_MSG_WARN([nvml header not found. Install appropriate cuda-nvml-devel package])])
                          nvml_happy="no"])

        # Check nvml library
        AS_IF([test "x$cuda_happy" = "xyes" -a "x$nvml_happy" = "xyes"],
              [AC_CHECK_LIB([nvidia-ml], [nvmlInit_v2],
                            [NVML_LIBS="-lnvidia-ml"],
                            [AS_IF([test "x$with_cuda" != "xguess"],
                                   [AC_MSG_WARN([libnvidia-ml not found. Install appropriate nvidia-driver package])])
                             nvml_happy="no"])])
        AS_IF([test "x$cuda_happy" = "xyes" -a "x$nvml_happy" = "xyes"],
              [AC_CHECK_DECL([nvmlDeviceGetNvLinkRemoteDeviceType],
                             [AC_CHECK_LIB([nvidia-ml], [nvmlDeviceGetNvLinkRemoteDeviceType],
                                           [AC_DEFINE([HAVE_NVML_REMOTE_DEVICE_TYPE],
                                                       1,
                                                      ["Use nvmlDeviceGetNvLinkRemoteDeviceType"])],
                                           [])],
                             [],
                             [[#include <nvml.h>]])])
        AC_CHECK_SIZEOF(cuFloatComplex,,[#include <cuComplex.h>])
        AC_CHECK_SIZEOF(cuDoubleComplex,,[#include <cuComplex.h>])

         # Check for NVCC
         AC_ARG_VAR(NVCC, [NVCC compiler command])
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_PATH_PROG([NVCC], [nvcc], [notfound], [$PATH:$check_cuda_dir/bin])])
         AS_IF([test "$NVCC" = "notfound"], [cuda_happy="no"])
         AS_IF([test "x$cuda_happy" = "xyes"],
               [CUDA_MAJOR_VERSION=`$NVCC --version | grep release | sed 's/.*release //' | sed 's/\,.*//' |  cut -d "." -f 1`
                CUDA_MINOR_VERSION=`$NVCC --version | grep release | sed 's/.*release //' | sed 's/\,.*//' |  cut -d "." -f 2`
                AC_MSG_RESULT([Detected CUDA version: $CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION])
                AS_IF([test $CUDA_MAJOR_VERSION -lt $CUDA_MIN_REQUIRED_MAJOR],
                      [AC_MSG_WARN([Minimum required CUDA version: $CUDA_MIN_REQUIRED_MAJOR.$CUDA_MIN_REQUIRED_MINOR])
                       cuda_happy=no])])
         AS_IF([test "x$enable_debug" = xyes],
               [NVCC_CFLAGS="$NVCC_CFLAGS -O0 -g"],
               [NVCC_CFLAGS="$NVCC_CFLAGS -O3 -g -DNDEBUG"])
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AS_IF([test "x$with_nvcc_gencode" = "xdefault"],
                      [AS_IF([test $CUDA_MAJOR_VERSION -eq 11],
                             [NVCC_ARCH="${ARCH8} ${ARCH9} ${ARCH10} ${ARCH11}"])],
                      [NVCC_ARCH="$with_nvcc_gencode"])
                AC_SUBST([NVCC_ARCH], ["$NVCC_ARCH"])])
         LDFLAGS="$save_LDFLAGS"
         CPPFLAGS="$save_CPPFLAGS"
         LDFLAGS="$save_LDFLAGS"
         LIBS="$save_LIBS"
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_SUBST([CUDA_CPPFLAGS], ["$CUDA_CPPFLAGS"])
                AC_SUBST([CUDA_LDFLAGS], ["$CUDA_LDFLAGS"])
                AC_SUBST([CUDA_LIBS], ["$CUDA_LIBS"])
                AC_SUBST([NVCC_CFLAGS], ["$NVCC_CFLAGS"])
                AC_DEFINE([HAVE_CUDA], 1, [Enable CUDA support])],
                AS_IF([test "x$nvml_happy" = "xyes"],
                        [AC_SUBST([NVML_LIBS], ["$NVML_LIBS"])
                         AC_DEFINE([HAVE_NVML], 1, [Enable NVML support])],[])
               [AS_IF([test "x$with_cuda" != "xguess"],
                      [AC_MSG_ERROR([CUDA support is requested but cuda packages cannot be found])],
                      [AC_MSG_WARN([CUDA not found])])])
        ]) # "x$with_cuda" = "xno"
        cuda_checked=yes
        AM_CONDITIONAL([HAVE_CUDA], [test "x$cuda_happy" != xno])
        AM_CONDITIONAL([HAVE_NVML], [test "x$nvml_happy" != xno])
   ]) # "x$cuda_checked" != "xyes"
]) # CHECK_CUDA
