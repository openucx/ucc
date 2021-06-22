#
# Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

ARCH5="-gencode=arch=compute_35,code=sm_35
ARCH6="-gencode=arch=compute_50,code=sm_50
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
    AS_IF([test "x$with_cuda" = "xno"],
        [
         cuda_happy=no
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
         # Check for NVCC
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_PROG(cuda_happy, [nvcc], ["yes"], ["no"])])
         AS_IF([test "x$cuda_happy" = "xyes"],
               [CUDA_MAJOR_VERSION=`nvcc  --version | grep release | sed 's/.*release //' | sed 's/\,.*//' |  cut -d "." -f 1`
                AS_IF([test $CUDA_MAJOR_VERSION -lt 8],
                      [cuda_happy=no])])
         AS_IF([test "x$enable_debug" = xyes],
               [NVCC_CFLAGS="$NVCC_CFLAGS -O0 -g"],
               [NVCC_CFLAGS="$NVCC_CFLAGS -O3 -g -DNDEBUG"])
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AS_IF([test $CUDA_MAJOR_VERSION -eq 8],
                      [NVCC_ARCH="${ARCH5} ${ARCH6} ${ARCH8}"])
                AS_IF([test $CUDA_MAJOR_VERSION -eq 9],
                      [NVCC_ARCH="${ARCH5} ${ARCH6} ${ARCH8} ${ARCH9}"])
                AS_IF([test $CUDA_MAJOR_VERSION -eq 10],
                      [NVCC_ARCH="${ARCH5} ${ARCH6} ${ARCH8} ${ARCH9} ${ARCH10}"])
                AS_IF([test $CUDA_MAJOR_VERSION -eq 11],
                      [NVCC_ARCH="${ARCH6} ${ARCH8} ${ARCH9} ${ARCH10} ${ARCH11}"])
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
               [AS_IF([test "x$with_cuda" != "xguess"],
                      [AC_MSG_ERROR([CUDA support is requested but cuda packages cannot be found])],
                      [AC_MSG_WARN([CUDA not found])])])
        ]) # "x$with_cuda" = "xno"
        cuda_checked=yes
        AM_CONDITIONAL([HAVE_CUDA], [test "x$cuda_happy" != xno])
   ]) # "x$cuda_checked" != "xyes"
]) # CHECK_CUDA