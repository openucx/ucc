#
# Copyright (c) 2001-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

CUDA_MIN_REQUIRED_MAJOR=11
CUDA_MIN_REQUIRED_MINOR=0

ARCH_NVLS_LIST="sm_90|sm_100"

ARCH7_CODE="-gencode=arch=compute_52,code=sm_52"
ARCH8_CODE="-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61"
ARCH9_CODE="-gencode=arch=compute_70,code=sm_70"
ARCH10_CODE="-gencode=arch=compute_75,code=sm_75"
ARCH110_CODE="-gencode=arch=compute_80,code=sm_80"
ARCH111_CODE="-gencode=arch=compute_86,code=sm_86"
ARCH120_CODE="-gencode=arch=compute_90,code=sm_90"
ARCH124_CODE="-gencode=arch=compute_89,code=sm_89"
ARCH128_CODE="-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120"
ARCH130_CODE="-gencode=arch=compute_110,code=sm_110"


ARCH8_PTX="-gencode=arch=compute_61,code=compute_61"
ARCH9_PTX="-gencode=arch=compute_70,code=compute_70"
ARCH10_PTX=""
ARCH110_PTX="-gencode=arch=compute_80,code=compute_80"
ARCH111_PTX="-gencode=arch=compute_86,code=compute_86"
ARCH120_PTX="-gencode=arch=compute_90,code=compute_90"
ARCH124_PTX="-gencode=arch=compute_90,code=compute_90"
ARCH128_PTX="-gencode=arch=compute_120,code=compute_120"
ARCH130_PTX="-gencode=arch=compute_120,code=compute_120"

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

        # Early check for NVML - required for TL CUDA
        AC_CHECK_HEADERS([nvml.h],
                         [nvml_happy="yes"],
                         [nvml_happy="no"])

        # Check nvml library if header found
        AS_IF([test "x$cuda_happy" = "xyes" -a "x$nvml_happy" = "xyes"],
              [AC_CHECK_LIB([nvidia-ml], [nvmlInit_v2],
                            [NVML_LIBS="-lnvidia-ml"],
                            [nvml_happy="no"])])
        AS_IF([test "x$cuda_happy" = "xyes" -a "x$nvml_happy" = "xyes"],
              [AC_CHECK_DECL([nvmlDeviceGetNvLinkRemoteDeviceType],
                             [AC_CHECK_LIB([nvidia-ml], [nvmlDeviceGetNvLinkRemoteDeviceType],
                                           [AC_DEFINE([HAVE_NVML_REMOTE_DEVICE_TYPE],
                                                       1,
                                                      ["Use nvmlDeviceGetNvLinkRemoteDeviceType"])],
                                           [])],
                             [],
                             [[#include <nvml.h>]])])

        # Determine TL CUDA availability early
        tl_cuda_will_be_available="no"
        AS_IF([test "x$cuda_happy" = "xyes" -a "x$nvml_happy" = "xyes"],
              [tl_cuda_will_be_available="yes"])

        # Provide early feedback about TL CUDA status
        AS_IF([test "x$cuda_happy" = "xyes" -a "x$nvml_happy" = "xno"],
              [AC_MSG_WARN([NVML headers/library not found - TL CUDA will be disabled])
               AS_IF([test "x$with_cuda" != "xguess"],
                     [AC_MSG_WARN([Install cuda-nvml-devel or nvidia-cuda-dev package to enable TL CUDA])])])
        AC_CHECK_SIZEOF(cuFloatComplex,,[#include <cuComplex.h>])
        AC_CHECK_SIZEOF(cuDoubleComplex,,[#include <cuComplex.h>])

         # Only proceed with detailed CUDA configuration if TL CUDA will be available
         AS_IF([test "x$tl_cuda_will_be_available" = "xyes"],
         [
             AC_MSG_RESULT([Proceeding with detailed CUDA configuration (TL CUDA will be enabled)])

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
         ],
         [
             AS_IF([test "x$cuda_happy" = "xyes"],
             [
                 AC_MSG_RESULT([Skipping detailed CUDA configuration (TL CUDA disabled - NVML missing)])
                 # Set minimal CUDA version for basic support (MC/EC)
                 AC_ARG_VAR(NVCC, [NVCC compiler command])
                 AC_PATH_PROG([NVCC], [nvcc], [notfound], [$PATH:$check_cuda_dir/bin])
                 AS_IF([test "$NVCC" != "notfound"],
                       [CUDA_MAJOR_VERSION=`$NVCC --version | grep release | sed 's/.*release //' | sed 's/\,.*//' |  cut -d "." -f 1`
                        CUDA_MINOR_VERSION=`$NVCC --version | grep release | sed 's/.*release //' | sed 's/\,.*//' |  cut -d "." -f 2`])
             ])
         ])

         # Advanced CUDA configuration only for TL CUDA builds
         AS_IF([test "x$tl_cuda_will_be_available" = "xyes" -a "x$cuda_happy" = "xyes"],
         [
             # Check if CUDA version is 13 or higher, which requires C++17 support.
             # If the compiler supports C++17, add the flag to NVCC_CFLAGS. If not, warn the user but still add the flag (build may fail).
             AS_IF([test $CUDA_MAJOR_VERSION -ge 13],
                   [AS_IF([test "x$cxx17_happy" = "xyes"],
                          [NVCC_CFLAGS="$NVCC_CFLAGS -std=c++17"],
                          [AC_MSG_WARN([CUDA $CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION requires C++17 but compiler does not support it. Build may fail.])
                           NVCC_CFLAGS="$NVCC_CFLAGS -std=c++17"])])

             # Generate appropriate CUDA architecture codes
             AS_IF([test "x$with_nvcc_gencode" = "xdefault"],
                   [AS_IF([test $CUDA_MAJOR_VERSION -eq 13],
                          # offline compilation support for architectures before '<compute/sm/lto>_75' is discontinued
                          [NVCC_ARCH="${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH120_CODE} ${ARCH124_CODE} ${ARCH128_CODE} ${ARCH130_CODE} ${ARCH130_PTX}"],
                   [AS_IF([test $CUDA_MAJOR_VERSION -eq 12],
                           [AS_IF([test $CUDA_MINOR_VERSION -ge 8],
                                 [NVCC_ARCH="${ARCH7_CODE} ${ARCH8_CODE} ${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH120_CODE} ${ARCH124_CODE} ${ARCH128_CODE} ${ARCH128_PTX}"],
                           [AS_IF([test $CUDA_MINOR_VERSION -ge 4],
                                 [NVCC_ARCH="${ARCH7_CODE} ${ARCH8_CODE} ${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH120_CODE} ${ARCH124_CODE} ${ARCH124_PTX}"],
                                 [NVCC_ARCH="${ARCH7_CODE} ${ARCH8_CODE} ${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH120_CODE} ${ARCH120_PTX}"])])],
                   [AS_IF([test $CUDA_MAJOR_VERSION -eq 11],
                           [AS_IF([test $CUDA_MINOR_VERSION -lt 1],
                                 [NVCC_ARCH="${ARCH7_CODE} ${ARCH8_CODE} ${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH110_PTX}"],
                                 [NVCC_ARCH="${ARCH7_CODE} ${ARCH8_CODE} ${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH111_PTX}"])])])])],
                   [NVCC_ARCH="$with_nvcc_gencode"])
            AC_SUBST([NVCC_ARCH], ["$NVCC_ARCH"])
            AC_MSG_RESULT([NVCC gencodes: $NVCC_ARCH])

            # Generate NVLS-specific architecture codes (SASS only, no PTX)
            # NVLS requires NVSwitch (datacenter only): Hopper (CC 9.0), Blackwell (CC 10.0)
            # Filter NVCC_ARCH to keep only sm_90 and sm_100 gencode flags
            # Handle both "-gencode=arch=..." (single token) and "-gencode arch=..." (two tokens)
            NVCC_ARCH_NVLS=""
            _ucc_pending_gencode=""
            for _ucc_nvls_flag in $NVCC_ARCH
            do
                if test "x$_ucc_nvls_flag" = "x-gencode"; then
                    _ucc_pending_gencode="-gencode"
                    continue
                fi
                if echo "$_ucc_nvls_flag" | grep -q -E "$ARCH_NVLS_LIST" 2>/dev/null; then
                    NVCC_ARCH_NVLS="$NVCC_ARCH_NVLS $_ucc_pending_gencode $_ucc_nvls_flag"
                fi
                _ucc_pending_gencode=""
            done
            NVCC_ARCH_NVLS=$(echo $NVCC_ARCH_NVLS)
            AC_SUBST([NVCC_ARCH_NVLS], ["$NVCC_ARCH_NVLS"])
            AC_MSG_RESULT([NVCC NVLS gencodes: $NVCC_ARCH_NVLS])
        ])


         LDFLAGS="$save_LDFLAGS"
         CPPFLAGS="$save_CPPFLAGS"
         LDFLAGS="$save_LDFLAGS"
         LIBS="$save_LIBS"
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_SUBST([CUDA_CPPFLAGS], ["$CUDA_CPPFLAGS"])
                AC_SUBST([CUDA_LDFLAGS], ["$CUDA_LDFLAGS"])
                AC_SUBST([CUDA_LIBS], ["$CUDA_LIBS"])
                AC_SUBST([NVCC_CFLAGS], ["$NVCC_CFLAGS"])
                AC_DEFINE([HAVE_CUDA], 1, [Enable CUDA support])

                # Report CUDA status with TL CUDA context
                AS_IF([test "x$tl_cuda_will_be_available" = "xyes"],
                      [AC_MSG_RESULT([CUDA support: yes (TL CUDA will be enabled)])],
                      [AC_MSG_RESULT([CUDA support: yes (TL CUDA disabled - NVML missing)])])

                AS_IF([test "x$nvml_happy" = "xyes"],
                      [AC_SUBST([NVML_LIBS], ["$NVML_LIBS"])
                       AC_DEFINE([HAVE_NVML], 1, [Enable NVML support])],[])],
               [AS_IF([test "x$with_cuda" != "xguess"],
                      [AC_MSG_ERROR([CUDA support is requested but cuda packages cannot be found])],
                      [AC_MSG_WARN([CUDA not found])])])
        ]) # "x$with_cuda" = "xno"
        cuda_checked=yes
        AM_CONDITIONAL([HAVE_CUDA], [test "x$cuda_happy" != xno])
        AM_CONDITIONAL([HAVE_NVML], [test "x$nvml_happy" != xno])
   ]) # "x$cuda_checked" != "xyes"
]) # CHECK_CUDA
