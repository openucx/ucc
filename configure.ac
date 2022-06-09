#
# Copyright (c) 2001-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This software product is a proprietary product of Mellanox Technologies Ltd.
# (the "Company") and all right, title, and interest and to the software product,
# including all associated intellectual property rights, are and shall
# remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
# $COPYRIGHT$
# $HEADER$

define([ucc_ver_major], 1)
define([ucc_ver_minor], 1)
define([ucc_ver_patch], 0)
define([ucc_ver_micro], esyscmd([sh -c "git rev-list HEAD | wc -l | sed -e 's/ *//g' | xargs -n1 printf"]))
define([ucc_git_sha], esyscmd([sh -c "git rev-parse --short HEAD"]))

# This is the API version (see libtool library versioning)
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
# current:rev:age
define([libucc_so_version], 1:0:0)

AC_INIT([ucc], [ucc_ver_major.ucc_ver_minor])
: ${CPPFLAGS=""}
: ${CFLAGS=""}
: ${CXXFLAGS=""}

AC_USE_SYSTEM_EXTENSIONS
AC_GNU_SOURCE
AC_CONFIG_HEADERS([config.h])

UCC_TOP_BUILDDIR="`pwd`"
AC_SUBST(UCC_TOP_BUILDDIR)

cd "$srcdir"
UCC_TOP_SRCDIR="`pwd`"
AC_SUBST(UCC_TOP_SRCDIR)
cd "$UCC_TOP_BUILDDIR"

AC_MSG_NOTICE([builddir: $UCC_TOP_BUILDDIR])
AC_MSG_NOTICE([srcdir: $UCC_TOP_SRCDIR])

AM_INIT_AUTOMAKE([1.10 foreign tar-ustar silent-rules subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE
AC_CONFIG_MACRO_DIR([m4])

define([ucc_prefix], [/opt/ucf/ucc])
AC_PREFIX_DEFAULT([ucc_prefix])

RPM_RELEASE=1
MAJOR_VERSION=ucc_ver_major
MINOR_VERSION=ucc_ver_minor
MICRO_VERSION=ucc_ver_micro
PATCH_VERSION=ucc_ver_patch
UCC_GIT_SHA=ucc_git_sha
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$PATCH_VERSION
SOVERSION=libucc_so_version
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST(MICRO_VERSION)
AC_SUBST(PATCH_VERSION)
AC_SUBST(RPM_RELEASE)
AC_SUBST(SOVERSION)
AC_SUBST(UCC_GIT_SHA)
AC_MSG_NOTICE([git sha: $UCC_GIT_SHA])

CFLAGS_save="$CFLAGS"
AC_PROG_CC
AC_PROG_CXX
AM_PROG_AS
AC_PROG_LN_S
AC_PROG_MKDIR_P
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
CFLAGS="$CFLAGS_save"

AC_CHECK_SIZEOF(float)
AC_CHECK_SIZEOF(double)
AC_CHECK_SIZEOF(long double)
AC_CHECK_SIZEOF(float _Complex)
AC_CHECK_SIZEOF(double _Complex)
AC_CHECK_SIZEOF(long double _Complex)
#
# Check if 'ln' supports creating relative links
#
AC_MSG_CHECKING([if ${LN_S} supports --relative])
AS_IF([${LN_S} --relative symlinktest 2>/dev/null],
      [AC_MSG_RESULT([yes])
       AC_SUBST([LN_RS], ["${LN_S} --relative"])
       rm symlinktest],
      [AC_MSG_RESULT([no])
       AC_SUBST([LN_RS], [${LN_S}])])

AC_SUBST([modulesubdir],   [${PACKAGE_NAME}])               # module directory names
AC_SUBST([moduledir],      [${libdir}/${modulesubdir}])     # module installation directory
AC_SUBST([localmoduledir], ['$(abs_top_builddir)/modules']) # local directory for module symlinks
AC_SUBST([objdir],         [${objdir}])                     # libtool objects dir, usually .libs
AC_SUBST([shrext],         [${shrext_cmds}])                # libtool shared library extension
AC_DEFINE_UNQUOTED([UCC_MODULE_SUBDIR], ["${modulesubdir}"], [UCC module sub-directory])

#
# Additional m4 files
#
m4_include([config/m4/ax_prog_doxygen.m4])

AC_ARG_WITH([docs_only],
        AS_HELP_STRING([--with-docs-only],
                       [Compile only documentation for UCC. [default=NO]]),
        ,[:],[with_docs_only=no])

AC_DEFUN([UCC_DX_ENABLE_CHECK],
         [AS_IF([DX_TEST_FEATURE($1)],
                [],
                [AS_IF([test "x$enable_doxygen_$1" == xyes],
                       [AC_MSG_ERROR([--enable-doxygen-$1 was specified, but $1
                       doxygen was not found])],
                       [])])])

#
# Doxygen options
#
DX_PS_FEATURE(OFF)
DX_HTML_FEATURE(ON)
DX_MAN_FEATURE(ON)
DX_PDF_FEATURE(ON)
DX_INIT_DOXYGEN([UCC],[docs/doxygen/ucc_doxygen.conf],[docs/doxygen-doc])

AS_IF([test "x$with_docs_only" = xyes],
    [AS_MESSAGE([Documents only requested])
     AS_IF([DX_TEST_FEATURE(doc)],
           [],
           [AC_MSG_ERROR([--with-only-docs was requested, but doxygen was not found])])
     UCC_DX_ENABLE_CHECK([html])
     UCC_DX_ENABLE_CHECK([man])
     UCC_DX_ENABLE_CHECK([pdf])
     AM_CONDITIONAL([DOCS_ONLY], [true])
     AM_CONDITIONAL([HAVE_GTEST], [false])
     AM_CONDITIONAL([HAVE_CXX11], [false])
     AM_CONDITIONAL([HAVE_GNUXX11], [false])
     AM_CONDITIONAL([HAVE_GLIBCXX_NOTHROW], [false])
     AM_CONDITIONAL([HAVE_AARCH64_THUNDERX2], [false])
     AM_CONDITIONAL([HAVE_AARCH64_THUNDERX1], [false])
     AM_CONDITIONAL([HAVE_AARCH64_HI1620], [false])
     AM_CONDITIONAL([HAVE_UCX], [false])
     AM_CONDITIONAL([HAVE_CUDA], [false])
     AM_CONDITIONAL([HAVE_ROCM], [false])
     AM_CONDITIONAL([HAVE_HIP], [false])
     AM_CONDITIONAL([HAVE_NVML], [false])
     AM_CONDITIONAL([HAVE_MPI], [false])
     AM_CONDITIONAL([HAVE_MPIRUN], [false])
     AM_CONDITIONAL([HAVE_MPICC], [false])
     AM_CONDITIONAL([HAVE_MPICXX], [false])
     AM_CONDITIONAL([HAVE_PROFILING],[false])
     AM_CONDITIONAL([HAVE_IBVERBS],[false])
     AM_CONDITIONAL([HAVE_MLX5DV],[false])
    ],
    [
     AM_CONDITIONAL([DOCS_ONLY], [false])
     m4_include([config/m4/compiler.m4])
     m4_include([config/m4/gtest.m4])
     m4_include([config/m4/sysdep.m4])
     m4_include([config/m4/ucx.m4])
     m4_include([config/m4/cuda.m4])
     m4_include([config/m4/nccl.m4])
     m4_include([config/m4/rocm.m4])
     m4_include([config/m4/rccl.m4])
     m4_include([config/m4/sharp.m4])
     m4_include([config/m4/mpi.m4])
     m4_include([config/m4/ibverbs.m4])
     m4_include([config/m4/configure.m4])
     m4_include([config/m4/tl_coll_plugins.m4])
     m4_include([config/m4/check_tls.m4])

     mc_modules=":cpu"
     tl_modules=""
     tlcp_modules=""
     AC_MSG_RESULT([MPI perftest: ${mpi_enable}])

     CHECK_UCX
     AC_MSG_RESULT([UCX support: $ucx_happy])
     if test $ucx_happy != "yes"; then
         AC_MSG_ERROR([UCX is not available])
     fi

     CHECK_CUDA
     AC_MSG_RESULT([CUDA support: $cuda_happy; $CUDA_CPPFLAGS $CUDA_LDFLAGS])
     if test $cuda_happy = "yes"; then
         mc_modules="${mc_modules}:cuda"
     fi

     CHECK_ROCM
     AC_MSG_RESULT([ROCM support: $rocm_happy; $ROCM_CPPFLAGS $ROCM_LDFLAGS])
     AC_MSG_RESULT([HIP support: $hip_happy; $HIP_CPPFLAGS $HIP_LDFLAGS])
     if test $rocm_happy = "yes"; then
         mc_modules="${mc_modules}:rocm"
     fi

     CHECK_GTEST
     AC_MSG_RESULT([GTEST support: $gtest_happy])

     CHECK_IBVERBS
     AC_MSG_RESULT([IBVERBS support: $ibverbs_happy, MLX5DV support: $mlx5dv_happy])
     ]) # Docs only


CHECK_TLS
CFLAGS="$CFLAGS -std=gnu11"
CPPFLAGS="$CPPFLAGS $UCS_CPPFLAGS $includes"
LDFLAGS="$LDFLAGS $UCS_LDFLAGS $UCS_LIBADD"

CHECK_TL_COLL_PLUGINS
AC_CONFIG_FILES([
                 Makefile
                 src/Makefile
                 src/ucc/api/ucc_version.h
                 src/core/ucc_version.c
                 src/components/cl/basic/Makefile
                 src/components/cl/hier/Makefile
                 src/components/mc/cpu/Makefile
                 src/components/mc/cuda/Makefile
                 src/components/ec/cpu/Makefile
                 src/components/ec/cuda/Makefile
                 src/components/ec/cuda/kernel/Makefile
                 src/components/mc/rocm/Makefile
                 src/components/mc/rocm/kernel/Makefile
                 src/components/ec/rocm/Makefile
                 src/components/ec/rocm/kernel/Makefile
                 test/mpi/Makefile
                 test/gtest/Makefile
                 tools/info/Makefile
                 tools/perf/Makefile
                 cmake/Makefile
                 cmake/ucc-config-version.cmake
                 cmake/ucc-config.cmake
                 cmake/ucc-targets.cmake
                 ])
AC_OUTPUT

#
# Print build condiguration
#
AC_MSG_NOTICE([=========================================================])
AS_IF([test "x$with_docs_only" = xyes],
[
AC_MSG_NOTICE([Building documents only])
],
[
AC_MSG_NOTICE([UCC build configuration:])
AC_MSG_NOTICE([      Build prefix:   ${prefix}])
AC_MSG_NOTICE([Preprocessor flags:   ${CPPFLAGS} ${BASE_CPPFLAGS}])
AC_MSG_NOTICE([        C compiler:   ${CC} ${CFLAGS} ${BASE_CFLAGS}])
AC_MSG_NOTICE([      C++ compiler:   ${CXX} ${CXXFLAGS} ${BASE_CXXFLAGS}])
AS_IF([test "x$cuda_happy" = "xyes"],[
AC_MSG_NOTICE([     NVCC gencodes:   ${NVCC_ARCH}])
])
AC_MSG_NOTICE([          Perftest:   ${mpi_enable}])
AC_MSG_NOTICE([             Gtest:   ${gtest_enable}])
AC_MSG_NOTICE([        MC modules:   <$(echo ${mc_modules}|tr ':' ' ') >])
AC_MSG_NOTICE([        TL modules:   <$(echo ${tl_modules}|tr ':' ' ') >])
AC_MSG_NOTICE([      TLCP modules:   <$(echo ${tlcp_modules}|tr ':' ' ') >])
AS_IF([test "x$enable_profiling" = xyes],[
AC_MSG_NOTICE([ Profiling modules:   <$(echo ${prof_modules}|tr ':' ' ') >])
])
])
AC_MSG_NOTICE([=========================================================])
