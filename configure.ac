#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
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
define([ucc_ver_minor], 0)
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
PATCH_VERSION=ucc_ver_micro
UCC_GIT_SHA=ucc_git_sha
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$MICRO_VERSION
SOVERSION=libucc_so_version
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST(MICRO_VERSION)
AC_SUBST(PATCH_VERSION)
AC_SUBST(RPM_RELEASE)
AC_SUBST(SOVERSION)
AC_SUBST(UCC_GIT_SHA)
AC_MSG_RESULT([git sha: $ucc_git_sha])

CFLAGS_save="$CFLAGS"
AC_PROG_CC
AC_PROG_CXX
AM_PROG_AS
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
CFLAGS="$CFLAGS_save"

#
# Additional m4 files
#
m4_include([config/m4/ax_prog_doxygen.m4])
m4_include([config/m4/compiler.m4])
m4_include([config/m4/gtest.m4])
m4_include([config/m4/sysdep.m4])
m4_include([config/m4/ucx.m4])

m4_include([test/gtest/configure.m4])

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
    ],
    [
     AM_CONDITIONAL([DOCS_ONLY], [false])
    ]) # Docs only


AC_ARG_ENABLE([debug],
              AS_HELP_STRING([--enable-debug], [Enable extra debugging code (default is NO).]),
              [], [enable_debug=no])

if test $enable_debug = yes; then
	AC_DEFINE([ENABLE_DEBUG], [1], [Enable debugging code])
	CFLAGS="$CFLAGS -O0 -g3"
	AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_POLL], [Highest log level])
else
	AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_INFO], [Highest log level])
	CFLAGS="$CFLAGS -O3 -DNDEBUG"
fi

CHECK_UCX
AC_MSG_RESULT([UCX support: $ucx_happy])
if test $ucx_happy != "yes"; then
   AC_MSG_ERROR([UCX is not available])
fi

includes="-I${UCC_TOP_SRCDIR}/src -I${UCC_TOP_BUILDDIR}/src"
CFLAGS="-std=gnu11"
CPPFLAGS="$UCS_CPPFLAGS $CPPFLAGS $includes -Wall -Werror"
LDFLAGS="$LDFLAGS $UCS_LDFLAGS $UCS_LIBADD"

AC_CONFIG_FILES([
                 Makefile
                 src/Makefile
                 tools/info/Makefile
                 src/cl/basic/Makefile
                 src/api/ucc_version.h
                 src/core/ucc_version.c
                 ])
AC_OUTPUT
