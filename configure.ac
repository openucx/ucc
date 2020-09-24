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

define([ucc_prefix], [/opt/ucc])
AC_PREFIX_DEFAULT([ucc_prefix])

RPM_RELEASE=1
MAJOR_VERSION=ucc_ver_major
MINOR_VERSION=ucc_ver_minor
MICRO_VERSION=ucc_ver_micro
UCC_GIT_SHA=ucc_git_sha
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$MICRO_VERSION
SOVERSION=libucc_so_version
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST(MICRO_VERSION)
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

AM_CONDITIONAL([HAVE_UCX], [false])
m4_include([m4/ucx.m4])
UCC_CHECK_UCX

AC_MSG_RESULT([UCX support: $ucx_happy])

includes="-I${UCC_TOP_SRCDIR}/src -I${UCC_TOP_SRCDIR}/src/core"
CPPFLAGS="$UCS_CPPFLAGS $CPPFLAGS $includes"
LDFLAGS="$LDFLAGS $UCS_LDFLAGS"
AC_CONFIG_FILES([
                 Makefile
                 src/Makefile
                 src/core/Makefile
                 src/team_lib/Makefile
                 src/team_lib/basic/Makefile
                 src/team_lib/dbg/Makefile
                 test/Makefile
                 ])
AC_OUTPUT
