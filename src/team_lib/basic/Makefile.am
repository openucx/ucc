#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
	ucc_tl_basic.h \
	ucc_basic_lib.c \
	ucc_basic_context.h \
	ucc_basic_context.c \
	ucc_basic_team.h \
	ucc_basic_team.c

component_noinst =
component_install = ucc_team_lib_basic.la

ucc_team_lib_basic_la_SOURCES =$(sources)
ucc_team_lib_basic_la_CPPFLAGS = $(AM_CPPFLAGS) $(VMC_CPPFLAGS)
ucc_team_lib_basic_la_LDFLAGS = -module -avoid-version  $(VMC_LDFLAGS)
pkglib_LTLIBRARIES = $(component_install)
