#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
	ucc_tl_debug.h \
	ucc_debug_lib.c \
	ucc_debug_context.h \
	ucc_debug_context.c

component_noinst =
component_install = ucc_team_lib_debug.la

ucc_team_lib_debug_la_SOURCES =$(sources)
ucc_team_lib_debug_la_CPPFLAGS = $(AM_CPPFLAGS) $(VMC_CPPFLAGS)
ucc_team_lib_debug_la_LDFLAGS = -module -avoid-version  $(VMC_LDFLAGS)
pkglib_LTLIBRARIES = $(component_install)
