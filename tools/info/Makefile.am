#
# Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) The University of Tennessee and the University of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

bin_PROGRAMS = ucc_info

BUILT_SOURCES  = build_config.h
DISTCLEANFILES = build_config.h

#
# Produce a C header file which contains all defined variables from config.h
#
build_config.h: $(top_builddir)/config.h Makefile
	$(SED) -nr 's:\s*#define\s+(\w+)(\s+(\w+)|\s+(".*")|\s*)$$:{"\1", UCC_PP_MAKE_STRING(\3\4)},:p' <$(top_builddir)/config.h >$@

ucc_info_SOURCES  = \
	build_info.c \
	ucc_info.c

noinst_HEADERS = \
	ucc_info.h

nodist_ucc_info_SOURCES = \
	build_config.h

ucc_info_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) -I${UCC_TOP_BUILDDIR}/src
ucc_info_CFLAGS = $(BASE_CFLAGS)
ucc_info_LDFLAGS = -Wl,--rpath-link=${UCS_LIBDIR}
ucc_info_LDADD = \
	$(UCC_TOP_BUILDDIR)/src/libucc.la
