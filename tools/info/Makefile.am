#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#
bin_PROGRAMS = ucc_info

ucc_info_SOURCES  = \
    build_info.c \
    ucc_info.c

noinst_HEADERS = \
    ucc_info.h

ucc_info_CPPFLAGS = $(AM_CPPFLAGS) -I${UCC_TOP_BUILDDIR}/src
ucc_info_LDADD = \
    $(UCC_TOP_BUILDDIR)/src/libucc.la
