#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
#

sources =          \
	ucc_ucp_ctx.h  \
	ucc_ucp_ctx.c

libucc_ucp_ctx_la_SOURCES  = $(sources)
libucc_ucp_ctx_la_CPPFLAGS = $(AM_CPPFLAGS) $(UCX_CPPFLAGS)
libucc_ucp_ctx_la_LDFLAGS  = $(UCX_LDFLAGS) -version-info $(SOVERSION) --as-needed
libucc_ucp_ctx_la_LIBADD   = $(UCX_LIBADD)

pkglib_LTLIBRARIES = libucc_ucp_ctx.la
