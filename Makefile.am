#
# Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
# $HEADER$
#

if !DOCS_ONLY
SUBDIRS =      \
	src        \
	tools/info \
	cmake

if HAVE_MPICXX
SUBDIRS +=     \
	tools/perf \
	test/mpi
endif

if HAVE_GTEST
SUBDIRS += test/gtest
gtest:
	@make -C test/gtest test
endif
endif #!DOCS_ONLY

include $(srcdir)/docs/doxygen/doxygen.am

.PHONY: docs docs-clean

DOCLIST = docs/doxygen/doxygen-doc/ucc.tag

FORMAT = pdf
DOT_CLEANFILES =

docs: $(DOCLIST)

docs-clean:
	$(RM) $(DX_CLEANFILES)
	$(RM) $(DOT_CLEANFILES)

docs/doxygen/doxygen-doc/ucc.tag: $(doxygen_doc_files) doxygen-doc

.dot.pdf:
	dot -T pdf -o $@ $<

MOSTLYCLEANFILES = $(DX_CLEANFILES) $(DOT_CLEANFILES)
