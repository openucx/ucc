#
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
# $HEADER$
#

if !DOCS_ONLY
SUBDIRS = src

PARALLEL_SUBDIRS = contrib tools/info cmake

if HAVE_MPICXX
PARALLEL_SUBDIRS += tools/perf test/mpi
endif

if HAVE_GTEST
PARALLEL_SUBDIRS += test/gtest
gtest:
	@make -C test/gtest test
endif

DIST_SUBDIRS = src contrib tools/info cmake tools/perf test/mpi test/gtest

all-local: $(PARALLEL_SUBDIRS:%=all-par/%)

# FORCE is a phony prerequisite that ensures pattern-rule targets are always
# re-evaluated even if a directory with a matching name exists on disk.
.PHONY: FORCE
FORCE:

all-par/%: FORCE
	+$(MAKE) -C $* all

install-data-local: $(PARALLEL_SUBDIRS:%=install-par/%)
install-exec-local: $(PARALLEL_SUBDIRS:%=install-par/%)
install-par/%: FORCE
	+$(MAKE) -C $* install

clean-local: $(PARALLEL_SUBDIRS:%=clean-par/%)
clean-par/%: FORCE
	+$(MAKE) -C $* clean

uninstall-local: $(PARALLEL_SUBDIRS:%=uninstall-par/%)
uninstall-par/%: FORCE
	+$(MAKE) -C $* uninstall

check-local: $(PARALLEL_SUBDIRS:%=check-par/%)
check-par/%: FORCE
	+$(MAKE) -C $* check

distclean-local: $(PARALLEL_SUBDIRS:%=distclean-par/%)
distclean-par/%: FORCE
	-+$(MAKE) -C $* distclean

endif #!DOCS_ONLY

include $(srcdir)/docs/doxygen/doxygen.am

.PHONY: docs docs-clean

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = ucc.pc

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
