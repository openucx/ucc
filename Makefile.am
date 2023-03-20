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
.PHONY: clean
clean:
    [tab]@rm -f $(PROGRAMS) *.o core
DOCLIST = docs/doxygen/doxygen-doc/ucc.tag

FORMAT = pdf
DOT_CLEANFILES =

docs: $(DOCLIST)

docs-clean:
	$(RM) $(DX_CLEANFILES)
	$(RM) $(DOT_CLEANFILES)

docs/doxygen/doxygen-doc/ucc.tag: $(doxygen_doc_files) doxygen-doc

clean-generic:

distclean-generic:
	-test -z "$(CONFIG_CLEAN_FILES)" || rm -f $(CONFIG_CLEAN_FILES)
	-test . = "$(srcdir)" || test -z "$(CONFIG_CLEAN_VPATH_FILES)" || rm -f $(CONFIG_CLEAN_VPATH_FILES)

maintainer-clean-generic:
	@echo "This command is intended for maintainers to use"
	@echo "it deletes files that may require special tools to rebuild."
clean: clean-recursive

clean-am: clean-generic clean-libtool mostlyclean-am

distclean: distclean-recursive
	-rm -f $(am__CONFIG_DISTCLEAN_FILES)
	-rm -f Makefile
distclean-am: clean-am distclean-generic distclean-libtool \
	distclean-tags

.dot.pdf:
	dot -T pdf -o $@ $<

MOSTLYCLEANFILES = $(DX_CLEANFILES) $(DOT_CLEANFILES)
