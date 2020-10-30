#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $HEADER$
#
SUBDIRS = \
       src \
       tools/info

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
