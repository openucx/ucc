#
# Copyright (c) 2001-2014, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and the University of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
# Copyright (C) Los Alamos National Security, LLC. 2018 ALL RIGHTS RESERVED.
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
# Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
#
# See file LICENSE for terms.
#

if HAVE_GTEST

# Set default configuration for running tests
UCC_HANDLE_ERRORS        ?= freeze
UCC_LOG_LEVEL            ?= warn
UCC_LOG_PRINT_ENABLE     ?= y
GTEST_FILTER             ?= *
GTEST_EXTRA_ARGS         ?=
LAUNCHER                 ?=
VALGRIND_EXTRA_ARGS      ?=

export UCC_HANDLE_ERRORS
export UCC_LOG_LEVEL
export UCC_LOG_PRINT_ENABLE

GTEST_ARGS = \
	--gtest_filter=$(GTEST_FILTER) \
	$(GTEST_EXTRA_ARGS)

VALGRIND_ARGS = \
	--tool=memcheck \
	--leak-check=full \
	--track-origins=yes \
	--fair-sched=try \
	--num-callers=25 \
	--error-exitcode=1 \
	--child-silent-after-fork=yes \
	--suppressions=$(top_srcdir)/contrib/valgrind.supp \
	$(VALGRIND_EXTRA_ARGS)

noinst_PROGRAMS = gtest

gtestdir = $(includedir)
gtest_LDADD = \
	$(top_builddir)/src/libucc.la \
	$(GTEST_LIBS)


gtest_CPPFLAGS = \
	$(BASE_CPPFLAGS) \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/test \
	-I$(top_builddir)/src \
	-I$(top_srcdir)/test/gtest \
	$(GTEST_CPPFLAGS)

gtest_LDFLAGS = $(GTEST_LDFLAGS) -pthread -no-install -Wl,-dynamic-list-data \
    -Wl,--rpath-link=${UCS_LIBDIR}

gtest_CFLAGS   = $(BASE_CFLAGS) $(AM_CPPFLAGS)
gtest_CXXFLAGS = -std=gnu++11 \
	$(BASE_CXXFLAGS) $(GTEST_CXXFLAGS) \
	-DGTEST_UCM_HOOK_LIB_DIR="\"${abs_builddir}/ucm/test_dlopen/.libs\"" \
	-DGTEST_UCC_TOP_SRCDIR="\"${UCC_TOP_SRCDIR}\""

gtest_SOURCES =                     \
	common/gtest-all.cc             \
	common/test_obj_size.cc         \
	common/main.cc                  \
	common/test_ucc.cc              \
	tl/tl_test.cc                   \
	core/test_lib_config.cc         \
	core/test_lib.cc                \
	core/test_context_config.cc     \
	core/test_context.cc            \
	core/test_mc.cc                 \
	core/test_mc_reduce.cc          \
	core/test_team.cc               \
	core/test_schedule.cc           \
	core/test_topo.cc               \
	core/test_service_coll.cc       \
	core/test_timeout.cc            \
	core/test_utils.cc              \
	coll/test_barrier.cc            \
	coll/test_alltoall.cc           \
	coll/test_alltoallv.cc          \
	coll/test_allgather.cc          \
	coll/test_allgatherv.cc         \
	coll/test_gather.cc         	\
	coll/test_bcast.cc              \
	coll/test_reduce.cc             \
	coll/test_allreduce.cc          \
	coll/test_reduce_scatter.cc     \
	coll/test_reduce_scatterv.cc    \
	utils/test_string.cc            \
	utils/test_ep_map.cc            \
	utils/test_lock_free_queue.cc   \
	utils/test_math.cc              \
	utils/test_cfg_file.cc          \
	coll_score/test_score.cc        \
	coll_score/test_score_str.cc    \
	coll_score/test_score_update.cc \
	active_set/test_active_set.cc

if HAVE_CUDA
gtest_SOURCES += \
	core/test_mc_cuda.cc
gtest_CPPFLAGS += \
	$(CUDA_CPPFLAGS)
gtest_LDFLAGS += \
	$(CUDA_LDFLAGS)
gtest_LDADD += \
	$(CUDA_LIBS)
endif

if HAVE_ROCM
gtest_SOURCES  += core/test_mc_rocm.cc
gtest_CPPFLAGS += $(HIP_CPPFLAGS)
gtest_LDFLAGS  += $(HIP_LDFLAGS)
gtest_LDADD    += $(HIP_LIBS)
endif


noinst_HEADERS =            \
	common/gtest.h          \
	common/test.h           \
	common/test_ucc.h       \
	core/test_context.h     \
	core/test_mc_reduce.h   \
	coll_score/test_score.h

.PHONY: test test gdb valgrind fix_rpath ucc


all-local: gtest

ucc:
	$(MAKE) -C $(top_builddir)


help:
	@echo
	@echo "Targets:"
	@echo "  list          : List unit tests."
	@echo "  test          : Run unit tests."
	@echo "  test_gdb      : Run unit tests with GDB."
	@echo "  test_valgrind : Run unit tests with Valgrind."
	@echo
	@echo "Environment variables:"
	@echo "  GTEST_FILTER        : Unit tests filter (\"$(GTEST_FILTER)\")"
	@echo "  GTEST_EXTRA_ARGS    : Additional arguments for gtest (\"$(GTEST_EXTRA_ARGS)\")"
	@echo "  LAUNCHER            : Custom launcher for gtest executable (\"$(LAUNCHER)\")"
	@echo "  VALGRIND_EXTRA_ARGS : Additional arguments for Valgrind (\"$(VALGRIND_EXTRA_ARGS)\")"
	@echo

#
# List unit tests
#
list: gtest
	$(abs_builddir)/gtest --gtest_list_tests $(GTEST_ARGS)

#
# Run unit tests
#
test: ucc gtest
	@rm -f core.*
	$(LAUNCHER) stdbuf -e0 -o0 $(abs_builddir)/gtest $(GTEST_ARGS)

#
# Run unit tests with GDB
#
test_gdb: ucc gtest
	echo -e 'r\ninit-if-undefined $$_exitcode=-1\nif $$_exitcode>=0\n\tq\nend' > .gdbcommands
	$(LAUNCHER) env UCC_HANDLE_ERRORS=none \
		gdb -x .gdbcommands --args $(GDB_ARGS) \
			$(abs_builddir)/gtest $(GTEST_ARGS)

#
# Run unit tests with valgrind
#
test_valgrind: ucc gtest
	$(LAUNCHER) env LD_LIBRARY_PATH="$(VALGRIND_LIBPATH):${LD_LIBRARY_PATH}" \
	stdbuf -e0 -o0 valgrind $(VALGRIND_ARGS) $(abs_builddir)/gtest $(GTEST_ARGS)
endif
