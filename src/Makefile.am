#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.

cl_dirs = components/cl/basic \
		  components/cl/hier

mc_dirs = components/mc/cpu
ec_dirs = components/ec/cpu
topo_dirs =

if HAVE_CUDA
mc_dirs += components/mc/cuda
ec_dirs += components/ec/cuda
topo_dirs += components/topo/cuda
endif

if HAVE_DOCA_UROM
cl_dirs += components/cl/doca_urom
endif

if HAVE_ROCM
mc_dirs += components/mc/rocm
ec_dirs += components/ec/rocm
endif

if HAVE_IBVERBS
topo_dirs += components/topo/ib
endif

SUBDIRS = .

# Component directories built in parallel after libucc.la.
# All components depend only on libucc.la, not on each other,
# so GNU make's job server can schedule them concurrently with -j.
COMPONENT_DIRS = $(cl_dirs) $(mc_dirs) $(ec_dirs) $(topo_dirs)

include components/tl/makefile.am

# Kernel-only subdirectories whose .cu compilation can overlap with libucc.la.
# NVCC only needs headers (source tree + config.h from configure), so there is
# no build-time dependency on the compiled core library.  The final link of the
# parent component (libucc_ec_cuda.la, etc.) still waits for libucc.la below.
early_cuda_kernel_dirs =
if HAVE_CUDA
early_cuda_kernel_dirs += components/ec/cuda/kernel
endif
if TL_CUDA_NVLS_ENABLED
early_cuda_kernel_dirs += components/tl/cuda/kernels
endif

# All possible component subdirectories (including conditional ones)
# needed for make dist to find all source files.
DIST_SUBDIRS = .                       \
    components/cl/basic                \
    components/cl/hier                 \
    components/cl/doca_urom            \
    components/mc/cpu                  \
    components/mc/cuda                 \
    components/mc/rocm                 \
    components/ec/cpu                  \
    components/ec/cuda                 \
    components/ec/rocm                 \
    components/tl/cuda                 \
    components/tl/mlx5                 \
    components/tl/nccl                 \
    components/tl/rccl                 \
    components/tl/self                 \
    components/tl/sharp                \
    components/tl/ucp                  \
    components/topo/cuda               \
    components/topo/ib

all-local: $(early_cuda_kernel_dirs:%=compile-kernels/%) $(COMPONENT_DIRS:%=all-component/%)

# Start CUDA kernel compilation immediately (no libucc.la dependency).
compile-kernels/%:
	+$(MAKE) -C $* all

# Components wait for libucc.la before linking.
all-component/%: libucc.la
	+$(MAKE) -C $* all

# Components that contain CUDA kernels must also wait for the early kernel
# compilation to finish, so two make processes never run in the same directory.
if HAVE_CUDA
all-component/components/ec/cuda: compile-kernels/components/ec/cuda/kernel
endif
if TL_CUDA_NVLS_ENABLED
all-component/components/tl/cuda: compile-kernels/components/tl/cuda/kernels
endif

# Component installs must wait for libucc.la to be installed to the prefix
# first, otherwise libtool relinking fails with "cannot find -lucc".
install-exec-local: install-libLTLIBRARIES $(COMPONENT_DIRS:%=install-component/%)
install-component/%: libucc.la install-libLTLIBRARIES
	+$(MAKE) -C $* install

clean-local: $(COMPONENT_DIRS:%=clean-component/%)
clean-component/%:
	+$(MAKE) -C $* clean

uninstall-local: $(COMPONENT_DIRS:%=uninstall-component/%)
uninstall-component/%:
	+$(MAKE) -C $* uninstall

distclean-local: $(COMPONENT_DIRS:%=distclean-component/%)
distclean-component/%:
	-+$(MAKE) -C $* distclean

lib_LTLIBRARIES  = libucc.la
noinst_LIBRARIES =

libucc_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS)
libucc_la_CFLAGS   = -c $(BASE_CFLAGS)
libucc_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed -pthread

nobase_dist_libucc_la_HEADERS =	\
	ucc/api/ucc.h                   \
	ucc/api/ucc_def.h               \
	ucc/api/ucc_version.h           \
	ucc/api/ucc_status.h

noinst_HEADERS =                       \
	core/ucc_global_opts.h             \
	core/ucc_lib.h                     \
	core/ucc_context.h                 \
	core/ucc_team.h                    \
	core/ucc_ee.h                      \
	core/ucc_progress_queue.h          \
	core/ucc_service_coll.h            \
	core/ucc_dt.h	                   \
	schedule/ucc_schedule.h            \
	schedule/ucc_schedule_pipelined.h  \
	coll_score/ucc_coll_score.h        \
	utils/arch/aarch64/cpu.h           \
	utils/arch/ppc64/cpu.h             \
	utils/arch/riscv64/cpu.h           \
	utils/arch/x86_64/cpu.h            \
	utils/arch/cpu.h                   \
	utils/arch/cuda_def.h              \
	utils/ucc_compiler_def.h           \
	utils/ucc_log.h                    \
	utils/ucc_parser.h                 \
	utils/ucc_component.h              \
	utils/ucc_datastruct.h             \
	utils/ucc_math.h                   \
	utils/ucc_coll_utils.h             \
	utils/ucc_list.h                   \
	utils/ucc_string.h                 \
	utils/ucc_queue.h                  \
	utils/ucc_proc_info.h              \
	utils/khash.h                      \
	utils/ini.h                        \
	utils/ucc_spinlock.h               \
	utils/ucc_mpool.h                  \
	utils/ucc_rcache.h                 \
	utils/profile/ucc_profile.h        \
	utils/profile/ucc_profile_on.h     \
	utils/profile/ucc_profile_off.h    \
	utils/ucc_time.h                   \
	utils/ucc_sys.h                    \
	utils/ucc_assert.h                 \
	utils/debug/log.h                  \
	components/base/ucc_base_iface.h   \
	components/cl/ucc_cl.h             \
	components/cl/ucc_cl_log.h         \
	components/cl/ucc_cl_type.h        \
	components/tl/ucc_tl.h             \
	components/tl/ucc_tl_log.h         \
	components/mc/ucc_mc.h             \
	components/mc/base/ucc_mc_base.h   \
	components/mc/ucc_mc_log.h         \
	components/ec/ucc_ec.h             \
	components/ec/base/ucc_ec_base.h   \
	components/ec/ucc_ec_log.h         \
	coll_patterns/recursive_knomial.h  \
	coll_patterns/sra_knomial.h        \
	coll_patterns/bruck_alltoall.h     \
	coll_patterns/double_binary_tree.h \
	components/topo/ucc_topo.h         \
	components/topo/ucc_sbgp.h         \
	components/topo/ucc_sysinfo.h      \
	components/topo/base/ucc_sysinfo_base.h

libucc_la_SOURCES =                   \
	core/ucc_lib.c                    \
	core/ucc_constructor.c            \
	core/ucc_global_opts.c            \
	core/ucc_version.c                \
	core/ucc_context.c                \
	core/ucc_team.c                   \
	core/ucc_ee.c                     \
	core/ucc_coll.c                   \
	core/ucc_progress_queue.c         \
	core/ucc_progress_queue_st.c      \
	core/ucc_progress_queue_mt.c      \
	core/ucc_service_coll.c           \
	core/ucc_dt.c                     \
	coll_patterns/ring.c              \
	schedule/ucc_schedule.c           \
	schedule/ucc_schedule_pipelined.c \
	coll_score/ucc_coll_score.c       \
	coll_score/ucc_coll_score_map.c   \
	utils/ini.c                       \
	utils/ucc_component.c             \
	utils/ucc_datastruct.c            \
	utils/ucc_status.c                \
	utils/ucc_mpool.c                 \
	utils/ucc_math.c                  \
	utils/ucc_proc_info.c             \
	utils/ucc_string.c                \
	utils/ucc_coll_utils.c            \
	utils/ucc_parser.c                \
	utils/profile/ucc_profile.c       \
	utils/ucc_sys.c                   \
	utils/arch/x86_64/cpu.c           \
	utils/arch/aarch64/cpu.c          \
	utils/ucc_assert.c                \
	utils/debug/log.c                 \
	components/base/ucc_base_iface.c  \
	components/cl/ucc_cl.c            \
	components/tl/ucc_tl.c            \
	components/mc/ucc_mc.c            \
	components/mc/base/ucc_mc_base.c  \
	components/ec/ucc_ec.c            \
	components/ec/base/ucc_ec_base.c  \
	components/topo/ucc_topo.c        \
	components/topo/ucc_sysinfo.c     \
	components/topo/ucc_sbgp.c

libucc_ladir = $(includedir)
