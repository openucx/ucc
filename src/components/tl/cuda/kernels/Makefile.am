#
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

NVCCFLAGS =                                      \
    ${AM_CPPFLAGS}                               \
    ${UCS_CPPFLAGS}                              \
    ${NVCC_CFLAGS}                               \
    -I${UCC_TOP_BUILDDIR}                        \
    -I${UCC_TOP_SRCDIR}/src                      \
    -I${UCC_TOP_BUILDDIR}/src                    \
    --compiler-options -fno-rtti,-fno-exceptions

LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(NVCC) -c $< -o $@  $(NVCCFLAGS) $(NVCC_ARCH)

.cu.lo:
	/bin/bash $(top_srcdir)/cuda_lt.sh "$(LIBTOOL)" $@ $(NVCC) -c  $< $(NVCCFLAGS) $(NVCC_ARCH)

comp_noinst = libucc_tl_cuda_kernels.la

if TL_CUDA_NVLS_ENABLED
libucc_tl_cuda_kernels_la_SOURCES  = reduce_scatter_kernel.cu allreduce_kernel.cu
else
libucc_tl_cuda_kernels_la_SOURCES  =
endif

libucc_tl_cuda_kernels_la_CPPFLAGS =

noinst_LTLIBRARIES = $(comp_noinst)

# Ensure kernels rebuild when helper header changes
reduce_scatter_kernel.lo: nvls.cuh
allreduce_kernel.lo: nvls.cuh
