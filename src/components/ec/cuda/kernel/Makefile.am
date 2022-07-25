#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

comp_noinst = libucc_ec_cuda_kernels.la

libucc_ec_cuda_kernels_la_SOURCES  = ec_cuda_wait_kernel.cu  \
                                     ec_cuda_executor.cu
libucc_ec_cuda_kernels_la_CPPFLAGS =

noinst_LTLIBRARIES = $(comp_noinst)
