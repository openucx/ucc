#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

HIPCCFLAGS =                                     \
    ${AM_CPPFLAGS}                               \
    ${UCS_CPPFLAGS}                              \
    ${HIP_CPPFLAGS}                              \
    ${ROCM_CPPFLAGS}                             \
    -I${UCC_TOP_BUILDDIR}                        \
    -I${UCC_TOP_SRCDIR}                          \
    -I${UCC_TOP_SRCDIR}/src                      \
    -I${UCC_TOP_BUILDDIR}/src                    \
    -I${UCC_TOP_SRCDIR}/src/components/mc/rocm

LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(HIPCC) -c $< -o $@  $(HIPCCFLAGS) 

.cu.lo:
	/bin/bash $(top_srcdir)/cuda_lt.sh "$(LIBTOOL)" $@ $(HIPCC) -c  $< $(HIPCCFLAGS)

comp_noinst = libucc_mc_rocm_kernels.la

libucc_mc_rocm_kernels_la_SOURCES  = mc_rocm_reduce.cu              \
                                     mc_rocm_reduce_multi.cu        \
                                     mc_rocm_reduce_multi_alpha.cu
libucc_mc_rocm_kernels_la_CPPFLAGS =

noinst_LTLIBRARIES = $(comp_noinst)
