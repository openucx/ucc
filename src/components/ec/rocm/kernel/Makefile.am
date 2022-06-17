#
# Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
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
    -I${UCC_TOP_SRCDIR}/src/components/ec/rocm


LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(HIPCC) -c $< -o $@  $(HIPCCFLAGS) 

.cu.lo:
	/bin/bash $(top_srcdir)/cuda_lt.sh "$(LIBTOOL)" $@ $(HIPCC) -c  $< $(HIPCCFLAGS) 

comp_noinst = libucc_ec_rocm_kernels.la

libucc_ec_rocm_kernels_la_SOURCES  = ec_rocm_wait_kernel.cu  \
                                     ec_rocm_executor_kernel.cu
libucc_ec_rocm_kernels_la_CPPFLAGS =

noinst_LTLIBRARIES = $(comp_noinst)
