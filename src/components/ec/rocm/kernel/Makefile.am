#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
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
	$(HIPCC) -c $< -o $@  $(ROCM_ARCH) $(HIPCCFLAGS)

.cu.lo:
	/bin/bash $(top_srcdir)/cuda_lt.sh "$(LIBTOOL)" $@ $(HIPCC) -c  $< $(ROCM_ARCH) $(HIPCCFLAGS)

comp_noinst = libucc_ec_rocm_kernels.la

libucc_ec_rocm_kernels_la_SOURCES  = ec_rocm_executor_kernel.cu \
                                     ec_rocm_reduce.cu
libucc_ec_rocm_kernels_la_CPPFLAGS =

noinst_LTLIBRARIES = $(comp_noinst)
