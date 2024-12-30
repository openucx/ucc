/**
 * Copyright (c) 2001-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "test_ucc.h"

int main(int argc, char **argv)
{
    int ret;

#ifdef HAVE_CUDA
    cudaSetDevice(0);
#endif
    ::testing::InitGoogleTest(&argc, argv);

    ret = RUN_ALL_TESTS();

    UccJob::cleanup();
    return ret;
}
