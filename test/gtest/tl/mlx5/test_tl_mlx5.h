/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#ifndef TEST_TL_MLX5_H
#define TEST_TL_MLX5_H

#include <dlfcn.h>
#include "common/test_ucc.h"
#include "components/tl/mlx5/tl_mlx5.h"
#include "components/tl/mlx5/tl_mlx5_dm.h"
#include "components/tl/mlx5/tl_mlx5_ib.h"

#define CHECK_TEST_STATUS() \
  if (Test::HasFatalFailure() || Test::IsSkipped()) { \
    return; \
  }

typedef ucc_status_t (*ucc_tl_mlx5_create_ibv_ctx_fn_t)(
    char **ib_devname, struct ibv_context **ctx, ucc_base_lib_t *lib);

typedef int (*ucc_tl_mlx5_get_active_port_fn_t)(struct ibv_context *ctx);



class test_tl_mlx5 : public ucc::test {
  protected:
    void *tl_mlx5_so_handle;
  public:
    ucc_base_lib_t                   lib;
    ucc_tl_mlx5_create_ibv_ctx_fn_t  create_ibv_ctx;
    ucc_tl_mlx5_get_active_port_fn_t get_active_port;
    struct ibv_port_attr             port_attr;
    struct ibv_context *             ctx;
    struct ibv_pd *                  pd;
    struct ibv_cq *                  cq;
    int                              port;
    test_tl_mlx5();
    virtual ~test_tl_mlx5();
    virtual void SetUp() override;

    // Check for Mellanox/NVIDIA vendor ID (0x02c9) and CX7 (MT4129) vendor_part_id
    bool is_cx7_vendor_id() const
    {
        struct ibv_device_attr device_attr;
        ibv_query_device(ctx, &device_attr);
        return device_attr.vendor_id == 0x02c9 &&
               device_attr.vendor_part_id == 4129;
    }
};

#endif
