/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_coll.h"
 
ucc_status_t ucc_tl_mlx5_mcast_service_bcast_post(void *arg, void *buf, size_t size, ucc_rank_t root,
                                                  ucc_service_coll_req_t **bcast_req);

ucc_status_t ucc_tl_mlx5_mcast_service_allgather_post(void *arg, void *sbuf, void *rbuf, size_t size,
                                                      ucc_service_coll_req_t **ag_req);

ucc_status_t ucc_tl_mlx5_mcast_service_barrier_post(void *arg, ucc_service_coll_req_t **barrier_req);

ucc_status_t ucc_tl_mlx5_mcast_service_allreduce_post(void *arg, void *sbuf, void *rbuf,
                                                       size_t count, ucc_datatype_t dt,
                                                       ucc_reduction_op_t op,
                                                       ucc_service_coll_req_t **red_req);

ucc_status_t ucc_tl_mlx5_mcast_service_coll_test(ucc_service_coll_req_t *req);

