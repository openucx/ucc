/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "test_mc_reduce.h"

TYPED_TEST(test_mc_reduce, ucc_reduce_single_host) {
    this->alloc_bufs(UCC_MEMORY_TYPE_HOST, 1);
    ucc_mc_reduce(this->buf1_h, this->buf2_h, this->res_h,
                  this->COUNT, TypeParam::dt,
                  TypeParam::redop, UCC_MEMORY_TYPE_HOST);
    for (int i = 0; i < this->COUNT; i++) {
        TypeParam::assert_equal(TypeParam::do_op(this->buf1_h[i],
                                this->buf2_h[i]), this->res_h[i]);
    }
}

TYPED_TEST(test_mc_reduce, ucc_reduce_multi_host) {
    const int num_vec = 3;
    this->alloc_bufs(UCC_MEMORY_TYPE_HOST, num_vec);
    ucc_mc_reduce_multi(this->buf1_h, this->buf2_h, this->res_h, num_vec,
                        this->COUNT, this->COUNT*sizeof(*this->buf2_h),
                        TypeParam::dt, TypeParam::redop, UCC_MEMORY_TYPE_HOST);
    for (int i = 0; i < this->COUNT; i++) {
        typename TypeParam::type res = TypeParam::do_op(this->buf1_h[i],
                                                        this->buf2_h[i]);
        for (int j = 1; j < num_vec; j++) {
            res = TypeParam::do_op(this->buf2_h[i + j * this->COUNT], res);
        }
        TypeParam::assert_equal(res, this->res_h[i]);
    }
}

TYPED_TEST(test_mc_reduce_alpha, ucc_reduce_multi_alpha_host) {
    const int num_vec = 3;
    const double alpha = 0.7;
    this->alloc_bufs(UCC_MEMORY_TYPE_HOST, num_vec);
    ucc_mc_reduce_multi_alpha(this->buf1_h, this->buf2_h, this->res_h, num_vec,
                              this->COUNT, this->COUNT*sizeof(*this->buf2_h),
                              TypeParam::dt, TypeParam::redop, UCC_OP_PROD, alpha,
                              UCC_MEMORY_TYPE_HOST);
    for (int i = 0; i < this->COUNT; i++) {
        typename TypeParam::type res = TypeParam::do_op(this->buf1_h[i],
                                                        this->buf2_h[i]);
        for (int j = 1; j < num_vec; j++) {
            res = TypeParam::do_op(this->buf2_h[i + j * this->COUNT], res);
        }
        if (TypeParam::dt == UCC_DT_BFLOAT16) {
            float32tobfloat16(bfloat16tofloat32(&res)*(float)alpha, &res);
        } else {
            res *= (typename TypeParam::type)alpha;
        }
        TypeParam::assert_equal(res, this->res_h[i]);
    }
}

TYPED_TEST(test_mc_reduce_alpha, ucc_reduce_multi_alpha_host_many_vectors) {
    const int num_vec = 20;
    const double alpha = 0.7;
    this->alloc_bufs(UCC_MEMORY_TYPE_HOST, num_vec);
    ucc_mc_reduce_multi_alpha(this->buf1_h, this->buf2_h, this->res_h, num_vec,
                              this->COUNT, this->COUNT*sizeof(*this->buf2_h),
                              TypeParam::dt, TypeParam::redop, UCC_OP_PROD, alpha,
                              UCC_MEMORY_TYPE_HOST);
    for (int i = 0; i < this->COUNT; i++) {
        typename TypeParam::type res = TypeParam::do_op(this->buf1_h[i],
                                                        this->buf2_h[i]);
        for (int j = 1; j < num_vec; j++) {
            res = TypeParam::do_op(this->buf2_h[i + j * this->COUNT], res);
        }
        if (TypeParam::dt == UCC_DT_BFLOAT16){
            float32tobfloat16(bfloat16tofloat32(&res)*(float)alpha, &res);
        } else {
            res *= (typename TypeParam::type)alpha;
        }
        TypeParam::assert_equal(res, this->res_h[i]);
    }
}
