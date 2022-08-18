/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_mc_reduce.h"
extern "C" {
#include "components/ec/ucc_ec.h"
}

template<typename T>
class test_mc_reduce : public testing::Test {
  protected:
    const int COUNT = 1024;
    ucc_memory_type_t mem_type;
    ucc_ee_executor_t *executor;

    virtual void SetUp() override
    {
        ucc_constructor();
        ucc_mc_params_t mc_params = {
            .thread_mode = UCC_THREAD_SINGLE,
        };
        ucc_ec_params_t ec_params = {
            .thread_mode = UCC_THREAD_SINGLE,
        };
        ucc_mc_init(&mc_params);
        ucc_ec_init(&ec_params);
        buf1_h = buf2_h = res_h = nullptr;
        buf1_d = buf2_d = res_d = nullptr;
        executor                = nullptr;
    }

    ucc_status_t alloc_executor(ucc_memory_type_t mtype)
    {
        ucc_ee_executor_params_t params;
        ucc_ee_type_t            coll_ee_type;
        ucc_status_t             status;

        switch (mtype) {
        case UCC_MEMORY_TYPE_CUDA:
            coll_ee_type = UCC_EE_CUDA_STREAM;
            break;
        case UCC_MEMORY_TYPE_HOST:
            coll_ee_type = UCC_EE_CPU_THREAD;
            break;
        default:
            std::cerr << "invalid executor mem type\n";
            return UCC_ERR_INVALID_PARAM;
            break;
        }
        params.mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
        params.ee_type = coll_ee_type;
        status         = ucc_ee_executor_init(&params, &executor);
        if (UCC_OK != status) {
            std::cerr << "failed to init executor: "
                      << ucc_status_string(status) << std::endl;
            return status;
        }
        status = ucc_ee_executor_start(executor, NULL);
        if (UCC_OK != status) {
            std::cerr << "failed to start executor: "
                      << ucc_status_string(status) << std::endl;
            ucc_ee_executor_finalize(executor);
        }
        return status;
    }

    ucc_status_t free_executor()
    {
        ucc_status_t status;

        status = ucc_ee_executor_stop(executor);
        if (UCC_OK != status) {
            std::cerr << "failed to stop executor: "
                      << ucc_status_string(status) << std::endl;
        }
        ucc_ee_executor_finalize(executor);
        return status;
    }

    ucc_status_t setup(ucc_memory_type_t mtype, size_t n)
    {
        ucc_status_t status;

        status = alloc_executor(mtype);
        if (UCC_OK != status) {
            return status;
        }
        return alloc_bufs(mtype, n);
    }

    ucc_status_t alloc_bufs(ucc_memory_type_t mtype, size_t n)
    {
        size_t n_bytes = COUNT*sizeof(typename T::type);
        mem_type = mtype;

        ucc_mc_alloc(&res_h_mc_header, n_bytes, UCC_MEMORY_TYPE_HOST);
        res_h = (typename T::type *)res_h_mc_header->addr;
        ucc_mc_alloc(&buf1_h_mc_header, n_bytes, UCC_MEMORY_TYPE_HOST);
        buf1_h = (typename T::type *)buf1_h_mc_header->addr;
        ucc_mc_alloc(&buf2_h_mc_header, n * n_bytes, UCC_MEMORY_TYPE_HOST);
        buf2_h = (typename T::type *)buf2_h_mc_header->addr;

        for (int i = 0; i < COUNT; i++) {
            res_h[i] = (typename T::type)(0);
        }
        for (int i = 0; i < COUNT; i++) {
            /* bFloat16 will be assigned with the floats matching the
               uint16_t bit pattern*/
            buf1_h[i] = (typename T::type)(i + 1);
        }
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < COUNT; i++) {
                buf2_h[i + j * COUNT] = (typename T::type)(2 * i + j + 1);
            }
        }
        if (mtype != UCC_MEMORY_TYPE_HOST) {
            ucc_mc_alloc(&res_d_mc_header, n_bytes, mtype);
            res_d = (typename T::type *)res_d_mc_header->addr;
            ucc_mc_alloc(&buf1_d_mc_header, n_bytes, mtype);
            buf1_d = (typename T::type *)buf1_d_mc_header->addr;
            ucc_mc_alloc(&buf2_d_mc_header, n * n_bytes, mtype);
            buf2_d = (typename T::type *)buf2_d_mc_header->addr;
            ucc_mc_memcpy(res_d, res_h, n_bytes, mtype, UCC_MEMORY_TYPE_HOST);
            ucc_mc_memcpy(buf1_d, buf1_h, n_bytes, mtype, UCC_MEMORY_TYPE_HOST);
            ucc_mc_memcpy(buf2_d, buf2_h, n * n_bytes, mtype,
                          UCC_MEMORY_TYPE_HOST);
            buf1 = buf1_d;
            buf2 = buf2_d;
            res  = res_d;
        } else {
            buf1 = buf1_h;
            buf2 = buf2_h;
            res  = res_h;
        }

        return UCC_OK;
    }

    ucc_status_t free_bufs(ucc_memory_type_t mtype)
    {
        if (buf1_h != nullptr) {
            ucc_mc_free(buf1_h_mc_header);
        }
        if (buf2_h != nullptr) {
            ucc_mc_free(buf2_h_mc_header);
        }
        if (res_h != nullptr) {
            ucc_mc_free(res_h_mc_header);
        }
        if (buf1_d != nullptr) {
            ucc_mc_free(buf1_d_mc_header);
        }
        if (buf2_d != nullptr) {
            ucc_mc_free(buf2_d_mc_header);
        }
        if (res_d != nullptr) {
            ucc_mc_free(res_d_mc_header);
        }

        return UCC_OK;
    }

    virtual void TearDown() override
    {
        free_bufs(mem_type);
        if (executor) {
            free_executor();
        }
        ucc_mc_finalize();
    }

    ucc_status_t do_reduce(void *src1, void *src2, void *dst, size_t count,
                           uint16_t n_src2, size_t stride, ucc_datatype_t dt,
                           ucc_reduction_op_t op, bool with_alpha, double alpha)
    {
        ucc_ee_executor_task_args_t eargs;
        ucc_status_t                status;
        ucc_ee_executor_task_t *    task;

        eargs.flags     = with_alpha ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0;
        eargs.task_type = UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED;
        eargs.reduce_strided.count  = count;
        eargs.reduce_strided.dt     = dt;
        eargs.reduce_strided.op     = op;
        eargs.reduce_strided.n_src2 = n_src2;
        eargs.reduce_strided.dst    = dst;
        eargs.reduce_strided.src1   = src1;
        eargs.reduce_strided.src2   = src2;
        eargs.reduce_strided.stride = stride;
        eargs.reduce_strided.alpha  = alpha;

        status = ucc_ee_executor_task_post(executor, &eargs, &task);
        if (UCC_OK != status) {
            std::cerr << "failed to post executor task: "
                      << ucc_status_string(status) << std::endl;
            return status;
        }

        while (0 < (status = ucc_ee_executor_task_test(task))) {
            ;
        }
        ucc_ee_executor_task_finalize(task);

        return status;
    }

    void test_reduce(ucc_memory_type_t mt) {
        ucc_status_t status;

        if (UCC_OK !=  ucc_mc_available(mt)) {
            GTEST_SKIP();
        }
        ASSERT_EQ(this->setup(mt, 1), UCC_OK);
        status = do_reduce(this->buf1, this->buf2, this->res, this->COUNT, 1, 0,
                           T::dt, T::redop, false, 0);
        if (UCC_ERR_NOT_SUPPORTED == status) {
            GTEST_SKIP();
        }
        ASSERT_EQ(status, UCC_OK);

        if (mt != UCC_MEMORY_TYPE_HOST) {
            ucc_mc_memcpy(this->res_h, this->res_d, this->COUNT * sizeof(*this->res_d),
                          UCC_MEMORY_TYPE_HOST, mt);
        }
        for (int i = 0; i < this->COUNT; i++) {
            T::assert_equal(T::do_op(this->buf1_h[i],
                                     this->buf2_h[i]), this->res_h[i]);
        }
    };

    void test_reduce_multi(ucc_memory_type_t mt) {
        const int    num_vec = 3;
        ucc_status_t status;

        if (UCC_OK !=  ucc_mc_available(mt)) {
            GTEST_SKIP();
        }
        ASSERT_EQ(this->setup(mt, num_vec), UCC_OK);
        status = do_reduce(this->buf1, this->buf2, this->res, this->COUNT,
                           num_vec, this->COUNT * sizeof(*this->buf2), T::dt,
                           T::redop, false, 0);
        if (UCC_ERR_NOT_SUPPORTED == status) {
            GTEST_SKIP();
        }
        ASSERT_EQ(status, UCC_OK);

        if (mt != UCC_MEMORY_TYPE_HOST) {
            ucc_mc_memcpy(this->res_h, this->res_d, this->COUNT * sizeof(*this->res_d),
                          UCC_MEMORY_TYPE_HOST, mt);
        }
        for (int i = 0; i < this->COUNT; i++) {
            typename T::type res = T::do_op(this->buf1_h[i],
                                                            this->buf2_h[i]);
            for (int j = 1; j < num_vec; j++) {
                res = T::do_op(this->buf2_h[i + j * this->COUNT], res);
            }
            T::assert_equal(res, this->res_h[i]);
        }
    };

    void test_reduce_multi_alpha(ucc_memory_type_t mt) {
        const int    num_vec = 20;
        const double alpha   = 0.7;
        ucc_status_t status;

        if (UCC_OK !=  ucc_mc_available(mt)) {
            GTEST_SKIP();
        }

        ASSERT_EQ(UCC_OK, this->setup(mt, num_vec));
        status = do_reduce(this->buf1, this->buf2, this->res, this->COUNT,
                           num_vec, this->COUNT * sizeof(*this->buf2), T::dt,
                           T::redop, true, alpha);

        if (UCC_ERR_NOT_SUPPORTED == status) {
            GTEST_SKIP();
        }
        ASSERT_EQ(status, UCC_OK);

        if (mt != UCC_MEMORY_TYPE_HOST) {
            ucc_mc_memcpy(this->res_h, this->res_d, this->COUNT * sizeof(*this->res_d),
                          UCC_MEMORY_TYPE_HOST, mt);
        }
        for (int i = 0; i < this->COUNT; i++) {
            typename T::type res = T::do_op(this->buf1_h[i], this->buf2_h[i]);
            for (int j = 1; j < num_vec; j++) {
                res = T::do_op(this->buf2_h[i + j * this->COUNT], res);
            }
            if (T::dt == UCC_DT_BFLOAT16) {
                float32tobfloat16(bfloat16tofloat32(&res)*(float)alpha, &res);
            } else {
                res *= (typename T::type)alpha;
            }
            T::assert_equal(res, this->res_h[i]);
        }
    }
    ucc_mc_buffer_header_t *buf1_h_mc_header, *buf2_h_mc_header,
        *res_h_mc_header, *buf1_d_mc_header, *buf2_d_mc_header,
        *res_d_mc_header;
    typename T::type *buf1_h;
    typename T::type *buf2_h;
    typename T::type *res_h;
    typename T::type *buf1_d;
    typename T::type *buf2_d;
    typename T::type *res_d;
    typename T::type *buf1;
    typename T::type *buf2;
    typename T::type *res;
};

#define INT_OP_PAIRS(_TYPE) ARITHMETIC_OP_PAIRS(_TYPE),         \
        TypeOpPair<UCC_DT_ ## _TYPE, land>,                     \
        TypeOpPair<UCC_DT_ ## _TYPE, lor>,                      \
        TypeOpPair<UCC_DT_ ## _TYPE, lxor>,                     \
        TypeOpPair<UCC_DT_ ## _TYPE, band>,                     \
        TypeOpPair<UCC_DT_ ## _TYPE, bor>,                      \
        TypeOpPair<UCC_DT_ ## _TYPE, bxor>

using TypeOpPairsInt = ::testing::Types<INT_OP_PAIRS(INT8), INT_OP_PAIRS(INT16),
                                      INT_OP_PAIRS(INT32), INT_OP_PAIRS(INT64)>;

using TypeOpPairsUint = ::testing::Types<INT_OP_PAIRS(UINT8), INT_OP_PAIRS(UINT16),
                                      INT_OP_PAIRS(UINT32), INT_OP_PAIRS(UINT64)>;

using TypeOpPairsFloat = ::testing::Types<ARITHMETIC_OP_PAIRS(FLOAT32),
                                          ARITHMETIC_OP_PAIRS(FLOAT64),
                                          ARITHMETIC_OP_PAIRS(FLOAT128),
                                          ARITHMETIC_OP_PAIRS(BFLOAT16),
                                          TypeOpPair<UCC_DT_FLOAT32_COMPLEX, sum>,
                                          TypeOpPair<UCC_DT_FLOAT32_COMPLEX, prod>,
                                          TypeOpPair<UCC_DT_FLOAT64_COMPLEX, sum>,
                                          TypeOpPair<UCC_DT_FLOAT64_COMPLEX, prod>,
                                          TypeOpPair<UCC_DT_FLOAT128_COMPLEX, sum>,
                                          TypeOpPair<UCC_DT_FLOAT128_COMPLEX, prod>,
                                          TypeOpPair<UCC_DT_FLOAT32, avg>,
                                          TypeOpPair<UCC_DT_FLOAT64, avg>,
                                          TypeOpPair<UCC_DT_BFLOAT16, avg>>;

template<typename T>
class test_mc_reduce_int : public test_mc_reduce<T> {};
TYPED_TEST_CASE(test_mc_reduce_int, TypeOpPairsInt);

template<typename T>
class test_mc_reduce_uint : public test_mc_reduce<T> {};
TYPED_TEST_CASE(test_mc_reduce_uint, TypeOpPairsUint);

template<typename T>
class test_mc_reduce_float : public test_mc_reduce<T> {};
TYPED_TEST_CASE(test_mc_reduce_float, TypeOpPairsFloat);

#define DECLARE_REDUCE_TEST(_type, _mt)             \
    TYPED_TEST(test_mc_reduce_ ## _type, _mt) {     \
        this->test_reduce(UCC_MEMORY_TYPE_ ## _mt); \
    }                                               \

#define DECLARE_REDUCE_MULTI_TEST(_type, _mt)               \
    TYPED_TEST(test_mc_reduce_ ## _type, multi_ ## _mt) {   \
        this->test_reduce_multi(UCC_MEMORY_TYPE_ ## _mt);   \
    }                                                       \

#define DECLARE_REDUCE_MULTI_ALPHA_TEST(_type, _mt)               \
    TYPED_TEST(test_mc_reduce_ ## _type, multi_alpha_ ## _mt) {   \
        this->test_reduce_multi_alpha(UCC_MEMORY_TYPE_ ## _mt);   \
    }                                                       \

DECLARE_REDUCE_TEST(int, HOST);
DECLARE_REDUCE_TEST(uint, HOST);
DECLARE_REDUCE_TEST(float, HOST);

DECLARE_REDUCE_MULTI_TEST(int, HOST);
DECLARE_REDUCE_MULTI_TEST(uint, HOST);
DECLARE_REDUCE_MULTI_TEST(float, HOST);

DECLARE_REDUCE_MULTI_ALPHA_TEST(float, HOST);

#ifdef HAVE_CUDA
DECLARE_REDUCE_TEST(int, CUDA);
DECLARE_REDUCE_TEST(uint, CUDA);
DECLARE_REDUCE_TEST(float, CUDA);

DECLARE_REDUCE_MULTI_TEST(int, CUDA);
DECLARE_REDUCE_MULTI_TEST(uint, CUDA);
DECLARE_REDUCE_MULTI_TEST(float, CUDA);

DECLARE_REDUCE_MULTI_ALPHA_TEST(float, CUDA);
#endif
