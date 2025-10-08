/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_COLL_H
#define UCC_PT_COLL_H

#include "ucc_pt_comm.h"
#include "generator/ucc_pt_generator.h"
#include <ucc/api/ucc.h>
extern "C" {
#include <components/ec/ucc_ec.h>
#include <components/mc/ucc_mc.h>
}

ucc_status_t ucc_pt_alloc(ucc_mc_buffer_header_t **h_ptr, size_t len,
                          ucc_memory_type_t mem_type);

ucc_status_t ucc_pt_free(ucc_mc_buffer_header_t *h_ptr);

typedef union {
    ucc_coll_args_t             coll_args;
    ucc_ee_executor_task_args_t executor_args;
} ucc_pt_test_args_t;

class ucc_pt_coll {
protected:
    bool has_inplace_;
    bool has_reduction_;
    bool has_range_;
    bool has_bw_;
    int  root_shift_;
    ucc_pt_comm *comm;
    ucc_pt_generator_base *generator;
    ucc_coll_args_t coll_args;
    ucc_ee_executor_task_args_t executor_args;
    ucc_mc_buffer_header_t *dst_header;
    ucc_mc_buffer_header_t *src_header;
    ucc_mem_map_mem_h src_memh;
    ucc_mem_map_mem_h dst_memh;
    ucc_mem_map_mem_h *dst_memh_global;
    ucc_mem_map_mem_h *src_memh_global;
public:
    ucc_pt_coll(ucc_pt_comm *communicator, ucc_pt_generator_base *generator)
    {
        this->comm = communicator;
        this->generator = generator;
        src_header = nullptr;
        dst_header = nullptr;
        src_memh = nullptr;
        dst_memh = nullptr;
        dst_memh_global = nullptr;
        src_memh_global = nullptr;
    }
    virtual ucc_status_t init_args(ucc_pt_test_args_t &args) = 0;
    virtual void free_args(ucc_pt_test_args_t &args) {}
    virtual float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args)
    {
        return 0.0;
    }
    bool has_reduction();
    bool has_inplace();
    bool has_range();
    bool has_bw();
    virtual ~ucc_pt_coll() {};
};

class ucc_pt_coll_allgather: public ucc_pt_coll {
public:
    ucc_pt_coll_allgather(ucc_datatype_t dt, ucc_memory_type mt,
                          bool is_inplace, bool is_persistent,
                          ucc_pt_map_type_t map_type,
                          ucc_pt_comm *communicator,
                          ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
    ~ucc_pt_coll_allgather();
};

class ucc_pt_coll_allgatherv: public ucc_pt_coll {
public:
    ucc_pt_coll_allgatherv(ucc_datatype_t dt, ucc_memory_type mt,
                           bool is_inplace, bool is_persistent,
                           ucc_pt_comm *communicator,
                           ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
};

class ucc_pt_coll_allreduce: public ucc_pt_coll {
public:
    ucc_pt_coll_allreduce(ucc_datatype_t dt, ucc_memory_type mt,
                          ucc_reduction_op_t op, bool is_inplace,
                          bool is_persistent, ucc_pt_comm *communicator,
                          ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
};

class ucc_pt_coll_alltoall: public ucc_pt_coll {
public:
    ucc_pt_coll_alltoall(ucc_datatype_t dt, ucc_memory_type mt,
                         bool is_inplace, bool is_persistent,
                         ucc_pt_map_type_t map_type,
                         ucc_pt_comm *communicator,
                         ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
    ~ucc_pt_coll_alltoall();
};

class ucc_pt_coll_alltoallv: public ucc_pt_coll {
public:
    ucc_pt_coll_alltoallv(ucc_datatype_t dt, ucc_memory_type mt,
                          bool is_inplace, bool is_persistent,
                          ucc_pt_comm *communicator,
                          ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
    ~ucc_pt_coll_alltoallv();
};

class ucc_pt_coll_barrier: public ucc_pt_coll {
public:
    ucc_pt_coll_barrier(ucc_pt_comm *communicator,
                        ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
};

class ucc_pt_coll_bcast: public ucc_pt_coll {
public:
    ucc_pt_coll_bcast(ucc_datatype_t dt, ucc_memory_type mt, int root_shift,
                      bool is_persistent, ucc_pt_comm *communicator,
                      ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
};

class ucc_pt_coll_gather: public ucc_pt_coll {
public:
    ucc_pt_coll_gather(ucc_datatype_t dt, ucc_memory_type mt,
                       bool is_inplace, bool is_persistent, int root_shift,
                       ucc_pt_comm *communicator,
                       ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
};

class ucc_pt_coll_gatherv: public ucc_pt_coll {
public:
    ucc_pt_coll_gatherv(ucc_datatype_t dt, ucc_memory_type mt,
                        bool is_inplace, bool is_persistent, int root_shift,
                        ucc_pt_comm *communicator,
                        ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
};

class ucc_pt_coll_reduce: public ucc_pt_coll {
public:
    ucc_pt_coll_reduce(ucc_datatype_t dt, ucc_memory_type mt,
                       ucc_reduction_op_t op, bool is_inplace, bool is_persistent,
                       int root_shift, ucc_pt_comm *communicator,
                       ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
};

class ucc_pt_coll_reduce_scatter: public ucc_pt_coll {
public:
    ucc_pt_coll_reduce_scatter(ucc_datatype_t dt, ucc_memory_type mt,
                               ucc_reduction_op_t op, bool is_inplace,
                               bool is_persistent, ucc_pt_comm *communicator,
                               ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
};

class ucc_pt_coll_reduce_scatterv: public ucc_pt_coll {
public:
    ucc_pt_coll_reduce_scatterv(ucc_datatype_t dt, ucc_memory_type mt,
                                ucc_reduction_op_t op, bool is_inplace,
                                bool is_persistent, ucc_pt_comm *communicator,
                                ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
};

class ucc_pt_coll_scatter: public ucc_pt_coll {
public:
    ucc_pt_coll_scatter(ucc_datatype_t dt, ucc_memory_type mt,
                        bool is_inplace, bool is_persistent, int root_shift,
                        ucc_pt_comm *communicator,
                        ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
};

class ucc_pt_coll_scatterv: public ucc_pt_coll {
public:
    ucc_pt_coll_scatterv(ucc_datatype_t dt, ucc_memory_type mt,
                         bool is_inplace, bool is_persistent, int root_shift,
                         ucc_pt_comm *communicator,
                         ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
};

class ucc_pt_op_memcpy: public ucc_pt_coll {
    ucc_memory_type_t mem_type;
    ucc_datatype_t    data_type;
    int               num_bufs;
public:
    ucc_pt_op_memcpy(ucc_datatype_t dt, ucc_memory_type mt, int nbufs,
                     ucc_pt_comm *communicator,
                     ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
};

class ucc_pt_op_reduce: public ucc_pt_coll {
    ucc_memory_type_t  mem_type;
    ucc_datatype_t     data_type;
    ucc_reduction_op_t reduce_op;
    int                num_bufs;
public:
    ucc_pt_op_reduce(ucc_datatype_t dt, ucc_memory_type mt,
                     ucc_reduction_op_t op, int nbufs,
                     ucc_pt_comm *communicator,
                     ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
};

class ucc_pt_op_reduce_strided: public ucc_pt_coll {
    ucc_memory_type_t  mem_type;
    ucc_datatype_t     data_type;
    ucc_reduction_op_t reduce_op;
    int                num_bufs;
public:
    ucc_pt_op_reduce_strided(ucc_datatype_t dt, ucc_memory_type mt,
                             ucc_reduction_op_t op, int nbufs,
                             ucc_pt_comm *communicator,
                             ucc_pt_generator_base *generator);
    ucc_status_t init_args(ucc_pt_test_args_t &args) override;
    void free_args(ucc_pt_test_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_pt_test_args_t args) override;
};

#endif
