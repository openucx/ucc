/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_COLL_H
#define UCC_PT_COLL_H

#include "ucc_pt_comm.h"
#include <ucc/api/ucc.h>
extern "C" {
#include <components/mc/ucc_mc.h>
}

class ucc_pt_coll {
protected:
    bool has_inplace_;
    bool has_reduction_;
    bool has_range_;
    bool has_bw_;
    ucc_pt_comm *comm;
    ucc_coll_args_t coll_args;
    ucc_mc_buffer_header_t *dst_header;
    ucc_mc_buffer_header_t *src_header;
public:
    ucc_pt_coll(ucc_pt_comm *communicator)
    {
        comm = communicator;
    }
    virtual ucc_status_t init_coll_args(size_t count,
                                        ucc_coll_args_t &args) = 0;
    virtual void free_coll_args(ucc_coll_args_t &args) = 0;
    virtual float get_bw(float time_ms, int grsize, ucc_coll_args_t args)
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
                          bool is_inplace, ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_coll_args_t args) override;
};

class ucc_pt_coll_allgatherv: public ucc_pt_coll {
public:
    ucc_pt_coll_allgatherv(ucc_datatype_t dt, ucc_memory_type mt,
                           bool is_inplace, ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
};

class ucc_pt_coll_allreduce: public ucc_pt_coll {
public:
    ucc_pt_coll_allreduce(ucc_datatype_t dt, ucc_memory_type mt,
                          ucc_reduction_op_t op, bool is_inplace,
                          ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_coll_args_t args) override;
};

class ucc_pt_coll_alltoall: public ucc_pt_coll {
public:
    ucc_pt_coll_alltoall(ucc_datatype_t dt, ucc_memory_type mt,
                         bool is_inplace, ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_coll_args_t args) override;
};

class ucc_pt_coll_alltoallv: public ucc_pt_coll {
public:
    ucc_pt_coll_alltoallv(ucc_datatype_t dt, ucc_memory_type mt,
                          bool is_inplace, ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
};

class ucc_pt_coll_barrier: public ucc_pt_coll {
public:
    ucc_pt_coll_barrier(ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
};

class ucc_pt_coll_bcast: public ucc_pt_coll {
public:
    ucc_pt_coll_bcast(ucc_datatype_t dt, ucc_memory_type mt,
                      ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_coll_args_t args) override;
};

class ucc_pt_coll_reduce: public ucc_pt_coll {
public:
    ucc_pt_coll_reduce(ucc_datatype_t dt, ucc_memory_type mt,
                       ucc_reduction_op_t op, bool is_inplace,
                       ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_coll_args_t args) override;
};

class ucc_pt_coll_reduce_scatter: public ucc_pt_coll {
public:
    ucc_pt_coll_reduce_scatter(ucc_datatype_t dt, ucc_memory_type mt,
                               ucc_reduction_op_t op, bool is_inplace,
                               ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_coll_args_t args) override;
};

class ucc_pt_coll_gather: public ucc_pt_coll {
public:
    ucc_pt_coll_gather(ucc_datatype_t dt, ucc_memory_type mt,
                          bool is_inplace, ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_coll_args_t args) override;
};

class ucc_pt_coll_gatherv: public ucc_pt_coll {
public:
    ucc_pt_coll_gatherv(ucc_datatype_t dt, ucc_memory_type mt,
                           bool is_inplace, ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
};

class ucc_pt_coll_scatter: public ucc_pt_coll {
public:
    ucc_pt_coll_scatter(ucc_datatype_t dt, ucc_memory_type mt,
                         bool is_inplace, ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    float get_bw(float time_ms, int grsize, ucc_coll_args_t args) override;
};

class ucc_pt_coll_scatterv: public ucc_pt_coll {
public:
    ucc_pt_coll_scatterv(ucc_datatype_t dt, ucc_memory_type mt,
                           bool is_inplace, ucc_pt_comm *communicator);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
};

#endif
