/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_COLL_H
#define UCC_PT_COLL_H

#include <ucc/api/ucc.h>
extern "C" {
#include <core/ucc_mc.h>
}

class ucc_pt_coll {
protected:
    bool has_inplace_;
    bool has_reduction_;
    bool has_range_;
    ucc_coll_args_t coll_args;
    ucc_mc_buffer_header_t *dst_header;
    ucc_mc_buffer_header_t *src_header;
public:
    virtual ucc_status_t init_coll_args(size_t count,
                                        ucc_coll_args_t &args) = 0;
    virtual void free_coll_args(ucc_coll_args_t &args) = 0;
    virtual double get_bus_bw(double time_us) = 0;
    bool has_reduction();
    bool has_inplace();
    bool has_range();
    virtual ~ucc_pt_coll() {};
};

class ucc_pt_coll_allgather: public ucc_pt_coll {
protected:
    int comm_size;
public:
    ucc_pt_coll_allgather(int size, ucc_datatype_t dt, ucc_memory_type mt,
                          bool is_inplace);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    double get_bus_bw(double time_us) override;
};

class ucc_pt_coll_allgatherv: public ucc_pt_coll {
protected:
    int comm_size;
public:
    ucc_pt_coll_allgatherv(int size, ucc_datatype_t dt, ucc_memory_type mt,
                           bool is_inplace);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    double get_bus_bw(double time_us) override;
};

class ucc_pt_coll_allreduce: public ucc_pt_coll {
public:
    ucc_pt_coll_allreduce(ucc_datatype_t dt, ucc_memory_type mt,
                          ucc_reduction_op_t op, bool is_inplace);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    double get_bus_bw(double time_us) override;
};

class ucc_pt_coll_alltoall: public ucc_pt_coll {
protected:
    int comm_size;
public:
    ucc_pt_coll_alltoall(int size, ucc_datatype_t dt, ucc_memory_type mt,
                         bool is_inplace);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    double get_bus_bw(double time_us) override;
};

class ucc_pt_coll_alltoallv: public ucc_pt_coll {
protected:
    int comm_size;
public:
    ucc_pt_coll_alltoallv(int size, ucc_datatype_t dt, ucc_memory_type mt,
                          bool is_inplace);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    double get_bus_bw(double time_us) override;
};

class ucc_pt_coll_barrier: public ucc_pt_coll {
public:
    ucc_pt_coll_barrier();
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    double get_bus_bw(double time_us) override;
};

class ucc_pt_coll_bcast: public ucc_pt_coll {
public:
    ucc_pt_coll_bcast(ucc_datatype_t dt, ucc_memory_type mt);
    ucc_status_t init_coll_args(size_t count, ucc_coll_args_t &args) override;
    void free_coll_args(ucc_coll_args_t &args) override;
    double get_bus_bw(double time_us) override;
};

#endif
