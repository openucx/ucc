/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_COLL_H
#define UCC_PT_COLL_H

#include <ucc/api/ucc.h>

class ucc_pt_coll {
protected:
    ucc_coll_args_t coll_args;
public:
    virtual ucc_status_t get_coll(size_t count, ucc_coll_args_t &args) = 0;
    virtual void free_coll(ucc_coll_args_t &args) = 0;
    virtual double get_bus_bw(double time_us) = 0;
    virtual ~ucc_pt_coll() {};
};

class ucc_pt_coll_allreduce: public ucc_pt_coll {
public:
    ucc_pt_coll_allreduce(ucc_datatype_t dt, ucc_memory_type mt,
                          ucc_reduction_op_t op, bool is_inplace);
    ucc_status_t get_coll(size_t count, ucc_coll_args_t &args) override;
    void free_coll(ucc_coll_args_t &args) override;
    double get_bus_bw(double time_us) override;
};

#endif
