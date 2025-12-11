/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_GENERATOR_H
#define UCC_PT_GENERATOR_H

#include "ucc_pt_config.h"
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

class ucc_pt_generator_base
{
public:
    virtual bool has_next() = 0;
    virtual void next() = 0;
    virtual size_t get_src_count() = 0; // src buffer count
    virtual size_t get_dst_count() = 0; // dst buffer count
    virtual size_t get_count_max() = 0; // max (src_count, dst_count) across all ranks
    virtual ucc_count_t *get_src_counts() = 0;
    virtual ucc_aint_t *get_src_displs() = 0;
    virtual ucc_count_t *get_dst_counts() = 0;
    virtual ucc_aint_t *get_dst_displs() = 0;
    virtual size_t get_src_count_max() = 0; // max src buffer count across iterations
    virtual size_t get_dst_count_max() = 0; // max dst buffer count across iterations
    virtual void reset() = 0;
    virtual ~ucc_pt_generator_base() {}
};

class ucc_pt_generator_exponential : public ucc_pt_generator_base
{
private:
    uint32_t comm_size;
    size_t min_count;
    size_t max_count;
    size_t mult_factor;
    size_t current_count;
    std::vector<uint32_t> src_counts;
    std::vector<uint32_t> src_displs;
    std::vector<uint32_t> dst_counts;
    std::vector<uint32_t> dst_displs;
    ucc_pt_op_type_t op_type;
public:
    ucc_pt_generator_exponential(size_t min, size_t max, size_t factor,
                                 uint32_t gsize, ucc_pt_op_type_t type);
    bool has_next() override;
    void next() override;
    void reset() override;
    size_t get_src_count() override;
    size_t get_dst_count() override;
    ucc_count_t *get_src_counts() override;
    ucc_aint_t *get_src_displs() override;
    ucc_count_t *get_dst_counts() override;
    ucc_aint_t *get_dst_displs() override;
    size_t get_src_count_max() override;
    size_t get_dst_count_max() override;
    size_t get_count_max() override;
};

class ucc_pt_generator_file : public ucc_pt_generator_base
{
private:
    uint32_t comm_size;
    uint32_t rank_id;
    std::string input_file;
    size_t nrep;
    size_t current_pattern;
    size_t current_rep;
    std::vector<std::vector<uint32_t>> pattern_counts;  // Store counts for each pattern
    std::vector<uint32_t> src_counts;
    std::vector<uint32_t> src_displs;
    std::vector<uint32_t> dst_counts;
    std::vector<uint32_t> dst_displs;
    ucc_pt_op_type_t op_type;
    void* counts_state_ptr = nullptr;
    void setup_counts_displs();
public:
    ucc_pt_generator_file(const std::string &file_path, uint32_t gsize,
                         uint32_t rank, ucc_pt_op_type_t type, size_t nrep);
    bool has_next() override;
    void next() override;
    void reset() override;
    size_t get_src_count() override;
    size_t get_dst_count() override;
    ucc_count_t *get_src_counts() override;
    ucc_aint_t *get_src_displs() override;
    ucc_count_t *get_dst_counts() override;
    ucc_aint_t *get_dst_displs() override;
    size_t get_src_count_max() override;
    size_t get_dst_count_max() override;
    size_t get_count_max() override;
};

class ucc_pt_generator_traffic_matrix : public ucc_pt_generator_base {
  private:
    uint32_t        comm_size;
    uint32_t        rank_id;
    int             kind;
    int             token_size_KB_mean;
    int             tgt_group_size_mean;
    int             num_tokens;
    int             tgt_group_size_std;
    int             token_size_KB_std;
    int             num_hl_ranks;
    double          bias_factor;
    size_t          nrep;
    size_t          current_pattern;
    size_t          current_rep;
    size_t          dt_size;
    std::mt19937_64 rng_;
    std::vector<std::vector<uint32_t>>
        pattern_counts; // Store counts for each pattern. vector of vectors of counts (#vectors = #matrices)
    std::vector<uint32_t>         src_counts;
    std::vector<uint32_t>         src_displs;
    std::vector<uint32_t>         dst_counts;
    std::vector<uint32_t>         dst_displs;
    std::vector<std::vector<int>> traffic_matrix;
    void                         *counts_state_ptr = nullptr;
    void                          setup_counts_displs();
    ucc_pt_op_type_t              op_type;

  public:
    ucc_pt_generator_traffic_matrix(
        int kind, uint32_t gsize, uint32_t rank, ucc_datatype_t dtype,
        ucc_pt_op_type_t type, size_t nrep, int token_size_KB_mean,
        int num_tokens, int tgt_group_size_mean, uint64_t seed);
    bool         has_next() override;
    void         next() override;
    void         reset() override;
    size_t       get_src_count() override;
    size_t       get_dst_count() override;
    ucc_count_t *get_src_counts() override;
    ucc_aint_t  *get_src_displs() override;
    ucc_count_t *get_dst_counts() override;
    ucc_aint_t  *get_dst_displs() override;
    size_t       get_src_count_max() override;
    size_t       get_dst_count_max() override;
    size_t       get_count_max() override;
};

#endif
