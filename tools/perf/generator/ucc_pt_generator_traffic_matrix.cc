#include "ucc_pt_generator.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <vector>

template <typename T>
void random_choice(
    const std::vector<T> &data, size_t size, std::vector<T> &result_vec,
    std::mt19937_64 &rng, const std::vector<double> &weights = {})
{
    if (data.empty()) {
        throw std::runtime_error("Cannot pick from an empty vector.");
    }

    result_vec.clear();
    result_vec.reserve(size);

    std::vector<double> final_weights;
    size_t              N = data.size();

    if (!weights.empty()) {
        if (weights.size() != N) {
            throw std::runtime_error(
                "Weights vector size must match data vector size.");
        }
        final_weights = weights;
    } else {
        final_weights.assign(N, 1.0);
    }

    std::discrete_distribution<int> distribution(
        final_weights.begin(), final_weights.end());

    for (size_t i = 0; i < size; ++i) {
        int index = distribution(rng);
        result_vec.push_back(data.at(index));
    }

    return;
}

std::vector<std::vector<int>> create_a2aV_traffic_matrix(
    int num_ranks, int token_size_KB_mean, int tgt_group_size_mean,
    int num_tokens, size_t dt_size, std::mt19937_64 &rng, bool add_bias = false,
    double bias_factor = 2, int num_hl_ranks = 2)
{
    // Create a random a2aV traffic matrix where each rank sends token_size_KB_mean messages
    // to a random group of tgt_group_size_mean other ranks.
    // If add_bias is true, the bias_factor is used to increase the probability of sending messages to the higher level
    // ranks. If num_hl_ranks is greater than 0, the num_hl_ranks highest level ranks will be used to send messages to
    // the lower level ranks. The traffic matrix is returned as a matrix of size num_ranks x num_ranks.
    std::vector<int>              bias_indices(num_hl_ranks);
    std::vector<int>              possible_targets(num_ranks);
    std::vector<std::vector<int>> traffic_matrix(
        num_ranks,
        std::vector<int>(num_ranks, 0)); // matrix of size num_ranks x num_ranks
    for (int i = 0; i < num_hl_ranks; i++) {
        bias_indices[i] = std::uniform_int_distribution<int>(
            0, num_ranks - 1)(rng);
    }
    for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
        // Choose random target ranks, excluding self
        possible_targets.clear();
        for (int i = 0; i < num_ranks; i++) {
            if (i != src_rank) {
                possible_targets.push_back(i);
            }
        }
        for (int token = 0; token < num_tokens; token++) {
            std::vector<int> target_ranks(tgt_group_size_mean);
            if (add_bias) {
                // Create biased probabilities for target selection
                std::vector<double> probabilities(
                    possible_targets.size(), 1.0 / possible_targets.size());
                for (int i = 0; i < num_hl_ranks; i++) {
                    int  bias_rank = bias_indices[i];
                    auto it        = std::find(
                        possible_targets.begin(),
                        possible_targets.end(),
                        bias_rank);
                    if (it != possible_targets.end()) {
                        int bias_index = std::distance(
                            possible_targets.begin(), it);
                        probabilities[bias_index] *= bias_factor;
                    }
                }
                double sum = std::accumulate(
                    probabilities.begin(), probabilities.end(), 0.0);
                for (int i = 0; i < probabilities.size(); i++) {
                    probabilities[i] = probabilities[i] / sum;
                }

                random_choice(
                    possible_targets,
                    tgt_group_size_mean,
                    target_ranks,
                    rng,
                    probabilities);
            } else {
                random_choice(
                    possible_targets, tgt_group_size_mean, target_ranks, rng);
            }
            for (int i = 0; i < target_ranks.size(); i++) {
                int target_rank = target_ranks[i];
                traffic_matrix
                    [src_rank]
                    [target_rank] += (token_size_KB_mean * (1000 / dt_size));
            }
        }
    }
    return traffic_matrix;
}

std::vector<std::vector<int>> create_random_tgt_group_a2aV_traffic_matrix(
    int num_ranks, int token_size_KB_mean, int tgt_group_size_mean,
    int tgt_group_size_std, int num_tokens, size_t dt_size,
    std::mt19937_64 &rng)
{
    // Create a random a2aV traffic matrix where each rank sends token_size_KB_mean messages
    // to a random group of ranks with random size.
    // The traffic matrix is returned as a matrix of size num_ranks x num_ranks.

    std::vector<std::vector<int>> traffic_matrix(
        num_ranks, std::vector<int>(num_ranks, 0));
    for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
        std::vector<int> possible_targets;
        possible_targets.reserve(num_ranks - 1);
        for (int i = 0; i < num_ranks; i++) {
            if (i != src_rank) {
                possible_targets.push_back(i);
            }
        }
        for (int token = 0; token < num_tokens; token++) {
            std::normal_distribution<double> distribution(
                tgt_group_size_mean, tgt_group_size_std);
            double normal_sample = distribution(rng);
            int    tgt_group_size;
            tgt_group_size = std::max(1, static_cast<int>(normal_sample));
            tgt_group_size = std::min(
                tgt_group_size, num_ranks - 1); // Cap at available targets

            std::vector<int> target_ranks(tgt_group_size);
            random_choice(possible_targets, tgt_group_size, target_ranks, rng);
            for (int i = 0; i < target_ranks.size(); i++) {
                traffic_matrix
                    [src_rank]
                    [target_ranks
                         [i]] += (token_size_KB_mean * (1000 / dt_size));
            }
        }
    }
    return traffic_matrix;
}

std::vector<std::vector<int>>
create_random_tgt_group_random_msg_size_a2aV_traffic_matrix(
    int num_ranks, int token_size_KB_mean, int token_size_KB_std,
    int tgt_group_size_mean, int tgt_group_size_std, int num_tokens,
    size_t dt_size, std::mt19937_64 &rng)
{
    // Create a random a2aV traffic matrix where each rank sends a random message size
    // to a random group of tgt_group_size_mean other ranks.
    // The traffic matrix is returned as a matrix of size num_ranks x num_ranks.

    std::vector<std::vector<int>> traffic_matrix(
        num_ranks, std::vector<int>(num_ranks, 0));
    std::normal_distribution<double> distribution_tgt_group_size(
        tgt_group_size_mean, tgt_group_size_std);
    std::normal_distribution<double> distribution_token_size(
        token_size_KB_mean, token_size_KB_std);

    for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
        std::vector<int> possible_targets(num_ranks - 1);
        possible_targets.clear();
        for (int i = 0; i < num_ranks; i++) {
            if (i != src_rank) {
                possible_targets.push_back(i);
            }
        }
        for (int token = 0; token < num_tokens; token++) {
            int normal_sample_tgt_group_size = static_cast<int>(
                distribution_tgt_group_size(rng));
            int tgt_group_size;
            tgt_group_size = std::max(1, normal_sample_tgt_group_size);
            tgt_group_size = std::min(
                tgt_group_size, num_ranks - 1); // Cap at available targets

            std::vector<int> target_ranks(tgt_group_size);
            random_choice(possible_targets, tgt_group_size, target_ranks, rng);
            for (int i = 0; i < target_ranks.size(); i++) {
                int normal_sample_token_size = static_cast<int>(
                    distribution_token_size(rng));
                traffic_matrix
                    [src_rank]
                    [target_ranks[i]] += std::max(0, normal_sample_token_size) *
                                         (1000 / dt_size);
            }
        }
    }
    return traffic_matrix;
}

void print_result(
    std::vector<std::vector<int>> traffic_matrix, bool print_full_result)
{
    // Print the traffic matrix
    // The first dimension is the source rank.
    // The second dimension is the target rank.
    for (int src_rank = 0; src_rank < traffic_matrix.size(); src_rank++) {
        for (int tgt_rank = 0; tgt_rank < traffic_matrix[0].size();
             tgt_rank++) {
            std::cout << traffic_matrix[src_rank][tgt_rank] << " ";
        }
        std::cout << std::endl;
    }
}

ucc_pt_generator_traffic_matrix::ucc_pt_generator_traffic_matrix(
    int kind, uint32_t gsize, uint32_t rank, ucc_datatype_t dtype,
    ucc_pt_op_type_t type, size_t nrepeats, int token_size_KB_mean_,
    int num_tokens_, int tgt_group_size_mean_, uint64_t seed)
{

    comm_size           = gsize;
    rank_id             = rank;
    op_type             = type;
    current_pattern     = 0;
    current_rep         = 0;
    nrep                = nrepeats;
    rng_                = std::mt19937_64(seed);

    token_size_KB_mean  = token_size_KB_mean_;
    tgt_group_size_mean = tgt_group_size_mean_;
    num_tokens          = num_tokens_;
    bias_factor         = 2;
    num_hl_ranks        = 2;
    tgt_group_size_std  = 1;
    token_size_KB_std   = 1;
    dt_size             = ucc_dt_size(dtype);

    if (kind == 0) {
        traffic_matrix = create_a2aV_traffic_matrix(
            comm_size,
            token_size_KB_mean,
            tgt_group_size_mean,
            num_tokens,
            dt_size,
            rng_,
            false,
            bias_factor,
            num_hl_ranks);
    } else if (kind == 1) {
        traffic_matrix = create_a2aV_traffic_matrix(
            comm_size,
            token_size_KB_mean,
            tgt_group_size_mean,
            num_tokens,
            dt_size,
            rng_,
            true,
            bias_factor,
            num_hl_ranks);
    } else if (kind == 2) {
        traffic_matrix = create_random_tgt_group_a2aV_traffic_matrix(
            comm_size,
            token_size_KB_mean,
            tgt_group_size_mean,
            tgt_group_size_std,
            num_tokens,
            dt_size,
            rng_);
    } else if (kind == 3) {
        traffic_matrix =
            create_random_tgt_group_random_msg_size_a2aV_traffic_matrix(
                comm_size,
                token_size_KB_mean,
                token_size_KB_std,
                tgt_group_size_mean,
                tgt_group_size_std,
                num_tokens,
                dt_size,
                rng_);
    }

    // print_result(traffic_matrix, false);
    pattern_counts.reserve(traffic_matrix.size());

    std::vector<uint32_t> pattern;
    if (!traffic_matrix.empty()) {
        pattern.reserve(comm_size * comm_size);
    }
    if (traffic_matrix[0].size() != comm_size ||
        traffic_matrix.size() != comm_size) {
        throw std::runtime_error(
            "Matrix size (" + std::to_string(traffic_matrix[0].size()) + "x" +
            std::to_string(traffic_matrix.size()) +
            ") is not equal to comm_size*comm_size (" +
            std::to_string(comm_size * comm_size) +
            "). "
            "Please check the traffic_matrix.");
    }

    for (const auto &row : traffic_matrix) {
        pattern.insert(pattern.end(), row.begin(), row.end());
    }
    pattern_counts.push_back(pattern);

    if (pattern_counts.empty()) {
        throw std::runtime_error(
            "No collective patterns provided in traffic_matrix.");
    }

    // Initialize arrays for counts and displacements
    src_counts.resize(comm_size);
    src_displs.resize(comm_size);
    dst_counts.resize(comm_size);
    dst_displs.resize(comm_size);
}

bool ucc_pt_generator_traffic_matrix::has_next()
{
    return current_rep < nrep;
}

void ucc_pt_generator_traffic_matrix::next()
{
    current_pattern++;
    if (current_pattern >= pattern_counts.size()) {
        current_pattern = 0;
        current_rep++;
    }
    if (has_next()) {
        setup_counts_displs();
    }
}

void ucc_pt_generator_traffic_matrix::reset()
{
    current_pattern = 0;
    current_rep     = 0;
    setup_counts_displs();
}

size_t ucc_pt_generator_traffic_matrix::get_src_count()
{
    size_t total = 0;
    for (int i = 0; i < comm_size; i++) {
        total += src_counts[i];
    }
    return total;
}

size_t ucc_pt_generator_traffic_matrix::get_dst_count()
{
    size_t total = 0;
    for (int i = 0; i < comm_size; i++) {
        total += dst_counts[i];
    }
    return total;
}

ucc_count_t *ucc_pt_generator_traffic_matrix::get_src_counts()
{
    return (ucc_count_t *)src_counts.data();
}

ucc_aint_t *ucc_pt_generator_traffic_matrix::get_src_displs()
{
    return (ucc_aint_t *)src_displs.data();
}

ucc_count_t *ucc_pt_generator_traffic_matrix::get_dst_counts()
{
    return (ucc_count_t *)dst_counts.data();
}

ucc_aint_t *ucc_pt_generator_traffic_matrix::get_dst_displs()
{
    return (ucc_aint_t *)dst_displs.data();
}

void ucc_pt_generator_traffic_matrix::setup_counts_displs()
{
    const auto &counts = pattern_counts[current_pattern];

    if (counts.size() < comm_size * comm_size) {
        throw std::runtime_error(
            "Pattern size (" + std::to_string(counts.size()) +
            ") is less than comm_size*comm_size (" +
            std::to_string(comm_size * comm_size) + ")");
    }

    for (int i = 0; i < comm_size; i++) {
        src_counts[i] = counts[rank_id * comm_size + i];
    }

    size_t displ = 0;
    for (int i = 0; i < comm_size; i++) {
        src_displs[i] = displ;
        displ += src_counts[i];
    }

    for (int i = 0; i < comm_size; i++) {
        dst_counts[i] = counts[i * comm_size + rank_id];
    }

    displ = 0;
    for (int i = 0; i < comm_size; i++) {
        dst_displs[i] = displ;
        displ += dst_counts[i];
    }
}

size_t ucc_pt_generator_traffic_matrix::get_src_count_max()
{
    size_t max_src_count = 0;

    for (size_t i = 0; i < pattern_counts.size(); i++) {
        const auto &counts    = pattern_counts[i];
        size_t      total_src = 0;
        for (int j = 0; j < comm_size; j++) {
            total_src += counts[rank_id * comm_size + j];
        }
        if (total_src > max_src_count) {
            max_src_count = total_src;
        }
    }
    return max_src_count;
}

size_t ucc_pt_generator_traffic_matrix::get_dst_count_max()
{
    size_t max_dst_count = 0;

    for (size_t i = 0; i < pattern_counts.size(); i++) {
        const auto &counts    = pattern_counts[i];
        size_t      total_dst = 0;
        for (int j = 0; j < comm_size; j++) {
            total_dst += counts[j * comm_size + rank_id];
        }
        if (total_dst > max_dst_count) {
            max_dst_count = total_dst;
        }
    }
    return max_dst_count;
}

size_t ucc_pt_generator_traffic_matrix::get_count_max()
{
    const auto &matrix    = pattern_counts[current_pattern];
    size_t      max_count = 0;
    size_t      cur_row_col;

    for (int i = 0; i < comm_size; i++) {
        cur_row_col = 0;
        for (int j = 0; j < comm_size; j++) {
            cur_row_col += matrix[i * comm_size + j];
        }
        if (cur_row_col > max_count) {
            max_count = cur_row_col;
        }
    }

    for (int i = 0; i < comm_size; i++) {
        cur_row_col = 0;
        for (int j = 0; j < comm_size; j++) {
            cur_row_col += matrix[j * comm_size + i];
        }
        if (cur_row_col > max_count) {
            max_count = cur_row_col;
        }
    }
    return max_count;
}
