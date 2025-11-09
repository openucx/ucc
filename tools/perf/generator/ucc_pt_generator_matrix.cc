#include "ucc_pt_generator.h"
#include "utils/ini.h"
#include <string>
#include <vector>
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <vector>
#include <iostream>
#include <algorithm> 
#include <random>
#include <numeric>
#include <stdexcept>

std::default_random_engine generator;

/**
 * @brief Randomly selects 'size' elements from a vector (with replacement), 
 * using weights if provided, or uniform probability otherwise.
 *
 * @tparam T The type of elements in the vector.
 * @param data The vector of elements to choose from.
 * @param size The number of elements to pick.
 * @param weights Optional vector of relative weights. 
 * @return A vector of m randomly selected elements of type T.
 */
template <typename T>
void random_choice(const std::vector<T>& data,size_t size, std::vector<T>& result_vec, const std::vector<double>& weights = {}) {
    if (data.empty()) {
        throw std::runtime_error("Cannot pick from an empty vector.");
    }

    result_vec.clear();
    result_vec.reserve(size);

    std::vector<double> final_weights;
    size_t N = data.size();

    if (!weights.empty()) {
        if (weights.size() != N) {
            throw std::runtime_error("Weights vector size must match data vector size.");
        }
        final_weights = weights;
    } else {
        final_weights.assign(N, 1.0);
    }

    std::discrete_distribution<int> distribution(final_weights.begin(), final_weights.end());

    for (size_t i = 0; i < size; ++i) {
        int index = distribution(generator);
        result_vec.push_back(data.at(index));
    }
    
    return;
}


std::vector<std::vector<double>> create_a2aV_traffic_matrix(int num_ranks, int token_size_KB_mean, int tgt_group_size_mean, int num_tokens, bool add_bias=false, double bias_factor=2, int num_hl_ranks=2) {
    // Create a random a2aV traffic matrix where each rank sends token_size_KB_mean messages
    // to a random group of tgt_group_size_mean other ranks.
    // If add_bias is true, the bias_factor is used to increase the probability of sending messages to the higher level ranks.
    // If num_hl_ranks is greater than 0, the num_hl_ranks highest level ranks will be used to send messages to the lower level ranks.
    // The traffic matrix is returned as a matrix of size num_ranks x num_ranks.

    std::vector<int> bias_indices(num_hl_ranks);
    std::vector<int> possible_targets(num_ranks);
    std::iota(possible_targets.begin(), possible_targets.end(), 0);
    std::vector<std::vector<double>> traffic_matrix(num_ranks, std::vector<double>(num_ranks, 0)); //matrix of size num_ranks x num_ranks
    
    random_choice(possible_targets, num_hl_ranks, bias_indices);
    for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
        // Choose random target ranks, excluding self
        possible_targets.clear();
        for (int i = 0; i < num_ranks; i++) {
            if (i != src_rank) {
                possible_targets.push_back(i);
            }
            for (int token = 0; token < num_tokens; token++) {
                std::vector<int> target_ranks(tgt_group_size_mean);
                if (add_bias) {
                    // Create biased probabilities for target selection
                    std::vector<double> probabilities(possible_targets.size(), 1.0 / possible_targets.size());
                    for (int i = 0; i < num_hl_ranks; i++) {
                        probabilities[bias_indices[i]] *= bias_factor;
                    }
                    double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
                    for (int i = 0; i < probabilities.size(); i++) {
                        probabilities[i] = probabilities[i] / sum;
                    }
                    
                        random_choice(possible_targets, tgt_group_size_mean, target_ranks, probabilities);
                    } else {
                        random_choice(possible_targets, tgt_group_size_mean, target_ranks);
                    }
                    for (int i = 0; i < target_ranks.size(); i++) {
                        traffic_matrix[src_rank][target_ranks[i]] += token_size_KB_mean;
                    }
                }
            }
        }
    return traffic_matrix;
}

std::vector<std::vector<double>> create_random_tgt_group_a2aV_traffic_matrix(int num_ranks, int token_size_KB_mean, int tgt_group_size_mean, int tgt_group_size_std, int num_tokens) {
    // Create a random a2aV traffic matrix where each rank sends token_size_KB_mean messages
    // to a random group of ranks with random size.
    // The traffic matrix is returned as a matrix of size num_ranks x num_ranks.

    std::vector<std::vector<double>> traffic_matrix(num_ranks, std::vector<double>(num_ranks, 0));
        for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
            std::vector<int> possible_targets(num_ranks-1);
            possible_targets.clear();
            for (int i = 0; i < num_ranks; i++) {
                if (i != src_rank) {
                    possible_targets.push_back(i);
                }
            }
            for (int token = 0; token < num_tokens; token++) {
                std::normal_distribution<double> distribution(tgt_group_size_mean, tgt_group_size_std);
                double normal_sample = distribution(generator);
                double tgt_group_size;
                tgt_group_size = std::max(1.0, normal_sample);
                tgt_group_size = std::min(tgt_group_size, num_ranks - 1.0);  // Cap at available targets
                
                std::vector<int> target_ranks(tgt_group_size);
                random_choice(possible_targets, tgt_group_size, target_ranks);
                for (int i = 0; i < target_ranks.size(); i++) {
                    traffic_matrix[src_rank][target_ranks[i]] += token_size_KB_mean;
                }
            }
        }
    return traffic_matrix;
}

std::vector<std::vector<double>> create_random_tgt_group_random_msg_size_a2aV_traffic_matrix(int num_ranks, int token_size_KB_mean, int token_size_KB_std, int tgt_group_size_mean, int tgt_group_size_std, int num_tokens) {
    // Create a random a2aV traffic matrix where each rank sends a random message size
    // to a random group of tgt_group_size_mean other ranks.
    // The traffic matrix is returned as a matrix of size num_ranks x num_ranks.

    std::vector<std::vector<double>> traffic_matrix(num_ranks, std::vector<double>(num_ranks, 0));
    std::normal_distribution<double> distribution_tgt_group_size(tgt_group_size_mean, tgt_group_size_std);
    std::normal_distribution<double> distribution_token_size(token_size_KB_mean, token_size_KB_std);


    for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
        std::vector<int> possible_targets(num_ranks-1);
        possible_targets.clear();
        for (int i = 0; i < num_ranks; i++) {
            if (i != src_rank) {
                possible_targets.push_back(i);
            }
        }
        for (int token = 0; token < num_tokens; token++) {
            double normal_sample_tgt_group_size = distribution_tgt_group_size(generator);
            double tgt_group_size;
            tgt_group_size = std::max(1.0, normal_sample_tgt_group_size);
            tgt_group_size = std::min(tgt_group_size, num_ranks - 1.0);  // Cap at available targets
            
            std::vector<int> target_ranks(tgt_group_size);
            random_choice(possible_targets, tgt_group_size, target_ranks);
            for (int i = 0; i < target_ranks.size(); i++) {
                double normal_sample_token_size = distribution_token_size(generator);
                traffic_matrix[src_rank][target_ranks[i]] += std::max(0.0, normal_sample_token_size);
            }
        }
    }
    return traffic_matrix;
}

void print_result(std::vector<std::vector<double>> traffic_matrix, bool print_full_result) {
    // Print the traffic matrix
    // The first dimension is the source rank.
    // The second dimension is the target rank.
    for (int src_rank = 0; src_rank < traffic_matrix.size(); src_rank++) {
        for (int tgt_rank = 0; tgt_rank < traffic_matrix[0].size(); tgt_rank++) {
            std::cout << traffic_matrix[src_rank][tgt_rank] << " ";
        }
        std::cout << std::endl;
    }
}


// use ucc_str_to_memunits - https://github.com/openucx/ucx/blob/78115a7edfe6af0a990873afbc46750b738cdece/src/ucs/sys/string.c#L169

ucc_pt_generator_matrix::ucc_pt_generator_matrix(
    int kind,
    uint32_t gsize,
    uint32_t rank,
    ucc_pt_op_type_t type,
    size_t nrepeats,
    int token_size_KB_mean,
    int num_tokens,
    int tgt_group_size_mean)
{

    comm_size = gsize;
    rank_id = rank;
    op_type = type;
    current_pattern = 0;
    current_rep = 0;
    nrep = nrepeats;
    
    token_size_KB_mean = token_size_KB_mean;
    tgt_group_size_mean = tgt_group_size_mean;
    num_tokens = num_tokens;
    bias_factor = 2;
    num_hl_ranks = 2;
    tgt_group_size_std = tgt_group_size_mean; 
    token_size_KB_std = token_size_KB_mean; 

    if (kind == 0) {
        traffic_matrix = create_a2aV_traffic_matrix(comm_size, token_size_KB_mean, tgt_group_size_mean, num_tokens, bias_factor, num_hl_ranks);
    } else if (kind == 1) {
        traffic_matrix = create_random_tgt_group_a2aV_traffic_matrix(comm_size, token_size_KB_mean, tgt_group_size_mean, tgt_group_size_std, num_tokens);
    } else if (kind == 2) {
        traffic_matrix = create_random_tgt_group_random_msg_size_a2aV_traffic_matrix(comm_size, token_size_KB_mean, token_size_KB_std, tgt_group_size_mean, tgt_group_size_std, num_tokens);
    }

    print_result(traffic_matrix, false);
    pattern_counts.reserve(traffic_matrix.size());

    // for multiple matrices option
    // for (const auto& matrix : traffic_matrix) {
    //     // 'matrix' is a std::vector<std::vector<uint32_t>> (a 2D matrix)
    //     std::vector<uint32_t> pattern;
    //     // Reserve memory based on expected size for optimization
    //     if (!matrix.empty()) {
    //         pattern.reserve(comm_size * comm_size);
    //     }
    //     if (matrix[0].size() != comm_size || matrix.size() != comm_size) {
    //         throw std::runtime_error("Matrix size (" + std::to_string(matrix[0].size()) +
    //                                  "x" + std::to_string(matrix.size()) +
    //                                  ") is not equal to comm_size*comm_size (" +
    //                                  std::to_string(comm_size * comm_size) + "). "
    //                                  "Please check the traffic_matrix.");
    //     }

    //     // Flatten the 2D matrix (row by row) into a 1D pattern vector
    //     for (const auto& row : matrix) {
    //         // Append all elements from the current row vector to the pattern vector
    //         pattern.insert(pattern.end(), row.begin(), row.end());
    //     }
        
    //     // Store the final flattened pattern
    //     pattern_counts.push_back(pattern);
    // }

 
    std::vector<uint32_t> pattern;
    // Reserve memory based on expected size for optimization
    if (!traffic_matrix.empty()) {
        pattern.reserve(comm_size * comm_size);
    }
    if (traffic_matrix[0].size() != comm_size || traffic_matrix.size() != comm_size) {
        throw std::runtime_error("Matrix size (" + std::to_string(traffic_matrix[0].size()) +
                                    "x" + std::to_string(traffic_matrix.size()) +
                                    ") is not equal to comm_size*comm_size (" +
                                    std::to_string(comm_size * comm_size) + "). "
                                    "Please check the traffic_matrix.");
    }

    // Flatten the 2D matrix (row by row) into a 1D pattern vector
    for (const auto& row : traffic_matrix) {
        // Append all elements from the current row vector to the pattern vector
        pattern.insert(pattern.end(), row.begin(), row.end());
    }
    
    // Store the final flattened pattern
    pattern_counts.push_back(pattern);





    // Ensure patterns were provided
    if (pattern_counts.empty()) {
        throw std::runtime_error("No collective patterns provided in traffic_matrix.");
    }

    // Initialize arrays for counts and displacements
    src_counts.resize(comm_size);
    src_displs.resize(comm_size);
    dst_counts.resize(comm_size);
    dst_displs.resize(comm_size);
}


bool ucc_pt_generator_matrix::has_next()
{
    return current_rep < nrep;
}

void ucc_pt_generator_matrix::next()
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

void ucc_pt_generator_matrix::reset()
{
    current_pattern = 0;
    current_rep = 0;
    setup_counts_displs();
}

size_t ucc_pt_generator_matrix::get_src_count()
{
    size_t total = 0;
    for (int i = 0; i < comm_size; i++) {
        total += src_counts[i];
    }
    return total;
}

size_t ucc_pt_generator_matrix::get_dst_count()
{
    size_t total = 0;
    for (int i = 0; i < comm_size; i++) {
        total += dst_counts[i];
    }
    return total;
}

ucc_count_t *ucc_pt_generator_matrix::get_src_counts()
{
    return (ucc_count_t *)src_counts.data();
}

ucc_aint_t *ucc_pt_generator_matrix::get_src_displs()
{
    return (ucc_aint_t *)src_displs.data();
}

ucc_count_t *ucc_pt_generator_matrix::get_dst_counts()
{
    return (ucc_count_t *)dst_counts.data();
}

ucc_aint_t *ucc_pt_generator_matrix::get_dst_displs()
{
    return (ucc_aint_t *)dst_displs.data();
}


void ucc_pt_generator_matrix::setup_counts_displs()
{
    const auto& counts = pattern_counts[current_pattern];

    if (counts.size() < comm_size * comm_size) {
        throw std::runtime_error("Pattern size (" + std::to_string(counts.size()) +
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


size_t ucc_pt_generator_matrix::get_src_count_max()
{
    size_t max_src_count = 0;

    for (size_t i = 0; i < pattern_counts.size(); i++) {
        const auto& counts = pattern_counts[i];
        size_t total_src = 0;
        for (int j = 0; j < comm_size; j++) {
            total_src += counts[rank_id * comm_size + j];
        }
        if (total_src > max_src_count) {
            max_src_count = total_src;
        }
    }
    return max_src_count;
}

size_t ucc_pt_generator_matrix::get_dst_count_max()
{
    size_t max_dst_count = 0;

    for (size_t i = 0; i < pattern_counts.size(); i++) {
        const auto& counts = pattern_counts[i];
        size_t total_dst = 0;
        for (int j = 0; j < comm_size; j++) {
            total_dst += counts[j * comm_size + rank_id];
        }
        if (total_dst > max_dst_count) {
            max_dst_count = total_dst;
        }
    }
    return max_dst_count;
}

size_t ucc_pt_generator_matrix::get_count_max()
{
    auto matrix = pattern_counts[current_pattern];
    size_t max_count = 0;
    size_t cur_row_col;

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

