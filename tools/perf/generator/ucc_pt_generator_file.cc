#include "ucc_pt_generator.h"
#include "utils/ini.h"
#include <string>
#include <vector>
#include <cstring>
#include <sstream>

ucc_pt_generator_file::ucc_pt_generator_file(const std::string &file_path,
                                           uint32_t gsize,
                                           uint32_t rank,
                                           ucc_pt_op_type_t type,
                                           size_t nrepeats)
{
    input_file = file_path;
    comm_size = gsize;
    rank_id = rank;
    op_type = type;
    current_pattern = 0;
    current_rep = 0;
    nrep = nrepeats;

    // Open and validate the INI file
    FILE* file = fopen(file_path.c_str(), "r");
    if (!file) {
        throw std::runtime_error("Failed to open pattern file: " + file_path);
    }

    struct counts_state_t {
        std::string counts_accum;
        bool in_counts = false;
    } counts_state;

    ini_handler handler = [](void* user, const char* section,
                             const char* name, const char* value) -> int {
        auto* self = static_cast<ucc_pt_generator_file*>(user);
        auto* state = static_cast<counts_state_t*>(self->counts_state_ptr);
        if (strcmp(section, "collective") == 0 && strcmp(name, "type") == 0) {
            if (state->in_counts && !state->counts_accum.empty()) {
                std::stringstream ss(state->counts_accum);
                std::string item;
                std::vector<uint32_t> pattern;
                while (std::getline(ss, item, ',')) {
                    item.erase(0, item.find_first_not_of(" \t\n\r"));
                    item.erase(item.find_last_not_of(" \t\n\r") + 1);
                    if (!item.empty()) {
                        pattern.push_back(std::stoull(item));
                    }
                }
                self->pattern_counts.push_back(pattern);
                state->counts_accum.clear();
                state->in_counts = false;
            }
            if (strcmp(value, "ALLTOALLV") != 0) {
                throw std::runtime_error("Unsupported collective type: " + std::string(value) +
                                       ". Only ALLTOALLV is supported.");
            }
        } else if (strcmp(section, "collective") == 0 && strcmp(name, "counts") == 0) {
            std::string line = value;
            line.erase(0, line.find_first_not_of(" \t\n\r"));
            line.erase(line.find_last_not_of(" \t\n\r") + 1);
            size_t open = line.find('{');
            if (open != std::string::npos) {
                state->in_counts = true;
                state->counts_accum.clear();
                line = line.substr(open + 1);
                line.erase(0, line.find_first_not_of(" \t\n\r"));
                line.erase(line.find_last_not_of(" \t\n\r") + 1);
            }
            size_t close = line.find('}');
            if (close != std::string::npos) {
                std::string before_brace = line.substr(0, close);
                before_brace.erase(0, before_brace.find_first_not_of(" \t\n\r"));
                before_brace.erase(before_brace.find_last_not_of(" \t\n\r") + 1);
                if (!before_brace.empty()) {
                    if (!state->counts_accum.empty())
                        state->counts_accum += ",";
                    state->counts_accum += before_brace;
                }
                if (state->in_counts && !state->counts_accum.empty()) {
                    std::stringstream ss(state->counts_accum);
                    std::string item;
                    std::vector<uint32_t> pattern;
                    while (std::getline(ss, item, ',')) {
                        item.erase(0, item.find_first_not_of(" \t\n\r"));
                        item.erase(item.find_last_not_of(" \t\n\r") + 1);
                        if (!item.empty()) {
                            pattern.push_back(std::stoull(item));
                        }
                    }
                    self->pattern_counts.push_back(pattern);
                    state->counts_accum.clear();
                }
                state->in_counts = false;
                return 1;
            }
            if (!line.empty()) {
                if (!state->counts_accum.empty())
                    state->counts_accum += ",";
                state->counts_accum += line;
            }
        }
        return 1;
    };

    this->counts_state_ptr = &counts_state;
    if (ucc_ini_parse_file(file, handler, this) < 0) {
        fclose(file);
        throw std::runtime_error("Failed to parse pattern file: " + file_path);
    }

    this->counts_state_ptr = nullptr;
    fclose(file);

    // After parsing, if there is any accumulated pattern, push it
    if (!counts_state.counts_accum.empty()) {
        std::stringstream ss(counts_state.counts_accum);
        std::string item;
        std::vector<uint32_t> pattern;
        while (std::getline(ss, item, ',')) {
            item.erase(0, item.find_first_not_of(" \t\n\r"));
            item.erase(item.find_last_not_of(" \t\n\r") + 1);
            if (!item.empty()) {
                pattern.push_back(std::stoull(item));
            }
        }
        pattern_counts.push_back(pattern);
    }

    if (pattern_counts.empty()) {
        throw std::runtime_error("No collective patterns found in file: " + file_path);
    }

    for (size_t i = 0; i < pattern_counts.size(); i++) {
        if (pattern_counts[i].size() != comm_size * comm_size) {
            throw std::runtime_error("Pattern size (" + std::to_string(pattern_counts[i].size()) +
                                     ") is not equal to comm_size*comm_size (" +
                                     std::to_string(comm_size * comm_size) + "). "
                                     "Please check the pattern file or the comm_size.");
        }
    }

    // Initialize arrays for counts and displacements
    src_counts.resize(comm_size);
    src_displs.resize(comm_size);
    dst_counts.resize(comm_size);
    dst_displs.resize(comm_size);
}

bool ucc_pt_generator_file::has_next()
{
    return current_rep < nrep;
}

void ucc_pt_generator_file::next()
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

void ucc_pt_generator_file::reset()
{
    current_pattern = 0;
    current_rep = 0;
    setup_counts_displs();
}

size_t ucc_pt_generator_file::get_src_count()
{
    size_t total = 0;
    for (int i = 0; i < comm_size; i++) {
        total += src_counts[i];
    }
    return total;
}

size_t ucc_pt_generator_file::get_dst_count()
{
    size_t total = 0;
    for (int i = 0; i < comm_size; i++) {
        total += dst_counts[i];
    }
    return total;
}

ucc_count_t *ucc_pt_generator_file::get_src_counts()
{
    return (ucc_count_t *)src_counts.data();
}

ucc_aint_t *ucc_pt_generator_file::get_src_displs()
{
    return (ucc_aint_t *)src_displs.data();
}

ucc_count_t *ucc_pt_generator_file::get_dst_counts()
{
    return (ucc_count_t *)dst_counts.data();
}

ucc_aint_t *ucc_pt_generator_file::get_dst_displs()
{
    return (ucc_aint_t *)dst_displs.data();
}

void ucc_pt_generator_file::setup_counts_displs()
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

size_t ucc_pt_generator_file::get_src_count_max()
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

size_t ucc_pt_generator_file::get_dst_count_max()
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

size_t ucc_pt_generator_file::get_count_max()
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