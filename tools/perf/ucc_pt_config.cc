/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_config.h"
BEGIN_C_DECLS
#include "utils/ucc_string.h"
#include <getopt.h>
END_C_DECLS
#include <algorithm>

ucc_pt_config::ucc_pt_config() {
    bootstrap.bootstrap  = UCC_PT_BOOTSTRAP_MPI;
    bench.op_type        = UCC_PT_OP_TYPE_ALLREDUCE;
    bench.min_count      = 128;
    bench.max_count      = 128;
    bench.dt             = UCC_DT_FLOAT32;
    bench.mt             = UCC_MEMORY_TYPE_HOST;
    bench.op             = UCC_OP_SUM;
    bench.inplace        = false;
    bench.persistent     = false;
    bench.map_type       = UCC_PT_MAP_TYPE_NONE;
    bench.triggered      = false;
    bench.n_iter_small   = 1000;
    bench.n_warmup_small = 100;
    bench.n_iter_large   = 200;
    bench.n_warmup_large = 20;
    bench.large_thresh   = 64 * 1024;
    bench.full_print     = false;
    bench.n_bufs         = UCC_PT_DEFAULT_N_BUFS;
    bench.root           = 0;
    bench.root_shift     = 0;
    bench.mult_factor    = 2;
    bench.seed           = 0;
    comm.mt              = bench.mt;
}

const std::map<std::string, ucc_reduction_op_t> ucc_pt_reduction_op_map = {
    {"sum", UCC_OP_SUM}, {"prod", UCC_OP_PROD}, {"min", UCC_OP_MIN},
    {"max", UCC_OP_MAX}, {"avg", UCC_OP_AVG},
};

const std::map<std::string, ucc_pt_op_type_t> ucc_pt_op_map = {
    {"allgather", UCC_PT_OP_TYPE_ALLGATHER},
    {"allgatherv", UCC_PT_OP_TYPE_ALLGATHERV},
    {"allreduce", UCC_PT_OP_TYPE_ALLREDUCE},
    {"alltoall", UCC_PT_OP_TYPE_ALLTOALL},
    {"alltoallv", UCC_PT_OP_TYPE_ALLTOALLV},
    {"barrier", UCC_PT_OP_TYPE_BARRIER},
    {"bcast", UCC_PT_OP_TYPE_BCAST},
    {"gather", UCC_PT_OP_TYPE_GATHER},
    {"gatherv", UCC_PT_OP_TYPE_GATHERV},
    {"reduce", UCC_PT_OP_TYPE_REDUCE},
    {"reduce_scatter", UCC_PT_OP_TYPE_REDUCE_SCATTER},
    {"reduce_scatterv", UCC_PT_OP_TYPE_REDUCE_SCATTERV},
    {"scatter", UCC_PT_OP_TYPE_SCATTER},
    {"scatterv", UCC_PT_OP_TYPE_SCATTERV},
    {"memcpy", UCC_PT_OP_TYPE_MEMCPY},
    {"reducedt", UCC_PT_OP_TYPE_REDUCEDT},
    {"reducedt_strided", UCC_PT_OP_TYPE_REDUCEDT_STRIDED},
};

const std::map<std::string, ucc_memory_type_t> ucc_pt_memtype_map = {
    {"host", UCC_MEMORY_TYPE_HOST},
    {"cuda", UCC_MEMORY_TYPE_CUDA},
    {"rocm", UCC_MEMORY_TYPE_ROCM},
    {"cuda-mng", UCC_MEMORY_TYPE_CUDA_MANAGED},
};

const std::map<std::string, ucc_datatype_t> ucc_pt_datatype_map = {
    {"int8", UCC_DT_INT8},
    {"uint8", UCC_DT_UINT8},
    {"int16", UCC_DT_INT16},
    {"uint16", UCC_DT_UINT16},
    {"float16", UCC_DT_FLOAT16},
    {"bfloat16", UCC_DT_BFLOAT16},
    {"int32", UCC_DT_INT32},
    {"uint32", UCC_DT_UINT32},
    {"float32", UCC_DT_FLOAT32},
    {"float32_complex", UCC_DT_FLOAT32_COMPLEX},
    {"int64", UCC_DT_INT64},
    {"uint64", UCC_DT_UINT64},
    {"float64", UCC_DT_FLOAT64},
    {"float64_complex", UCC_DT_FLOAT64_COMPLEX},
    {"int128", UCC_DT_INT128},
    {"uint128", UCC_DT_UINT128},
    {"float128", UCC_DT_FLOAT64},
    {"float128_complex", UCC_DT_FLOAT128_COMPLEX},
};

const std::map<std::string, ucc_pt_map_type_t> ucc_pt_map_type_map = {
    {"none", UCC_PT_MAP_TYPE_NONE},
    {"local", UCC_PT_MAP_TYPE_LOCAL},
    {"global", UCC_PT_MAP_TYPE_GLOBAL},
};

ucc_status_t ucc_pt_config::process_args(int argc, char *argv[])
{
    int c;
    ucc_status_t st;
    int option_index = 0;
    static struct option long_options[] = {
        {"gen", required_argument, 0, 0},
        {"seed", required_argument, 0, 0},
        {0, 0, 0, 0}};

    // Reset getopt state
    optind = 1;

    while (1) {
        c = getopt_long(argc, argv, "c:b:e:d:f:m:n:w:o:N:r:S:M:iphFT", long_options, &option_index);
        if (c == -1)
            break;
        if (c == 0) { // long option
            if (strcmp(long_options[option_index].name, "gen") == 0) {
                std::string gen_arg(optarg);
                if (gen_arg.rfind("exp:", 0) == 0) {
                    bench.gen.type = UCC_PT_GEN_TYPE_EXP;
                    auto min_pos = gen_arg.find("min=", 4);
                    if (min_pos == std::string::npos) {
                        std::cerr << "Invalid format for --gen exp:min=N[@max=M]" << std::endl;
                        return UCC_ERR_INVALID_PARAM;
                    }
                    auto at_pos = gen_arg.find("@", min_pos);
                    if (at_pos != std::string::npos) {
                        auto max_pos = gen_arg.find("max=", at_pos);
                        if (max_pos == std::string::npos) {
                            std::cerr << "Invalid format for --gen exp:min=N@max=M" << std::endl;
                            return UCC_ERR_INVALID_PARAM;
                        }
                        try {
                            ucc_status_t st = ucc_str_to_memunits(
                                gen_arg
                                    .substr(min_pos + 4, at_pos - (min_pos + 4))
                                    .c_str(),
                                (void *)&bench.gen.exp.min);
                            if (st != UCC_OK) {
                                std::cerr << "Failed to parse min value" << std::endl;
                                return st;
                            }
                            st = ucc_str_to_memunits(
                                gen_arg.substr(max_pos + 4).c_str(),
                                (void *)&bench.gen.exp.max);
                            if (st != UCC_OK) {
                                std::cerr << "Failed to parse max value" << std::endl;
                                return st;
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "Invalid values in --gen exp:min=N@max=M" << std::endl;
                            return UCC_ERR_INVALID_PARAM;
                        }
                    } else {
                        try {
                            ucc_status_t st = ucc_str_to_memunits(
                                gen_arg.substr(min_pos + 4).c_str(),
                                (void *)&bench.gen.exp.min);
                            if (st != UCC_OK) {
                                std::cerr << "Failed to parse min value" << std::endl;
                                return st;
                            }
                            bench.gen.exp.max = bench.gen.exp.min;
                        } catch (const std::exception& e) {
                            std::cerr << "Invalid value in --gen exp:min=N" << std::endl;
                            return UCC_ERR_INVALID_PARAM;
                        }
                    }
                    bench.min_count = bench.gen.exp.min;
                    bench.max_count = bench.gen.exp.max;
                } else if (gen_arg.rfind("file:", 0) == 0) {
                    bench.gen.type = UCC_PT_GEN_TYPE_FILE;
                    auto name_pos = gen_arg.find("name=", 5);
                    if (name_pos == std::string::npos) {
                        std::cerr << "Invalid format for --gen file:name=filename[@nrep=N]" << std::endl;
                        return UCC_ERR_INVALID_PARAM;
                    }
                    auto at_pos = gen_arg.find("@", name_pos);
                    if (at_pos != std::string::npos) {
                        bench.gen.file_name = gen_arg.substr(name_pos + 5, at_pos - (name_pos + 5));
                        auto nrep_str = gen_arg.substr(at_pos + 1);
                        if (nrep_str.rfind("nrep=", 0) == 0) {
                            try {
                                bench.gen.nrep = std::stoull(nrep_str.substr(5));
                            } catch (const std::exception& e) {
                                std::cerr << "Invalid nrep value in --gen file:name=filename@nrep=N" << std::endl;
                                return UCC_ERR_INVALID_PARAM;
                            }
                        } else {
                            std::cerr << "Invalid format for --gen file:name=filename@nrep=N" << std::endl;
                            return UCC_ERR_INVALID_PARAM;
                        }
                    } else {
                        bench.gen.file_name = gen_arg.substr(name_pos + 5);
                        bench.gen.nrep = 1; // Default value if nrep is not specified
                    }
                } else if (gen_arg.rfind("matrix:", 0) == 0) {
                    bench.gen.type        = UCC_PT_GEN_TYPE_TRAFFIC_MATRIX;
                    /* Defaults (all parameters optional) */
                    bench.gen.nrep        = 1;
                    bench.gen.matrix.kind = 0;
                    bench.gen.matrix.token_size_KB_mean = 16;
                    bench.gen.matrix
                        .token_size_KB_std = bench.gen.matrix
                                                 .token_size_KB_mean;
                    bench.gen.matrix.num_tokens          = 2048;
                    bench.gen.matrix.tgt_group_size_mean = 8;

                    auto find_param = [&](const std::string &key,
                                          std::string       &out) -> bool {
                        auto pos = gen_arg.find(key + "=");
                        if (pos == std::string::npos) {
                            return false;
                        }
                        pos += key.size() + 1;
                        auto end = gen_arg.find("@", pos);
                        if (end == std::string::npos) {
                            end = gen_arg.size();
                        }
                        out = gen_arg.substr(pos, end - pos);
                        return true;
                    };

                    std::string kind_str;
                    if (find_param("kind", kind_str)) {
                        std::string ks = kind_str;
                        std::transform(
                            ks.begin(), ks.end(), ks.begin(), ::tolower);
                        if (ks == "0" ||
                            ks.find("normal") != std::string::npos) {
                            bench.gen.matrix.kind = 0;
                        } else if (
                            ks == "1" ||
                            ks.find("biased") != std::string::npos) {
                            bench.gen.matrix.kind = 1;
                        } else if (
                            ks == "2" ||
                            ks.find("random_tgt_group") != std::string::npos) {
                            bench.gen.matrix.kind = 2;
                        } else if (
                            ks == "3" ||
                            ks.find("random_tgt_group_random_msg_size") !=
                                std::string::npos) {
                            bench.gen.matrix.kind = 3;
                        } else {
                            std::cerr << "Invalid kind value in --gen matrix: "
                                         "accepts 0,1,2,3 or names {normal, "
                                         "biased, random_tgt_group, "
                                         "random_tgt_group_random_msg_size}"
                                      << std::endl;
                            return UCC_ERR_INVALID_PARAM;
                        }
                    }

                    std::string nrep_str;
                    if (find_param("nrep", nrep_str)) {
                        try {
                            bench.gen.nrep = std::stoull(nrep_str);
                        } catch (const std::exception &) {
                            std::cerr << "Invalid nrep value in --gen matrix"
                                      << std::endl;
                            return UCC_ERR_INVALID_PARAM;
                        }
                    }

                    std::string token_size_str;
                    if (find_param("token_size", token_size_str)) {
                        try {
                            bench.gen.matrix.token_size_KB_mean = std::stoull(
                                token_size_str);
                        } catch (const std::exception &) {
                            std::cerr
                                << "Invalid token_size value in --gen matrix"
                                << std::endl;
                            return UCC_ERR_INVALID_PARAM;
                        }
                        bench.gen.matrix
                            .token_size_KB_std = bench.gen.matrix
                                                     .token_size_KB_mean;
                    }

                    std::string num_tokens_str;
                    if (find_param("num_tokens", num_tokens_str)) {
                        try {
                            bench.gen.matrix.num_tokens = std::stoull(
                                num_tokens_str);
                        } catch (const std::exception &) {
                            std::cerr
                                << "Invalid num_tokens value in --gen matrix"
                                << std::endl;
                            return UCC_ERR_INVALID_PARAM;
                        }
                    }

                    std::string tgt_group_size_str;
                    if (find_param("tgt_group_size", tgt_group_size_str)) {
                        try {
                            bench.gen.matrix.tgt_group_size_mean = std::stoull(
                                tgt_group_size_str);
                        } catch (const std::exception &) {
                            std::cerr << "Invalid tgt_group_size value in "
                                         "--gen matrix"
                                      << std::endl;
                            return UCC_ERR_INVALID_PARAM;
                        }
                    }
                } else {
                    std::cerr
                        << "Invalid value for --gen. Use exp:min=N[@max=M] or "
                           "file:name=filename[@nrep=N] or "
                           "matrix:kind=mat_kind[@nrep=N@token_size=M@num_"
                           "tokens=K]"
                        << std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
            } else if (strcmp(long_options[option_index].name, "seed") == 0) {
                std::string seed_str(optarg);
                try {
                    bench.seed = std::stoull(seed_str);
                } catch (const std::exception &) {
                    std::cerr << "Invalid seed value in --seed" << std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
            } else {
                std::cerr << "Unknown long option" << std::endl;
                return UCC_ERR_INVALID_PARAM;
            }
            continue;
        }
        switch (c) {
            case 'c':
                if (ucc_pt_op_map.count(optarg) == 0) {
                    std::cerr << "invalid operation: " << optarg
                              << std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
                bench.op_type = ucc_pt_op_map.at(optarg);
                break;
            case 'o':
                if (ucc_pt_reduction_op_map.count(optarg) == 0) {
                    std::cerr << "invalid reduction operation: " << optarg
                              <<  std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
                bench.op = ucc_pt_reduction_op_map.at(optarg);
                break;
            case 'm':
                if (ucc_pt_memtype_map.count(optarg) == 0) {
                    std::cerr << "invalid memory type: " << optarg
                              <<std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
                bench.mt = ucc_pt_memtype_map.at(optarg);
                comm.mt  = bench.mt;
                break;
            case 'd':
                if (ucc_pt_datatype_map.count(optarg) == 0) {
                    std::cerr << "invalid datatype:" << optarg
                              << std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
                bench.dt = ucc_pt_datatype_map.at(optarg);
                break;
            case 'b':
                st = ucc_str_to_memunits(optarg, (void*)&bench.min_count);
                if (st != UCC_OK) {
                    std::cerr << "failed to parse min count" << std::endl;
                    return st;
                }
                break;
            case 'e':
                st = ucc_str_to_memunits(optarg, (void*)&bench.max_count);
                if (st != UCC_OK) {
                    std::cerr << "failed to parse max count" << std::endl;
                    return st;
                }
                break;
            case 'r':
                std::stringstream(optarg) >> bench.root;
                break;
            case 'S':
                std::stringstream(optarg) >> bench.root_shift;
                break;
            case 'n':
                std::stringstream(optarg) >> bench.n_iter_small;
                bench.n_iter_large = bench.n_iter_small;
                break;
            case 'w':
                std::stringstream(optarg) >> bench.n_warmup_small;
                bench.n_warmup_large = bench.n_warmup_small;
                break;
            case 'f':
                std::stringstream(optarg) >> bench.mult_factor;
                break;
            case 'N':
                std::stringstream(optarg) >> bench.n_bufs;
                break;
            case 'i':
                bench.inplace = true;
                break;
            case 'p':
                bench.persistent = true;
                break;
            case 'M':
                if (ucc_pt_map_type_map.count(optarg) == 0) {
                    std::cerr << "invalid map type: " << optarg
                              << std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
                bench.map_type = ucc_pt_map_type_map.at(optarg);
                break;
            case 'T':
                bench.triggered = true;
                break;
            case 'F':
                bench.full_print = true;
                break;
            case 'h':
            default:
                print_help();
                std::exit(0);
        }
    }
    return UCC_OK;
}

void ucc_pt_config::print_help()
{
    std::cout << "Usage: ucc_perftest [options]"<<std::endl;
    std::cout << "  -c <collective name>: Collective type"<<std::endl;
    std::cout << "  -b <count>: Min number of elements"<<std::endl;
    std::cout << "  -e <count>: Max number of elements"<<std::endl;
    std::cout << "  -i: inplace collective"<<std::endl;
    std::cout << "  -p: persistent collective"<<std::endl;
    std::cout << "  -d <dt name>: datatype"<<std::endl;
    std::cout << "  -o <op name>: reduction operation type"<<std::endl;
    std::cout << "  -r <number>: root for rooted collectives"<<std::endl;
    std::cout << "  -m <mtype name>: memory type"<<std::endl;
    std::cout << "  -n <number>: number of iterations"<<std::endl;
    std::cout << "  -w <number>: number of warmup iterations"<<std::endl;
    std::cout << "  -f <number>: multiplication factor between sizes. Default : 2."<<std::endl;
    std::cout << "  -N <number>: number of buffers"<<std::endl;
    std::cout << "  -M: use local memory registration for collectives"<<std::endl;
    std::cout << "  -T: triggered collective"<<std::endl;
    std::cout << "  -F: enable full print"<<std::endl;
    std::cout << "  -S: <number>: root shift for rooted collectives"<<std::endl;
    std::cout << "  --gen "
                 "<exp:min=N[@max=M]|file:name=filename[@nrep=N]|matrix:[kind=K][@nrep=N][@token_size=M][@num_tokens=K]"
                 "[@tgt_group_size=G]>: Pattern generator (exponential or file-based or matrix-based)"
              << std::endl;
    std::cout << " --seed <number>: seed for the random distributions" << std::endl;
    std::cout << "  -h: show this help message"<<std::endl;
    std::cout << std::endl;
}
