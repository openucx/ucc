#include "ucc_pt_config.h"
BEGIN_C_DECLS
#include "utils/ucc_string.h"
END_C_DECLS

ucc_pt_config::ucc_pt_config() {
    bootstrap.bootstrap  = UCC_PT_BOOTSTRAP_MPI;
    bench.coll_type      = UCC_COLL_TYPE_ALLREDUCE;
    bench.min_count      = 128;
    bench.max_count      = 128;
    bench.dt             = UCC_DT_FLOAT32;
    bench.mt             = UCC_MEMORY_TYPE_HOST;
    bench.op             = UCC_OP_SUM;
    bench.inplace        = false;
    bench.n_iter_small   = 1000;
    bench.n_warmup_small = 100;
    bench.n_iter_large   = 200;
    bench.n_warmup_large = 20;
    bench.large_thresh   = 64 * 1024;
    bench.full_print     = false;
}

const std::map<std::string, ucc_reduction_op_t> ucc_pt_op_map = {
    {"sum", UCC_OP_SUM}, {"prod", UCC_OP_PROD}, {"min", UCC_OP_MIN},
    {"max", UCC_OP_MAX}, {"avg", UCC_OP_AVG},
};

const std::map<std::string, ucc_coll_type_t> ucc_pt_coll_map = {
    {"allgather", UCC_COLL_TYPE_ALLGATHER},
    {"allgatherv", UCC_COLL_TYPE_ALLGATHERV},
    {"allreduce", UCC_COLL_TYPE_ALLREDUCE},
    {"alltoall", UCC_COLL_TYPE_ALLTOALL},
    {"alltoallv", UCC_COLL_TYPE_ALLTOALLV},
    {"barrier", UCC_COLL_TYPE_BARRIER},
    {"bcast", UCC_COLL_TYPE_BCAST},
    {"gather", UCC_COLL_TYPE_GATHER},
    {"gatherv", UCC_COLL_TYPE_GATHERV},
    {"reduce", UCC_COLL_TYPE_REDUCE},
    {"reduce_scatter", UCC_COLL_TYPE_REDUCE_SCATTER},
    {"scatter", UCC_COLL_TYPE_SCATTER},
    {"scatterv", UCC_COLL_TYPE_SCATTERV},
};

const std::map<std::string, ucc_memory_type_t> ucc_pt_memtype_map = {
    {"host", UCC_MEMORY_TYPE_HOST},
    {"cuda", UCC_MEMORY_TYPE_CUDA},
    {"rocm", UCC_MEMORY_TYPE_ROCM},
};

const std::map<std::string, ucc_datatype_t> ucc_pt_datatype_map = {
    {"int8", UCC_DT_INT8},
    {"uint8", UCC_DT_UINT8},
    {"int16", UCC_DT_INT16},
    {"uint16", UCC_DT_UINT16},
    {"float16", UCC_DT_FLOAT16},
    {"bfloat16", UCC_DT_BFLOAT16},
    {"int32", UCC_DT_INT32},
    {"float32", UCC_DT_FLOAT32},
    {"int64", UCC_DT_INT64},
    {"uint64", UCC_DT_UINT64},
    {"float64", UCC_DT_FLOAT64},
    {"int128", UCC_DT_INT128},
    {"uint128", UCC_DT_UINT128},
};

ucc_status_t ucc_pt_config::process_args(int argc, char *argv[])
{
    int c;
    ucc_status_t st;

    while ((c = getopt(argc, argv, "c:b:e:d:m:n:w:o:ihF")) != -1) {
        switch (c) {
            case 'c':
                if (ucc_pt_coll_map.count(optarg) == 0) {
                    std::cerr << "invalid collective" << std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
                bench.coll_type = ucc_pt_coll_map.at(optarg);
                break;
            case 'o':
                if (ucc_pt_op_map.count(optarg) == 0) {
                    std::cerr << "invalid operation" << std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
                bench.op = ucc_pt_op_map.at(optarg);
                break;
            case 'm':
                if (ucc_pt_memtype_map.count(optarg) == 0) {
                    std::cerr << "invalid memory type" << std::endl;
                    return UCC_ERR_INVALID_PARAM;
                }
                bench.mt = ucc_pt_memtype_map.at(optarg);
                comm.mt  = bench.mt;
                break;
            case 'd':
                if (ucc_pt_datatype_map.count(optarg) == 0) {
                    std::cerr << "invalid datatype" << std::endl;
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
            case 'n':
                std::stringstream(optarg) >> bench.n_iter_small;
                bench.n_iter_large = bench.n_iter_small;
                break;
            case 'w':
                std::stringstream(optarg) >> bench.n_warmup_small;
                bench.n_warmup_large = bench.n_warmup_small;
                break;
            case 'i':
                bench.inplace = true;
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
    std::cout << "  -d <dt name>: datatype"<<std::endl;
    std::cout << "  -o <op name>: reduction operation type"<<std::endl;
    std::cout << "  -m <mtype name>: memory type"<<std::endl;
    std::cout << "  -n <number>: number of iterations"<<std::endl;
    std::cout << "  -w <number>: number of warmup iterations"<<std::endl;
    std::cout << "  -F: enable full print"<<std::endl;
    std::cout << "  -h: show this help message"<<std::endl;
    std::cout << std::endl;
}
