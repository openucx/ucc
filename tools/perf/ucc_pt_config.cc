#include "ucc_pt_config.h"

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
}

const std::map<std::string, ucc_reduction_op_t> ucc_pt_op_map = {
    {"sum", UCC_OP_SUM},
    {"prod", UCC_OP_PROD},
    {"min", UCC_OP_MIN},
    {"max", UCC_OP_MAX},
};

const std::map<std::string, ucc_coll_type_t> ucc_pt_coll_map = {
    {"allgather", UCC_COLL_TYPE_ALLGATHER},
    {"allgatherv", UCC_COLL_TYPE_ALLGATHERV},
    {"allreduce", UCC_COLL_TYPE_ALLREDUCE},
    {"alltoall", UCC_COLL_TYPE_ALLTOALL},
};

ucc_status_t ucc_pt_config::process_args(int argc, char *argv[])
{
    int c;

    while ((c = getopt(argc, argv, "c:b:e:d:m:n:w:o:ih")) != -1) {
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
            case 'b':
                std::stringstream(optarg) >> bench.min_count;
                break;
            case 'e':
                std::stringstream(optarg) >> bench.max_count;
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
    std::cout<<"Usage: ucc_perftest [options]"<<std::endl;
    std::cout<<"  -c <collective name>: Collective type"<<std::endl;
    std::cout<<"  -b <count>: Min number of elements"<<std::endl;
    std::cout<<"  -e <count>: Max number of elements"<<std::endl;
    std::cout<<"  -i: inplace collective"<<std::endl;
    std::cout<<"  -o <op name>: operation type for rudction"<<std::endl;
    std::cout<<"  -n <number>: number of iterations"<<std::endl;
    std::cout<<"  -w <number>: number of warmup iterations"<<std::endl;
    std::cout<<"  -h: show this help message"<<std::endl;
    std::cout<<std::endl;
}
