#include <ucc/api/ucc.h>
#include "ucc_pt_comm.h"
#include "ucc_pt_config.h"
#include "ucc_pt_coll.h"
#include "ucc_pt_benchmark.h"

int main(int argc, char *argv[])
{
    ucc_pt_config pt_config;
    ucc_pt_comm *comm;
    ucc_pt_benchmark *bench;
    ucc_status_t st;

    pt_config.process_args(argc, argv);
    try {
        comm = new ucc_pt_comm(pt_config.comm);
    } catch(std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::exit(1);
    }
    st = comm->init(pt_config.bench.n_threads);
    if (st != UCC_OK) {
        delete comm;
        std::exit(1);
    }
    try {
        bench = new ucc_pt_benchmark(pt_config.bench, comm);
    } catch(std::exception &e) {
        std::cerr << e.what() << std::endl;
        comm->finalize();
        delete comm;
        std::exit(1);
    }
    bench->run_bench();
    delete bench;
    comm->finalize();
    delete comm;
    return 0;
}
