#include <ucc/api/ucc.h>
#include "ucc_pt_comm.h"
#include "ucc_pt_config.h"
#include "ucc_pt_coll.h"
#include "ucc_pt_benchmark.h"

int main(int argc, char *argv[])
{
    ucc_pt_config pt_config;
    pt_config.process_args(argc, argv);
    ucc_pt_comm comm;
    comm.init();
    ucc_pt_benchmark bench(pt_config.bench, &comm);
    bench.run_bench();
    comm.finalize();
    return 0;
}
