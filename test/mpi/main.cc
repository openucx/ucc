/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Advanced Micro Devices, Inc. 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <getopt.h>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "test_mpi.h"

int test_rand_seed = -1;
static size_t test_max_size = TEST_UCC_RANK_BUF_SIZE_MAX;
ucc_test_mpi_data_t ucc_test_mpi_data;

static std::vector<ucc_coll_type_t> colls = {
    UCC_COLL_TYPE_BARRIER,        UCC_COLL_TYPE_BCAST,
    UCC_COLL_TYPE_REDUCE,         UCC_COLL_TYPE_ALLREDUCE,
    UCC_COLL_TYPE_ALLGATHER,      UCC_COLL_TYPE_ALLGATHERV,
    UCC_COLL_TYPE_ALLTOALL,       UCC_COLL_TYPE_ALLTOALLV,
    UCC_COLL_TYPE_REDUCE_SCATTER, UCC_COLL_TYPE_REDUCE_SCATTERV,
    UCC_COLL_TYPE_GATHER,         UCC_COLL_TYPE_GATHERV,
    UCC_COLL_TYPE_SCATTER,        UCC_COLL_TYPE_SCATTERV};

static std::vector<ucc_coll_type_t> onesided_colls = {
    UCC_COLL_TYPE_ALLTOALL, UCC_COLL_TYPE_ALLTOALLV};

static std::vector<ucc_memory_type_t> mtypes = {
    UCC_MEMORY_TYPE_HOST};

static std::vector<ucc_datatype_t> dtypes = {
    UCC_DT_INT16,   UCC_DT_INT32,   UCC_DT_INT64,
    UCC_DT_UINT16,  UCC_DT_UINT32,  UCC_DT_UINT64,
    UCC_DT_FLOAT32, UCC_DT_FLOAT64, UCC_DT_FLOAT64_COMPLEX};

static std::vector<ucc_reduction_op_t> ops = {
    UCC_OP_SUM, UCC_OP_MAX, UCC_OP_AVG};

static std::vector<ucc_test_mpi_team_t> teams = {
    TEAM_WORLD, TEAM_REVERSE, TEAM_SPLIT_HALF, TEAM_SPLIT_ODD_EVEN};

static std::vector<ucc_test_vsize_flag_t> counts_vsize = {
    TEST_FLAG_VSIZE_32BIT, TEST_FLAG_VSIZE_64BIT};

static std::vector<ucc_test_vsize_flag_t> displs_vsize = {
    TEST_FLAG_VSIZE_32BIT, TEST_FLAG_VSIZE_64BIT};

static size_t msgrange[3] = {
    8, (1ULL << 21), 8};

static std::vector<bool>   inplace      = {false};
static std::vector<bool>   persistent   = {false};
static std::vector<bool>   triggered    = {false};
static std::vector<bool>   local_reg    = {false};
static ucc_test_mpi_root_t root_type    = ROOT_RANDOM;
static int                 root_value   = 10;
static ucc_thread_mode_t   thread_mode  = UCC_THREAD_SINGLE;
static int                 iterations   = 1;
static int                 show_help    = 0;
static int                 num_tests    = 1;
static bool                has_onesided = true;
static bool                verbose      = false;

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
extern test_set_gpu_device_t test_gpu_set_device;
#endif

static std::vector<std::string> str_split(const char *value, const char *delimiter)
{
    std::vector<std::string> rst;
    std::string str(value);
    std::string delim(delimiter);
    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delim)) != std::string::npos) {
        token = str.substr(0, pos);
        rst.push_back(token);
        str.erase(0, pos + delim.length());
    }
    rst.push_back(str);
    return rst;
}

void print_help()
{
    std::cout <<
       "-c, --colls            <c1,c2,..>\n\tlist of collectives: "
            "barrier, allreduce, allgather, allgatherv, bcast, alltoall, alltoallv "
            "reduce, reduce_scatter, reduce_scatterv, gather, gatherv, scatter, scatterv\n\n"
       "-t, --teams            <t1,t2,..>\n\tlist of teams: world,half,reverse,odd_even\n\n"
       "-M, --mtypes           <m1,m2,..>\n\tlist of mtypes: host,cuda,cudaManaged,rocm\n\n"
       "-d, --dtypes           <d1,d2,..>\n\tlist of dtypes: (u)int8(16,32,64),float32(64,128),float32(64,128)_complex\n\n"
       "-o, --ops              <o1,o2,..>\n\tlist of ops:sum,prod,max,min,land,lor,lxor,band,bor,bxor\n\n"
       "-I, --inplace          <value>\n\t0 - no inplace, 1 - inplace, 2 - both\n\n"
       "-P, --persistent       <value>\n\t0 - no persistent, 1 - persistent, 2 - both\n\n"
       "-m, --msgsize          <min:max[:power]>\n\tmesage sizes range\n\n"
       "-r, --root             <type:[value]>\n\ttype of root selection: single:<value>, random:<value>, all\n\n"
       "-s, --seed             <value>\n\tuser defined random seed\n\n"
       "-Z, --max_size         <value>\n\tmaximum send/recv buffer allocation size\n\n"
       "-C, --count_bits       <c1,c2,..>\n\tlist of counts bits: 32,64          (alltoallv only)\n\n"
       "-D, --displ_bits       <d1,d2,..>\n\tlist of displacements bits: 32,64   (alltoallv only)\n\n"
       "-S, --set_device       <value>\n\t0 - don't set, 1 - cuda_device = local_rank, 2 - cuda_device = local_rank % cuda_device_count\n\n"
       "-N, --num_tests        <value>\n\tnumber of tests to run in parallel\n\n"
       "-O, --onesided         <value>\n\t0 - no onesided tests, 1 - onesided tests\n\n"
       "-i, --iter             <value>\n\tnumber of iterations each test cases is executed\n\n"
       "-T, --thread-multiple\n\tenable multi-threaded testing\n\n"
       "-v, --verbose\n\tlog all test cases\n\n"
       "--triggered            <value>\n\t0 - use post, 1 - use triggered post, 2 - both\n\n"
       "--local_reg            <value>\n\t0 - no local registration, 1 - local registration, 2 - both\n\n"
       "-h, --help\n\tShow help\n";
}


template<typename T>
static std::vector<T> process_arg(const char *value, T (*str_to_type)(std::string value))
{
    std::vector<T> rst;
    for (auto &c : str_split(value, ",")) {
        rst.push_back(str_to_type(c));
    }
    return rst;
}

static ucc_test_mpi_team_t team_str_to_type(std::string team)
{
    if (team == "world") {
        return TEAM_WORLD;
    } else if (team == "half") {
        return TEAM_SPLIT_HALF;
    } else if (team == "odd_even") {
        return TEAM_SPLIT_ODD_EVEN;
    } else if (team == "reverse") {
        return TEAM_REVERSE;
    }
    throw std::string("incorrect team type: ") + team;
}

static std::string team_type_to_str(ucc_test_mpi_team_t team)
{
    switch (team) {
    case TEAM_WORLD:
        return "world";
    case TEAM_SPLIT_HALF:
        return "half";
    case TEAM_SPLIT_ODD_EVEN:
        return "odd_even";
    case TEAM_REVERSE:
        return "reverse";
    default:
        break;
    }
    throw std::string("incorrect team type: ");
}

static ucc_coll_type_t coll_str_to_type(std::string coll)
{
    if (coll == "barrier") {
        return UCC_COLL_TYPE_BARRIER;
    } else if (coll == "allreduce") {
        return UCC_COLL_TYPE_ALLREDUCE;
    } else if (coll == "allgather") {
        return UCC_COLL_TYPE_ALLGATHER;
    } else if (coll == "allgatherv") {
        return UCC_COLL_TYPE_ALLGATHERV;
    } else if (coll == "bcast") {
        return UCC_COLL_TYPE_BCAST;
    } else if (coll == "reduce") {
        return UCC_COLL_TYPE_REDUCE;
    } else if (coll == "alltoall") {
        return UCC_COLL_TYPE_ALLTOALL;
    } else if (coll == "alltoallv") {
        return UCC_COLL_TYPE_ALLTOALLV;
    } else if (coll == "reduce_scatter") {
        return UCC_COLL_TYPE_REDUCE_SCATTER;
    } else if (coll == "reduce_scatterv") {
        return UCC_COLL_TYPE_REDUCE_SCATTERV;
    } else if (coll == "reduce") {
        return UCC_COLL_TYPE_REDUCE;
    } else if (coll == "gather") {
        return UCC_COLL_TYPE_GATHER;
    } else if (coll == "gatherv") {
        return UCC_COLL_TYPE_GATHERV;
    } else if (coll == "scatter") {
        return UCC_COLL_TYPE_SCATTER;
    } else if (coll == "scatterv") {
        return UCC_COLL_TYPE_SCATTERV;
    } else {
        throw std::string("incorrect coll type: ") + coll;
    }
}

static ucc_memory_type_t mtype_str_to_type(std::string mtype)
{
    if (mtype == "host") {
        return UCC_MEMORY_TYPE_HOST;
    } else if (mtype == "cuda") {
        return UCC_MEMORY_TYPE_CUDA;
    } else if (mtype == "cudaManaged") {
        return UCC_MEMORY_TYPE_CUDA_MANAGED;
    } else if (mtype == "rocm") {
        return UCC_MEMORY_TYPE_ROCM;
    }
    throw std::string("incorrect memory type: ") + mtype;
}

static ucc_datatype_t dtype_str_to_type(std::string dtype)
{
    if (dtype == "int8") {
        return UCC_DT_INT8;
    } else if (dtype == "uint8") {
        return UCC_DT_UINT8;
    } else if (dtype == "int16") {
        return UCC_DT_INT16;
    } else if (dtype == "uint16") {
        return UCC_DT_UINT16;
    } else if (dtype == "int32") {
        return UCC_DT_INT32;
    } else if (dtype == "uint32") {
        return UCC_DT_UINT32;
    } else if (dtype == "int64") {
        return UCC_DT_INT64;
    } else if (dtype == "uint64") {
        return UCC_DT_UINT64;
    } else if (dtype == "float32") {
        return UCC_DT_FLOAT32;
    } else if (dtype == "float64") {
        return UCC_DT_FLOAT64;
    } else if (dtype == "float128") {
        return UCC_DT_FLOAT128;
    } else if (dtype == "bfloat16") {
        return UCC_DT_BFLOAT16;
    } else if (dtype == "float16") {
        return UCC_DT_FLOAT16;
    } else if (dtype == "int128") {
        return UCC_DT_INT128;
    } else if (dtype == "uint128") {
        return UCC_DT_UINT128;
    } else if (dtype == "float32_complex") {
        return UCC_DT_FLOAT32_COMPLEX;
    } else if (dtype == "float64_complex") {
        return UCC_DT_FLOAT64_COMPLEX;
    } else if (dtype == "float128_complex") {
        return UCC_DT_FLOAT128_COMPLEX;
    }
    throw std::string("incorrect  dtype: ") + dtype;
}

static ucc_reduction_op_t op_str_to_type(std::string op)
{
    if (op == "sum") {
        return UCC_OP_SUM;
    } else if (op == "prod") {
        return UCC_OP_PROD;
    } else if (op == "max") {
        return UCC_OP_MAX;
    } else if (op == "min") {
        return UCC_OP_MIN;
    } else if (op == "land") {
        return UCC_OP_LAND;
    } else if (op == "lor") {
        return UCC_OP_LOR;
    } else if (op == "lxor") {
        return UCC_OP_LXOR;
    } else if (op == "band") {
        return UCC_OP_BAND;
    } else if (op == "bor") {
        return UCC_OP_BOR;
    } else if (op == "bxor") {
        return UCC_OP_BXOR;
    } else if (op == "avg") {
        return UCC_OP_AVG;
    }
    throw std::string("incorrect  op: ") + op;
}

static ucc_test_vsize_flag_t bits_str_to_type(std::string vsize)
{
    if (vsize == "32") {
        return TEST_FLAG_VSIZE_32BIT;
    } else if (vsize == "64") {
        return TEST_FLAG_VSIZE_64BIT;
    }
    throw std::string("incorrect vsize") + vsize;
}

static void process_msgrange(const char *arg)
{
    auto tokens = str_split(arg, ":");

    try {
        if (tokens.size() == 1) {
            msgrange[0] = std::stol(tokens[0]);
            msgrange[1] = msgrange[0];
            msgrange[2] = 0;
        } else if (tokens.size() >= 2) {
            msgrange[0] = std::stol(tokens[0]);
            msgrange[1] = std::stol(tokens[1]);
            msgrange[2] = 2;
            if (tokens.size() == 3) {
                msgrange[2] = std::stol(tokens[2]);
            }
        }
    } catch (std::exception &e) {
        throw std::string("incorrect msgrange: ") + arg;
    }
}

static void process_inplace(const char *arg)
{
    int value = std::stoi(arg);
    switch(value) {
    case 0:
        inplace = {false};
        return;
    case 1:
        inplace = {true};
        return;
    case 2:
        inplace = {false, true};
        return;
    default:
        break;
    }
    throw std::string("incorrect inplace: ") + arg;
}

static void process_persistent(const char *arg)
{
    int value = std::stoi(arg);
    switch(value) {
    case 0:
        persistent = {false};
        return;
    case 1:
        persistent = {true};
        return;
    case 2:
        persistent = {false, true};
        return;
    default:
        break;
    }
    throw std::string("incorrect persistent: ") + arg;
}

static void process_triggered(const char *arg)
{
    int value = std::stoi(arg);
    switch(value) {
    case 0:
        triggered = {false};
        return;
    case 1:
        triggered = {true};
        return;
    case 2:
        triggered = {false, true};
        return;
    default:
        break;
    }
    throw std::string("incorrect triggered: ") + arg;
}

static void process_local_reg(const char *arg)
{
    int value = std::stoi(arg);
    switch(value) {
    case 0:
        local_reg = {false};
        return;
    case 1:
        local_reg = {true};
        return;
    case 2:
        local_reg = {false, true};
        return;
    default:
        break;
    }
    throw std::string("incorrect local_reg: ") + arg;
}

static void process_root(const char *arg)
{
    auto tokens = str_split(arg, ":");
    std::string root_type_str = tokens[0];
    if (root_type_str == "all") {
        if (tokens.size() != 1) {
            goto err;
        }
        root_type = ROOT_ALL;
    } else if (root_type_str == "random") {
        if (tokens.size() != 2) {
            goto err;
        }
        root_type = ROOT_RANDOM;
        root_value = std::atoi(tokens[1].c_str());
    } else if (root_type_str == "single") {
        if (tokens.size() != 2) {
            goto err;
        }
        root_type = ROOT_SINGLE;
        root_value = std::atoi(tokens[1].c_str());
    } else {
        goto err;
    }
    return;
err:
    throw std::string("incorrect root: ") + arg;
}

int init_rand_seed(int user_seed)
{
    int rank, seed;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (0 > user_seed) {
        if (0 == rank) {
            seed = time(NULL) % 32768;
        }
    } else {
        seed = user_seed;
    }
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (0 != rank) {
        seed += rank;
    }
    return seed;
}

void print_info()
{
    int world_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank) {
        return;
    }

    std::cout << "===== UCC MPI TEST INFO =======" << std::endl;
    std::cout <<"seed:         " << std::to_string(test_rand_seed) << std::endl;
    std::cout <<"collectives:  ";
    for (const auto &c : colls) {
        std::cout << ucc_coll_type_str(c);
        if (c != colls.back()) {
            std::cout << ", ";
        } else {
            std::cout << std::endl;
        }
    }
    std::cout <<"data types:   ";
    for (const auto &d : dtypes) {
        std::cout << ucc_datatype_str(d);
        if (d != dtypes.back()) {
            std::cout << ", ";
        } else {
            std::cout << std::endl;
        }
    }

    std::cout <<"memory types: ";
    for (const auto &m : mtypes) {
        std::cout << ucc_mem_type_str(m);
        if (m != mtypes.back()) {
            std::cout << ", ";
        } else {
            std::cout << std::endl;
        }
    }

    std::cout <<"teams:        ";
    for (const auto &t : teams) {
        std::cout << team_type_to_str(t);
        if (t != teams.back()) {
            std::cout << ", ";
        } else {
            std::cout << std::endl;
        }
    }
}

void ProcessArgs(int argc, char** argv)
{
    const char *const short_opts  = "c:t:m:d:o:M:I:P:N:r:s:C:D:i:Z:G:ThvS:O:L:";
    const option      long_opts[] = {
                                {"colls", required_argument, nullptr, 'c'},
                                {"teams", required_argument, nullptr, 't'},
                                {"mtypes", required_argument, nullptr, 'M'},
                                {"dtypes", required_argument, nullptr, 'd'},
                                {"ops", required_argument, nullptr, 'o'},
                                {"msgsize", required_argument, nullptr, 'm'},
                                {"inplace", required_argument, nullptr, 'I'},
                                {"persistent", required_argument, nullptr, 'P'},
                                {"root", required_argument, nullptr, 'r'},
                                {"seed", required_argument, nullptr, 's'},
                                {"max_size", required_argument, nullptr, 'Z'},
                                {"count_bits", required_argument, nullptr, 'C'},
                                {"displ_bits", required_argument, nullptr, 'D'},
                                {"iter", required_argument, nullptr, 'i'},
                                {"thread-multiple", no_argument, nullptr, 'T'},
                                {"num_tests", required_argument, nullptr, 'N'},
                                {"triggered", required_argument, nullptr, 'G'},
                                {"verbose", no_argument, nullptr, 'v'},
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
                                {"set_device", required_argument, nullptr, 'S'},
#endif
                                {"onesided", required_argument, nullptr, 'O'},
                                {"local_registration", required_argument, nullptr, 'L'},
                                {"help", no_argument, nullptr, 'h'},
                                {nullptr, no_argument, nullptr, 0}
    };

    while (true)
    {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch (opt)
        {
        case 'c':
            colls = process_arg<ucc_coll_type_t>(optarg, coll_str_to_type);
            break;
        case 't':
            teams = process_arg<ucc_test_mpi_team_t>(optarg, team_str_to_type);
            break;
        case 'M':
            mtypes = process_arg<ucc_memory_type_t>(optarg, mtype_str_to_type);
            break;
        case 'd':
            dtypes = process_arg<ucc_datatype_t>(optarg, dtype_str_to_type);
            break;
        case 'o':
            ops = process_arg<ucc_reduction_op_t>(optarg, op_str_to_type);
            break;
        case 'm':
            process_msgrange(optarg);
            break;
        case 'I':
            process_inplace(optarg);
            break;
        case 'P':
            process_persistent(optarg);
            break;
        case 'G':
            process_triggered(optarg);
            break;
        case 'L':
            process_local_reg(optarg);
            break;
        case 'r':
            process_root(optarg);
            break;
        case 's':
            test_rand_seed = std::stoi(optarg);
            break;
        case 'Z':
            test_max_size = std::stoi(optarg);
            break;
        case 'C':
            counts_vsize = process_arg<ucc_test_vsize_flag_t>(optarg, bits_str_to_type);
            break;
        case 'D':
            displs_vsize = process_arg<ucc_test_vsize_flag_t>(optarg, bits_str_to_type);
            break;
        case 'T':
            thread_mode = UCC_THREAD_MULTIPLE;
            break;
        case 'i':
            iterations = std::stoi(optarg);
            break;
        case 'N':
            num_tests = std::stoi(optarg);
            break;
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
        case 'S':
            test_gpu_set_device = (test_set_gpu_device_t)std::stoi(optarg);
            break;
#endif
        case 'O':
            has_onesided = std::stoi(optarg);
            break;

        case 'v':
            verbose = true;
            break;
        case 'h':
            show_help = 1;
            break;
        case '?': // Unrecognized option
        default:
            throw std::string("unrecognized option");
        }
    }
}

int main(int argc, char *argv[])
{
    int failed = 0;
    int total_done_skipped_failed[ucc_ilog2(UCC_COLL_TYPE_LAST) + 1][4];
    std::chrono::steady_clock::time_point begin;
    int size, required, provided, completed, rank;
    UccTestMpi *test;
    MPI_Request req;
    std::string err;

    begin = std::chrono::steady_clock::now();
    memset(total_done_skipped_failed, 0,
           sizeof(total_done_skipped_failed));
    try {
        ProcessArgs(argc, argv);
    } catch (const std::string &s) {
        failed = 1;
        err = s;
    }
    required = (thread_mode == UCC_THREAD_SINGLE) ? MPI_THREAD_SINGLE
        : MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided != required) {
        std::cerr << "could not initialize MPI in thread multiple\n";
        return 1;
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (!err.empty() || show_help) {
        if (rank == 0) {
            std::cerr << "ParseArgs error:" << err << "\n\n";
            print_help();
        }
        goto mpi_exit;
    }

    if (size < 2) {
        std::cerr << "test requires at least 2 ranks\n";
        goto mpi_exit;
    }

    init_test_mpi_data();
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
    set_gpu_device(test_gpu_set_device);
#endif
    test = new UccTestMpi(argc, argv, thread_mode, 0, has_onesided);
    for (auto &m : mtypes) {
        if (UCC_MEMORY_TYPE_HOST != m && UCC_OK != ucc_mc_available(m)) {
            std::cerr << "requested memory type " << ucc_memory_type_names[m]
                      << " is not supported " << std::endl;
            failed = -1;
            goto test_exit;
        }
    }
    test->create_teams(teams);
    if (has_onesided) {
        test->create_teams(teams, true);
    }
    test->set_verbose(verbose);
    test->set_iter(iterations);
    test->set_num_tests(num_tests);
    test->set_colls(colls);
    test->set_dtypes(dtypes);
    test->set_mtypes(mtypes);
    test->set_ops(ops);
    test->set_root(root_type, root_value);
    test->set_count_vsizes(counts_vsize);
    test->set_displ_vsizes(displs_vsize);
    test->set_msgsizes(msgrange[0],msgrange[1],msgrange[2]);
    test->set_max_size(test_max_size);
    test_rand_seed = init_rand_seed(test_rand_seed);

    print_info();

    for (auto inpl : inplace) {
        for (auto pers : persistent) {
            for (auto trig: triggered) {
                for (auto lr : local_reg) {
                    test->set_triggered(trig);
                    test->set_inplace(inpl);
                    test->set_persistent(pers);
                    test->set_local_registration(lr);
                    test->run_all();
                }
            }
        }
    }

    if (has_onesided) {
        std::vector<ucc_coll_type_t>           os_colls(onesided_colls.size());
        std::vector<ucc_coll_type_t>::iterator it_start;

        std::sort(colls.begin(), colls.end());
        std::sort(onesided_colls.begin(), onesided_colls.end());

        it_start = std::set_intersection(
            colls.begin(), colls.end(), onesided_colls.begin(),
            onesided_colls.end(), os_colls.begin());
        os_colls.resize(it_start - os_colls.begin());
        test->set_colls(os_colls);
        for (auto inpl : inplace) {
            for (auto pers : persistent) {
                test->set_triggered(false);
                test->set_inplace(inpl);
                test->set_persistent(pers);
                test->run_all(true);
            }
        }
    }
    std::cout << std::flush;

    for (auto s : test->results) {
        int coll_num = ucc_ilog2(std::get<0>(s));
        switch(std::get<1>(s)) {
        case UCC_OK:
            total_done_skipped_failed[coll_num][1]++;
            break;
        case UCC_ERR_NOT_IMPLEMENTED:
        case UCC_ERR_LAST:
            total_done_skipped_failed[coll_num][2]++;
            break;
        default:
            total_done_skipped_failed[coll_num][3]++;
        }
        total_done_skipped_failed[coll_num][0]++;
    }
    MPI_Iallreduce(MPI_IN_PLACE, total_done_skipped_failed,
                   sizeof(total_done_skipped_failed)/sizeof(int),
                   MPI_INT, MPI_MAX, MPI_COMM_WORLD, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        test->progress_ctx();
    } while(!completed);

    if (0 == rank) {
        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();
        ucc_coll_type_t coll_type;
        int num_all = 0, num_skipped = 0, num_done =0, num_failed = 0;
        std::ios iostate(nullptr);

        iostate.copyfmt(std::cout);
        std::cout << "\n===== UCC MPI TEST REPORT =====\n" <<
            std::setw(22) << std::left << "collective" <<
            std::setw(10) << std::right << "tests" <<
            std::setw(10) << std::right << "passed" <<
            std::setw(10) << std::right << "failed" <<
            std::setw(10) << std::right << "skipped" << std::endl;

        for (coll_type =  (ucc_coll_type_t)1;
             coll_type < UCC_COLL_TYPE_LAST;
             coll_type = (ucc_coll_type_t)(coll_type << 1))
        {
            int coll_num = ucc_ilog2(coll_type);
            if (total_done_skipped_failed[coll_num][0] == 0) {
                continue;
            }
            num_all += total_done_skipped_failed[coll_num][0];
            num_done += total_done_skipped_failed[coll_num][1];
            num_skipped += total_done_skipped_failed[coll_num][2];
            num_failed += total_done_skipped_failed[coll_num][3];
            std::cout <<
                std::setw(22) << std::left << ucc_coll_type_str(coll_type) <<
                std::setw(10) << std::right << total_done_skipped_failed[coll_num][0] <<
                std::setw(10) << std::right << total_done_skipped_failed[coll_num][1] <<
                std::setw(10) << std::right << total_done_skipped_failed[coll_num][3] <<
                std::setw(10) << std::right << total_done_skipped_failed[coll_num][2] <<
                std::endl;

        }
        std::cout <<
            " \n===== UCC MPI TEST SUMMARY =====\n" <<
            "total tests:  " << num_all << "\n" <<
            "passed:       " << num_done << "\n" <<
            "skipped:      " << num_skipped << "\n" <<
            "failed:       " << num_failed << "\n" <<
            "elapsed:      " <<
            std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
                  << "s" << std::endl;
        std::cout.copyfmt(iostate);

        /* check if all tests have been skipped */
        if (num_all == num_skipped) {
            std::cout << "\n All tests have been skipped, indicating most likely "
                         "a problem\n";
            failed = 1;
        }

        if (num_failed != 0) {
            failed = 1;
        }
    }

test_exit:
    delete test;
mpi_exit:
    MPI_Finalize();
    return failed;
}

int init_test_mpi_data(void)
{
    MPI_Comm local_comm;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &ucc_test_mpi_data.local_node_rank);
    MPI_Comm_free(&local_comm);
    return 0;
}
