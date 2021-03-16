#include <getopt.h>
#include <sstream>
#include "test_mpi.h"
#include <chrono>
#include "mpi_util.h"

static std::vector<ucc_coll_type_t> colls = {UCC_COLL_TYPE_BARRIER,
                                             UCC_COLL_TYPE_ALLREDUCE,
                                             UCC_COLL_TYPE_ALLGATHER,
                                             UCC_COLL_TYPE_ALLGATHERV,
                                             UCC_COLL_TYPE_BCAST};
static std::vector<ucc_memory_type_t> mtypes = {UCC_MEMORY_TYPE_HOST};
static std::vector<ucc_datatype_t> dtypes = {UCC_DT_INT32, UCC_DT_INT64,
                                             UCC_DT_FLOAT32, UCC_DT_FLOAT64};
static std::vector<ucc_reduction_op_t> ops = {UCC_OP_SUM, UCC_OP_MAX};
static std::vector<ucc_test_mpi_team_t> teams = {TEAM_WORLD, TEAM_REVERSE,
                                                 TEAM_SPLIT_HALF, TEAM_SPLIT_ODD_EVEN};
static size_t msgrange[3] = {8, (1ULL << 21), 8};
static char *cls = NULL;
static std::vector<ucc_test_mpi_inplace_t> inplace = {TEST_NO_INPLACE, TEST_INPLACE};
static ucc_test_mpi_root_t root_type = ROOT_RANDOM;
static int root_value = 10;
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

void PrintHelp()
{
    std::cout <<
       "--colls    <c1,c2,..>:        list of collectives: barrier,allreduce,allgatherv\n"
       "--teams    <t1,t2,..>:        list of teams: world,half,reverse,odd_even\n"
       "--mtypes   <m1,m2,..>:        list of mtypes: host,cuda\n"
       "--dtypes   <d1,d2,..>:        list of dtypes: (u)int8(16,32,64),float32(64)\n"
       "--ops      <o1,o2,..>:        list of ops:sum,prod,max,min,land,lor,lxor,band,bor,bxor\n"
       "--inplace  <value>:           0 - no inplace, 1 - inplace, 2 - both\n"
       "--msgsize  <min:max[:power]>  mesage sizes range:\n"
       "--root     <type:[value]>     type of root selection: single:<value>, random:<value>, all\n"
       "--help:              Show help\n";
    exit(1);
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
    } else {
        std::cerr << "incorrect team type: " << team << std::endl;
        PrintHelp();
    }
    abort();
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
    } else {
        std::cerr << "incorrect coll type: " << coll << std::endl;
        PrintHelp();
    }
    abort();
}

static ucc_memory_type_t mtype_str_to_type(std::string mtype)
{
    if (mtype == "host") {
        return UCC_MEMORY_TYPE_HOST;
    } else if (mtype == "cuda") {
        return UCC_MEMORY_TYPE_CUDA;
    } else {
        std::cerr << "incorrect memory type: " << mtype << std::endl;
        PrintHelp();
    }
    abort();
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
    } else {
        std::cerr << "incorrect dtype: " << dtype << std::endl;
        PrintHelp();
    }
    abort();
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
    } else {
        std::cerr << "incorrect op: " << op << std::endl;
        PrintHelp();
    }
    abort();
}

static void process_msgrange(const char *arg)
{
    auto tokens = str_split(arg, ":");
    if (tokens.size() == 1) {
        std::stringstream s(tokens[0]);
        s >> msgrange[0];
        msgrange[1] = msgrange[0];
        msgrange[2] = 0;
    } else if (tokens.size() >= 2) {
        std::stringstream s1(tokens[0]);
        std::stringstream s2(tokens[1]);
        s1 >> msgrange[0];
        s2 >> msgrange[1];
        msgrange[2] = 2;
        if (tokens.size() == 3) {
            std::stringstream s3(tokens[2]);
            s3 >> msgrange[2];
        }
    }
}

static void process_inplace(const char *arg)
{
    int value = std::stoi(arg);
    switch(value) {
    case 0:
        inplace = {TEST_NO_INPLACE};
        break;
    case 1:
        inplace = {TEST_INPLACE};
        break;
    case 2:
        inplace = {TEST_NO_INPLACE, TEST_INPLACE};
        break;
    default:
        break;
    }
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
    std::cerr << "incorrect root setting" << arg << std::endl;
    PrintHelp();
}

void ProcessArgs(int argc, char** argv)
{
    const char* const short_opts = "c:t:m:d:o:M:I:r:h";
    const option long_opts[] = {
        {"colls",   required_argument, nullptr, 'c'},
        {"teams",   required_argument, nullptr, 't'},
        {"mtypes",  required_argument, nullptr, 'M'},
        {"dtypes",  required_argument, nullptr, 'd'},
        {"ops",     required_argument, nullptr, 'o'},
        {"msgsize", required_argument, nullptr, 'm'},
        {"inplace", required_argument, nullptr, 'I'},
        {"root",    required_argument, nullptr, 'r'},
        {"help",    no_argument,       nullptr, 'h'},
        {nullptr,   no_argument,       nullptr, 0}
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
        case 'r':
            process_root(optarg);
            break;
        case 'h': // -h or --help
        case '?': // Unrecognized option
        default:
            PrintHelp();
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    int rank;
    ProcessArgs(argc, argv);

    UccTestMpi test(argc, argv, UCC_THREAD_SINGLE, teams, cls);
    test.set_colls(colls);
    test.set_dtypes(dtypes);
    test.set_mtypes(mtypes);
    test.set_ops(ops);
    test.set_root(root_type, root_value);
    test.set_msgsizes(msgrange[0],msgrange[1],msgrange[2]);
    for (auto &inpl : inplace) {
        test.set_inplace(inpl);
        if (UCC_OK != test.run_all()) {
            return -1;
        }
    }
    std::cout << std::flush;
    MPI_Allreduce(MPI_IN_PLACE, test.results.data(), test.results.size(),
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (0 == rank) {
        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();
        int failed = 0;
        for (auto s : test.results) {
            if (s < 0) failed++;
        }
        std::cout << "\n===== UCC MPI TEST =====\n" <<
            "   total tests : " << test.results.size() << "\n" << 
            "   passed      : " << test.results.size() - failed << "\n" <<
            "   failed      : " << failed << "\n" <<
            "   elapsed     : " <<
            std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
                  << "s" << std::endl;
    }
    mpi_progress_cleanup();
    return 0;
}
