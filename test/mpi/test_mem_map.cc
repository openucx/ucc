/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"
#include <climits>
#include <cstring>
#include <vector>
#include "components/mc/ucc_mc.h"

class TestMemMap : public TestCase
{
private:
    void *             test_buffer;
    size_t             buffer_size;
    ucc_mem_map_mem_h  memh;
    size_t             memh_size;
    ucc_mem_map_mode_t mode;
    bool               is_export_test;

public:
    TestMemMap(ucc_test_team_t &_team, TestCaseParams &params,
               ucc_mem_map_mode_t _mode = UCC_MEM_MAP_MODE_EXPORT)
        : TestCase(_team, UCC_COLL_TYPE_BARRIER, params), // Using barrier as placeholder
          test_buffer(nullptr), buffer_size(0), memh(nullptr), memh_size(0),
          mode(_mode), is_export_test(_mode == UCC_MEM_MAP_MODE_EXPORT)
    {
        buffer_size = params.msgsize;
        int rank;
        if (buffer_size == 0) {
            buffer_size = 1024 * 1024; // Default 1MB
        }

        if (skip_reduce(test_max_size < buffer_size, TEST_SKIP_MEM_LIMIT,
                        team.comm)) {
            return;
        }

        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, buffer_size, mem_type));
        test_buffer = rbuf_mc_header->addr;
        UCC_MALLOC_CHECK(test_buffer);

        // Initialize buffer with rank-specific data
        MPI_Comm_rank(team.comm, &rank);
        memset(test_buffer, 0xAA + rank, buffer_size);
    }

    ~TestMemMap()
    {
        if (memh) {
            ucc_mem_unmap(&memh);
        }
    }

    ucc_status_t set_input(int iter_persistent = 0) override
    {
        int rank;
        MPI_Comm_rank(team.comm, &rank);

        // Initialize buffer with rank and iteration specific data
        memset(test_buffer, 0xAA + rank + iter_persistent, buffer_size);
        return UCC_OK;
    }

    ucc_status_t check() override
    {
        unsigned char *buf = (unsigned char *)test_buffer;
        unsigned char  expected;
        int            rank;
        size_t         i;

        MPI_Comm_rank(team.comm, &rank);

        expected = 0xAA + rank;
        for (i = 0; i < buffer_size; i++) {
            if (buf[i] != expected) {
                return UCC_ERR_INVALID_PARAM;
            }
        }
        return UCC_OK;
    }

    void run(bool triggered) override
    {
        (void)triggered; /* mem_map tests don't support triggered mode */
        ucc_mem_map_mem_h    export_memh = nullptr;
        ucc_mem_map_params_t map_params = {};
        ucc_mem_map_t        segment    = {};
        size_t               export_size = 0;
        uint64_t             recv_size   = 0;
        int                  comm_size, peer, rank;
        ucc_status_t         status;

        MPI_Comm_rank(team.comm, &rank);
        MPI_Comm_size(team.comm, &comm_size);

        /* Set up memory map parameters */
        segment.address       = test_buffer;
        segment.len           = buffer_size;
        map_params.segments   = &segment;
        map_params.n_segments = 1;

        if (!is_export_test) {
            /* Import operates on an exchanged serialized export handle. Send
             * each rank's blob to its next peer so import never receives a
             * NULL or otherwise uninitialized handle.
             */
            status = ucc_mem_map(team.ctx, UCC_MEM_MAP_MODE_EXPORT,
                                 &map_params, &export_size, &export_memh);
            if (status != UCC_OK) {
                if (status == UCC_ERR_NOT_SUPPORTED ||
                    status == UCC_ERR_NOT_IMPLEMENTED) {
                    test_skip = TEST_SKIP_NOT_SUPPORTED;
                    return;
                }
                UCC_CHECK(status);
            }
            if (!export_memh || export_size == 0) {
                UCC_CHECK(UCC_ERR_INVALID_PARAM);
            }

            peer = (rank + 1) % comm_size;
            uint64_t send_size = export_size;
            MPI_Sendrecv(&send_size, 1, MPI_UINT64_T, peer, 1001,
                         &recv_size, 1, MPI_UINT64_T,
                         (rank + comm_size - 1) % comm_size, 1001,
                         team.comm, MPI_STATUS_IGNORE);

            if (send_size > INT_MAX || recv_size > INT_MAX) {
                UCC_CHECK(UCC_ERR_INVALID_PARAM);
            }
            memh = ucc_malloc(recv_size, "import memh blob");
            UCC_MALLOC_CHECK(memh);
            MPI_Sendrecv(export_memh, static_cast<int>(send_size), MPI_BYTE,
                         peer, 1002, memh, static_cast<int>(recv_size), MPI_BYTE,
                         (rank + comm_size - 1) % comm_size, 1002,
                         team.comm, MPI_STATUS_IGNORE);
            memh_size = recv_size;
        }

        /* Test memory map */
        status = ucc_mem_map(team.ctx, mode, &map_params, &memh_size, &memh);
        if (status != UCC_OK) {
            if (export_memh) {
                UCC_CHECK(ucc_mem_unmap(&export_memh));
            }
            if (status == UCC_ERR_NOT_SUPPORTED ||
                status == UCC_ERR_NOT_IMPLEMENTED) {
                if (memh) {
                    ucc_free(memh);
                    memh = nullptr;
                }
                test_skip = TEST_SKIP_NOT_SUPPORTED;
                return;
            }
            UCC_CHECK(status);
        }

        if (!memh) {
            std::cerr << "Rank " << rank << ": Memory handle is NULL"
                      << std::endl;
            UCC_CHECK(UCC_ERR_INVALID_PARAM);
        }
        if (is_export_test && memh_size == 0) {
            std::cerr << "Rank " << rank << ": Memory handle size is 0"
                      << std::endl;
            UCC_CHECK(UCC_ERR_INVALID_PARAM);
        }

        /* Verify data integrity after mapping */
        UCC_CHECK(check());

        /* Test unmap */
        UCC_CHECK(ucc_mem_unmap(&memh));
        if (export_memh) {
            UCC_CHECK(ucc_mem_unmap(&export_memh));
        }
        if (memh != nullptr) {
            std::cerr << "Rank " << rank
                      << ": Memory handle not NULL after unmap" << std::endl;
            return;
        }

        /* Verify data integrity after unmapping */
        UCC_CHECK(check());
    }

    std::string str() override
    {
        return std::string("mem_map mode=") + std::to_string(mode) +
               " team=" + team_str(team.type) +
               " buffer_size=" + std::to_string(buffer_size) +
               " mem_type=" + std::to_string(mem_type);
    }
};

class TestMemMapExport : public TestMemMap
{
public:
    TestMemMapExport(ucc_test_team_t &_team, TestCaseParams &params)
        : TestMemMap(_team, params, UCC_MEM_MAP_MODE_EXPORT)
    {
    }
    static std::shared_ptr<TestCase> init_single(ucc_test_team_t &_team,
                                                 ucc_coll_type_t _type,
                                                 TestCaseParams params);
};

class TestMemMapImport : public TestMemMap
{
public:
    TestMemMapImport(ucc_test_team_t &_team, TestCaseParams &params)
        : TestMemMap(_team, params, UCC_MEM_MAP_MODE_IMPORT)
    {
    }
    static std::shared_ptr<TestCase> init_single(ucc_test_team_t &_team,
                                                 ucc_coll_type_t _type,
                                                 TestCaseParams params);
};

class TestMemMapStress : public TestCase
{
private:
    void                          *test_buffer;
    size_t                         buffer_size;
    std::vector<ucc_mem_map_mem_h> memhs;
    int                            num_iterations = 10;

public:
    TestMemMapStress(ucc_test_team_t &_team, TestCaseParams &params)
        : TestCase(_team, UCC_COLL_TYPE_BARRIER, params),
          test_buffer(nullptr), buffer_size(0), num_iterations(10)
    {
        buffer_size = params.msgsize;
        if (buffer_size == 0) {
            buffer_size = 1024 * 1024; /* Default 1MB */
        }

        if (skip_reduce(test_max_size < buffer_size, TEST_SKIP_MEM_LIMIT,
                        team.comm)) {
            return;
        }

        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, buffer_size, mem_type));
        test_buffer = rbuf_mc_header->addr;
        UCC_MALLOC_CHECK(test_buffer);

        memhs.reserve(num_iterations);
    }

    ~TestMemMapStress()
    {
        for (auto memh : memhs) {
            if (memh) {
                ucc_mem_unmap(&memh);
            }
        }
    }

    ucc_status_t set_input(int iter_persistent = 0) override
    {
        int rank;
        MPI_Comm_rank(team.comm, &rank);

        // Initialize buffer with iteration-specific data
        memset(test_buffer, 0xBB + rank + iter_persistent, buffer_size);
        return UCC_OK;
    }

    ucc_status_t check() override
    {
        unsigned char *buf      = (unsigned char *)test_buffer;
        unsigned char  expected;
        size_t         i;
        int            rank;

        MPI_Comm_rank(team.comm, &rank);

        /* Verify buffer integrity */
        expected = 0xBB + rank;
        for (i = 0; i < buffer_size; i++) {
            if (buf[i] != expected) {
                return UCC_ERR_INVALID_PARAM;
            }
        }
        return UCC_OK;
    }

    void run(bool triggered) override
    {
        (void)triggered; /* mem_map tests don't support triggered mode */
        ucc_mem_map_params_t map_params;
        ucc_mem_map_t        segment;
        int                  rank;
        int                  i;

        MPI_Comm_rank(team.comm, &rank);

        /* Set up memory map parameters */
        segment.address = test_buffer;
        segment.len     = buffer_size;
        map_params.segments = &segment;
        map_params.n_segments = 1;

        /* Stress test: multiple map/unmap operations */
        for (i = 0; i < num_iterations; i++) {
            ucc_mem_map_mem_h memh;
            size_t             memh_size;

            /* Keep the contents stable while repeatedly mapping the buffer. */
            memset(test_buffer, 0xBB + rank, buffer_size);

            ucc_status_t status = ucc_mem_map(team.ctx, UCC_MEM_MAP_MODE_EXPORT,
                                              &map_params, &memh_size, &memh);
            if (status != UCC_OK) {
                if (status == UCC_ERR_NOT_SUPPORTED ||
                    status == UCC_ERR_NOT_IMPLEMENTED) {
                    test_skip = TEST_SKIP_NOT_SUPPORTED;
                    return;
                }
                UCC_CHECK(status);
            }

            if (!memh) {
                std::cerr << "Rank " << rank
                          << ": Memory handle is NULL in stress test"
                          << std::endl;
                return;
            }
            if (memh_size == 0) {
                std::cerr << "Rank " << rank
                          << ": Memory handle size is 0 in stress test"
                          << std::endl;
                return;
            }

            /* Store memh for cleanup */
            memhs.push_back(memh);

            /* Verify data integrity */
            UCC_CHECK(check());
        }

        /* Cleanup all memory handles */
        for (auto &memh : memhs) {
            UCC_CHECK(ucc_mem_unmap(&memh));
        }
        memhs.clear();

        /* Final verification */
        UCC_CHECK(check());
    }

    std::string str() override
    {
        return std::string("mem_map_stress") +
               " team=" + team_str(team.type) +
               " buffer_size=" + std::to_string(buffer_size) +
               " iterations=" + std::to_string(num_iterations) +
               " mem_type=" + std::to_string(mem_type);
    }
    static std::shared_ptr<TestCase> init_single(ucc_test_team_t &_team,
                                                 ucc_coll_type_t _type,
                                                 TestCaseParams params);
};

class TestMemMapMultiSize : public TestCase
{
private:
    std::vector<size_t>                      buffer_sizes;
    std::vector<void *>                      test_buffers;
    std::vector<ucc_mem_map_mem_h>           memhs;

public:
    TestMemMapMultiSize(ucc_test_team_t &_team, TestCaseParams &params)
        : TestCase(_team, UCC_COLL_TYPE_BARRIER, params)
    {
        size_t i;
        /* Test different buffer sizes */
        buffer_sizes = {1024, 4096, 65536, 1024 * 1024};

        if (skip_reduce(test_max_size < buffer_sizes.back(), TEST_SKIP_MEM_LIMIT,
                        team.comm)) {
            return;
        }

        test_buffers.resize(buffer_sizes.size());
        memhs.resize(buffer_sizes.size());

        /* Allocate buffers */
        for (i = 0; i < buffer_sizes.size(); i++) {
            test_buffers[i] = ucc_malloc(buffer_sizes[i], "test buffer");
            UCC_MALLOC_CHECK(test_buffers[i]);

            /* Initialize with size-specific pattern */
            memset(test_buffers[i], 0xDD + i, buffer_sizes[i]);
        }
    }

    ~TestMemMapMultiSize()
    {
        for (auto memh : memhs) {
            if (memh) {
                ucc_mem_unmap(&memh);
            }
        }
        for (auto buf : test_buffers) {
            if (buf) {
                ucc_free(buf);
            }
        }
    }

    ucc_status_t set_input(int iter_persistent = 0) override
    {
        size_t i;
        int    rank;

        MPI_Comm_rank(team.comm, &rank);
        /* Initialize all buffers with rank and iteration specific data */
        for (i = 0; i < test_buffers.size(); i++) {
            memset(test_buffers[i], 0xDD + rank + iter_persistent,
                   buffer_sizes[i]);
        }
        return UCC_OK;
    }

    ucc_status_t check() override
    {
        size_t i;
        size_t j;
        int    rank;

        MPI_Comm_rank(team.comm, &rank);
        for (i = 0; i < test_buffers.size(); i++) {
            unsigned char *buf      = (unsigned char *)test_buffers[i];
            unsigned char  expected = 0xDD + rank;

            for (j = 0; j < buffer_sizes[i]; j++) {
                if (buf[j] != expected) {
                    return UCC_ERR_INVALID_PARAM;
                }
            }
        }
        return UCC_OK;
    }

    void run(bool triggered) override
    {
        (void)triggered; /* mem_map tests don't support triggered mode */
        ucc_mem_map_params_t map_params;
        ucc_mem_map_t        segment;
        int                  rank;
        size_t               i;
        ucc_mem_map_mem_h    memh;
        size_t               memh_size;

        MPI_Comm_rank(team.comm, &rank);

        /* Test memory mapping with different buffer sizes */
        for (i = 0; i < buffer_sizes.size(); i++) {
            segment.address       = test_buffers[i];
            segment.len           = buffer_sizes[i];
            map_params.segments   = &segment;
            map_params.n_segments = 1;
            memh                  = nullptr;
            memh_size             = 0;
            ucc_status_t status = ucc_mem_map(team.ctx, UCC_MEM_MAP_MODE_EXPORT,
                                              &map_params, &memh_size, &memh);
            if (status != UCC_OK) {
                if (status == UCC_ERR_NOT_SUPPORTED ||
                    status == UCC_ERR_NOT_IMPLEMENTED) {
                    test_skip = TEST_SKIP_NOT_SUPPORTED;
                    return;
                }
                UCC_CHECK(status);
            }

            if (!memh) {
                std::cerr << "Rank " << rank
                          << ": Memory handle is NULL in multi-size test"
                          << std::endl;
                return;
            }
            if (memh_size == 0) {
                std::cerr << "Rank " << rank
                          << ": Memory handle size is 0 in multi-size test"
                          << std::endl;
                return;
            }

            /* Store memh for cleanup */
            memhs[i] = memh;

            /* Verify data integrity */
            UCC_CHECK(check());
        }

        /* Cleanup all memory handles */
        for (auto &memh : memhs) {
            UCC_CHECK(ucc_mem_unmap(&memh));
        }

        /* Final verification */
        UCC_CHECK(check());
    }

    std::string str() override
    {
        return std::string("mem_map_multi_size") +
               " team=" + team_str(team.type) +
               " num_sizes=" + std::to_string(buffer_sizes.size()) +
               " mem_type=" + std::to_string(mem_type);
    }
    static std::shared_ptr<TestCase> init_single(ucc_test_team_t &_team,
                                                 ucc_coll_type_t _type,
                                                 TestCaseParams params);
};

// Factory functions for creating test instances
std::shared_ptr<TestCase> TestMemMapExport::init_single(ucc_test_team_t &_team,
                                                        ucc_coll_type_t _type,
                                                        TestCaseParams params)
{
    (void)_type; /* mem_map tests only use BARRIER */
    return std::make_shared<TestMemMapExport>(_team, params);
}

std::shared_ptr<TestCase> TestMemMapImport::init_single(ucc_test_team_t &_team,
                                                        ucc_coll_type_t _type,
                                                        TestCaseParams params)
{
    (void)_type; /* mem_map tests only use BARRIER */
    return std::make_shared<TestMemMapImport>(_team, params);
}

std::shared_ptr<TestCase> TestMemMapStress::init_single(ucc_test_team_t &_team,
                                                        ucc_coll_type_t _type,
                                                        TestCaseParams params)
{
    (void)_type; /* mem_map tests only use BARRIER */
    return std::make_shared<TestMemMapStress>(_team, params);
}

std::shared_ptr<TestCase> TestMemMapMultiSize::init_single(ucc_test_team_t &_team,
                                                            ucc_coll_type_t _type,
                                                            TestCaseParams params)
{
    (void)_type; /* mem_map tests only use BARRIER */
    return std::make_shared<TestMemMapMultiSize>(_team, params);
}

void run_mem_map_tests(UccTestMpi *test, int tally[4])
{
    static std::shared_ptr<TestCase> (*mem_map_factories[])(
        ucc_test_team_t &, ucc_coll_type_t, TestCaseParams) = {
            TestMemMapExport::init_single,
            TestMemMapImport::init_single,
            TestMemMapStress::init_single,
            TestMemMapMultiSize::init_single
    };

    /* tally = {tests, passed, skipped, failed}. Reported separately from the
       collective results so a mem_map failure does not masquerade as BARRIER. */
    tally[0] = tally[1] = tally[2] = tally[3] = 0;

    for (auto &team : test->teams) {
        TestCaseParams params;
        memset(&params, 0, sizeof(params));
        params.max_size = 8 * 1024 * 1024;
        params.msgsize  = 0;
        params.inplace  = false;
        params.persistent = false;
        params.memh_mode = UCC_TEST_MEMH_NONE;

        for (auto &factory : mem_map_factories) {
            auto tc = factory(team, UCC_COLL_TYPE_BARRIER, params);
            if (!tc) continue;

            tally[0]++;
            if (TEST_SKIP_NONE != tc->test_skip) {
                tally[2]++;
                continue;
            }

            UCC_CHECK(tc->set_input());
            tc->run(false);
            if (TEST_SKIP_NONE != tc->test_skip) {
                tally[2]++;
                continue;
            }

            if (tc->check() == UCC_OK) {
                tally[1]++;
            } else {
                tally[3]++;
            }
        }
    }
}
