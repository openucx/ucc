/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"
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
        ucc_mem_map_params_t map_params;
        ucc_mem_map_t        segment;
        int                  rank;

        MPI_Comm_rank(team.comm, &rank);

        /* Set up memory map parameters */
        segment.address       = test_buffer;
        segment.len           = buffer_size;
        map_params.segments   = &segment;
        map_params.n_segments = 1;

        /* Test memory map */
        ucc_status_t status = ucc_mem_map(team.ctx, mode, &map_params,
                                          &memh_size, &memh);
        if (status != UCC_OK) {
            if (status == UCC_ERR_NOT_SUPPORTED ||
                status == UCC_ERR_NOT_IMPLEMENTED) {
                test_skip = TEST_SKIP_NOT_SUPPORTED;
                return;
            }
            UCC_CHECK(status);
        }

        if (!memh) {
            std::cerr << "Rank " << rank << ": Memory handle is NULL"
                      << std::endl;
            return;
        }
        if (memh_size == 0) {
            std::cerr << "Rank " << rank << ": Memory handle size is 0"
                      << std::endl;
            return;
        }

        /* Verify data integrity after mapping */
        UCC_CHECK(check());

        /* Test unmap */
        UCC_CHECK(ucc_mem_unmap(&memh));
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
        if (test_buffer) {
            ucc_free(test_buffer);
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

            /* Fill buffer with iteration-specific pattern */
            memset(test_buffer, 0xCC + rank + i, buffer_size);

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
    return std::make_shared<TestMemMapExport>(_team, params);
}

std::shared_ptr<TestCase> TestMemMapImport::init_single(ucc_test_team_t &_team,
                                                        ucc_coll_type_t _type,
                                                        TestCaseParams params)
{
    return std::make_shared<TestMemMapImport>(_team, params);
}

std::shared_ptr<TestCase> TestMemMapStress::init_single(ucc_test_team_t &_team,
                                                        ucc_coll_type_t _type,
                                                        TestCaseParams params)
{
    return std::make_shared<TestMemMapStress>(_team, params);
}

std::shared_ptr<TestCase> TestMemMapMultiSize::init_single(ucc_test_team_t &_team,
                                                           ucc_coll_type_t _type,
                                                           TestCaseParams params)
{
    return std::make_shared<TestMemMapMultiSize>(_team, params);
}
