/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_BOOTSTRAP_H
#define UCC_PT_BOOTSTRAP_H

#include <ucc/api/ucc.h>
#include <unistd.h>
#include <functional>
#include <string>
#include <iostream>

class ucc_pt_bootstrap {
protected:
    size_t node_hash;
    ucc_context_oob_coll_t context_oob;
    ucc_team_oob_coll_t team_oob;
    int ppn;
    int local_rank;
    void find_ppn()
    {
        int     comm_size = get_size();
        int     comm_rank = get_rank();
        size_t *hashes    = new size_t[comm_size];
        ucc_status_t st;
        void *req;

        ppn = 0;
        local_rank = 0;
        st = team_oob.allgather(&node_hash, hashes, sizeof(node_hash),
                                team_oob.coll_info, &req);
        if (st != UCC_OK) {
            goto exit_err;
        }
        do {
            st = team_oob.req_test(req);
        } while (st == UCC_INPROGRESS);
        if (st != UCC_OK) {
            goto exit_err;
        }
        team_oob.req_free(req);
        for (int i = 0; i < comm_size; i++) {
            if (i == comm_rank) {
                break;
            }
            if (node_hash == hashes[i]) {
                local_rank++;
            }
        }
        for (int i = 0; i < comm_size; i++) {
            if (node_hash == hashes[i]) {
                ppn++;
            }
        }
        delete[] hashes;
        return;
exit_err:
        std::cerr <<"failed to find ppn" <<std::endl;
        delete[] hashes;
    }
public:
    ucc_pt_bootstrap()
    {
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        node_hash = std::hash<std::string>{}(std::string(hostname));
        ppn = -1;
        local_rank = -1;
    }
    virtual int get_rank() = 0;
    virtual int get_size() = 0;
    int get_ppn()
    {
        if (ppn == -1) {
            find_ppn();
        }
        return ppn;
    }
    int get_local_rank()
    {
        if (local_rank == -1) {
            find_ppn();
        }
        return local_rank;
    }
    virtual ~ucc_pt_bootstrap() {};
    ucc_context_oob_coll_t get_context_oob()
    {
        return context_oob;
    }

    ucc_team_oob_coll_t get_team_oob() {
        return team_oob;
    }
};

#endif
