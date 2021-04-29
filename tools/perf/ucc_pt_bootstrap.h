/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
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
    int find_ppn()
    {
        int     comm_size = get_size();
        size_t *hashes    = new size_t[comm_size];
        int     ppn       = 0;
        ucc_status_t st;
        void *req;

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
            if (node_hash == hashes[i]) {
                ppn++;
            }
        }
        delete[] hashes;
        return ppn;
exit_err:
        std::cerr <<"failed to find ppn" <<std::endl;
        delete[] hashes;
        return ppn;
    }
public:
    ucc_pt_bootstrap()
    {
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        node_hash = std::hash<std::string>{}(std::string(hostname));
        ppn = -1;
    }
    virtual int get_rank() = 0;
    virtual int get_size() = 0;
    int get_ppn()
    {
        if (ppn == -1) {
            ppn = find_ppn();
        }
        return ppn;
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
