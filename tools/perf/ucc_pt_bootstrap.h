/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_BOOTSTRAP_H
#define UCC_PT_BOOTSTRAP_H

#include <ucc/api/ucc.h>

class ucc_pt_bootstrap {
protected:
    ucc_context_oob_coll_t context_oob;
    ucc_team_oob_coll_t team_oob;
public:
    virtual int get_rank() = 0;
    virtual int get_size() = 0;
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
