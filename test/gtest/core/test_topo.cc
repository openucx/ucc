/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
extern "C" {
#include <core/ucc_context.h>
#include <core/ucc_team.h>
#include <core/ucc_global_opts.h>
}

#include <common/test.h>
#include <vector>
#include <algorithm>
#include <random>

class addr_storage {
  public:
    ucc_addr_storage_t                     storage;
    std::vector<ucc_context_addr_header_t> h;
    addr_storage(int size)
    {
        h.resize(size);
        storage.storage  = h.data();
        storage.size     = size;
        storage.addr_len = sizeof(ucc_context_addr_header_t);
    }
};

class test_topo : public ucc::test {
  public:
    ucc_context_topo_t *ctx_topo;
    ucc_topo_t  *topo;
    test_topo()
    {
        ucc_constructor();
    }
    ~test_topo()
    {
        ucc_topo_cleanup(topo);
        ucc_context_topo_cleanup(ctx_topo);
    }
};

#define SET_PI(_s, _i, _host, _sock, _pid)                                     \
    _s.h[_i].ctx_id.pi.host_hash = _host;                                      \
    _s.h[_i].ctx_id.pi.socket_id = _sock;                                      \
    _s.h[_i].ctx_id.pi.pid       = _pid;

UCC_TEST_F(test_topo, single_node)
{
    const ucc_rank_t ctx_size = 4;
    addr_storage     s(ctx_size);
    ucc_sbgp_t *     sbgp;
    ucc_subset_t     set;

    /* simulates world proc array */
    SET_PI(s, 0, 0xabcd, 0, 0);
    SET_PI(s, 1, 0xabcd, 0, 1);
    SET_PI(s, 2, 0xabcd, 0, 2);
    SET_PI(s, 3, 0xabcd, 0, 3);

    /* subset from the world */
    set.map.ep_num = ctx_size;
    set.myrank     = 0;
    set.map.type   = UCC_EP_MAP_FULL;

    /* Init topo for such subset */
    EXPECT_EQ(UCC_OK, ucc_context_topo_init(&s.storage, &ctx_topo));
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));

    /* Check subgroups */

    /* NODE subgroup  - ALL on the same node*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(sbgp->group_size, ctx_size);
    EXPECT_EQ(sbgp->group_rank, 0);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_FULL);

    /* NODE_LEADERS subgroup */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);
    EXPECT_EQ(UCC_SBGP_NOT_EXISTS, sbgp->status);

    /* SOCKET subgroup - ALL on the same socket */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(sbgp->group_size, ctx_size);
    EXPECT_EQ(sbgp->group_rank, 0);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_FULL);

    /* SOCKET_LEADERS subgroup - just 1 socket - no socket_leaders group*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET_LEADERS);
    EXPECT_EQ(UCC_SBGP_NOT_EXISTS, sbgp->status);
}

UCC_TEST_F(test_topo, node_reordered)
{
    const ucc_rank_t ctx_size              = 4;
    const ucc_rank_t team_size             = 3;
    ucc_rank_t       team_ranks[team_size] = {2, 3, 1};
    addr_storage     s(ctx_size);
    ucc_sbgp_t *     sbgp;
    ucc_subset_t     set;

    /* simulates world proc array */
    SET_PI(s, 0, 0xabcd, 0, 0);
    SET_PI(s, 1, 0xabcd, 0, 1);
    SET_PI(s, 2, 0xabcd, 0, 2);
    SET_PI(s, 3, 0xabcd, 0, 3);

    /* subset from the world */
    set.map.ep_num = team_size;
    set.myrank     = 2; //will build subgroups from rank 2 perspective
    set.map.type   = UCC_EP_MAP_ARRAY;
    set.map.array.map       = team_ranks;
    set.map.array.elem_size = sizeof(ucc_rank_t);

    /* Init topo for such subset */
    EXPECT_EQ(UCC_OK, ucc_context_topo_init(&s.storage, &ctx_topo));
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));

    /* Check subgroups */

    /* NODE subgroup  - ALL on the same node*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(team_size, sbgp->group_size);
    EXPECT_EQ(2, sbgp->group_rank);
}

UCC_TEST_F(test_topo, 1node_2sockets)
{
    const ucc_rank_t ctx_size  = 6;
    const ucc_rank_t team_size = 6;
    addr_storage     s(ctx_size);
    ucc_sbgp_t *     sbgp;
    ucc_subset_t     set;

    /* simulates world proc array */
    SET_PI(s, 0, 0xabcd, 0, 0);
    SET_PI(s, 1, 0xabcd, 1, 1);
    SET_PI(s, 2, 0xabcd, 0, 2);
    SET_PI(s, 3, 0xabcd, 1, 3);
    SET_PI(s, 4, 0xabcd, 0, 4);
    SET_PI(s, 5, 0xabcd, 1, 5);

    /* subset from the world */
    set.map.ep_num = team_size;
    set.myrank     = 3; // from rank 1 perspective
    set.map.type   = UCC_EP_MAP_FULL;

    /* Init topo for such subset */
    EXPECT_EQ(UCC_OK, ucc_context_topo_init(&s.storage, &ctx_topo));
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));

    /* Check subgroups */

    /* NODE subgroup  - ALL on the same node*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(team_size, sbgp->group_size);
    EXPECT_EQ(3, sbgp->group_rank);

    /* SOCKET subgroup - must contain ranks 1, 3, 5 */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(sbgp->group_size, ctx_size / 2);
    EXPECT_EQ(sbgp->group_rank, 1); //rank 3 is rank 1 in subgroup 1, 3, 5
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 1);
    EXPECT_EQ(sbgp->map.strided.stride, 2);

    /* SOCKET_LEADERS subgroup - ranks 0 and 1. Rank 3 does not participate, so
       the SBGP is disabled for him */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET_LEADERS);
    EXPECT_EQ(UCC_SBGP_DISABLED, sbgp->status);

    ucc_topo_cleanup(topo);
    set.myrank = 1;
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));
    /* SOCKET_LEADERS subgroup - ranks 0 and 1. Rank 1 is also rank 1
       in the SBGP*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET_LEADERS);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(sbgp->group_size, 2);
    EXPECT_EQ(sbgp->group_rank, 1);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 0);
    EXPECT_EQ(sbgp->map.strided.stride, 1);
}

UCC_TEST_F(test_topo, 2nodes)
{
    const ucc_rank_t ctx_size  = 8;
    const ucc_rank_t team_size = 8;
    addr_storage     s(ctx_size);
    ucc_sbgp_t *     sbgp;
    ucc_subset_t     set;

    /* simulates world proc array : 2 nodes, 5 ranks on 1st node and
       3 on 2nd*/
    SET_PI(s, 0, 0xaaa, 0, 0);
    SET_PI(s, 1, 0xaaa, 1, 1);
    SET_PI(s, 2, 0xaaa, 0, 2);
    SET_PI(s, 3, 0xaaa, 1, 3);
    SET_PI(s, 4, 0xaaa, 0, 4);

    SET_PI(s, 5, 0xbbb, 0, 5);
    SET_PI(s, 6, 0xbbb, 1, 6);
    SET_PI(s, 7, 0xbbb, 0, 7);


    /* subset from the world */
    set.map.ep_num = team_size;
    set.map.type   = UCC_EP_MAP_FULL;
    set.myrank     = 3; // from rank 1 perspective

    /* Init topo for such subset */
    EXPECT_EQ(UCC_OK, ucc_context_topo_init(&s.storage, &ctx_topo));
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));

    /* NODE subgroup  - ALL on the same node*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(5, sbgp->group_size);
    EXPECT_EQ(3, sbgp->group_rank);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 0);
    EXPECT_EQ(sbgp->map.strided.stride, 1);

    /* SOCKET subgroup - must contain ranks 1, 3, 5 */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(sbgp->group_size, 2);
    EXPECT_EQ(sbgp->group_rank, 1); //rank 3 is rank 1 in subgroup 1, 3, 5
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 1);
    EXPECT_EQ(sbgp->map.strided.stride, 2);

    /* SOCKET_LEADERS subgroup - ranks 0 and 1. Rank 3 does not participate, so
       the SBGP is disabled for him */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET_LEADERS);
    EXPECT_EQ(UCC_SBGP_DISABLED, sbgp->status);

    /* NODE LEADERS subgroup - ranks 0 and 5. Rank 3 does not participate, so
       the SBGP is disabled for him */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);
    EXPECT_EQ(UCC_SBGP_DISABLED, sbgp->status);

    /* NET subgroup - there is no process with local rank 3 on 2nd node */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NET);
    EXPECT_EQ(UCC_SBGP_NOT_EXISTS, sbgp->status);

    /* RANK 6 perspective */
    ucc_topo_cleanup(topo);
    set.myrank = 6;
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));
    /* NODE subgroup  - ALL on the same node*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(3, sbgp->group_size);
    EXPECT_EQ(1, sbgp->group_rank);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 5);
    EXPECT_EQ(sbgp->map.strided.stride, 1);

    /* SOCKET subgroup - has only 1 rank on socket 1, so SBGP NOT EXISTS */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET);
    EXPECT_EQ(UCC_SBGP_NOT_EXISTS, sbgp->status);

    /* SOCKET_LEADERS subgroup - ranks 5 and 6*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET_LEADERS);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(2, sbgp->group_size);
    EXPECT_EQ(1, sbgp->group_rank);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 5);
    EXPECT_EQ(sbgp->map.strided.stride, 1);

    /* NODE LEADERS subgroup - ranks 0 and 5. Rank 6 does not participate, so
       the SBGP is disabled for him */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);
    EXPECT_EQ(UCC_SBGP_DISABLED, sbgp->status);

    /* NET subgroup - ranks 1 and 6 (local ranks 0 on nodes) */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NET);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(2, sbgp->group_size);
    EXPECT_EQ(1, sbgp->group_rank);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 1);
    EXPECT_EQ(sbgp->map.strided.stride, 5);
}

UCC_TEST_F(test_topo, 4nodes_half)
{
    const ucc_rank_t ctx_size  = 8;
    addr_storage     s(ctx_size);
    ucc_sbgp_t *     sbgp;
    ucc_subset_t     set;

    /* simulates world proc array : 4 nodes, 2  ranks per node*/
    SET_PI(s, 0, 0xaaa, 0, 0);
    SET_PI(s, 1, 0xaaa, 1, 1);
    SET_PI(s, 2, 0xbbb, 0, 2);
    SET_PI(s, 3, 0xbbb, 1, 3);
    SET_PI(s, 4, 0xccc, 0, 4);
    SET_PI(s, 5, 0xccc, 1, 5);
    SET_PI(s, 6, 0xddd, 0, 6);
    SET_PI(s, 7, 0xddd, 1, 7);

    /* subset from the world */
    set.map.ep_num = 4;
    set.map.type   = UCC_EP_MAP_STRIDED;
    set.map.strided.start = 0;
    set.map.strided.stride = 1;

    set.myrank = 1; // from rank 1 perspective
    /* Init topo for such subset */
    EXPECT_EQ(UCC_OK, ucc_context_topo_init(&s.storage, &ctx_topo));
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));

    /* NODE subgroup  - 2 ranks on the same node*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(2, sbgp->group_size);
    EXPECT_EQ(1, sbgp->group_rank);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 0);
    EXPECT_EQ(sbgp->map.strided.stride, 1);

    /* NODE LEADERS subgroup - ranks 0 and 2. Rank 1 does not participate, so
       the SBGP is disabled for him */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);
    EXPECT_EQ(UCC_SBGP_DISABLED, sbgp->status);

    /* RANK 2 perspective */
    ucc_topo_cleanup(topo);
    set.myrank = 2;
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE_LEADERS);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(2, sbgp->group_size);
    EXPECT_EQ(1, sbgp->group_rank);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 0);
    EXPECT_EQ(sbgp->map.strided.stride, 2);
}

UCC_TEST_F(test_topo, 4sockets_half)
{
    const ucc_rank_t ctx_size  = 8;
    addr_storage     s(ctx_size);
    ucc_sbgp_t *     sbgp;
    ucc_subset_t     set;

    /* simulates world proc array : 4 sockets, 2  ranks per socket*/
    SET_PI(s, 0, 0xaaa, 0, 0);
    SET_PI(s, 1, 0xaaa, 0, 1);
    SET_PI(s, 2, 0xaaa, 2, 2);
    SET_PI(s, 3, 0xaaa, 2, 3);
    SET_PI(s, 4, 0xaaa, 3, 4);
    SET_PI(s, 5, 0xaaa, 3, 5);
    SET_PI(s, 6, 0xaaa, 4, 6);
    SET_PI(s, 7, 0xaaa, 4, 7);


    /* team from the world */
    set.map.ep_num = 4;
    set.map.type   = UCC_EP_MAP_STRIDED;
    set.map.strided.start = 0;
    set.map.strided.stride = 1;

    set.myrank = 1; // from rank 1 perspective
    /* Init topo for such team */
    EXPECT_EQ(UCC_OK, ucc_context_topo_init(&s.storage, &ctx_topo));
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));

    /* SOCKET subgroup  - 2 ranks on the same socket*/
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(2, sbgp->group_size);
    EXPECT_EQ(1, sbgp->group_rank);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 0);
    EXPECT_EQ(sbgp->map.strided.stride, 1);

    /* SOCKET LEADERS subgroup - ranks 0 and 2. Rank 1 does not participate, so
       the SBGP is disabled for him */
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET_LEADERS);
    EXPECT_EQ(UCC_SBGP_DISABLED, sbgp->status);

    /* RANK 2 perspective */
    ucc_topo_cleanup(topo);
    set.myrank = 2;
    EXPECT_EQ(UCC_OK, ucc_topo_init(set, ctx_topo, &topo));
    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_SOCKET_LEADERS);
    EXPECT_EQ(UCC_SBGP_ENABLED, sbgp->status);
    EXPECT_EQ(2, sbgp->group_size);
    EXPECT_EQ(1, sbgp->group_rank);
    EXPECT_EQ(sbgp->map.type, UCC_EP_MAP_STRIDED);
    EXPECT_EQ(sbgp->map.strided.start, 0);
    EXPECT_EQ(sbgp->map.strided.stride, 2);
}
