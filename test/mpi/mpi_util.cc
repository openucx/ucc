/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "mpi_util.h"

static MPI_Comm create_half_comm()
{
    int world_rank, world_size;
    MPI_Comm new_comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < world_size / 2,
                   world_rank, &new_comm);
    return new_comm;
}

static MPI_Comm create_odd_even_comm()
{
    int world_rank;
    MPI_Comm new_comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2,
                   world_rank, &new_comm);
    return new_comm;
}

static MPI_Comm create_reverse_comm()
{
    int world_rank, world_size;
    MPI_Comm new_comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_split(MPI_COMM_WORLD, 0,
                   world_size - world_rank - 1, &new_comm);
    return new_comm;
}

MPI_Comm create_mpi_comm(ucc_test_mpi_team_t t)
{
    MPI_Comm comm = MPI_COMM_NULL;

    switch(t) {
    case TEAM_WORLD:
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        break;
    case TEAM_REVERSE:
        comm = create_reverse_comm();
        break;
    case TEAM_SPLIT_HALF:
        comm = create_half_comm();
        break;
    case TEAM_SPLIT_ODD_EVEN:
        comm = create_odd_even_comm();
        break;
    default:
        break;
    }
    return comm;
}
