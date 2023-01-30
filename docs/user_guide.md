# UCC User Guide

This guide describes how to leverage UCC to accelerate collectives within a parallel
programming model, e.g. MPI or OpenSHMEM, contingent on support from the particular
implementation of the programming model. For simplicity, this guide uses Open MPI as
one such example implementation. However, the described concepts are sufficiently
general, so they should transfer to other MPI implementations or programming models
as well.

Note that this is not a guide on how to use the UCC API or contribute to UCC, for
that see the UCC API documentation available [here](https://openucx.github.io/ucc/)
and consider the technical and legal guidelines in the [contributing](../CONTRIBUTING.md)
file.

## Getting Started

Build Open MPI with UCC as described [here](../README.md#open-mpi-and-ucc-collectives).
To check if your Open MPI build supports UCC accelerated collectives, you can check for
the MCA coll `ucc` component:

```
$ ompi_info | grep ucc
                MCA coll: ucc (MCA v2.1.0, API v2.0.0, Component v4.1.4)
```

To execute your MPI program with UCC accelerated collectives, the `ucc` MCA component
needs to be enabled:

```
export OMPI_MCA_coll_ucc_enable=1
```

Currently, it is also required to set 

```
export OMPI_MCA_coll_ucc_priority=100 
```

to work around https://github.com/open-mpi/ompi/issues/9885. 

In most situations, this is all that is needed to leverage UCC accelerated collectives
from your MPI program. UCC heuristics aim to always select the highest performing
implementation for a given collective, and UCC aims to support execution at all scales,
from a single node to a full supercomputer. 

However, because there are many different system setups, collectives, and message sizes,
these heuristics can't be perfect in all cases. The remainder of this User Guide therefore
describes the parts of UCC which are necessary for basic UCC tuning. If manual tuning is
necessary, an issue report is appreciated at
[the Github tracker](https://github.com/openucx/ucc/issues) so that this can be considered
for future tuning of UCC heuristics.

## CLs and TLs

UCC collective implementations are compositions of one or more **T**eam **L**ayers (TLs).
TLs are designed as thin composable abstraction layers with no dependencies between
different TLs. To fulfill semantic requirements of programming models like MPI and because
not all TLs cover the full functionality required by a given collective (e.g. the SHARP TL
does not support intra-node collectives), TLs are composed by
**C**ollective **L**ayers (CLs). The list of CLs and TLs supported by the available UCC
installation can be queried with:

```
$ ucc_info -s
Default CLs scores: basic=10 hier=50
Default TLs scores: cuda=40 nccl=20 self=50 ucp=10
```

This UCC implementations supports two CLs:
- `basic`: Basic CL available for all supported algorithms and good for most use cases.
- `hier`: Hierarchical CL exploiting the hierarchy on a system, e.g. NVLINK within a node
and SHARP for the network. The `hier` CL exposes two hierarchy levels: `NODE` containing
all ranks running on the same node and `NET` containing one rank from each node. In addition
to that, there is the `FULL` subgroup with all ranks. A concrete example of a hierarchical
CL is a pipeline of shared memory UCP reduce with inter-node SHARP and UCP broadcast.
The `basic` CL can leverage the same TLs but would execute in a non-pipelined,
less efficient fashion.

and four TLs:
- `cuda`: TL supporting CUDA device memory exploiting NVLINK connections between GPUs.
- `nccl`: TL leveraging [NCCL](https://github.com/NVIDIA/nccl) for collectives on CUDA
   device memory. In many cases, UCC collectives are directly mapped to NCCL collectives.
   If that is not possible, a combination of NCCL collectives might be used.
- `self`: TL to support collectives with only 1 participant.
- `ucp`: TL building on UCP point to point communication routines from
   [UCX](https://github.com/openucx/ucx). This is the most general TL which supports all
  memory types. If required computation happens local to the memory, e.g. for CUDA device
  memory CUDA kernels are used for computation.

In addition to those TLs supported by the example Open MPI implementation used in this guide,
UCC also supports the following TLs:
- `sharp`: TL leveraging the
  [NVIDIA **S**calable **H**ierarchical **A**ggregation and **R**eduction **P**rotocol (SHARP)™](https://docs.nvidia.com/networking/category/mlnxsharp)
  in-network computing features to accelerate inter-node collectives.
- `rccl`: TL leveraging [RCCL](https://github.com/ROCmSoftwarePlatform/rccl) for collectives
  on ROCm device memory. 

UCC is extensible so vendors can provide additional TLs. For example the UCC binaries shipped
with [HPC-X](https://developer.nvidia.com/networking/hpc-x) add the `shm` TL with optimized
CPU shared memory collectives.

UCC exposes environment variables to tune CL and TL selection and behavior. The list of all
environment variables with a description is available from `ucc_info`:

```
$ ucc_info -caf | head -15
# UCX library configuration file
# Uncomment to modify values

#
# UCC configuration
#

#
# Comma separated list of CL components to be used
#
# syntax:    comma-separated list of: [basic|hier|all]
#
UCC_CLS=basic
```

In this guide we will focus on how TLs are selected based on a score. Every time UCC needs
to select a TL the TL with the highest score is selected considering:

- The collective type
- The message size
- The memory type
- The team size (number of ranks participating in the collective)

A user can set the `UCC_TL_<NAME>_TUNE` environment variables to override the default scores
following this syntax:

```
UCC_TL_<NAME>_TUNE=token1#token2#...#tokenN,
```

Passing a `# ` separated list of tokens to the environment variable. Each token is a `:`
separated list of qualifiers:

```
token=coll_type:msg_range:mem_type:team_size:score:alg
```

Where each qualifier is optional. The only requirement is that either `score` or `alg`
is provided. The qualifiers are

- `coll_type = coll_type_1,coll_type_2,...,coll_type_n` - a `,` separated list of
  collective types.
- `msg_range = m_start_1-m_end_1,m_start_2-m_end_2,..,m_start_n-m_end_n` - a `,`
  separated list of msg ranges in byte, where each range is represented by `start`
  and `end` values separated by `-`. Values can be integers using optional binary
  prefixes. Supported prefixes are `K=1<<10`, `M=1<<20`, `G=1<<30` and, `T=1<<40`.
  Parsing is case indepdent and a `b` can be optionally added. The special value
  `inf` means MAX msg size. E.g. `128`, `256b`, `4K`, `1M` are valid sizes. 
- `mem_type = m1,m2,..,mN` - a `,` separated list of memory types
- `team_size = [t_start_1-t_end_1,t_start_2-t_end_2,...,t_start_N-t_end_N]` - a
  `,` separated list of team size ranges enclosed with `[]`.
- `score =` , a `int` value from `0` to `inf`
- `alg = @<value|str>` - character `@` followed by either the `int` or string
  representing the collective algorithm.

Supported memory types are:
- `cpu`: for CPU memory. 
- `cuda`: for pinned CUDA Device memory (`cudaMalloc`).
- `cuda_managed`: for CUDA Managed Memory (`cudaMallocManaged`).
- `rocm`: for pinned ROCm Device memory.
- `rocm_managed`: for ROCm Managed Memory.

The supported collective types and algorithms can be queried with

```
$ ucc_info -A
cl/hier algorithms:
  Allreduce
    0 :              rab : intra-node reduce, followed by inter-node allreduce, followed by innode broadcast
    1 :       split_rail : intra-node reduce_scatter, followed by PPN concurrent  inter-node allreduces, followed by intra-node allgather
  Alltoall
    0 :       node_split : splitting alltoall into two concurrent a2av calls withing the node and outside of it
  Alltoallv
    0 :       node_split : splitting alltoallv into two concurrent a2av calls withing the node and outside of it
[...] snip
```

See the [FAQ](https://github.com/openucx/ucc/wiki/FAQ#6-what-is-tl-scoring-and-how-to-select-a-certain-tl)
in the [UCC Wiki](https://github.com/openucx/ucc/wiki) for more information and concrete examples.
If for a given combination, multiple TLs have the same highest score, it is implementation-defined
which of those TLs with the highest score is selected.

## Logging

To debug the choices made by UCC heuristics, setting `UCC_LOG_LEVEL=INFO` provides valuable
information. E.g. it prints score map with all collectives, TLs and memory types supported
```
[...] snip
       ucc_team.c:452  UCC  INFO  ===== COLL_SCORE_MAP (team_id 32768) =====
ucc_coll_score_map.c:185  UCC  INFO  Allgather:
ucc_coll_score_map.c:185  UCC  INFO       Host: {0..inf}:TL_UCP:10
ucc_coll_score_map.c:185  UCC  INFO       Cuda: {0..inf}:TL_NCCL:10
ucc_coll_score_map.c:185  UCC  INFO       CudaManaged: {0..inf}:TL_UCP:10
ucc_coll_score_map.c:185  UCC  INFO  Allgatherv:
ucc_coll_score_map.c:185  UCC  INFO       Host: {0..inf}:TL_UCP:10
ucc_coll_score_map.c:185  UCC  INFO       Cuda: {0..16383}:TL_NCCL:10 {16K..1048575}:TL_NCCL:10 {1M..inf}:TL_NCCL:10
ucc_coll_score_map.c:185  UCC  INFO       CudaManaged: {0..inf}:TL_UCP:10
ucc_coll_score_map.c:185  UCC  INFO  Allreduce:
ucc_coll_score_map.c:185  UCC  INFO       Host: {0..4095}:TL_UCP:10 {4K..inf}:TL_UCP:10
ucc_coll_score_map.c:185  UCC  INFO       Cuda: {0..4095}:TL_NCCL:10 {4K..inf}:TL_NCCL:10
ucc_coll_score_map.c:185  UCC  INFO       CudaManaged: {0..4095}:TL_UCP:10 {4K..inf}:TL_UCP:10
ucc_coll_score_map.c:185  UCC  INFO       Rocm: {0..4095}:TL_UCP:10 {4K..inf}:TL_UCP:10
ucc_coll_score_map.c:185  UCC  INFO       RocmManaged: {0..4095}:TL_UCP:10 {4K..inf}:TL_UCP:10
[...] snip
```

## Known Issues

- For the CUDA and NCCL TL CUDA device dependent data structures are created when UCC
  is initialized which usually happens during `MPI_Init`. For these TLs it is therefore
  important that the GPU used by an MPI rank does not change after `MPI_Init` is called.
- UCC does not support CUDA managed memory for all TLs and collectives.
- Logging of collective tasks as described above using NCCL as example is not unified.
  E.g. some TLs do not log when a collective is started and finalized.

## Other useful information

- UCC FAQ: https://github.com/openucx/ucc/wiki/FAQ 
- Output of `ucc_info -caf`
