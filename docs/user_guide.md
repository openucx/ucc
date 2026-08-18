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

Also a MPI or other programming model implementation might need to execute a collective not
supported by UCC, e.g. because the datatype or reduction operator support is lacking. In
these cases the implementation can't call into UCC but need to leverage another collective
implementation. See [Logging](#logging) for an example how to detect this in case of Open MPI.
If UCC support is missing an issue report describing the use case is appreciated at
[the Github tracker](https://github.com/openucx/ucc/issues) so that this can be considered
for future UCC development.

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

The supported collective types and algorithms can be queried with

```
$ ucc_info -A
cl/hier algorithms:
  Allreduce
    0 :              rab : intra-node reduce, followed by inter-node allreduce, followed by innode broadcast
    1 :       split_rail : intra-node reduce_scatter, followed by PPN concurrent  inter-node allreduces, followed by intra-node allgather
  Alltoall
    0 :       node_split : splitting alltoall into two concurrent a2av calls within the node and outside of it
  Alltoallv
    0 :       node_split : splitting alltoallv into two concurrent a2av calls within the node and outside of it
[...] snip
```

See the [FAQ](https://github.com/openucx/ucc/wiki/FAQ#6-what-is-tl-scoring-and-how-to-select-a-certain-tl)
in the [UCC Wiki](https://github.com/openucx/ucc/wiki) for more information and concrete examples.
If for a given combination, multiple TLs have the same highest score, it is implementation-defined
which of those TLs with the highest score is selected.

Tuning UCC heuristics is also possible with the UCC configuration file (`ucc.conf`). This file provides a unified way of tailoring the behavior of UCC components - CLs, TLs, and ECs. It can contain any UCC variables of the format `VAR = VALUE`, e.g. `UCC_TL_NCCL_TUNE=allreduce:cuda:inf#alltoall:0` to force NCCL allreduce for "cuda" buffers and disable NCCL for alltoall. See [`contrib/ucc.conf`](../contrib/ucc.conf) for an example and the [FAQ](https://github.com/openucx/ucc/wiki/FAQ#13-ucc-configuration-file-and-priority) for further details.

## Team (communicator) caching

Building a UCC team (collective communicator structure) is an expensive operation involving out-of-band communication, unique ID allocation, and per-component initialization. UCC can transparently cache and reuse teams built for identical-membership communicator groups, avoiding repeated reconstruction for the same process sets.

### When caching helps

Caching accelerates **create → use → free → recreate-identical** workloads via two
complementary mechanisms:

1. **Dormant reuse** — a freed team is retained and re-adopted when an
   identical-membership communicator is recreated:
   - An MPI program repeatedly creates and destroys communicators with the same rank membership (e.g., iterative solvers splitting a rank subset each iteration).
   - Repeated `MPI_Comm_split` or `MPI_Comm_create` with stable rank sets or colors (common in ScaLAPACK-style process-grid splitting and domain decomposition).

2. **Derived teams (simultaneous coexistence)** — when a second identical-membership
   communicator is created while the first is *still live* (e.g., `MPI_Comm_dup` of a
   live communicator), UCC builds a lightweight **derived team** that shares the
   parent's immutable membership and topology while getting its **own** team id (its
   own tag/sequence domain, so the two live teams never alias). This skips the
   address exchange and topology build — the bulk of team-creation cost.

Both mechanisms are transparent and work for externally-provided team ids (e.g. an
MPI communicator's context id): dormant reuse re-adopts only when membership **and**
id match, while derived-create keys coexistence on membership alone (a dup legitimately
carries a different id than its live parent).

**Order-dependence note:** teams are keyed by their materialized membership — the rank-to-endpoint mapping. If the same process set is reordered (a different `MPI_Comm_create` with a different `ranks` argument order), it is treated as a *different* team, so no reuse occurs. Only identical membership in identical rank order is eligible for reuse.

### Configuration reference

UCC team caching is controlled via environment variables (`UCC_*`) or the `ucc.conf` configuration file. It is **experimental and disabled by default** (`UCC_TEAM_CACHE_ENABLE=n`). Use `ucc_info -c` to list the current settings:

```text
$ ucc_info -c | grep -i team_cache
UCC_TEAM_CACHE_ENABLE: n
UCC_TEAM_CACHE_AGREEMENT: y
UCC_TEAM_CACHE_MAX_SIZE: 128
UCC_TEAM_CACHE_EVICTION: fifo
UCC_TEAM_CACHE_DISABLE_LINEAR_CHECK: n
UCC_TEAM_CACHE_DUMP_STATS: n
UCC_TEAM_CACHE_DERIVED: y
UCC_TEAM_CACHE_RESEAT: n
```

| Knob | Type | Default | Description |
|------|------|---------|-------------|
| `UCC_TEAM_CACHE_ENABLE` | Boolean | `n` | Enable team caching. When disabled, each communicator creation rebuilds its team from scratch. Experimental. |
| `UCC_TEAM_CACHE_AGREEMENT` | Boolean | `y` | Run a cross-rank agreement on every cacheable create so all members reach an identical reuse-vs-fresh decision, making reuse safe for overlapping subcommunicators (adds one small member-scoped allreduce per create). Set to `n` only when communicator scopes never overlap. |
| `UCC_TEAM_CACHE_MAX_SIZE` | Unsigned integer | `128` | Maximum number of teams (live plus dormant) tracked by the cache. Each holds a unique team ID, so the effective limit is also clamped by `UCC_TEAM_IDS_POOL_SIZE` (see [Team-ID pool and cache sizing](#team-id-pool-and-cache-sizing)). |
| `UCC_TEAM_CACHE_EVICTION` | Enum: `fifo`, `lfu`, `lru`, `none` | `fifo` | Eviction policy at capacity (only dormant teams are evictable): `none` never evicts; `fifo` evicts the oldest dormant team; `lfu` evicts the least-used (fewest collectives, by `seq_num`); `lru` is an alias for `lfu` (UCC has no wall-clock recency). |
| `UCC_TEAM_CACHE_DISABLE_LINEAR_CHECK` | Boolean | `n` | Trust the 64-bit membership hash alone and skip the exact rank-array compare after a hash match. Faster, but a hash collision would reuse the wrong team. Enable only if collisions cannot occur. |
| `UCC_TEAM_CACHE_DUMP_STATS` | Boolean | `n` | Log cache statistics (lookups, hits and hit rate, misses, inserts, evictions) at context destruction. |
| `UCC_TEAM_CACHE_DERIVED` | Boolean | `y` | Cache and reuse derived teams: when a create duplicates the membership of a still-live team (e.g. `MPI_Comm_dup`), build it with its own team ID but borrow the parent's shared membership/topology artifacts. No effect unless `UCC_TEAM_CACHE_ENABLE=y`. |
| `UCC_TEAM_CACHE_RESEAT` | Boolean | `n` | Experimental: recover reuse under context-id drift by re-adopting a cached dormant derived team of identical membership but a different external id, re-seating its id/tag domain instead of rebuilding. Requires `UCC_TEAM_CACHE_DERIVED=y`. |
| `UCC_TEAM_IDS_POOL_SIZE` | Unsigned integer | `32` | Size of the team-ID pool in 64-ID blocks; total unique IDs per context is `pool_size × 64` (32 → 2048). See [Team-ID pool and cache sizing](#team-id-pool-and-cache-sizing). |

### Team-ID pool and cache sizing

Dormant cached teams retain their unique team IDs, so the team-ID pool (`UCC_TEAM_IDS_POOL_SIZE`, default 32 blocks × 64 = 2048 IDs) must accommodate both live and cached dormant teams. If the cache grows too large it risks pool exhaustion, failing team creation with `UCC_ERR_NO_RESOURCE`. To guard against this, `UCC_TEAM_CACHE_MAX_SIZE` is automatically clamped at runtime to stay safely below the pool size, reserving headroom for live and in-flight teams.

If your workload needs a large cache, raise `UCC_TEAM_IDS_POOL_SIZE` to match. The clamp reserves a small ID headroom for live/in-flight teams, so a pool of `N × 64` IDs caches somewhat fewer than `N × 64` dormant teams — size the pool with margin (e.g. `UCC_TEAM_IDS_POOL_SIZE=5` comfortably caches 256 teams) rather than exactly. Monitor with `UCC_TEAM_CACHE_DUMP_STATS`: high eviction counts mean the cache is thrashing and the pool is undersized. Verify both values with `ucc_info -c`.

### Reading cache statistics

Set `UCC_TEAM_CACHE_DUMP_STATS=y` to log per-context cache statistics at context destruction. Each context (typically one per MPI process) logs a line like:

```text
team_cache stats: lookups=1000 hits=950 (95.0%) misses=50 inserts=50 evictions=0
```

`hits` counts reuses of a cached dormant team (derived-team creates are new teams sharing a parent's artifacts, not hits). If the hit rate is low and evictions are high, raise `UCC_TEAM_CACHE_MAX_SIZE` (and `UCC_TEAM_IDS_POOL_SIZE` if the cache is pool-clamped — see [Team-ID pool and cache sizing](#team-id-pool-and-cache-sizing)).

### Configuration example

To enable aggressive caching with increased pool size and detailed statistics:

```bash
export UCC_TEAM_CACHE_ENABLE=y
export UCC_TEAM_CACHE_MAX_SIZE=256
export UCC_TEAM_CACHE_EVICTION=lfu
export UCC_TEAM_IDS_POOL_SIZE=5
export UCC_TEAM_CACHE_DUMP_STATS=y
```

Or in `ucc.conf`:

```ini
UCC_TEAM_CACHE_ENABLE=y
UCC_TEAM_CACHE_MAX_SIZE=256
UCC_TEAM_CACHE_EVICTION=lfu
UCC_TEAM_IDS_POOL_SIZE=5
UCC_TEAM_CACHE_DUMP_STATS=y
```

## Logging

To detect if Open MPI leverages UCC for a given collective one can set `OMPI_MCA_coll_ucc_verbose=3` checking for output like

```
coll_ucc_alltoall.c:70 - mca_coll_ucc_alltoall() running ucc alltoall
```

For example Open MPI leverages UCC for `MPI_Alltoall` used by `osu_alltoall` from the
[OSU Microbenchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) as can be seen by the log message

```
coll_ucc_alltoall.c:70 - mca_coll_ucc_alltoall() running ucc alltoall
```

in the output of

```
$ OMPI_MCA_coll_ucc_verbose=3 srun ./c/mpi/collective/osu_alltoall -i 1 -x 0 -d cuda -m 1048576:1048576

[...] snip

# OSU MPI-CUDA All-to-All Personalized Exchange Latency Test v7.0
# Size       Avg Latency(us)
 coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
 coll_ucc_alltoall.c:70 - mca_coll_ucc_alltoall() running ucc alltoall
 coll_ucc_alltoall.c:70 - mca_coll_ucc_alltoall() running ucc alltoall
 coll_ucc_alltoall.c:70 - mca_coll_ucc_alltoall() running ucc alltoall
 coll_ucc_alltoall.c:70 - mca_coll_ucc_alltoall() running ucc alltoall
 coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
 coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
 coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
 coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
 coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
1048576           2586333.78

[...] snip

```

For `MPI_Alltoallw` Open MPI can't leverage UCC so the output of `osu_alltoallw` looks different.
It only contains the log messages for the barriers needed for correct timing and reduces needed
to calculate timing statistics:

```
$ OMPI_MCA_coll_ucc_verbose=3 srun ./c/mpi/collective/osu_alltoallw -i 1 -x 0 -d cuda -m 1048576:1048576

[...] snip

# OSU MPI-CUDA All-to-Allw Personalized Exchange Latency Test v7.0
# Size       Avg Latency(us)
coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_barrier.c:31 - mca_coll_ucc_barrier() running ucc barrier
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
coll_ucc_reduce.c:70 - mca_coll_ucc_reduce() running ucc reduce
1048576             11434.97

[...] snip

```

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

UCC 1.2.0 or newer supports the `UCC_COLL_TRACE` environment variable:

```
$ ucc_info -caf | grep -B6 UCC_COLL_TRACE
#
# UCC collective logging level. Higher level will result in more verbose collective info.
#  Possible values are: fatal, error, warn, info, debug, trace, data, func, poll.
#
# syntax:    [FATAL|ERROR|WARN|DIAG|INFO|DEBUG|TRACE|REQ|DATA|ASYNC|FUNC|POLL]
#
UCC_COLL_TRACE=WARN
```

With `UCC_COLL_TRACE=INFO` UCC reports for every collective which CL and TL has been selected:

```
$ UCC_COLL_TRACE=INFO srun ./c/mpi/collective/osu_allreduce -i 1 -x 0 -d cuda -m 1048576:1048576

# OSU MPI-CUDA Allreduce Latency Test v7.0
# Size       Avg Latency(us)
[1678205653.808236] [node_name:903  :0]        ucc_coll.c:255  UCC_COLL INFO  coll_init: Barrier; CL_BASIC {TL_UCP}, team_id 32768
[1678205653.809882] [node_name:903  :0]        ucc_coll.c:255  UCC_COLL INFO  coll_init: Allreduce sum: src={0x7fc1f3a03800, 262144, float32, Cuda}, dst={0x7fc195800000, 262144, float32, Cuda}; CL_BASIC {TL_NCCL}, team_id 32768
[1678205653.810344] [node_name:903  :0]        ucc_coll.c:255  UCC_COLL INFO  coll_init: Barrier; CL_BASIC {TL_UCP}, team_id 32768
[1678205653.810582] [node_name:903  :0]        ucc_coll.c:255  UCC_COLL INFO  coll_init: Reduce min root 0: src={0x7ffef34d5898, 1, float64, Host}, dst={0x7ffef34d58b0, 1, float64, Host}; CL_BASIC {TL_UCP}, team_id 32768
[1678205653.810641] [node_name:903  :0]        ucc_coll.c:255  UCC_COLL INFO  coll_init: Reduce max root 0: src={0x7ffef34d5898, 1, float64, Host}, dst={0x7ffef34d58a8, 1, float64, Host}; CL_BASIC {TL_UCP}, team_id 32768
[1678205653.810651] [node_name:903  :0]        ucc_coll.c:255  UCC_COLL INFO  coll_init: Reduce sum root 0: src={0x7ffef34d5898, 1, float64, Host}, dst={0x7ffef34d58a0, 1, float64, Host}; CL_BASIC {TL_UCP}, team_id 32768
1048576               582.07
[1678205653.810705] [node_name:903  :0]        ucc_coll.c:255  UCC_COLL INFO  coll_init: Barrier; CL_BASIC {TL_UCP}, team_id 32768
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
