# Implementation Plan: src/dst memh support in the UCC MPI tests

## Goal

Extend the UCC MPI test suite (`test/mpi/`) so that collectives can run with
user-registered source/destination memory handles (`memh`), covering both the
**local** handle path (`src_memh.local_memh` / `dst_memh.local_memh`) and the
**global** handle-array path (`src_memh.global_memh` / `dst_memh.global_memh`
with `UCC_COLL_ARGS_FLAG_SRC/DST_MEMH_GLOBAL`).

## Current state (baseline)

- **Infrastructure already present** (no scaffolding needed):
  - `TestCaseParams.local_registration` (`test_mpi.h`)
  - `TestCase` members `src_memh`, `dst_memh`, `src_memh_size`, `dst_memh_size`,
    `local_registration` (`test_mpi.h:271-276`)
  - `TestCase::~TestCase` already unmaps both handles (`test_case.cc:235-240`)
  - CLI option `--local_reg <0|1|2>` -> `local_reg` vector, looped in
    `main.cc:679` and applied via `set_local_registration` (`main.cc:683`)
  - Params propagation: `params.local_registration = local_registration`
    (`test_mpi.cc:554`)
- **Collective coverage today:** only `allgather` wires memh into a real
  collective (`test_allgather.cc:46-71`), local path only.
- **Standalone coverage:** `test_mem_map.cc` exercises `ucc_mem_map`/
  `ucc_mem_unmap` directly (export/import/stress/multi-size) using barrier as a
  placeholder; it does not feed handles into collectives.
- **Global path:** unused anywhere in the codebase — no precedent.

## API reference (`src/ucc/api/ucc.h`)

- `ucc_coll_args_t.src_memh` / `.dst_memh` are unions of `local_memh` and
  `global_memh` (`ucc.h:1895-1914`).
- Mask bits: `UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH` (bit 5),
  `UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH` (bit 6) (`ucc.h:1817-1818`).
- Flags: `UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL` (bit 8),
  `UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL` (bit 9) (`ucc.h:1719-1726`).
- Map/unmap: `ucc_mem_map(ctx, mode, params, &memh_size, &memh)` /
  `ucc_mem_unmap(&memh)` (`ucc.h:2304-2320`). `memh_size` is the serialized
  handle length when exported — this is the payload to exchange for the global
  path.

## Design

### 1. Base-class helper (`test_case.cc` / `test_mpi.h`)

Add a single reusable method so each collective registers with one call and the
sizing logic lives with the collective (buffer sizes differ per collective).

```c
// test_mpi.h (in class TestCase, protected)
ucc_status_t register_memhs(void *sbuf, size_t ssize,
                            void *dbuf, size_t dsize,
                            bool global);
```

Behavior:
- No-op unless `local_registration` is set.
- For each non-NULL buffer, `ucc_mem_map(team.ctx, UCC_MEM_MAP_MODE_EXPORT,
  ...)` into `src_memh`/`dst_memh` (stored on the base class, so the existing
  dtor unmap keeps working).
- **Local mode:** set `args.src_memh.local_memh` / `args.dst_memh.local_memh`
  and OR in the `UCC_COLL_ARGS_FIELD_MEM_MAP_SRC/DST_MEMH` mask bits.
- **Global mode:** see section 2.
- Skip src when inplace (caller passes `sbuf == NULL`).

### 2. Global handle-array path

New, no precedent — prototype on **one** collective first (allgather or
alltoall) and confirm the TL actually consumes the global array before rolling
out.

Per registered buffer:
1. Export the local handle (as above); `memh_size` gives the serialized length.
2. `MPI_Allgather` the exported handle blob across the team comm (may require an
   `MPI_Allgather` of sizes first if `memh_size` is not uniform, then
   `MPI_Allgatherv`).
3. Reconstruct an array of `ucc_mem_map_mem_h` of length `team_size` (import
   remote handles as needed via `UCC_MEM_MAP_MODE_IMPORT`).
4. Assign the array to `args.src_memh.global_memh` / `.dst_memh.global_memh`,
   OR in the field mask bit **and** `UCC_COLL_ARGS_FLAG_SRC/DST_MEMH_GLOBAL`.
5. Track the array + imported handles for cleanup (extend dtor / helper state).

Open questions to resolve during prototype:
- Does the target TL expect imported remote handles, or raw exported blobs, in
  `global_memh`? Confirm against the TL/CUDA mem-map consumer on this branch.
- Ownership/lifetime of imported handles vs. the exported local handle.

### 3. Per-collective call sites

In each `test_<coll>.cc` ctor, after `args.src/dst.info` are populated and
before `ucc_collective_init`, call `register_memhs(...)` with that collective's
buffer sizing. Notable sizings:
- `allgather`: dst = `count * size`; src = `count` (skip if inplace).
- `alltoall`: src + dst = `count * size`.
- `allreduce`/`reduce`/`bcast`: src + dst = `count` (reduce dst root-only).
- `reduce_scatter`: dst = `count`, src = `count * size`.
- `*v` variants: size from `info_v` per-rank counts/displacements.
- `barrier`: no buffers — excluded.

Migrate the existing inline block in `test_allgather.cc:46-71` to the helper as
the reference conversion.

### 4. CLI / driver

- Extend `--local_reg` handling in `main.cc` (`process_local_reg`,
  `main.cc:368`) to select local vs. global vs. both, threading a mode (not just
  a bool) through `TestCaseParams` and the `for (auto lr : local_reg)` loop
  (`main.cc:679`). Keep `0/1/2` backward compatible; add a value (e.g. `3`) or a
  separate `--memh_global` flag for the global path.
- Update the usage string (`main.cc:112`).

## Phasing

1. Add `register_memhs` helper (local path only); convert `allgather` to use it.
   Verify parity with current behavior.
2. Roll the helper out to all applicable collectives (local path).
3. Prototype the global path on one collective; resolve the TL-consumption open
   questions.
4. Generalize global-path cleanup and roll out; wire the CLI mode selector.

## Testing / verification

- Build on ryzen (HPC-X) and/or with CUDA per the branch's mem-map target.
- Run with `--local_reg 2` (both) across teams/msgsizes; confirm `check()`
  passes and no `ucc_mem_unmap` leaks.
- Add the global mode once implemented; compare results against the non-memh
  baseline for the same collective/params.

## Files touched

- `test/mpi/test_mpi.h` — helper decl, any new params/state.
- `test/mpi/test_case.cc` — `register_memhs`, cleanup extensions.
- `test/mpi/test_<coll>.cc` — call sites (allgather first, then the rest).
- `test/mpi/main.cc` — CLI mode selector + usage text.
- `test/mpi/test_mpi.cc` — params propagation if the mode replaces the bool.
