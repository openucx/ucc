# Checkpoint: src/dst memh support in UCC MPI tests

Branch: `topic/tlcuda-mem-map`  ·  Date: 2026-07-16
Scope: `test/mpi/` (bench/ ignored)

## Where we are

Implemented the **base-class-helper** approach: a new
`TestCase::register_memhs(sbuf, ssize, dbuf, dsize)` that, when
`local_registration` is set, maps each non-NULL buffer via
`ucc_mem_map(EXPORT)`, stores handles in the base-class `src_memh`/`dst_memh`,
and sets `args.*_memh.local_memh` + the `UCC_COLL_ARGS_FIELD_MEM_MAP_SRC/DST_MEMH`
mask bits. Cleanup rides on the existing `~TestCase` dtor.

Only the **local** handle path is done. The **global** handle-array path
(`global_memh` + `UCC_COLL_ARGS_FLAG_SRC/DST_MEMH_GLOBAL`) is still TODO.

Full design/plan lives in `MEMH_TEST_PLAN.md` (same dir).

## Working-tree changes (uncommitted)

Modified (14): `test_case.cc`, `test_mpi.h`, and call sites in
`test_allgather.cc`, `test_allgatherv.cc`, `test_allreduce.cc`,
`test_alltoall.cc`, `test_bcast.cc`, `test_gather.cc`, `test_gatherv.cc`,
`test_reduce.cc`, `test_reduce_scatter.cc`, `test_reduce_scatterv.cc`,
`test_scatter.cc`, `test_scatterv.cc`.
Untracked: `MEMH_TEST_PLAN.md`, `checkpoint.md`, `bench/`.

- `test_case.cc`: added `register_memhs()` (gated on `local_registration`;
  maps src if `sbuf && ssize>0`, dst if `dbuf && dsize>0`).
- `test_mpi.h`: declared `register_memhs` in `protected:`.
- `test_allgather.cc`: replaced the old inline `if (local_registration){...}`
  block with a `register_memhs(...)` call (reference conversion).
- Other 12: added one `register_memhs(...)` call each before
  `ucc_collective_init`.

Note: `test_alltoallv.cc` was NOT touched (only buffered collective without a
call; barrier correctly excluded).

## Review findings

### Real bugs — need fixing (both unambiguous)

1. **`test_reduce_scatter.cc` dst over-registration (non-inplace).**
   Registers `msgsize`, but non-inplace `rbuf` is only `msgsize/comm_size`
   (alloc at `test_reduce_scatter.cc:41`). Maps comm_size× past the buffer.
   Fix: `rbuf, inplace ? msgsize : msgsize / comm_size`.

2. **`test_reduce_scatterv.cc` dst under-registration (inplace).**
   Registers `counts[rank]*dt_size`, but inplace `rbuf` is full `msgsize`
   (alloc at `test_reduce_scatterv.cc:52`); reduction input spans whole buffer.
   Fix: `rbuf, inplace ? msgsize : (size_t)counts[rank] * dt_size`.

   (The two files have inplace/non-inplace dst sizing inverted relative to each
   other. The other 10 call sites verified correct.)

### Minor / decisions pending

3. **`test_mpi.h` stray indentation** — `class TestCase {` became
   `    class TestCase {`. Cosmetic; revert.
4. **Abort vs skip on unsupported mem_map** — `register_memhs` uses `UCC_CHECK`,
   which `MPI_Abort`s if the TL lacks `ucc_mem_map`. Now hits all 13 collectives
   under `--local_reg 1` instead of skipping. Consider mapping
   `UCC_ERR_NOT_SUPPORTED/NOT_IMPLEMENTED -> test_skip = TEST_SKIP_NOT_SUPPORTED`
   (like `test_mem_map.cc`).
5. **`test_alltoallv.cc` not covered** — intentional gap or oversight?
6. **Global memh path** — not implemented (local-only). Still TODO per plan.
7. **onesided + local_registration double-map** — `test_alltoall.cc` calls
   `register_memhs` gated only on `local_registration`; in the onesided path
   sbuf/rbuf are already mapped context segments. Consider guarding
   `!is_onesided`. Low-probability combo.

## Decisions to make (next session)

- [ ] Apply fixes 1 + 2 (dst sizing) and 3 (whitespace revert)? — recommended yes.
- [ ] Decide on #4 abort-vs-skip behavior.
- [ ] Cover `test_alltoallv.cc` (#5)?
- [ ] Guard onesided (#7)?
- [ ] Schedule the global-memh path (#6): prototype on one collective
      (allgather/alltoall), resolve TL-consumption question (does the TL expect
      imported remote handles or raw exported blobs in `global_memh`?), then
      wire `--local_reg` mode selector.
- [ ] Build/run verification: ryzen (HPC-X) and/or CUDA target; run
      `--local_reg 2` across teams/msgsizes; confirm `check()` passes, no leaks.

## Key references

- API: `src/ucc/api/ucc.h` — `ucc_coll_args_t.src_memh/.dst_memh` (1895-1914),
  field mask bits (1817-1818), global flags (1719-1726),
  `ucc_mem_map`/`ucc_mem_unmap` (2304-2320).
- Driver: `main.cc` `--local_reg` (`process_local_reg` ~368, loop ~679);
  `test_mpi.cc:554` params propagation.
- Standalone map tests: `test_mem_map.cc` (export/import/stress/multi-size).
