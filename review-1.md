The new push algorithms can silently produce incorrect alltoall/alltoallv results on supported proxy topologies, and the perftest global memh path has an allocator mismatch on teardown. These are correctness/runtime issues that should be fixed before considering the patch correct.

Full review comments:

- [P1] Reject non-direct topologies in alltoall push - /work/nvidia/curr-work/ucc-cuda-mem-map/src/components/tl/cuda/alltoall/alltoall_push.c:272-273
  On GPU topologies where this rank needs a proxy for any peer, this `continue` lets the push algorithm initialize without mapping that peer; the progress loop applies the same direct-only check and never posts the copy, but the final barrier can still complete with `UCC_OK`. For nonzero alltoall traffic to a non-direct peer, that peer's receive block is left unwritten, so this path should return `UCC_ERR_NOT_SUPPORTED` unless the CUDA topology is fully connected.

- [P1] Reject non-direct topologies in alltoallv push - /work/nvidia/curr-work/ucc-cuda-mem-map/src/components/tl/cuda/alltoallv/alltoallv_push.c:331-333
  When a peer is not directly reachable, this skip allows alltoallv push to initialize and later skip posting sends for that peer, while still completing after the final barrier. For any nonzero send count to a non-direct peer on a proxy topology, the destination segment remains stale, so the push algorithm needs to reject non-fully-connected CUDA topologies instead of silently omitting those peers.

- [P2] Allocate global alltoallv memh blobs with ucc_malloc - /work/nvidia/curr-work/ucc-cuda-mem-map/tools/perf/ucc_pt_coll_alltoallv.cc:108-108
  With `-M global`, each imported handle blob is later passed to `ucc_mem_unmap`, which frees the handle object with `ucc_free` and nulls the pointer. Allocating this storage with `new[]` makes unmap call `free()` on a C++ allocation, and the following `delete[]` is a no-op because the pointer was nulled, so perftest can corrupt the heap on teardown; use `ucc_malloc` for these blobs, including the src path below.
