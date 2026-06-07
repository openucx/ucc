# Design: UCC MPI + DLRM CI on Slurm

**Date:** 2026-06-04
**Author:** Daniel Pressler
**Status:** Approved (design); pending implementation plan

## Summary

Convert the bare-metal `ucc` CI job — which today builds the `ngc_pytorch`
image and runs the UCC MPI unit-test matrix plus the DLRM/torch_ucc GPU test on
two bare-metal agents (`swx-clx01`, `swx-clx02`) — into a Slurm-based pipeline
modeled on the existing `test_gtest_matrix.yaml` / `test_nvls_matrix.yaml`
pipelines.

The new pipeline is added as a **separate job** (`ucc-test-mpi`) using a new
config file `.ci/pipeline/test_mpi_matrix.yaml`. The existing bare-metal `ucc`
job (`.ci/job_matrix.yaml`) is **left running in parallel** until the Slurm
version is validated in CI, after which the bare-metal flow is retired in a
follow-up.

## Goals

- Run the existing MPI unit-test matrix and the DLRM test under Slurm on the
  `funk` partition (head node `scctl`), with **2 nodes** for genuine multi-node
  coverage (parity with today's 2-node bare-metal run).
- Reuse the proven Slurm plumbing from `test_gtest_matrix.yaml` /
  `test_nvls_matrix.yaml`: `runs_on_dockers` image build, `build_helper` driver
  container, and the `slurmCI` allocate → run → stop lifecycle.
- Keep changes to the existing test *logic* minimal — add new Slurm-native
  wrapper scripts rather than rewriting collective coverage.

## Non-goals

- Retiring the bare-metal `ucc` job / `job_matrix.yaml` (deferred to a follow-up
  once the Slurm job is validated).
- Changing which collectives / transports are exercised (coverage parity is the
  target, not expansion).
- Multi-node coverage beyond 2 nodes.

## Current state (bare-metal)

`.ci/job_matrix.yaml` (`job: ucc`), loaded by `.ci/Jenkinsfile.shlib`, triggered
as a child of the `ucc-ci-dispatcher`:

- `runs_on_dockers`: builds `ngc_pytorch` (centos8, x86_64, CUDA 12.9, from a
  Harbor `:base` image) + `build_helper`.
- `runs_on_agents`: bare-metal `swx-clx01`, `swx-clx02`.
- Steps: pull images on both hosts → `run_docker.sh` (start a detached `sshd`
  container per host, `--gpus all --device=/dev/infiniband`) → MPI tests →
  DLRM tests → `pipeline_stop` runs `clean_docker.sh`.
- The hostfiles (`.ci/configs/swx-clx01/hostfile.txt`,
  `.ci/configs/swx-clx02/hostfile.txt`) list **both** hosts, so the run is
  genuinely 2-node (`NNODES=2`).

Test scripts (bare-metal launch model):

- `run_tests_ucc_mpi_docker.sh` → ssh into the head-node container →
  `run_tests_ucc_mpi.sh <hostfile>`, which drives ~12 `mpirun --hostfile`
  invocations (ssh launcher, `ibstat` device discovery) across configurations:
  default (host,cuda), NCCL, TL/UCP, TL/CUDA, TL/MLX5 (2-node only), and several
  CL/HIER variants + 2-step bcast — the whole matrix run twice (`MT=""` and
  `-T`).
- `run_dlrm_docker.sh` → `run_dlrm.sh gpu <hostfile>` → `mpirun -np <nnodes>
  --map-by node` launching `run_dlrm_s_pytorch.sh` (torch `dlrm_s_pytorch.py`,
  `--dist-backend=ucc`), i.e. one rank per node.

## Target architecture (Slurm)

### 1. Job registration

- Add a `ucc-test-mpi` job-template in `.ci/proj_jjb.yaml` mirroring the
  `ucc-test-gtest` template, with `CONF_FILE` defaulting to
  `.ci/pipeline/test_mpi_matrix.yaml` and `BUILD_DOCKERS=true`.
- Add `"{jjb_proj}-test-mpi"` to the `jobs:` list of the `ucc-build` project.
- Add `"ucc-test-mpi"` to the dispatcher's `childJobNames` and add a parallel
  `build job: 'ucc-test-mpi'` branch in the dispatcher DSL.
- The existing `ucc` job is untouched.

### 2. New pipeline file: `.ci/pipeline/test_mpi_matrix.yaml`

Structured like `test_gtest_matrix.yaml`:

- `job: "ucc-mpi"`.
- `kubernetes`, `registry_*` blocks copied from gtest.
- `pvc_volumes` (`hpcx-pvc` → `/mnt/pvc`) + `empty_volumes` (`/root`).
- `runs_on_dockers`:
  - The `ngc_pytorch` image via `file: .ci/Dockerfile.ngc_pytorch`, tag
    `${DOCKER_IMAGE_TAG}` (= `${BUILD_NUMBER}`), `uri: ${UCC_URI_SUFFIX}`, with
    `--build-arg CUDA_VER`, `_UID=149917 _GID=30
    _LOGIN=svcnbu-swx-hpcx _GROUP=svcnbu-swx-hpcx`, and
    `UCC_ENABLE_GTEST=yes` (so `ucc_test_mpi` is built). The image must also
    carry the torch_ucc + DLRM stack (it does today).
  - The `build_helper` image via `file: .ci/dockerfiles/Dockerfile.build_helper`,
    tag `mpi`.
- `env` (`SLURM_*`) mirroring gtest:
  - `SLURM_PARTITION: funk`, `SLURM_HEAD_NODE: scctl`, `SLURM_NODES: 2`,
    `SLURM_GRES: gpu:<N>` (N = GPUs per funk node, confirmed during impl),
    `SLURM_JOB_TIMEOUT`, `SLURM_IMMEDIATE_TIMEOUT`,
    `SLURM_JOB_NAME/SLURM_CONTAINER_NAME: ${BUILD_TAG}`,
    `JOB_ID_FILE: /mnt/pvc/job-id-${BUILD_TAG}.txt`,
    `SCCTL_CREDENTIALS_ID: svcnbu-swx-hpcx-corporate-user-pass`.
  - `UCC_URI_SUFFIX` for the ngc_pytorch image, `SRC_DIR`, `DOCKER_IMAGE_TAG`,
    per-step `TEST_TIMEOUT_MINUTES`.

### 3. Steps (all `containerSelector: build_helper`, `module: slurmCI`)

| Step | slurmCI | ntasks-per-node | nodes | Notes |
|------|---------|-----------------|-------|-------|
| Allocate Slurm job | `allocation` | — | 2 | `extraArgs: --gres=${SLURM_GRES}` |
| Read Slurm job ID | groovy | — | — | read `JOB_ID_FILE` into `env.SLURM_JOB_ID` |
| MPI tests (bulk) | `run` | 4 | 2 | default, TL/UCP, all CL/HIER variants, 2-step bcast, TL/MLX5 (self-skips without CX7) |
| MPI tests (NCCL) | `run` | `<NGPUS>` | 2 | NCCL block |
| MPI tests (TL/CUDA) | `run` | 4 | 1 | the current `mpi_params $PPN 1` single-node case |
| DLRM test | `run` | 1 | 2 | one rank per node, torch + ucc backend |
| pipeline_stop | `stop` | — | — | deallocate |

Rationale for grouping: under one `srun` step `--ntasks-per-node` is fixed, so
configurations that need a different rank geometry become separate `run` steps
(the same reason `test_nvls_matrix.yaml` splits allreduce / reduce_scatter).
Within a single step, the wrapper script may run many sequential `$EXE`
invocations — each is a fresh collective MPI job over the srun-provided ranks.

Each `run` step passes `dockerImage:
${registry_host}#torch-ucc/${UCC_URI_SUFFIX}:${DOCKER_IMAGE_TAG}`,
`testScript`, `headNode`, `credentialsId`, `containerName`, and the
`--ntasks-per-node` `extraArgs`.

### 4. New Slurm-native test scripts

Add new wrappers (bare-metal scripts stay untouched):

- `.ci/scripts/run_tests_ucc_mpi_slurm.sh` — accepts a "group" argument
  (`bulk` / `nccl` / `tlcuda`) selecting which configurations to run, or three
  thin scripts. Transformation rules from `run_tests_ucc_mpi.sh`:
  - Each `mpirun $(mpi_params ...) <-x VARS> $EXE <args>` becomes
    `export VARS; $EXE <args>` — ranks come from `srun`/PMIx, not `mpirun`.
  - `-c <colls>`, `--mtypes ...`, `-T`, `--triggered`, etc. stay as binary args.
  - IB device discovery (`ibstat` over ssh in the bare-metal script) is replaced
    by per-node discovery guarded on `SLURM_LOCALID==0`, or a fixed
    `UCX_NET_DEVICES` / UCX auto-detection. No ssh, no `DOCKER_SSH_PORT`,
    no hostfile.
- `.ci/scripts/run_dlrm_slurm.sh` — runs `run_dlrm_s_pytorch.sh` directly under
  srun (`--ntasks-per-node=1`), mapping `SLURM_PROCID`/`SLURM_NTASKS`/first node
  → torch `RANK`/`WORLD_SIZE`/`MASTER_ADDR` (+ `MASTER_PORT`), with the same
  `UCC_CLS=basic UCC_CL_BASIC_TLS=nccl,ucp` env the bare-metal path sets.

`run_docker.sh` / `stop_docker.sh` / `clean_docker.sh` and the `_docker.sh`
wrappers are **not** used by the Slurm pipeline (pyxis/enroot provides the
container).

## Risks & validation order

1. **srun/PMIx launch compatibility (make-or-break).** The `ngc_pytorch`
   (centos8 / CUDA 12.9) image's MPI must launch under `srun --mpi=pmix` on
   funk/scctl. nvls proves the pattern with HPC-X on dlcluster, but this image
   on this partition is unproven. **Smoke-test first**: a trivial 2-node
   `srun` of `ucc_test_mpi` (or even `hostname` + `MPI_Init`) before wiring the
   full matrix. If the image's MPI/PMIx is incompatible, options are: install
   HPC-X in the image (as `Dockerfile.nvls` does) or build OMPI `--with-pmix`
   matching the cluster.
2. **GPU arch gencode** vs funk's GPUs (`DEBUG_GTEST_SLURM.md` #6) — may require
   a `UCC_NVCC_GENCODE` build-arg.
3. **GPUs per node on funk** — sets `SLURM_GRES` and the NCCL step's
   `--ntasks-per-node` (`NGPUS`).
4. **CX7 IB presence on funk** — if absent, TL/MLX5 self-skips (acceptable,
   matches existing guard).
5. **DLRM torch distributed init under srun** — env var mapping and that the
   ucc backend initializes with 2 ranks across nodes.

## Validation results (2026-06-07, on funk32 via reservation BF3_tests)

Validated directly on the cluster using image
`harbor.mellanox.com/torch-ucc/ucc/1.0.0/x86_64/centos8/cuda12.9:1460`:

- **Risk #1 (srun/PMIx) — RESOLVED.** The centos8/CUDA-12.9 image ships HPC-X
  OpenMPI (`/opt/hpcx/ompi`, PMIx-capable). `srun --mpi=pmix` launches it in
  pyxis with CUDA working; a 4-rank smoke (barrier+allreduce, host+cuda) passed
  3524/3524.
- **funk topology:** nodes have **1 A100 80GB GPU each**, 64 CPUs → confirms
  `SLURM_GRES=gpu:1` and `MPI_NCCL_PPN=1`. The bulk group runs 4 ranks sharing
  the single GPU for cuda mtype (same geometry the bare-metal job used).
- **bulk default (host,cuda) + TL/CUDA — pass** (3948/0 repeatably).
- **TL/CUDA node prerequisite — `kernel.yama.ptrace_scope=0`.** TL/CUDA uses
  CUDA IPC between the same-GPU ranks; rootless enroot has no effective
  CAP_SYS_PTRACE, so with `ptrace_scope=1` the IPC fails (flaky exit-137 /
  fallback). `--container-remap-root` grants caps but breaks PMIx, so it is NOT
  usable. Fix is node-side: `ptrace_scope=0` (set by cluster admin via sysctl/
  prolog). After that TL/CUDA is deterministic. The leading
  `failed to create tl context for cuda` log line is benign (initial UCC context
  before the test sets a CUDA device). See [[funk-tlcuda-ptrace-scope]].
- **Not yet validated:** multi-node (2-node) PMIx + IB fabric, the NCCL group
  (needs 2 nodes), and DLRM — funk had only the single reserved node `funk32`
  available; the rest of the partition was fully allocated. These remain for the
  real 2-node CI run (Tasks 7, 10).

## Success criteria

- `ucc-test-mpi` runs from the dispatcher, builds both images, allocates a
  2-node funk job, runs all MPI groups + DLRM, and deallocates cleanly.
- The MPI collective/transport coverage matches the bare-metal matrix (modulo
  TL/MLX5 when CX7 is unavailable, which already self-skips).
- The DLRM test completes its 10 batches with `--dist-backend=ucc` across 2
  ranks.
- The bare-metal `ucc` job continues to run unchanged in parallel.

## Follow-ups (out of scope here)

- Retire the bare-metal `ucc` job, `job_matrix.yaml`, and the `_docker.sh` /
  `run_docker.sh` / `clean_docker.sh` scripts once the Slurm job is trusted.
