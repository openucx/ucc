# UCC MPI + DLRM Slurm Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new Slurm-based `ucc-test-mpi` CI job that builds the `ngc_pytorch` image and runs the UCC MPI unit-test matrix + DLRM test on a 2-node `funk` allocation, modeled on `test_gtest_matrix.yaml`, leaving the bare-metal `ucc` job intact.

**Architecture:** A new pipeline config `.ci/pipeline/test_mpi_matrix.yaml` uses `runs_on_dockers` to build the image, then `slurmCI` allocate → run → stop steps. The MPI matrix is split into geometry groups (smoke/bulk/nccl/tlcuda) implemented as thin wrapper scripts over a shared `mpi_slurm_common.sh`; each runs under `srun --ntasks-per-node=N` (ranks via PMIx, test binary invoked directly — no `mpirun`/ssh/hostfile). DLRM runs one rank per node under srun with Slurm→torch env mapping. A new `ucc-test-mpi` job-template + dispatcher entry registers it.

**Tech Stack:** Blossom/Jenkins pipeline YAML, `slurmCI` module, pyxis/enroot, Slurm (`funk`/`scctl`), HPC-X/OpenMPI + PMIx, bash, JJB (`proj_jjb.yaml`).

---

## Reference spec

`docs/superpowers/specs/2026-06-04-ucc-mpi-dlrm-slurm-pipeline-design.md`

## Validation note (read first)

This is CI infrastructure; there is no local runtime to unit-test against. Each
code task is verified by **static checks** only:
- Shell scripts: `bash -n <file>` (syntax). `shellcheck` is not installed; skip it.
- YAML: `python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" <file>`.

Real validation happens at the **CI CHECKPOINT** tasks (4, 7, 10), which you
(the human) trigger by running the `ucc-test-mpi` job from a PR / the dispatcher.
Do not mark a checkpoint complete until its CI run is green.

## File structure

- Create `.ci/scripts/mpi_slurm_common.sh` — shared setup (diagnostics, IB device discovery, `$EXE` definition) + `mpi_slurm_run_*` functions. Sourced, not executed.
- Create `.ci/scripts/run_tests_ucc_mpi_slurm_smoke.sh` — sources common, runs smoke group.
- Create `.ci/scripts/run_tests_ucc_mpi_slurm_bulk.sh` — bulk group (ppn=4, multi-node).
- Create `.ci/scripts/run_tests_ucc_mpi_slurm_nccl.sh` — NCCL group (ppn=NGPUS).
- Create `.ci/scripts/run_tests_ucc_mpi_slurm_tlcuda.sh` — TL/CUDA group (1 node).
- Create `.ci/scripts/run_dlrm_slurm.sh` — DLRM, one rank/node, Slurm→torch env.
- Create `.ci/pipeline/test_mpi_matrix.yaml` — the pipeline.
- Modify `.ci/proj_jjb.yaml` — new `ucc-test-mpi` job-template, project jobs entry, dispatcher branch.

Bare-metal files (`job_matrix.yaml`, `run_*_docker.sh`, `run_docker.sh`, `clean_docker.sh`, `run_tests_ucc_mpi.sh`, `run_dlrm*.sh`) are **not** modified.

Per-group wrapper scripts (rather than one script taking a `$1` group arg)
because `slurmCI run`'s `testScript` is invoked as a bare path in the existing
pipelines; passing an argument is unproven. Wrappers keep each `testScript` a
plain path while the logic stays DRY in the common file.

---

## Task 1: Shared common script + smoke wrapper

**Files:**
- Create: `.ci/scripts/mpi_slurm_common.sh`
- Create: `.ci/scripts/run_tests_ucc_mpi_slurm_smoke.sh`

- [ ] **Step 1: Create `mpi_slurm_common.sh` with setup + smoke function**

```bash
#!/bin/bash
# Shared helpers for Slurm-native UCC MPI tests.
#
# Sourced by run_tests_ucc_mpi_slurm_<group>.sh wrappers. Each wrapper is the
# script that slurmCI 'run' launches via `srun --ntasks-per-node=N` inside a
# pyxis/enroot container. srun/PMIx provides the MPI ranks, so the test binary
# is invoked directly (no mpirun, no hostfile, no ssh).

UCC_SRC_DIR="/opt/nvidia/src/ucc"
EXE="${UCC_SRC_DIR}/build/test/mpi/ucc_test_mpi"
EXE_ARGS="--inplace 2 --set_device 2 --root random:2 --count_bits 32,64 --displ_bits 32,64"

# DEV is set by mpi_slurm_setup and used by the run_* functions.
DEV=""

mpi_slurm_setup() {
    export UCX_WARN_UNUSED_ENV_VARS=n
    # Disable NCCL by default; the NCCL group re-enables it explicitly.
    export UCC_TL_NCCL_TUNE=0

    echo "=== UCC MPI slurm (job ${SLURM_JOB_ID:-?} rank ${SLURM_PROCID:-0}/${SLURM_NTASKS:-?} nodes ${SLURM_NNODES:-?}) ==="
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>&1 || echo "nvidia-smi failed"
    fi

    # Local IB device discovery (no ssh): first Active device.
    DEV=""
    if command -v ibstat >/dev/null 2>&1; then
        for d in $(ibstat -l 2>/dev/null); do
            state=$(ibstat "$d" 2>/dev/null | awk '/State:/{print $2; exit}')
            if [ "$state" = "Active" ]; then DEV="$d"; break; fi
        done
    fi
    if [ -n "$DEV" ]; then
        export UCX_NET_DEVICES="${DEV}:1"
        echo "INFO: using IB device ${DEV}"
    else
        echo "WARNING: no Active IB device found; UCX will auto-select"
    fi
}

# Minimal end-to-end sanity: confirms srun/PMIx launch + CUDA memtype work.
mpi_slurm_run_smoke() {
    echo "INFO: smoke - barrier + small allreduce on host,cuda ..."
    # shellcheck disable=SC2086
    UCX_TLS="^cuda_ipc" $EXE $EXE_ARGS --mtypes host,cuda -c barrier,allreduce -m 1:1024
    echo "INFO: smoke ... DONE"
}
```

- [ ] **Step 2: Create `run_tests_ucc_mpi_slurm_smoke.sh`**

```bash
#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=/dev/null
. "${SCRIPT_DIR}/mpi_slurm_common.sh"

mpi_slurm_setup
mpi_slurm_run_smoke
```

- [ ] **Step 3: Make wrapper executable and syntax-check both**

Run:
```bash
chmod +x .ci/scripts/run_tests_ucc_mpi_slurm_smoke.sh
bash -n .ci/scripts/mpi_slurm_common.sh
bash -n .ci/scripts/run_tests_ucc_mpi_slurm_smoke.sh
```
Expected: no output (exit 0) from both `bash -n`.

- [ ] **Step 4: Commit**

```bash
git add .ci/scripts/mpi_slurm_common.sh .ci/scripts/run_tests_ucc_mpi_slurm_smoke.sh
git commit -m "CI: add slurm-native UCC MPI common helpers + smoke wrapper"
```

---

## Task 2: Pipeline skeleton with smoke step

**Files:**
- Create: `.ci/pipeline/test_mpi_matrix.yaml`

- [ ] **Step 1: Create `test_mpi_matrix.yaml`**

```yaml
---
job: "ucc-mpi"

step_allow_single_selector: true

registry_host: "harbor.mellanox.com"
registry_path: "/torch-ucc"
registry_auth: "ucc-harbor-credentials"

kubernetes:
  cloud: il-ipp-blossom-prod
  namespace: hpcx
  limits: "{memory: 16Gi, cpu: 16000m}"
  requests: "{memory: 8Gi, cpu: 8000m}"

pvc_volumes:
  - {claimName: hpcx-pvc, mountPath: /mnt/pvc, readOnly: false}

empty_volumes:
  - {mountPath: /root, memory: false}

env:
  CUDA_VER: 12.9
  UCC_URI_SUFFIX: "ucc/${UCC_VERSION}/x86_64/centos8/cuda${CUDA_VER}"
  SRC_DIR: "/opt/nvidia/src"
  DOCKER_IMAGE_TAG: "${BUILD_NUMBER}"
  SLURM_NODES: 2
  SLURM_PARTITION: 'funk'
  SLURM_GRES: 'gpu:1'  # GPUs per node; confirm funk capacity and bump if >1
  SLURM_HEAD_NODE: 'scctl'
  SLURM_JOB_NAME: "${BUILD_TAG}"
  SLURM_CONTAINER_NAME: "${BUILD_TAG}"
  SLURM_JOB_TIMEOUT: '2:00:00'
  SLURM_IMMEDIATE_TIMEOUT: 3600 # 1 hour
  TEST_TIMEOUT_MINUTES: 90
  MPI_BULK_PPN: 4      # ranks per node for the bulk/tlcuda groups
  MPI_NCCL_PPN: 1      # ranks per node for the NCCL group = GPUs per node
  JOB_ID_FILE: "/mnt/pvc/job-id-${BUILD_TAG}.txt"
  SCCTL_CREDENTIALS_ID: 'svcnbu-swx-hpcx-corporate-user-pass'

# cloud pod to build the shared docker image
runs_on_dockers:
  - {
      file: ".ci/Dockerfile.ngc_pytorch",
      name: "ngc_pytorch",
      tag: "${DOCKER_IMAGE_TAG}",
      arch: "x86_64",
      uri: "${UCC_URI_SUFFIX}",
      build_args: "--no-cache \
                   --build-arg _UID=149917 \
                   --build-arg _GID=30 \
                   --build-arg _LOGIN=svcnbu-swx-hpcx \
                   --build-arg _GROUP=svcnbu-swx-hpcx \
                   --build-arg CUDA_VER=${CUDA_VER} \
                   --build-arg UCC_ENABLE_GTEST=yes",
    }
  - {
      file: ".ci/dockerfiles/Dockerfile.build_helper",
      name: "build_helper",
      tag: "mpi",
      arch: "x86_64",
      uri: "$arch/$name",
      build_args: "--no-cache",
    }

timeout_minutes: 180

steps:
  - name: Allocate Slurm job
    containerSelector: "{name: 'build_helper'}"
    timeout: 30
    parallel: false
    shell: action
    module: slurmCI
    run: allocation
    args:
      partition: "${SLURM_PARTITION}"
      headNode: "${SLURM_HEAD_NODE}"
      nodes: "${SLURM_NODES}"
      jobTimeout: "${SLURM_JOB_TIMEOUT}"
      immediateTimeout: "${SLURM_IMMEDIATE_TIMEOUT}"
      jobName: "${SLURM_JOB_NAME}"
      jobIdFile: "${JOB_ID_FILE}"
      credentialsId: "${SCCTL_CREDENTIALS_ID}"
      extraArgs: [
        "--gres=${SLURM_GRES}",
      ]

  - name: Read Slurm job ID
    containerSelector: "{name: 'build_helper'}"
    parallel: false
    shell: action
    module: groovy
    run: env.SLURM_JOB_ID = readFile(JOB_ID_FILE).trim()

  - name: Run UCC MPI smoke
    containerSelector: "{name: 'build_helper'}"
    timeout: "${TEST_TIMEOUT_MINUTES}"
    parallel: false
    shell: action
    module: slurmCI
    run: run
    args:
      jobId: "${SLURM_JOB_ID}"
      testScript: "${SRC_DIR}/ucc/.ci/scripts/run_tests_ucc_mpi_slurm_smoke.sh"
      headNode: "${SLURM_HEAD_NODE}"
      dockerImage: "${registry_host}#torch-ucc/${UCC_URI_SUFFIX}:${DOCKER_IMAGE_TAG}"
      credentialsId: "${SCCTL_CREDENTIALS_ID}"
      containerName: "${SLURM_CONTAINER_NAME}"
      extraArgs: [
        "--ntasks-per-node=${MPI_BULK_PPN}",
      ]

pipeline_stop:
  containerSelector: "{name: 'build_helper'}"
  shell: action
  module: slurmCI
  run: stop
  args:
    credentialsId: "${SCCTL_CREDENTIALS_ID}"
    headNode: "${SLURM_HEAD_NODE}"
    jobId: "${SLURM_JOB_ID}"
```

- [ ] **Step 2: Validate YAML parses**

Run:
```bash
python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1])); print('OK')" .ci/pipeline/test_mpi_matrix.yaml
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .ci/pipeline/test_mpi_matrix.yaml
git commit -m "CI: add slurm MPI pipeline skeleton (build + allocate + smoke + stop)"
```

---

## Task 3: Register the `ucc-test-mpi` job

**Files:**
- Modify: `.ci/proj_jjb.yaml`

- [ ] **Step 1: Add the `ucc-test-mpi` job-template**

Insert a new job-template block immediately after the `{jjb_proj}-test-gtest`
template ends (after its `script-path: "{jjb_jenkinsfile}"` line, before the
`- project:` block). Add:

```yaml
- job-template:
    name: "{jjb_proj}-test-mpi"
    project-type: pipeline
    folder: "{jjb_folder}"
    properties:
      - github:
          url: "{jjb_git_web}"
      - build-discarder:
          days-to-keep: 30
          num-to-keep: 20
      - inject:
          keep-system-variables: true
          properties-content: |
            jjb_proj="{jjb_proj}-test-mpi"
    description: Do NOT edit this job through the Web GUI !
    concurrent: true
    sandbox: true
    parameters:
      - string:
          name: "sha1"
          default: "{jjb_branch}"
          description: "Commit to be checked, set by PR"
      - string:
          name: "UCC_VERSION"
          default: "1.0.0"
          description: "UCC version"
      - string:
          name: "githubData"
          default: ""
          description: "Blossom CI Variables from post"
      - bool:
          name: "BUILD_DOCKERS"
          default: true
          description: "Rebuild docker containers"
      - string:
          name: "CONF_FILE"
          default: ".ci/pipeline/test_mpi_matrix.yaml"
          description: "Regex to select job config file"
      - string:
          name: "DEBUG"
          default: 0
          description: "Enable debug prints and traces, valid values are 0-9"
    pipeline-scm:
      scm:
        - git:
            url: "{jjb_git}"
            credentials-id: "{jjb_gh_auth_id}"
            branches: [ '$sha1' ]
            shallow-clone: true
            depth: 10
            refspec: "+refs/heads/*:refs/remotes/origin/* +refs/pull/*:refs/remotes/origin/pr/*"
            browser: githubweb
            browser-url: "{jjb_git}"
      script-path: "{jjb_jenkinsfile}"
```

- [ ] **Step 2: Add the job to the project `jobs:` list**

In the `- project:` block at the bottom, change the `jobs:` list. Replace:

```yaml
      - "{jjb_proj}-run-coverity"
      - "{jjb_proj}-test-gtest"
```

with:

```yaml
      - "{jjb_proj}-run-coverity"
      - "{jjb_proj}-test-gtest"
      - "{jjb_proj}-test-mpi"
```

- [ ] **Step 3: Add `ucc-test-mpi` to the dispatcher's childJobNames**

In the `{jjb_proj}-ci-dispatcher` `dsl:` block, replace:

```groovy
        def childJobNames = ["ucc", "ucc-build-hpcsdk", "ucc-test-nvls", "ucc-run-coverity", "ucc-test-gtest"]
```

with:

```groovy
        def childJobNames = ["ucc", "ucc-build-hpcsdk", "ucc-test-nvls", "ucc-run-coverity", "ucc-test-gtest", "ucc-test-mpi"]
```

- [ ] **Step 4: Add the dispatcher parallel branch**

In the same `dsl:` `parallel` block, replace the `ucc-test-gtest` branch tail
(note the doubled braces — this is a JJB template):

```groovy
        }}, 'ucc-test-gtest': {{
            def jobName = 'ucc-test-gtest'
            build job: jobName, parameters: [
                string(name: 'sha1', value: githubHelper.getMergedSHA()),
                string(name: 'githubData', value: VARIABLE_FROM_POST)
            ], propagate: false, wait: true
        }}
        githubHelper.updateCommitStatus(blueOceanUrl, "Blossom CI ended", GitHubCommitState.SUCCESS)
```

with:

```groovy
        }}, 'ucc-test-gtest': {{
            def jobName = 'ucc-test-gtest'
            build job: jobName, parameters: [
                string(name: 'sha1', value: githubHelper.getMergedSHA()),
                string(name: 'githubData', value: VARIABLE_FROM_POST)
            ], propagate: false, wait: true
        }}, 'ucc-test-mpi': {{
            def jobName = 'ucc-test-mpi'
            build job: jobName, parameters: [
                string(name: 'sha1', value: githubHelper.getMergedSHA()),
                string(name: 'githubData', value: VARIABLE_FROM_POST)
            ], propagate: false, wait: true
        }}
        githubHelper.updateCommitStatus(blueOceanUrl, "Blossom CI ended", GitHubCommitState.SUCCESS)
```

- [ ] **Step 5: Validate YAML parses**

Run:
```bash
python3 -c "import yaml,sys; list(yaml.safe_load_all(open(sys.argv[1]))); print('OK')" .ci/proj_jjb.yaml
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add .ci/proj_jjb.yaml
git commit -m "CI: register ucc-test-mpi job and wire it into the dispatcher"
```

---

## Task 4: CI CHECKPOINT — plumbing + srun/PMIx smoke (make-or-break)

No code. This validates the highest risk from the spec: that the centos8/CUDA
12.9 `ngc_pytorch` image's MPI launches under `srun --mpi=pmix` on funk/scctl.

- [ ] **Step 1: Push the branch and run the job**

Open a PR (or trigger `ucc-test-mpi` directly with the branch `sha1`). Confirm
the job: builds both images, allocates a 2-node funk job, runs the smoke step,
and deallocates in `pipeline_stop`.

- [ ] **Step 2: Inspect the smoke step log**

Confirm in the `Run UCC MPI smoke` step:
- The diagnostics line shows `SLURM_NTASKS` > 1 across 2 nodes (PMIx world formed).
- `CUDA_VISIBLE_DEVICES` is set and `nvidia-smi` lists a GPU.
- `barrier + small allreduce on host,cuda ... DONE` prints with no MPI/PMIx init error and exit 0.

- [ ] **Step 3: If it fails, triage before continuing**

- `no kernel image is available` (CUDA 209) → GPU arch mismatch. Add a
  `--build-arg UCC_NVCC_GENCODE="..."` for funk's GPU to the `ngc_pytorch`
  `build_args` (see `.ci/docs/DEBUG_GTEST_SLURM.md` #6), rebuild, rerun.
- `CUDA_VISIBLE_DEVICES` unset / `nvidia-smi` fails → bump `SLURM_GRES` and/or
  check enroot GPU passthrough (DEBUG doc #2).
- PMIx / MPI_Init failure → the image's MPI is incompatible with the cluster
  PMIx. Resolve by installing HPC-X in `Dockerfile.ngc_pytorch` (as
  `Dockerfile.nvls` does) or building OMPI `--with-pmix`. This is a design-level
  change — pause and report back before proceeding.

Do not proceed to Task 5 until the smoke step is green.

---

## Task 5: Full MPI matrix functions + group wrappers

**Files:**
- Modify: `.ci/scripts/mpi_slurm_common.sh`
- Create: `.ci/scripts/run_tests_ucc_mpi_slurm_bulk.sh`
- Create: `.ci/scripts/run_tests_ucc_mpi_slurm_nccl.sh`
- Create: `.ci/scripts/run_tests_ucc_mpi_slurm_tlcuda.sh`

- [ ] **Step 1: Append the bulk/nccl/tlcuda functions to `mpi_slurm_common.sh`**

Add these functions at the end of `.ci/scripts/mpi_slurm_common.sh` (after
`mpi_slurm_run_smoke`). Each `$EXE` invocation is a complete collective MPI job
over the srun-provided ranks; the outer `MT` loop runs each config twice
(non-triggered then `-T`), matching the bare-metal `run_tests_ucc_mpi.sh`.

```bash
# Bulk group (ppn=4, multi-node): default, TL/UCP, CL/HIER variants, 2-step
# bcast, and TL/MLX5 (self-skips without >=2 nodes + IB device).
mpi_slurm_run_bulk() {
    local MT TG
    for MT in "" "-T"; do
        TG="--triggered 0"

        echo "INFO: default configuration ..."
        # shellcheck disable=SC2086
        UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda
        echo "INFO: default configuration ... DONE"

        echo "INFO: TL/UCP ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic UCC_CL_BASIC_TLS=ucp UCX_LOG_LEVEL=info UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda
        echo "INFO: TL/UCP ... DONE"

        echo "INFO: CL/HIER ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=inf UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c alltoall,alltoallv,allreduce,barrier
        echo "INFO: CL/HIER ... DONE"

        echo "INFO: CL/HIER+ucp ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=inf UCC_CL_HIER_TLS=ucp UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c alltoall,alltoallv,allreduce,barrier
        echo "INFO: CL/HIER+ucp ... DONE"

        echo "INFO: CL/HIER+rab ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=allreduce:@rab:inf UCC_CL_HIER_TLS=ucp UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c allreduce
        echo "INFO: CL/HIER+rab ... DONE"

        echo "INFO: CL/HIER+split_rail ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=allreduce:@split_rail:inf UCC_CL_HIER_TLS=ucp UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c allreduce
        echo "INFO: CL/HIER+split_rail ... DONE"

        echo "INFO: CL/HIER+split_rail+pipeline ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=allreduce:@split_rail:inf UCC_CL_HIER_TLS=ucp UCC_TL_NCCL_TUNE=0 \
            UCC_CL_HIER_ALLREDUCE_SPLIT_RAIL_PIPELINE=thresh=0:fragsize=256K UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c allreduce
        echo "INFO: CL/HIER+split_rail+pipeline ... DONE"

        echo "INFO: CL/HIER+2step bcast ..."
        # shellcheck disable=SC2086
        UCC_CLS=all UCC_TLS="^sharp" UCC_CL_HIER_TUNE="bcast:0-inf:@2step" UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c bcast
        echo "INFO: CL/HIER+2step bcast ... DONE"

        if [ "${SLURM_NNODES:-1}" -ge 2 ] && [ -n "$DEV" ]; then
            echo "INFO: TL/MLX5 ..."
            # shellcheck disable=SC2086
            UCC_CLS=basic UCC_CL_BASIC_TLS=ucp,mlx5 UCC_TL_MLX5_NET_DEVICES="${DEV}:1" UCC_TL_MLX5_TUNE=inf \
                $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c alltoall -t world -d uint8 -O 0 -m 1:128
            echo "INFO: TL/MLX5 ... DONE"
        else
            echo "INFO: TL/MLX5 ... SKIPPED (needs >=2 nodes + Active IB device)"
        fi
    done
}

# NCCL group (ppn = GPUs per node): cuda-only collectives over TL/NCCL.
mpi_slurm_run_nccl() {
    local MT TG
    for MT in "" "-T"; do
        TG="--triggered 0"
        echo "INFO: NCCL ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic UCC_CL_BASIC_TLS=ucp,nccl UCC_TL_NCCL_TUNE=cuda:inf \
            NCCL_IB_HCA="${DEV}" NCCL_DEBUG=WARN \
            $EXE $EXE_ARGS $MT $TG --mtypes cuda
        echo "INFO: NCCL ... DONE"
    done
}

# TL/CUDA group (single node): cuda collectives over TL/CUDA.
mpi_slurm_run_tlcuda() {
    local MT TG
    for MT in "" "-T"; do
        TG="--triggered 0"
        echo "INFO: TL/CUDA ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic UCC_CL_BASIC_TLS=ucp,cuda UCC_TL_CUDA_TUNE=cuda:inf UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes cuda \
            -c alltoall,alltoallv,allgather,allgatherv,reduce_scatter,reduce_scatterv
        echo "INFO: TL/CUDA ... DONE"
    done
}
```

- [ ] **Step 2: Create `run_tests_ucc_mpi_slurm_bulk.sh`**

```bash
#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=/dev/null
. "${SCRIPT_DIR}/mpi_slurm_common.sh"

mpi_slurm_setup
mpi_slurm_run_bulk
```

- [ ] **Step 3: Create `run_tests_ucc_mpi_slurm_nccl.sh`**

```bash
#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=/dev/null
. "${SCRIPT_DIR}/mpi_slurm_common.sh"

mpi_slurm_setup
mpi_slurm_run_nccl
```

- [ ] **Step 4: Create `run_tests_ucc_mpi_slurm_tlcuda.sh`**

```bash
#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=/dev/null
. "${SCRIPT_DIR}/mpi_slurm_common.sh"

mpi_slurm_setup
mpi_slurm_run_tlcuda
```

- [ ] **Step 5: Make wrappers executable and syntax-check all four files**

Run:
```bash
chmod +x .ci/scripts/run_tests_ucc_mpi_slurm_bulk.sh \
         .ci/scripts/run_tests_ucc_mpi_slurm_nccl.sh \
         .ci/scripts/run_tests_ucc_mpi_slurm_tlcuda.sh
for f in .ci/scripts/mpi_slurm_common.sh \
         .ci/scripts/run_tests_ucc_mpi_slurm_bulk.sh \
         .ci/scripts/run_tests_ucc_mpi_slurm_nccl.sh \
         .ci/scripts/run_tests_ucc_mpi_slurm_tlcuda.sh; do
  bash -n "$f" && echo "OK $f"
done
```
Expected: `OK` for each of the four files.

- [ ] **Step 6: Commit**

```bash
git add .ci/scripts/mpi_slurm_common.sh \
        .ci/scripts/run_tests_ucc_mpi_slurm_bulk.sh \
        .ci/scripts/run_tests_ucc_mpi_slurm_nccl.sh \
        .ci/scripts/run_tests_ucc_mpi_slurm_tlcuda.sh
git commit -m "CI: add full UCC MPI matrix groups (bulk/nccl/tlcuda) for slurm"
```

---

## Task 6: Wire the full MPI groups into the pipeline

**Files:**
- Modify: `.ci/pipeline/test_mpi_matrix.yaml`

- [ ] **Step 1: Replace the smoke step with the three group steps**

In `.ci/pipeline/test_mpi_matrix.yaml`, replace the entire `Run UCC MPI smoke`
step block:

```yaml
  - name: Run UCC MPI smoke
    containerSelector: "{name: 'build_helper'}"
    timeout: "${TEST_TIMEOUT_MINUTES}"
    parallel: false
    shell: action
    module: slurmCI
    run: run
    args:
      jobId: "${SLURM_JOB_ID}"
      testScript: "${SRC_DIR}/ucc/.ci/scripts/run_tests_ucc_mpi_slurm_smoke.sh"
      headNode: "${SLURM_HEAD_NODE}"
      dockerImage: "${registry_host}#torch-ucc/${UCC_URI_SUFFIX}:${DOCKER_IMAGE_TAG}"
      credentialsId: "${SCCTL_CREDENTIALS_ID}"
      containerName: "${SLURM_CONTAINER_NAME}"
      extraArgs: [
        "--ntasks-per-node=${MPI_BULK_PPN}",
      ]
```

with these three steps:

```yaml
  - name: Run UCC MPI tests (bulk)
    containerSelector: "{name: 'build_helper'}"
    timeout: "${TEST_TIMEOUT_MINUTES}"
    parallel: false
    shell: action
    module: slurmCI
    run: run
    args:
      jobId: "${SLURM_JOB_ID}"
      testScript: "${SRC_DIR}/ucc/.ci/scripts/run_tests_ucc_mpi_slurm_bulk.sh"
      headNode: "${SLURM_HEAD_NODE}"
      dockerImage: "${registry_host}#torch-ucc/${UCC_URI_SUFFIX}:${DOCKER_IMAGE_TAG}"
      credentialsId: "${SCCTL_CREDENTIALS_ID}"
      containerName: "${SLURM_CONTAINER_NAME}"
      extraArgs: [
        "--ntasks-per-node=${MPI_BULK_PPN}",
      ]

  - name: Run UCC MPI tests (NCCL)
    containerSelector: "{name: 'build_helper'}"
    timeout: "${TEST_TIMEOUT_MINUTES}"
    parallel: false
    shell: action
    module: slurmCI
    run: run
    args:
      jobId: "${SLURM_JOB_ID}"
      testScript: "${SRC_DIR}/ucc/.ci/scripts/run_tests_ucc_mpi_slurm_nccl.sh"
      headNode: "${SLURM_HEAD_NODE}"
      dockerImage: "${registry_host}#torch-ucc/${UCC_URI_SUFFIX}:${DOCKER_IMAGE_TAG}"
      credentialsId: "${SCCTL_CREDENTIALS_ID}"
      containerName: "${SLURM_CONTAINER_NAME}"
      extraArgs: [
        "--ntasks-per-node=${MPI_NCCL_PPN}",
      ]

  - name: Run UCC MPI tests (TL/CUDA, single node)
    containerSelector: "{name: 'build_helper'}"
    timeout: "${TEST_TIMEOUT_MINUTES}"
    parallel: false
    shell: action
    module: slurmCI
    run: run
    args:
      jobId: "${SLURM_JOB_ID}"
      testScript: "${SRC_DIR}/ucc/.ci/scripts/run_tests_ucc_mpi_slurm_tlcuda.sh"
      headNode: "${SLURM_HEAD_NODE}"
      dockerImage: "${registry_host}#torch-ucc/${UCC_URI_SUFFIX}:${DOCKER_IMAGE_TAG}"
      credentialsId: "${SCCTL_CREDENTIALS_ID}"
      containerName: "${SLURM_CONTAINER_NAME}"
      extraArgs: [
        "--nodes=1",
        "--ntasks-per-node=${MPI_BULK_PPN}",
      ]
```

- [ ] **Step 2: Validate YAML parses**

Run:
```bash
python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1])); print('OK')" .ci/pipeline/test_mpi_matrix.yaml
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .ci/pipeline/test_mpi_matrix.yaml
git commit -m "CI: run full UCC MPI matrix (bulk/nccl/tlcuda) on slurm"
```

---

## Task 7: CI CHECKPOINT — full MPI matrix

No code. Trigger `ucc-test-mpi` again.

- [ ] **Step 1: Confirm all three MPI steps pass**

In CI, verify `bulk`, `NCCL`, and `TL/CUDA` steps each complete with `... DONE`
lines and exit 0. TL/MLX5 inside the bulk step may print `SKIPPED` if funk has
no CX7 / Active IB — that is acceptable and matches the spec.

- [ ] **Step 2: Tune geometry if needed**

- If funk nodes have >1 GPU, set `MPI_NCCL_PPN` (and `SLURM_GRES`) in
  `test_mpi_matrix.yaml` to the GPU count so the NCCL step uses all GPUs, then
  rerun and commit the change.
- If a step exceeds `TEST_TIMEOUT_MINUTES`, raise that step's `timeout` and/or
  `SLURM_JOB_TIMEOUT` and commit.

Do not proceed to Task 8 until all three MPI steps are green.

---

## Task 8: DLRM slurm script

**Files:**
- Create: `.ci/scripts/run_dlrm_slurm.sh`

- [ ] **Step 1: Create `run_dlrm_slurm.sh`**

Launched by slurmCI with `--ntasks-per-node=1`, so it runs once per node (one
rank per node). It maps Slurm rank env → the torch distributed env that
`dlrm_s_pytorch.py` expects, then execs the existing
`run_dlrm_s_pytorch.sh` (which runs `python ... --dist-backend=ucc` directly,
no mpirun).

```bash
#!/bin/bash -eEx
set -o pipefail

# Slurm-native DLRM (torch_ucc) test. One rank per node under
# `srun --ntasks-per-node=1`. Maps Slurm env -> torch distributed env, then
# runs the existing dlrm python launcher.

UCC_SRC_DIR="/opt/nvidia/src/ucc"

# Master address = first node of the Slurm allocation.
if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
else
    MASTER_ADDR=$(hostname -s)
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-12346}"
export RANK="${SLURM_PROCID:-0}"
export WORLD_SIZE="${SLURM_NTASKS:-1}"
export LOCAL_RANK="${SLURM_LOCALID:-0}"
export CPU_GPU_MODE="gpu"

# Same UCC configuration as the bare-metal DLRM path.
export UCC_CLS=basic
export UCC_CL_BASIC_TLS=nccl,ucp

echo "=== DLRM slurm (job ${SLURM_JOB_ID:-?}) RANK=${RANK}/${WORLD_SIZE} LOCAL_RANK=${LOCAL_RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT} ==="

exec "${UCC_SRC_DIR}/.ci/scripts/run_dlrm_s_pytorch.sh"
```

- [ ] **Step 2: Make executable and syntax-check**

Run:
```bash
chmod +x .ci/scripts/run_dlrm_slurm.sh
bash -n .ci/scripts/run_dlrm_slurm.sh && echo OK
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .ci/scripts/run_dlrm_slurm.sh
git commit -m "CI: add slurm-native DLRM (torch_ucc) test launcher"
```

---

## Task 9: Wire the DLRM step into the pipeline

**Files:**
- Modify: `.ci/pipeline/test_mpi_matrix.yaml`

- [ ] **Step 1: Add the DLRM step after the TL/CUDA step**

In `.ci/pipeline/test_mpi_matrix.yaml`, insert this step immediately after the
`Run UCC MPI tests (TL/CUDA, single node)` step and before `pipeline_stop:`:

```yaml
  - name: Run DLRM tests (UCC/GPU)
    containerSelector: "{name: 'build_helper'}"
    timeout: "${TEST_TIMEOUT_MINUTES}"
    parallel: false
    shell: action
    module: slurmCI
    run: run
    args:
      jobId: "${SLURM_JOB_ID}"
      testScript: "${SRC_DIR}/ucc/.ci/scripts/run_dlrm_slurm.sh"
      headNode: "${SLURM_HEAD_NODE}"
      dockerImage: "${registry_host}#torch-ucc/${UCC_URI_SUFFIX}:${DOCKER_IMAGE_TAG}"
      credentialsId: "${SCCTL_CREDENTIALS_ID}"
      containerName: "${SLURM_CONTAINER_NAME}"
      extraArgs: [
        "--ntasks-per-node=1",
      ]
```

- [ ] **Step 2: Validate YAML parses**

Run:
```bash
python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1])); print('OK')" .ci/pipeline/test_mpi_matrix.yaml
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .ci/pipeline/test_mpi_matrix.yaml
git commit -m "CI: run DLRM (UCC/GPU) test on slurm"
```

---

## Task 10: CI CHECKPOINT — full pipeline (MPI + DLRM)

No code. Final end-to-end validation.

- [ ] **Step 1: Run `ucc-test-mpi` and confirm the whole pipeline is green**

Verify: both images build, allocation succeeds, all three MPI steps pass, the
DLRM step trains its 10 batches with `--dist-backend=ucc` across 2 ranks, and
`pipeline_stop` deallocates the job.

- [ ] **Step 2: DLRM triage if needed**

- torch distributed init hang/timeout → check `RANK`/`WORLD_SIZE`/`MASTER_ADDR`
  in the step log are correct (2 distinct ranks, one master addr); adjust the
  env mapping in `run_dlrm_slurm.sh` if the torch ucc backend expects different
  variable names, then rerun and commit.
- `UCX_NET_DEVICES` mismatch (the dlrm python script hardcodes `mlx5_0:1`) → if
  funk's device differs, override `UCX_NET_DEVICES` in `run_dlrm_slurm.sh`
  before the `exec`, then rerun and commit.

- [ ] **Step 3: Confirm the bare-metal `ucc` job is unaffected**

Confirm the existing `ucc` job still runs (it shares the dispatcher) and was not
modified by this work.

---

## Self-review

- **Spec coverage:** job registration (Task 3) ✓; ngc_pytorch + build_helper build (Task 2) ✓; pvc/empty volumes + SLURM_* env (Task 2) ✓; allocate→read→run→stop (Tasks 2,6,9) ✓; geometry-grouped MPI steps (Tasks 5,6) ✓; new slurm-native scripts replacing mpirun/ssh/hostfile (Tasks 1,5,8) ✓; DLRM Slurm→torch env mapping (Task 8) ✓; risk validation order incl. srun/PMIx make-or-break first (Task 4), gencode/gres (Tasks 4,7), CX7 self-skip (Task 7), DLRM init (Task 10) ✓; bare-metal left intact (file-structure note + Task 10 step 3) ✓.
- **Placeholders:** `gpu:1` / `MPI_NCCL_PPN: 1` are real, working defaults with explicit "confirm/tune" steps (Tasks 4,7) — not unfilled placeholders. No TBD/TODO remain.
- **Type/name consistency:** function names `mpi_slurm_setup`, `mpi_slurm_run_smoke|bulk|nccl|tlcuda` match between common file and wrappers; script paths match between scripts and pipeline `testScript` entries; env var names (`SLURM_GRES`, `MPI_BULK_PPN`, `MPI_NCCL_PPN`, `JOB_ID_FILE`, `SCCTL_CREDENTIALS_ID`, `DOCKER_IMAGE_TAG`, `UCC_URI_SUFFIX`) consistent across all steps; job name `ucc-test-mpi` consistent across template/project/dispatcher.
