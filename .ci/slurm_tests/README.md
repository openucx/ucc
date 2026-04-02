# Slurm Tests

This directory contains configuration and scripts for running UCC tests on Slurm clusters.

## Required Environment Variables for Job Matrix

| Variable | Description |
|----------|-------------|
| `UCC_ENROOT_IMAGE_NAME` | The container image name to use for running tests. Must be in Enroot-compatible format with `#` after the registry host (e.g., `harbor.mellanox.com#torch-ucc/ucc/1.0.0/x86_64/centos8/cuda12.9:31`). |
| `SLM_JOB_NAME` | The name assigned to the Slurm job allocation. Used with `salloc --job-name`. Typically includes the build number for traceability (e.g., `ucc_tests_${BUILD_NUMBER}`). |
| `SLM_NODES` | The number of nodes to allocate for the Slurm job. Used with `salloc -N`. |
| `SLM_HEAD_NODE` | The Slurm head node to connect to. Can be `scctl` (uses scctl client) or a hostname for direct SSH access (e.g., `hpchead`). |
| `SLM_PARTITION` | The Slurm partition to submit the job to. Used with `salloc -p` (e.g., `funk`, `soul`). |

