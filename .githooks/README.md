# Git Hooks

This directory contains shared Git hooks to enforce code quality and standards.

## Installation

After cloning the repository, run:

```bash
./.githooks/setup.sh
```

This will configure Git to use these hooks automatically.

## Architecture

The commit message validation uses a **shared script** located at `.github/scripts/check-commit-msg.sh`. This same script is used by:

1. **Local git hooks** (`.githooks/commit-msg`) - validates commits before they are created
2. **GitHub Actions CI** (`.github/workflows/codestyle.yaml`) - validates all commits in pull requests

This ensures that commit message standards are consistently enforced both locally and in CI.

## Hooks

### commit-msg

Enforces commit message standards:

- **Commit title length**: Maximum 50 characters (merge commits are exempt)
- **Commit title prefix**: Must start with an approved prefix (see list below)
- **No trailing period**: Title must NOT end with a period

#### Approved Prefixes

**Category 1:**
`CODESTYLE`, `REVIEW`, `CORE`, `UTIL`, `TEST`, `API`, `DOCS`, `TOOLS`, `BUILD`, `MC`, `EC`, `SCHEDULE`, `TOPO`

**Category 2:**
`CI`, `CL/`, `TL/`, `MC/`, `EC/`, `UCP`, `SHM`, `NCCL`, `SHARP`, `BASIC`, `HIER`, `DOCA_UROM`, `CUDA`, `CPU`, `EE`, `RCCL`, `ROCM`, `SELF`, `MLX5`

**Format:** `PREFIX: description`

If a commit message violates these rules, the commit will be rejected with an error message.

## Examples

**❌ Bad commits (will be rejected):**
```
CI: fix command for clang-format in codestyle workflow
```
(54 characters - too long)

```
Fix typo in README
```
(Missing approved prefix)

```
TEST: add new unit tests.
```
(Ends with a period)

**✅ Good commits (will be accepted):**
```
CI: fix clang-format command in codestyle
```
(42 characters, valid prefix, no period)

```
TL/CUDA: add support for multinode NVLS
```
(Valid prefix with slash notation)

```
BUILD: update clang format
```
(Short and follows all rules)

## Bypassing Hooks (Not Recommended)

In rare cases where you need to bypass the hooks, you can use:
```bash
git commit --no-verify
```

**Warning**: Only use this in exceptional circumstances and ensure your commit messages still follow the project standards.

## Uninstalling

To disable the hooks:
```bash
git config --unset core.hooksPath
```
