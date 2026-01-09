# GitHub Scripts

This directory contains shared scripts used by both local development tools and CI workflows.

## check-commit-msg.sh

Validates commit message format and standards.

### Usage

**Direct invocation:**
```bash
.github/scripts/check-commit-msg.sh "COMMIT_MESSAGE_HERE"
```

**From file:**
```bash
.github/scripts/check-commit-msg.sh --file path/to/commit-msg-file
```

### Validation Rules

1. **Title Length**: Maximum 50 characters (merge commits exempt)
2. **Title Prefix**: Must start with approved prefix
3. **No Period**: Title must NOT end with a period

### Approved Prefixes

**Category 1:** `CODESTYLE`, `REVIEW`, `CORE`, `UTIL`, `TEST`, `API`, `DOCS`, `TOOLS`, `BUILD`, `MC`, `EC`, `SCHEDULE`, `TOPO`

**Category 2:** `CI`, `CL/`, `TL/`, `MC/`, `EC/`, `UCP`, `SHM`, `NCCL`, `SHARP`, `BASIC`, `HIER`, `DOCA_UROM`, `CUDA`, `CPU`, `EE`, `RCCL`, `ROCM`, `SELF`, `MLX5`

### Used By

- **Local Git Hooks**: `.githooks/commit-msg`
- **GitHub Actions**: `.github/workflows/codestyle.yaml`

### Exit Codes

- `0`: Commit message is valid
- `1`: Commit message validation failed

### Examples

**Valid:**
```bash
$ .github/scripts/check-commit-msg.sh "CI: fix typo in workflow"
Good commit title: 'CI: fix typo in workflow'
```

**Invalid:**
```bash
$ .github/scripts/check-commit-msg.sh "fix typo"
Commit title is too long: 8

Bad commit title: 'fix typo'

  ✗ Wrong header - must start with one of: ...
```
