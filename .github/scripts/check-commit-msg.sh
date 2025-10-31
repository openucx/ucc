#!/bin/bash
#
# Shared script to validate commit messages
# Used by both git hooks and GitHub Actions CI
#
# Usage:
#   check-commit-msg.sh <commit-message>
#   check-commit-msg.sh --file <commit-msg-file>

set -e

# Parse arguments
if [ "$1" = "--file" ]; then
    commit_title=$(head -n1 "$2")
else
    commit_title="$1"
fi

if [ -z "$commit_title" ]; then
    echo "Error: No commit message provided"
    echo "Usage: $0 <commit-message>"
    echo "   or: $0 --file <commit-msg-file>"
    exit 1
fi

title_length=${#commit_title}
max_length=50
errors=()

# Check title length (except for merge commits)
if [ $title_length -gt $max_length ]; then
    if ! echo "$commit_title" | grep -qP '^Merge'; then
        errors+=("Commit title is too long: $title_length characters (max: $max_length)")
    fi
fi

# Check title prefix
H1="CODESTYLE|REVIEW|CORE|UTIL|TEST|API|DOCS|TOOLS|BUILD|MC|EC|SCHEDULE|TOPO"
H2="CI|CL/|TL/|MC/|EC/|UCP|SHM|NCCL|SHARP|BASIC|HIER|DOCA_UROM|CUDA|CPU|EE|RCCL|ROCM|SELF|MLX5"
if ! echo "$commit_title" | grep -qP '^Merge |^'"(($H1)|($H2))"'+: \w'; then
    errors+=("Wrong header - must start with one of: $H1, $H2")
fi

# Check for period at the end
if [ "${commit_title: -1}" = "." ]; then
    errors+=("Commit title must NOT end with a period")
fi

# If there are errors, display them and exit
if [ ${#errors[@]} -gt 0 ]; then
    echo "Commit title is too long: ${#commit_title}"
    echo ""
    echo "Bad commit title: '$commit_title'"
    echo ""
    for error in "${errors[@]}"; do
        echo "  ✗ $error"
    done
    echo ""
    exit 1
fi

echo "Good commit title: '$commit_title'"
exit 0
