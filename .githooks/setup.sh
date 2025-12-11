#!/bin/bash
#
# Setup script to install git hooks
# Run this script after cloning the repository to enable commit message checks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Setting up git hooks..."

# Configure git to use .githooks directory
git config --local core.hooksPath "$SCRIPT_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Git hooks installed successfully!"
    echo ""
    echo "The following checks are now enabled:"
    echo "  - Commit title length must be ≤ 50 characters"
    echo ""
else
    echo "✗ Failed to install git hooks"
    exit 1
fi
