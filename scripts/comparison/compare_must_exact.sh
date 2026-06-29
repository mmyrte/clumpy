#!/usr/bin/env bash
# Exact (pixel-perfect) cross-language check of the pivot test (MuST / GART):
# export clumpy's GART inputs+output, replay them through evoland's must_cpp.
#
# Usage (clumpy repo root):
#   bash scripts/comparison/compare_must_exact.sh [--evoland ../evoland-plus]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
EVOLAND="../evoland-plus"
[[ "${1:-}" == "--evoland" ]] && EVOLAND="$2"

echo "=== Step 1: export clumpy GART (P, x, V) ==="
MPLBACKEND=Agg uv run python scripts/comparison/compare_must_exact.py

echo "=== Step 2: replay through evoland must_cpp and diff ==="
Rscript scripts/comparison/compare_must_exact.R --evoland "$EVOLAND"
