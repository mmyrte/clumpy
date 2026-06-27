#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# COMPARISON wrapper: run the Python clumpy reference, then compare it against
# the evoland uSAM / uPAM allocators (compiled from the evoland-plus sources).
#
# Usage (from the clumpy repo root):
#   bash scripts/comparison/compare_evoland.sh \
#       [--seed 42] [--nrep 200] [--evoland ../evoland-plus]
#
# Prerequisites:
#   - uv-managed Python env with clumpy/ekde/hyperclip (uv sync)
#   - R with Rcpp + a C++ toolchain (binaries: https://p3m.dev)
#   - an evoland-plus checkout (default sibling dir ../evoland-plus)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SEED=42
NREP=200
EVOLAND="../evoland-plus"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --nrep) NREP="$2"; shift 2 ;;
    --evoland) EVOLAND="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--seed N] [--nrep N] [--evoland PATH]"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

PYDIR="scripts/output/csv"

echo "=== Step 1: Python reference allocation (seed=$SEED) ==="
MPLBACKEND=Agg uv run python scripts/run_allocation.py --seed "$SEED" --verbose 1

echo
echo "=== Step 2: evoland uSAM/uPAM comparison ==="
Rscript scripts/comparison/compare_evoland.R \
  --pydir "$PYDIR" --evoland "$EVOLAND" --nrep "$NREP" --seed "$SEED"
