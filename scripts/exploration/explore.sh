#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# EXPLORATION — wrapper: run Python allocator, then the standalone Rcpp
# re-implementation, and diff cell-by-cell.
#
# NOTE: this exercises the original standalone-Rcpp *exploration*
# (scripts/exploration/), not the evoland allocator.  For the Python-vs-evoland
# uSAM/uPAM comparison see scripts/comparison/compare_evoland.sh.
#
# Usage (from repo root):
#   bash scripts/exploration/explore.sh [--seed 42] [--verbose 2]
#
# Prerequisites:
#   - uv-managed Python environment with clumpy/ekde/hyperclip installed
#   - rv-activated R environment with Rcpp installed
#   - Both environments set up per PROGRESS.md steps 1 & 4
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- Defaults --------------------------------------------------------------
SEED=42
VERBOSE=2

# ---- Parse arguments -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)    SEED="$2";    shift 2 ;;
        --verbose) VERBOSE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--seed N] [--verbose N]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PYDIR="scripts/output/csv"
R_OUTDIR="scripts/output/r_output"

# ---- Colour helpers --------------------------------------------------------
if [[ -t 1 ]]; then
    BOLD="\033[1m"
    GREEN="\033[32m"
    RED="\033[31m"
    YELLOW="\033[33m"
    RESET="\033[0m"
else
    BOLD="" GREEN="" RED="" YELLOW="" RESET=""
fi

banner() { printf "\n${BOLD}=== %s ===${RESET}\n\n" "$*"; }
info()   { printf "${GREEN}✓${RESET} %s\n" "$*"; }
warn()   { printf "${YELLOW}⚠${RESET} %s\n" "$*"; }
fail()   { printf "${RED}✗${RESET} %s\n" "$*"; }

# ---- Step 1: Run Python allocation -----------------------------------------
banner "Step 1: Python allocation (seed=$SEED)"

MPLBACKEND=Agg uv run python scripts/run_allocation.py \
    --seed "$SEED" \
    --verbose "$VERBOSE"

if [[ ! -f "$PYDIR/luc_allocated.csv" ]]; then
    fail "Python did not produce $PYDIR/luc_allocated.csv"
    exit 1
fi
info "Python outputs in $PYDIR/"

# ---- Step 1b: Verify Python exported random draws for replay ----------------
# run_allocation.py now saves areas_sampled.csv and eccentricities_sampled.csv
# directly (via an instrumented _sample method), so no separate export needed.

if [[ -f "$PYDIR/areas_sampled.csv" && -f "$PYDIR/eccentricities_sampled.csv" ]]; then
    info "Python exported area/eccentricity draws for deterministic replay"
else
    warn "Area draws not found — deterministic replay will be skipped in R"
fi

# ---- Step 2: Run R allocation -----------------------------------------------
banner "Step 2: R allocation (Rcpp)"

# Ensure rv scripts are generated so .Rprofile can source them
rv activate 2>/dev/null || true

# Run R with --no-save --no-restore (not --vanilla, which skips .Rprofile).
# We let .Rprofile run normally to set up the rv library path, then
# source the driver script.  Arguments after --args are passed to
# commandArgs(trailingOnly=TRUE) inside the R session.
Rscript --no-save --no-restore \
    -e "source('scripts/exploration/run_allocation.R')" \
    --pydir "$PYDIR" --verbose "$VERBOSE" 2>&1

if [[ ! -f "$R_OUTDIR/luc_allocated_r.csv" ]]; then
    fail "R did not produce $R_OUTDIR/luc_allocated_r.csv"
    exit 1
fi
info "R outputs in $R_OUTDIR/"

# ---- Step 3: Final summary -------------------------------------------------
banner "Step 3: Final comparison summary"

# Quick cell-by-cell diff using Python (since it can read both CSVs easily)
MPLBACKEND=Agg uv run python -c "
import numpy as np
import os

pydir = '${PYDIR}'
r_outdir = '${R_OUTDIR}'

luc_py = np.loadtxt(os.path.join(pydir, 'luc_allocated.csv'), delimiter=',', dtype=int)

# R's write.csv includes a header row and row names — handle both formats
r_file = os.path.join(r_outdir, 'luc_allocated_r.csv')
with open(r_file) as f:
    first_line = f.readline()
# If first line contains non-numeric chars (header), skip it
try:
    [int(x) for x in first_line.strip().split(',')]
    header_rows = 0
except ValueError:
    header_rows = 1

luc_r = np.loadtxt(r_file, delimiter=',', dtype=int, skiprows=header_rows)

# If R wrote row names as first column, detect and strip
if luc_r.shape[1] == luc_py.shape[1] + 1:
    luc_r = luc_r[:, 1:]

n_total = luc_py.size
diff = luc_py != luc_r

# Check for replay results too
replay_file = os.path.join(r_outdir, 'luc_allocated_r.csv')

print(f'Python allocated map shape: {luc_py.shape}')
print(f'R allocated map shape:      {luc_r.shape}')
print(f'Cells that differ:          {diff.sum()} / {n_total}')

if diff.sum() == 0:
    print()
    print('🎉  PERFECT MATCH — Python and R outputs are identical!')
else:
    print()
    print(f'Differences found in {diff.sum()} cells.')
    print('This is expected when R uses its own RNG for area sampling.')
    print('Check the deterministic replay section in the R output above')
    print('for a comparison using identical random draws.')

    # Show locations of differences
    rows, cols = np.where(diff)
    for r, c in zip(rows[:10], cols[:10]):
        print(f'  [{r},{c}]: Python={luc_py[r,c]}, R={luc_r[r,c]}')
    if len(rows) > 10:
        print(f'  ... and {len(rows)-10} more')
"

echo ""
echo "Done. All outputs are in:"
echo "  Python: $PYDIR/"
echo "  R:      $R_OUTDIR/"
