#!/usr/bin/env python3
"""
Exact (pixel-perfect) cross-language check of the pivot test.

Exports a probability matrix P, the exact uniform draws x that clumpy's
`generalized_allocation_rejection_test` (GART) uses for a given seed, and the
resulting per-row final states V. The R side (compare_must_exact.R) replays the
SAME (P, x) through evoland's `must_cpp` and asserts the two assignments are
identical cell-by-cell.

This isolates the pivot mechanism (MuST / GART, Mazy 2022 App. 3.B): because both
implementations are fed the same uniforms, any difference would be a genuine
logic difference, not RNG divergence.

Usage (clumpy repo root):
    MPLBACKEND=Agg uv run python scripts/comparison/compare_must_exact.py
"""

from pathlib import Path

import numpy as np

from clumpy.allocation._gart import generalized_allocation_rejection_test as gart

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "scripts" / "output" / "must_exact"
OUT.mkdir(parents=True, exist_ok=True)

N, K = 2000, 3
SEED = 42
list_v = [1, 2, 3]  # final states; one of them doubles as the "stay" column

# A clean probability matrix with rows summing to exactly 1 (so every row is
# assigned a column; the row-sum < 1 "unaffected" sentinel path is exercised by
# the dedicated unit tests, not here).
gen = np.random.RandomState(123)
P = gen.random((N, K))
P = P / P.sum(axis=1, keepdims=True)

# Reproduce the exact uniforms GART draws for SEED: it does
# `np.random.seed(SEED); x = np.random.random(n)` internally.
np.random.seed(SEED)
x = np.random.random(N)
np.random.seed(None)

V = gart(P.copy(), list_v, seed=SEED)

np.savetxt(OUT / "P.csv", P, delimiter=",", fmt="%.17g")
np.savetxt(OUT / "x.csv", x, delimiter=",", fmt="%.17g")
np.savetxt(OUT / "states.csv", np.array(list_v, dtype=int), fmt="%d", delimiter=",")
np.savetxt(OUT / "V.csv", V.astype(int), fmt="%d", delimiter=",")

print(f"Exported {N} rows x {K} states (seed={SEED}) to {OUT}")
print(f"  state counts (python GART): {np.bincount(V)[1:]}")
