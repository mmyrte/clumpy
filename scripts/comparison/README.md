# Python clumpy ↔ evoland allocation comparison

This directory compares the **Python clumpy reference allocation** against the
**evoland `uSAM` / `uPAM` allocators** (the C++ backend `allocate_clumpy_cpp`
from the `evoland-plus` repository).

It is distinct from [`../exploration/`](../exploration/), which was the original
*exploration*: a standalone Rcpp re-implementation of clumpy's patch grower,
verified against Python by replaying identical random draws for a cell-by-cell
match. That exploration did **not** touch the evoland code; this comparison
drives the actual evoland backend.

## What it does

1. `../run_allocation.py` builds a 20×20 synthetic landscape, fits a Bayes TPE,
   computes per-pixel `P(urban | forest, z)`, and runs the Python pivot +
   `GaussianPatcher` allocation. It writes everything to
   `../output/csv/` (the shared interface).
2. `compare_evoland.R` compiles the evoland backend straight from the
   `evoland-plus` sources, feeds it the **same** probability field and patch
   parameters, and compares:
   - **GART (pivot mechanism)** — `gart_cpp` vs Python GART vs the analytic
     expectation `E[#pivots] = Σ P(v|u,z)`.
   - **Quantity of change** — Python vs evoland `uSAM` (rarefy on/off) and
     `uPAM`, averaged over `--nrep` seeds (different RNGs ⇒ no cell-by-cell
     match is expected; that was the exploration's job).
   - **Patch structure** — count / mean area / elongation of the newly created
     patches, measured identically for both via `calculate_class_stats_cpp`.

## Exact (pixel-perfect) checks

`compare_evoland.*` is a *statistical* comparison (different RNGs ⇒ no
cell-by-cell match). For a deterministic, RNG-controlled equivalence:

- **Pivot test (MuST / GART) — done.** `compare_must_exact.{py,R,sh}` exports
  clumpy's GART inputs and the exact uniforms it draws, then replays them through
  evoland's `must_cpp` (`u=` argument) and asserts identical per-cell assignment:

  ```sh
  bash scripts/comparison/compare_must_exact.sh --evoland ../evoland-plus
  # -> "PERFECT MATCH: evoland must_cpp == clumpy GART on identical (P, x)."
  ```

- **Patch growth — planned.** See [`PIXEL_PERFECT_PLAN.md`](PIXEL_PERFECT_PLAN.md):
  evoland's grower is a re-implementation (no hollow-fill, different tie-breaks),
  so a cell-by-cell patch comparison needs a restricted-scenario battery; the
  plan lays out the design and the differences to control for.

## Running

```sh
# from the clumpy repo root
uv sync                                   # build the Python env once
bash scripts/comparison/compare_evoland.sh --seed 42 --nrep 200 --evoland ../evoland-plus
```

`compare_evoland.R` can be run on its own once the Python CSVs exist:

```sh
Rscript scripts/comparison/compare_evoland.R --evoland ../evoland-plus --nrep 200
```

Requirements: a `uv`-managed Python env (`uv sync`), R with `Rcpp` and a C++
toolchain (binary packages from <https://p3m.dev>), and an `evoland-plus`
checkout (default `../evoland-plus`). Results are printed and written to
`../output/comparison/results.csv`.

## Headline findings (300 reps, seed 42)

The direct analogue of the Python reference is **`evoland uPAM normal +agg`**:
same area distribution (normal/Gaussian) and aggregation avoidance as clumpy's
`GaussianPatcher`.

| method | changed | n_patch | patch area |
|---|---|---|---|
| python clumpy (Gaussian +agg) | 10.0 | 3.0 | 3.3 |
| evoland uPAM normal +agg | 8.0 | 2.2 | 4.0 |
| evoland uPAM normal −agg | 9.5 | 1.7 | 6.3 |
| evoland uSAM (mono-pixel) | 9.8 | 3.4 | 3.3 |

- **Pivot mechanism is equivalent.** evoland `gart_cpp` reproduces the analytic
  pivot expectation and the Python GART draw (differences are pure RNG noise).
- **Rarefaction matters.** With the `1/E(σ)` rarefaction the uPAM quota tracks
  the calibrated target quantity of change (`rate × #forest` = 10); without it,
  allocation over-shoots by ~mean patch area.
- **Aggregation avoidance now implemented.** With `avoid_aggregation = TRUE`
  evoland rejects patches that would merge, yielding **more, smaller** patches
  (2.2 patches, area 4.0) — close to the Python `GaussianPatcher` (3.0, 3.3) —
  versus **fewer, larger** merged blobs without it (1.7, area 6.3). (Aggregation
  avoidance is rook-based, while the patch-count stat uses 8-connectivity, so
  diagonally touching patches are still counted as one — hence the modest patch
  counts.) Avoidance can leave the quantity slightly short of target as the map
  saturates (8.0 vs 10).
- Patch-area *distributions* are now a user choice (`area_dist`): `normal`
  matches `GaussianPatcher`; `lognormal` is right-skewed. Elongation is measured
  the same way for all tools.
