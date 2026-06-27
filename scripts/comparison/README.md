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

## Headline findings (seed 42)

- **Pivot mechanism is equivalent.** evoland `gart_cpp` reproduces the analytic
  pivot expectation and the Python GART draw (differences are pure RNG noise).
- **Rarefaction matters.** `uSAM(rarefy=TRUE)` and `uPAM` track the calibrated
  target quantity of change (`rate × #forest`); `uPAM` enforces it exactly via
  the quota. Without the `1/E(σ)` rarefaction, `uSAM` over-allocates.
- **Aggregation avoidance is the main behavioural gap.** The Python
  `GaussianPatcher` runs with `avoid_aggregation=TRUE` and rejects patches that
  would merge, producing several small patches; evoland's grower has no merge
  avoidance, so adjacent patches coalesce into fewer, larger blobs. This is the
  "no merge-failure rollback" point from
  [`evoland-plus/dev/pivot-mechanism-verification.md`](https://github.com/ethzplus/evoland-plus/blob/copilot/discuss-transition-probability-estimation/dev/pivot-mechanism-verification.md)
  and the main thing to weigh next (expander vs patcher semantics).
- Patch-area *distributions* differ by construction (Python Gaussian vs evoland
  log-normal); elongation is measured the same way for both.
