# Clumpy → Rcpp Replication: Progress Tracker

## Goal

Replicate the allocation logic from Mazy's `clumpy` Python package in an Rcpp module,
possibly using R's `terra` library, achieving numerical consistency between the Python
and R implementations.

## Plan

1. ✅ **Set up Python environment** — `uv` with local packages `hyperclip`, `ekde`, `clumpy`
2. ✅ **Dead-code removal** — use basedpyright to identify and remove unreachable code
3. ✅ **Python allocation script** — standalone script with synthetic data exercising `GaussianPatcher` / allocation pipeline
4. ✅ **Shell wrapper + Rcpp module** — C++ allocation via `Rcpp::sourceCpp()`, R driver, shell wrapper; **perfect cell-by-cell match** with Python
5. ⬜ **Full Rcpp module** — expand to cover calibration / TPE fitting, optimised data structures, `terra` raster I/O
6. ⬜ **Dinamica EGO compatibility** — map clumpy's inputs/outputs to Dinamica EGO conventions so the two tools can be used interchangeably or in tandem

---

## Step 1: Python Environment Setup — ✅ DONE

### What was done

- Created `/pyproject.toml` at the repository root (workspace-level) using `uv`.
- Python **3.10** selected (compatible with all three packages and numpy <2.0).
- All three local packages install as editable/path sources:
  - `hyperclip` (v0.2.4) — Cython C++ extension for hypercube-hyperplane volume computation
  - `ekde` (v0.0.10) — Cython C++ extension for efficient kernel density estimation
  - `clumpy` (v0.1.0) — pure Python LUCC modeling framework
- **numpy 1.26.4** resolved (last numpy 1.x release), avoiding numpy 2.0 C-API breakage.

### Changes to source code

| File | Change | Reason |
|------|--------|--------|
| `clumpy/setup.py` | `rasterio==1.1.8` → `rasterio>=1.1.8` | Exact pin couldn't compile on modern macOS / Python 3.10+ |
| `ekde/cython/ekdefunc.pyx` (4 locations) | `/ 2` → `// 2` for integer operands | Newer Cython treats `int / int` as float division; assigning to `cdef int` is a compile error |

### Key files

| File | Purpose |
|------|---------|
| `/pyproject.toml` | uv workspace definition with local sources and build deps |
| `/uv.lock` | Generated lockfile |
| `/.venv/` | Virtual environment (Python 3.10) |

### How to activate

```sh
cd /Users/jhartman/github-repos/clumpy
# Imports that touch matplotlib need a non-GUI backend:
MPLBACKEND=Agg uv run python -c "import clumpy; print('OK')"
```

---

## Step 2: Dead-Code Removal (basedpyright) — ✅ DONE

### What was done

Added `basedpyright` as a dev dependency and ran static analysis on `clumpy/clumpy/`.
Initial scan: **4701 diagnostics** (332 errors, 4369 warnings).
After cleanup: **4309 diagnostics** (310 errors, 3999 warnings).

The bulk of the remaining diagnostics are type-annotation warnings (`reportUnknownMemberType`,
`reportUnknownArgumentType`, etc.) — not dead code.  The actionable categories were
addressed as follows:

| Category | Before | After | Notes |
|----------|--------|-------|-------|
| `reportUnusedImport` | 102 | 32 | Remaining 32 are intentional re-exports in `__init__.py` files |
| `reportMissingImports` | 3 | 0 | Dead files that imported nonexistent modules were deleted |
| `reportUndefinedVariable` | 6 | 6 | Genuine bugs in broken methods (`nb_monte_carlo`, `Unbiased.allocate`, `Calibrator.__init__`) — not on our critical path, left as-is |
| `reportUnusedVariable` | 33 | 31 | Mostly loop variables (`for i, x in ...`) or tuple unpacking; harmless |
| `reportPossiblyUnboundVariable` | 18 | 18 | Mostly conditional initialisation patterns (e.g. `proba_layer`, `P_v__Y`); runtime-valid |
| `reportInvalidStringEscapeSequence` | 17 | 17 | LaTeX strings in `_cramer_mrmr.py` plot labels; cosmetic only |

### Files deleted

| File | Reason |
|------|--------|
| `clumpy/clumpy/_base/_feature_old.py` | Imports from nonexistent modules (`._layer`, `..feature_selection`); dead code |
| `clumpy/clumpy/ev_selection/_old_pipeline.py` | Imports from nonexistent `._feature_selector`; dead code |
| `clumpy/clumpy/calibration/_compute_patches.py` | Imports from nonexistent `._patch`; dead code |
| `hyperclip/cython/hyperfunc_save.pyx` | Backup copy of the Cython source |
| `ekde/=0.0.8` | Stray file (pip typo artefact) |

### Import cleanup (files edited)

Unused imports were removed from these files:

- `clumpy/clumpy/allocation/_allocator.py` — removed `State`, `create_proba_layer`, `path_split`
- `clumpy/clumpy/allocation/_unbiased.py` — removed `tqdm`, `deepcopy`, `TransitionMatrix`, `_update_P_v__Y_u`, `generalized_allocation_rejection_test`, `_weighted_neighbors_patcher`
- `clumpy/clumpy/_base/_area.py` — removed `LandUseLayer`, `Region`, `path_split`
- `clumpy/clumpy/_base/_land.py` — removed ~20 unused imports; stripped ~300 lines of commented-out code
- `clumpy/clumpy/_base/_region.py` — removed ~6 unused imports; stripped ~300 lines of commented-out code
- `clumpy/clumpy/calibration/_calibrator.py` — removed `time`, `ndimage`, `Layer`, `EVLayer`, `LandUseLayer`, `RegionsLayer`, `State`, `EVSelectors`
- `clumpy/clumpy/patch/_patcher.py` — removed `stats`, `np_drop_duplicates_from_column`
- `clumpy/clumpy/patch/_log_norm_patcher.py` — removed `scipy_lognorm` (only used in commented-out code)
- `clumpy/clumpy/transition_probability_estimation/_bayes.py` — removed `title_heading`, `Palette`
- `clumpy/clumpy/transition_probability_estimation/_tpe.py` — removed `np`, `Palette`, `title_heading`
- `clumpy/clumpy/ev_selection/_ev_selectors.py` — removed `pd`, `EVLayer`, `State`
- `clumpy/clumpy/ev_selection/_cramer_mrmr.py` — removed `KBinsDiscretizer`, `warnings`, `sys`
- `clumpy/clumpy/case/_case.py` — removed `LandUseLayer`, `RegionsLayer`, `start_log`, `stop_log`, `Palette`, `load_palette`, `load_transition_matrix`, `datetime`, `json`, `logging`
- `clumpy/clumpy/metrics/_rec.py` — removed `simpson`

### Known bugs found (not fixed — not on critical path)

| Location | Issue |
|----------|-------|
| `allocation/_allocator.py:91,93` (`nb_monte_carlo`) | References `lul_origin` which is not a parameter or local variable |
| `allocation/_unbiased.py:68,74,76` (`Unbiased.allocate`) | References `tm`, `mask`, `features` which are not defined |
| `calibration/_calibrator.py:29` (`Calibrator.__init__`) | References `FeatureSelectors` which is not imported |
| `calibration/_calibrator.py:68` (`Calibrator.check`) | References `Pipeline` which is not imported |

These are broken methods that would crash at runtime but are not called by any current working path.

---

## Step 3: Python Allocation Script — ✅ DONE

### What was done

Created `scripts/run_allocation.py` — a standalone script that exercises the full
clumpy allocation pipeline with synthetic data, bypassing the broken `Calibrator`/`Case`
orchestration layer and assembling the pipeline components manually.

The script:

1. **Generates synthetic data** — a 20×20 land-use grid (forest/urban/water) with two
   explanatory variables (distance-to-urban, random slope).
2. **Fits a Bayes TPE** using `ekde.KDE` on calibration data (observed forest→urban
   transitions).
3. **Computes per-pixel transition probabilities** P(v|u,Z) via Bayes rule.
4. **Runs GART** (`generalized_allocation_rejection_test`) to select pivot pixels.
5. **Grows patches** around pivots using `GaussianPatcher` (the core inner loop in
   `Patcher.allocate()`).
6. **Saves all inputs and outputs** as `.npz` and CSV files for later comparison with R.

### Key findings during implementation

- The `Calibrator` class has naming inconsistencies: `__init__` stores
  `self.transition_probability_estimator` but other methods reference `self.tpe`;
  similarly `self.ev_selector` vs `self.feature_selector`. These are bugs that would
  crash at runtime. The script bypasses `Calibrator` entirely.
- `Patcher.allocate()` expects `self.initial_state` and `self.final_state` to be set
  externally (not in `__init__`). The script sets them directly on the patcher instance.
- `ekde/ekde/base.py` used `np.bool` which was removed in numpy 1.24+. Fixed to
  `np.bool_` in both the source file and the installed copy in `.venv`.

### Changes to source code

| File | Change | Reason |
|------|--------|--------|
| `ekde/ekde/base.py` | `np.bool` → `np.bool_` | `np.bool` removed in numpy ≥1.24; causes `AttributeError` at runtime |

### Pipeline results (seed=42, 20×20 grid)

- 351 forest pixels in initial map, 10 observed forest→urban transitions in calibration
- Bayes TPE fitted with ekde KDE (box kernel, Terrel bandwidth)
- GART selected 11 pivot pixels for transition
- 3 of 11 patches succeeded (aggregation avoidance rejects the rest), 10 pixels allocated
- **Deterministic**: two runs with the same seed produce bit-identical results

### Output files

| File | Purpose |
|------|---------|
| `scripts/run_allocation.py` | The allocation script |
| `scripts/output/allocation_seed42.npz` | All arrays in numpy format |
| `scripts/output/patch_log_seed42.txt` | Human-readable patch log |
| `scripts/output/csv/*.csv` | All arrays as CSV for R consumption |
| `scripts/output/csv/params.txt` | Scalar parameters (seed, grid size, patcher config) |
| `scripts/output/csv/patch_log.csv` | Patch log as proper CSV |

### How to run

```sh
cd /Users/jhartman/github-repos/clumpy
MPLBACKEND=Agg uv run python scripts/run_allocation.py --seed 42 --verbose 2
```

### Core allocation architecture (from code reading)

The allocation pipeline works as follows:

1. **Calibration**: Given initial + final LUC maps and explanatory variables (EVs),
   fit a Bayesian transition probability estimator (`Bayes` using `ekde.KDE`) per
   land/transition.

2. **Transition probability estimation**: For a new LUC map, compute per-pixel
   transition probabilities P(v|u,Y) using the calibrated KDE densities and Bayes rule.

3. **Allocation** (`Unbiased` or `UnbiasedMonoPixel`):
   - Use GART (generalized allocation rejection test) to sample which pixels transition.
   - For patch-based allocation (`Unbiased`), grow patches around kernel pixels using
     `Patcher.allocate()`, which:
     - Samples a patch area from the configured distribution (Gaussian/LogNorm/Bootstrap).
     - Grows the patch by iteratively selecting neighbours weighted by their transition
       probability and eccentricity constraint.
     - Checks aggregation avoidance.

4. **Patch growth** (`_patcher.py:Patcher.allocate`): This is the core inner loop we
   need to port. It uses a convolution-based neighbour finder, probability weighting,
   and eccentricity-based shape control.

---

## Step 4: Shell Wrapper + Rcpp Module — ✅ DONE

### What was done

Created a complete cross-language verification pipeline:

1. **`scripts/rcpp/allocate.cpp`** — standalone C++ source compiled at runtime via
   `Rcpp::sourceCpp()`.  Implements:
   - `gart_cpp()` / `gart_with_u_cpp()` — GART (generalized allocation rejection test).
   - `patch_allocate_cpp()` — single-patch growth (convolution-based neighbour finding,
     moment-based eccentricity weighting, aggregation avoidance, hollow filling).
   - `run_allocation_cpp()` — orchestrator that iterates over pivot pixels and calls
     `patch_allocate_cpp` for each.

2. **`scripts/run_allocation.R`** — R driver script that:
   - Reads all CSV inputs produced by the Python pipeline (Step 3).
   - Compiles `allocate.cpp` via `Rcpp::sourceCpp()`.
   - Runs the allocation with R's own RNG (structural comparison).
   - If `areas_sampled.csv` exists (exported by Python), replays the exact draws
     for a **deterministic cell-by-cell comparison**.
   - Prints patch logs, diff grids, and summary statistics.

3. **`scripts/compare.sh`** — end-to-end shell wrapper that:
   - Runs `run_allocation.py` (Python, via `uv run`).
   - Activates `rv`, then runs `run_allocation.R` (R, via `Rscript`).
   - Performs a final Python-based cell-by-cell diff of both output maps.

4. **`scripts/run_allocation.py` (updated)** — instrumented `GaussianPatcher._sample()`
   to record the exact area/eccentricity drawn for each pivot, saved as
   `areas_sampled.csv` and `eccentricities_sampled.csv` for deterministic R replay.

### Key findings during implementation

- **Float vs int area comparison**: Python's `Patcher.allocate()` uses
  `while len(J_allocated) < area` where `area` is a float (e.g. 2.249).
  An initial C++ implementation rounded to `int`, causing patches to be 1 pixel
  smaller.  Fixed by keeping the float comparison — **this was the only logic bug**.

- **RNG divergence is expected**: R and numpy use different MT19937 implementations,
  so `set.seed(42)` and `np.random.seed(42)` produce different uniform sequences.
  The deterministic replay sidesteps this by passing Python's exact area draws to the
  C++ code.

- **No hollow-fill RNG consumed**: In the seed=42 test case, no patch growth step
  triggers the hollow-fill branch (`np.random.choice(j_hollows)`), so the area draws
  are the only stochastic element.  The argmax-based neighbour selection is fully
  deterministic given the same probability map and area.

### Results (seed=42, 20×20 grid)

| Metric | Python | R (own RNG) | R (replay) |
|--------|--------|-------------|------------|
| Total allocated | 10 | 10 | 10 |
| Patches succeeded | 3/11 | 4/11 | 3/11 |
| Cells differing vs Python | — | 6/400 | **0/400** |

**Perfect cell-by-cell match** when using identical random draws.

### Changes to source code

| File | Change | Reason |
|------|--------|--------|
| `scripts/run_allocation.py` | Added instrumented `_sample()` + CSV export of area/eccentricity draws | Enables deterministic R replay |

### New files

| File | Purpose |
|------|---------|
| `scripts/rcpp/allocate.cpp` | Rcpp C++ source: GART + patch growth + allocation driver |
| `scripts/run_allocation.R` | R driver: load CSVs, compile Rcpp, run allocation, compare |
| `scripts/compare.sh` | Shell wrapper: Python → R → diff |

### How to run

```sh
cd /Users/jhartman/github-repos/clumpy

# Full pipeline (Python + R + comparison):
bash scripts/compare.sh --seed 42 --verbose 2

# R only (assumes Python CSVs already exist):
rv activate
Rscript scripts/run_allocation.R --pydir scripts/output/csv --verbose 2

# Python only (regenerate CSVs):
MPLBACKEND=Agg uv run python scripts/run_allocation.py --seed 42 --verbose 2
```

---

## Step 5: Full Rcpp Module — ⬜ TODO

- Expand Rcpp module to cover calibration / TPE fitting (currently uses Python-exported probabilities).
- Optimised data structures (priority queues, neighbour lookup) in C++ for larger grids.
- Read/write rasters via `terra`'s R-level SpatRaster ↔ matrix interface.
- Package skeleton (`R CMD build`) if needed for deployment.

---

## Step 6: Dinamica EGO Compatibility — ⬜ TODO

Dinamica EGO is a widely-used LUCC modelling platform.  Making clumpy inputs and
outputs interchangeable with Dinamica EGO conventions would allow practitioners to
compare models or use clumpy as a drop-in calibration / allocation step within a
Dinamica workflow.

### Planned work

- **Transition matrix format**: verify that clumpy's transition matrix CSV conventions
  match what Dinamica EGO expects; add import/export helpers if needed.
- **Patcher parameter tables**: map clumpy's `GaussianPatcher` / `LogNormPatcher`
  configuration (mean area, variance, isometry/eccentricity) to Dinamica's patch-size
  and isometry table format (`.csv`).
- **Probability maps**: ensure per-pixel transition probability rasters produced by
  clumpy can be consumed directly by Dinamica EGO's allocation step (and vice versa).
- **Aggregation / expander flags**: confirm that clumpy's `avoid_aggregation` flag
  corresponds to Dinamica's expander/patcher distinction; document the mapping.
- **Two-pass allocation**: investigate whether a two-pass scheme (expander first,
  then patcher) is needed to replicate Dinamica EGO's patch-seeding behaviour, and
  implement if required.
- **Validation**: run both tools on a common test case and compare allocated maps
  (patch count, size distribution, spatial pattern metrics).

---

## Files That May Still Be Deletable

| File | Reason |
|------|--------|
| `ekde/ekde/new_whitening_transformer_illustration.py` | Illustration script, not library code |
| `ekde/setup_annotate.py` / `ekde/setup_annotate.py.save` | Development-only Cython annotation helpers |
| `hyperclip/setup_annotate.py` | Development-only Cython annotation helper |

---

*Last updated: added Step 6 (Dinamica EGO compatibility) to plan.*
