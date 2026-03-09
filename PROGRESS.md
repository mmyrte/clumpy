# Clumpy → Rcpp Replication: Progress Tracker

## Goal

Replicate the allocation logic from Mazy's `clumpy` Python package in an Rcpp module,
possibly using R's `terra` library, achieving numerical consistency between the Python
and R implementations.

## Plan

1. ✅ **Set up Python environment** — `uv` with local packages `hyperclip`, `ekde`, `clumpy`
2. ⬜ **Dead-code removal** — use basedpyright to identify and remove unreachable code
3. ⬜ **Python allocation script** — standalone script with synthetic data exercising `GaussianPatcher` / allocation pipeline
4. ⬜ **Shell wrapper** — script that runs both the Python allocator and the R function, compares outputs
5. ⬜ **Rcpp module** — R script with Rcpp module implementing the core allocation possibly making use of `SpatRaster` from `terra` for raster handling.

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

## Step 2: Dead-Code Removal (basedpyright) — ⬜ TODO

Plan:
- Run basedpyright on the `clumpy/clumpy/` package tree.
- Identify unreachable code, unused imports, dead branches.
- Remove conservatively — the codebase uses dynamic patterns (star imports, metaclasses)
  that may cause false positives.
- Verify `import clumpy` still works after each round of changes.
- Be prepared to revert if the analysis breaks runtime behaviour.

---

## Step 3: Python Allocation Script — ⬜ TODO

Plan:
- Identify the minimal pipeline to run an allocation:
  calibration → transition probability estimation → density estimation → allocation.
- Write a self-contained script that generates synthetic raster data (land-use map +
  explanatory variables) and runs `GaussianPatcher`-based allocation.
- Key entry points to investigate:
  - `clumpy.case._engine.Engine` — appears to be the top-level orchestrator
  - `clumpy.allocation._allocator.Allocator` — the allocation dispatcher
  - `clumpy.patch._gaussian_patcher.GaussianPatcher` — patch-level change placement

---

## Step 4: Shell Wrapper — ⬜ TODO

- Wrap the Python script from step 3 in a shell script.
- Add an R invocation of the Rcpp module from step 5.
- Compare outputs numerically (cell-by-cell diff of output rasters).

---

## Step 5: Rcpp Module — ⬜ TODO

- R package skeleton with `Rcpp` and `terra` dependencies.
- Port the core allocation loop (probability sorting + patch placement).
- Optimised data structures (priority queues, neighbour lookup) in C++.
- Read/write rasters via `terra`'s C++ API or R-level SpatRaster ↔ matrix.

---

## Files That May Be Deletable

These appear to be ad-hoc or stale artefacts found during exploration:

| File | Reason |
|------|--------|
| `clumpy/after_alloc.pdf` | Looks like a one-off visualisation |
| `clumpy/before_alloc.pdf` | Looks like a one-off visualisation |
| `clumpy/architecture.drawio` | Diagram, not used by code |
| `clumpy/new_params.json` | Looks like a one-off experiment config |
| `clumpy/clumpy/_base/_feature_old.py` | Filename suggests deprecated |
| `ekde/=0.0.8` | Stray file (looks like a pip typo artifact) |
| `ekde/ekde/new_whitening_transformer_illustration.py` | Illustration script, not library code |
| `ekde/setup_annotate.py` / `ekde/setup_annotate.py.save` | Development-only Cython annotation helpers |
| `hyperclip/setup_annotate.py` | Development-only Cython annotation helper |
| `hyperclip/cython/hyperfunc_save.pyx` | Backup copy of the Cython source |

---

*Last updated after completing Step 1.*
