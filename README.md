# clumpy (fork) — Modern Python + Rcpp Allocation

This is a **fork of the [clumpy](https://gitlab.inria.fr/fmazy/clumpy/) Land Use and
Cover Change (LUCC) modelling framework** by François-Rémi Mazy and Pierre-Yves
Longaretti (INRIA/LJK Grenoble).  It bundles three packages that were originally
developed as separate repositories:

| Sub-package  | Upstream                                                                   | Role                                                 |
| ------------ | -------------------------------------------------------------------------- | ---------------------------------------------------- |
| `clumpy/`    | [gitlab.inria.fr/fmazy/clumpy](https://gitlab.inria.fr/fmazy/clumpy)       | LUCC modelling (calibration + allocation)            |
| `ekde/`      | [gitlab.inria.fr/fmazy/ekde](https://gitlab.inria.fr/fmazy/ekde)           | Efficient Kernel Density Estimation (Cython/C++)     |
| `hyperclip/` | [gitlab.inria.fr/fmazy/hyperclip](https://gitlab.inria.fr/fmazy/hyperclip) | Hypercube–hyperplane volume computation (Cython/C++) |

---

## Goals of this fork

### 1 · Modern Python compatibility

The upstream packages target Python 3.8 / NumPy 1.x and use Cython idioms that fail
on current toolchains.  This fork:

- Manages all three packages as a **[uv](https://github.com/astral-sh/uv) workspace**
  with a single `pyproject.toml` at the repository root.
- Pins **NumPy < 2.0** (last stable 1.x release: 1.26.4) to avoid C-API breakage in
  the Cython extensions.
- Fixes integer-division Cython compile errors in `ekde` (`/ 2` → `// 2`).
- Loosens `rasterio==1.1.8` to `rasterio>=1.1.8` so the package builds on macOS /
  Python 3.10+.
- Fixes a `np.bool` → `np.bool_` runtime error in `ekde` (removed in NumPy 1.24).

### 2 · Rcpp translation of the allocation kernel

The patch-growth allocation loop (`Patcher.allocate`) is the computational hot-path
and a natural candidate for a standalone C++ implementation callable from R.  This
fork provides:

- A **standalone Python reference script** (`scripts/run_allocation.py`) that exercises
  the full pipeline (KDE calibration → Bayes TPE → GART → GaussianPatcher) on
  synthetic data, saving all inputs and outputs as CSV files.
- An **Rcpp C++ module** (`scripts/rcpp/allocate.cpp`, ~430 lines) implementing GART
  and the patch-growth loop (`patch_allocate_cpp`, `run_allocation_cpp`), compiled at
  runtime via `Rcpp::sourceCpp()`.
- An **R driver** (`scripts/run_allocation.R`) that reads the Python-generated CSVs,
  compiles the Rcpp module, runs the allocation, and compares outputs.
- A **shell wrapper** (`scripts/compare.sh`) that runs the full Python → R → diff
  pipeline end-to-end.

Deterministic replay (passing Python's exact area draws to C++) achieves a **0/400
cell difference** between Python and R on the reference test case.

---

## Repository layout

```text
.
├── pyproject.toml          # uv workspace definition (Python deps for all 3 packages)
├── uv.lock                 # pinned Python dependency lockfile
├── rproject.toml           # rv R package manager definition
├── rv.lock                 # R dependency lockfile
├── PROGRESS.md             # detailed engineering log
│
├── clumpy/                 # clumpy Python package (upstream fork)
│   ├── clumpy/             # source tree
│   │   ├── allocation/     # GART + Unbiased allocator
│   │   ├── calibration/    # Calibrator, Bayes TPE wrapper
│   │   ├── patch/          # Patcher, GaussianPatcher, LogNormPatcher
│   │   ├── density_estimation/
│   │   ├── layer/          # LandUseLayer, EVLayer, ProbaLayer
│   │   └── transition_probability_estimation/
│   └── setup.py
│
├── ekde/                   # Efficient KDE (Cython extension, upstream fork)
│   ├── ekde/
│   └── cython/ekdefunc.pyx
│
├── hyperclip/              # Hypercube clipping (Cython extension, upstream fork)
│   ├── hyperclip/
│   └── cython/hyperfunc.pyx
│
└── scripts/
    ├── run_allocation.py   # Python reference pipeline (synthetic data)
    ├── run_allocation.R    # R driver: reads CSVs, runs Rcpp, compares outputs
    ├── compare.sh          # End-to-end shell wrapper (Python → R → diff)
    └── rcpp/
        └── allocate.cpp    # Rcpp C++ allocation module (~430 lines)
```

---

## Quick start (Python)

Requires **Python 3.10+** and [uv](https://github.com/astral-sh/uv).

```sh
# Install all packages (builds Cython extensions automatically)
uv sync

# Run the reference allocation script
MPLBACKEND=Agg uv run python scripts/run_allocation.py --seed 42 --verbose 2
```

Output is written to `scripts/output/` (`.npz` + CSV files).

> **macOS note:** Set `MPLBACKEND=Agg` to prevent matplotlib from trying to open a
> GUI window in headless/scripted contexts.

---

## Quick start (R / Rcpp)

Requires **R** with the `Rcpp` package.  The repo uses
[rv](https://github.com/A2-ai/rv) for R dependency management, but any R install
with Rcpp will work.

```sh
# 1. Generate Python reference outputs (CSVs):
MPLBACKEND=Agg uv run python scripts/run_allocation.py --seed 42

# 2. Run the Rcpp allocation and compare:
Rscript scripts/run_allocation.R --pydir scripts/output/csv --verbose 2

# Or run the full pipeline in one step:
bash scripts/compare.sh --seed 42 --verbose 2
```

---

## Allocation pipeline architecture

The allocation workflow has four stages, each corresponding to a clumpy module:

```
 Observed land-use maps + explanatory variables (EVs)
            │
            ▼
  ┌─────────────────────┐
  │   Calibration       │  Bayes TPE fit using ekde.KDE
  │   (_bayes.py)       │  → density f(Z | u→v) per transition
  └─────────────────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Transition         │  Bayes rule: P(v|u,Z) ∝ f(Z|u→v) · P(v|u)
  │  Probability Maps   │  hyperclip normalises the KDE weights
  └─────────────────────┘
            │
            ▼
  ┌─────────────────────┐
  │  GART               │  Generalised Allocation Rejection Test
  │  (_gart.py)         │  Stochastic pixel sampling from P(v|u,Z)
  └─────────────────────┘
            │  pivot pixels
            ▼
  ┌─────────────────────┐
  │  Patch growth       │  GaussianPatcher / LogNormPatcher
  │  (patch/_patcher.py)│  Iterative weighted-neighbour expansion
  │                     │  with eccentricity + aggregation constraints
  └─────────────────────┘
            │
            ▼
       Allocated land-use map
```

The **patch growth loop** (`Patcher.allocate`) is the stochastic inner kernel that is
replicated in C++/Rcpp.  It:

1. Samples a patch **area** from a Gaussian (or log-normal) distribution.
2. Identifies **neighbour pixels** via 3×3 rook/queen convolution.
3. Selects the next pixel by **probability-weighted sampling** subject to an
   **eccentricity constraint** (moment-of-inertia based).
4. Enforces **aggregation avoidance** (patches may not merge into already-transited
   areas).

---

## Rcpp module (`scripts/rcpp/allocate.cpp`)

Exported functions (compiled via `Rcpp::sourceCpp()`):

| Function | Description |
|----------|-------------|
| `gart_cpp(P, J, area)` | Basic GART: select pivot pixels |
| `gart_with_u_cpp(P, J, area, u_draws)` | GART with pre-supplied uniform draws (deterministic replay) |
| `patch_allocate_cpp(...)` | Grow a single patch around one pivot pixel |
| `run_allocation_cpp(...)` | Iterate over all pivots; return allocated grid |

---

## Current status

| Milestone | Status |
|-----------|--------|
| uv workspace + Cython builds on Python 3.10 | ✅ Done |
| Dead-code removal (basedpyright) | ✅ Done |
| Python reference allocation script | ✅ Done |
| Rcpp C++ module + R driver | ✅ Done — 0/400 cell diff on reference case |
| Full Rcpp pipeline (calibration + TPE in C++) | ⬜ Planned |
| terra raster I/O in R driver | ⬜ Planned |
| Optimised data structures (priority queue, spatial index) | ⬜ Planned |
| Dinamica EGO input/output compatibility | ⬜ Planned |

See [`PROGRESS.md`](PROGRESS.md) for a detailed engineering log.

---

## Known issues / limitations

- The `Calibrator` class in upstream clumpy has internal naming inconsistencies
  (`self.tpe` vs `self.transition_probability_estimator`, `self.ev_selector` vs
  `self.feature_selector`) that cause runtime crashes.  The reference script bypasses
  `Calibrator` and assembles the pipeline components manually.
- A few methods (`nb_monte_carlo`, `Unbiased.allocate`) reference undefined variables;
  they are not on the critical allocation path and have been left unfixed.
- The Rcpp module currently relies on Python-exported probability maps (CSVs); the
  calibration and TPE stages are not yet ported to C++/R.

---

## References

- Mazy, F.-R. and Longaretti, P.-Y. (2022). *A Formally Correct and Algorithmically
  Efficient LULC Change Model-building Environment.* Proceedings of the 8th
  International Conference on Geographical Information Systems Theory, Applications and
  Management (GISTAM), pp. 25–36.
  doi: [10.5220/0011000000003185](https://doi.org/10.5220/0011000000003185)

- Cho, Y. and Kim, S. (2020). *Volume of Hypercubes Clipped by Hyperplanes and
  Combinatorial Identities.*
  [arXiv:1512.07768](https://arxiv.org/abs/1512.07768)

---

## Original authors

François-Rémi Mazy and Pierre-Yves Longaretti
Université Grenoble Alpes / CNRS / Inria / LJK — Grenoble, France
