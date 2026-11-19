# clumpy: reference implementation for the allocators of evoland.plus

This repository is a fork of [CLUMPY](https://gitlab.inria.fr/fmazy/clumpy), the land use and cover change model by François-Rémi Mazy and Pierre-Yves Longaretti (Université Grenoble Alpes, CNRS, Inria, LJK).
CLUMPY is the reference implementation of the unbiased allocation methods uSAM and uPAM of [Mazy (2022)](https://theses.hal.science/tel-04382012v1), Ch. 3.

The fork exists for one purpose: to run CLUMPY on a current Python toolchain, so that the C++ reimplementation of these allocators in [evoland.plus](https://github.com/ethzplus/evoland.plus) can be compared against it.
It is not a maintained version of CLUMPY and adds no methods to it.

## What is in here

| Path | Content | License |
| --- | --- | --- |
| `clumpy/` | CLUMPY: calibration, transition probability estimation, allocation | GPL-3.0 (`clumpy/LICENSE`; the classifier in `clumpy/setup.py` says MIT) |
| `ekde/` | Efficient kernel density estimation (Cython), used by CLUMPY | BSD-3-Clause (`ekde/LICENSE.txt`) |
| `hyperclip/` | Volume of hypercubes clipped by hyperplanes (Cython), used by `ekde` | GPL-3.0 (`hyperclip/LICENSE.txt`) |
| `scripts/` | Comparison scripts added by this fork, see below | |
| `docs/` | Notes of the fork: the [engineering log](docs/engineering-log.md) and an early [code reading of the pivot test](docs/pivot-mechanism-verification.md) | |
| `mazy-thesis.pdf`, `mazy-thesis.txt` | The thesis the methods are specified in, and its plain text | |

The three packages are separate repositories upstream ([clumpy](https://gitlab.inria.fr/fmazy/clumpy), [ekde](https://gitlab.inria.fr/fmazy/ekde), [hyperclip](https://gitlab.inria.fr/fmazy/hyperclip)); here they form one [uv](https://docs.astral.sh/uv/) workspace.

## Changes to the upstream code

The first commit of this repository (`d47b21d`) holds the upstream sources as imported; the upstream revisions are not recorded in the history.
All later changes to the package code were made in two commits, and are listed file by file in the [engineering log](docs/engineering-log.md):

- **Build fixes**, so that the packages install on Python 3.10 and later with NumPy 1.26:
  - `clumpy/setup.py`: `rasterio==1.1.8` relaxed to `rasterio>=1.1.8`;
  - `ekde/cython/ekdefunc.pyx`: integer division written as `//` in four places (a compile error in current Cython);
  - `ekde/ekde/base.py`: `np.bool` replaced by `np.bool_` (removed in NumPy 1.24).
- **Dead code removed** (commit `79552cb`): three modules that imported modules that do not exist, a backup copy of a Cython source, unused imports and commented-out code.
  The files touched were also reformatted, so the diff of this commit is larger than the change.

According to that log, no algorithm was changed; this has not been checked again against the upstream repositories.
The package code has not changed since commit `e9f73b9`, the revision [evoland-experiments](https://github.com/mmyrte/evoland-experiments) pins.

Known defects of the upstream code, left as they are:

- `clumpy.allocation.Unbiased.allocate`, the iterative uPAM allocator, does not run: its body uses the variables `tm`, `mask` and `features`, which are not defined.
  The comparisons therefore assemble uPAM from its parts, the pivot test (`generalized_allocation_rejection_test`) and the patch grower (`Patcher.allocate`).
- `Calibrator` and `nb_monte_carlo` refer to undefined names as well; `scripts/run_allocation.py` sets up the calibration by hand.

## Comparisons

All scripts run from the repository root.
They need uv, R with the Rcpp package and a C++ compiler; the comparisons with evoland.plus also need a source checkout of it, by default in `../evoland.plus`.

```sh
uv sync
```

| Script | What it compares | Result |
| --- | --- | --- |
| `scripts/comparison/compare_must_exact.sh` | The pivot test (MuST, Mazy 2022, App. 3.B; CLUMPY calls it GART): CLUMPY's uniform draws are replayed through `must_cpp()` of evoland.plus | identical in all 2000 rows |
| `scripts/comparison/compare_evoland.sh` | Quantity of change and patch statistics of uSAM and uPAM in evoland.plus against one CLUMPY allocation, over many seeds | statistical, see [`scripts/comparison/README.md`](scripts/comparison/README.md) |
| `scripts/exploration/explore.sh` | A standalone Rcpp port of CLUMPY's patch grower (`scripts/exploration/allocate.cpp`, not the evoland.plus code) against CLUMPY, replaying its area draws | identical in all 400 cells |

These results were reproduced on 2026-11-19 with evoland.plus at commit `b06e3d1`.

The comparison of the allocators as evoland.plus runs them is part of [evoland-experiments](https://github.com/mmyrte/evoland-experiments), which installs the packages from this repository: `2026-09-paper-figures/010-verify-clumpy.r` replays CLUMPY's random numbers into the uSAM allocator and compares the patch grower scenario by scenario, and `010-verify-upam-bias.r` measures the effect of the fixed potentials in uPAM.
Where evoland.plus departs from CLUMPY on purpose, the differences are listed in its documentation, `?alloc_clumpy_spec`.

## How to cite

Cite the methods and the software by their authors: Mazy (2022) for uSAM, uPAM and the patch construction, Mazy and Longaretti (2022) for CLUMPY.
To refer to the exact code the comparisons ran against, name this repository and the commit, see [`CITATION.cff`](CITATION.cff).

## References

- Mazy, F.-R. (2022). *Towards a coherent algorithmic theory and implementation of pattern-based land use and land cover change modeling.* PhD thesis, Université Grenoble Alpes, NNT 2022GRALM043. <https://theses.hal.science/tel-04382012v1>
- Mazy, F.-R. and Longaretti, P.-Y. (2022). A formally correct and algorithmically efficient LULC change model-building environment. *Proceedings of the 8th International Conference on Geographical Information Systems Theory, Applications and Management (GISTAM)*, 25–36. <https://doi.org/10.5220/0011000000003185>
- Cho, Y. and Kim, S. (2020). Volume of hypercubes clipped by hyperplanes and combinatorial identities. <https://arxiv.org/abs/1512.07768>
