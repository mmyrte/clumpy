# Plan: deterministic pixel-perfect equivalence (evoland ↔ clumpy)

Goal: deterministically demonstrate that evoland re-implements the CLUMPY
allocation faithfully, cell-by-cell, controlling for RNG.

## Status

| Stage | Status |
|---|---|
| **Pivot test (MuST / GART)** | ✅ DONE — `compare_must_exact.{py,R,sh}` replays clumpy's exact uniforms through evoland's `must_cpp` (the new `u=` argument) and asserts identical per-cell assignment. 2000 rows, 0 differing. |
| **Patch growth (the patcher)** | ⬜ TODO — this document. |
| **Full allocation map** | ⬜ TODO — follows once patch growth is pinned. |

The pivot test is clean because the inverse-CDF mapping is a pure function of
`(P, u)`, so feeding both implementations the same uniforms gives an exact check.
Patch growth is harder, because **evoland's grower is a re-implementation, not a
line-by-line port** of clumpy's `_weighted_neighbors_patcher`. The differences
below must be either avoided (restricted test scenarios) or reconciled before a
general cell-by-cell match is meaningful.

## Known differences to reconcile (evoland `grow_one_patch` vs clumpy `_weighted_neighbors_patcher`)

1. **Hollow-fill.** clumpy fills "hollows" (border cells almost surrounded by the
   patch, `B >= n_neighbors_to_fill`) by `np.random.choice(j_hollows)` — a random
   pick. evoland has **no** hollow-fill branch. → Avoid by testing only small
   convex patches that never create hollows, or port hollow-fill (with replayable
   RNG) into a reference build.
2. **Tie-breaking.** clumpy picks `j_neighbors[np.argmax(P)]` where `j_neighbors`
   is in convolution/`np.where` order; evoland iterates an ordered
   `std::set<int>` (ascending cell index) and keeps the first max. On ties these
   differ. → Use scenarios with a **unique** argmax (distinct candidate
   potentials), or align the tie-break rule.
3. **Eccentricity/elongation.** Both compute `e = 1 - sqrt(lambda_min/lambda_max)`
   from the second central moments; values should match. But the score is
   `prob / |ecc_target - e|` in clumpy (no epsilon → `de == 0` gives `inf`, that
   neighbour always wins) vs `prob / (|...| + 1e-6)` in evoland. → Pick an
   `ecc_target` that never produces `de == 0` for candidate shapes in the test.
4. **Partial vs all-or-nothing on "no eligible neighbour".** clumpy returns a
   *failed* patch (allocates nothing) whenever no suitable neighbour remains,
   regardless of aggregation settings; evoland only fails in that case when
   `avoid_aggregation = TRUE` (otherwise it keeps the partial patch). → Compare
   with `avoid_aggregation = TRUE` so both are all-or-nothing, or restrict to
   patches that always reach their target area.
5. **Area draw.** clumpy `GaussianPatcher` draws `area ~ Normal(mean, cov)` (cov =
   std-dev), clamped `>= 1`; evoland draws `Normal(mean, sd = sqrt(area_var))`.
   For the grower comparison this is sidestepped: evoland's `grow_patch_cpp`
   takes an explicit integer `target_area`, so export clumpy's *sampled* area and
   pass it directly.

## Recommended approach: restricted-equivalence battery (tractable)

Compare the two **single-patch growers** directly, on a battery of hand-built
deterministic scenarios chosen so the differences above don't apply. Assert the
allocated cell *sets* are identical.

### Python side (`compare_patch_exact.py`)
For each scenario, set up `clumpy.patch._patcher` and call the patch grower
(`Patcher.allocate` / `_weighted_neighbors_patcher`) with:
- `equi_neighbors_proba = False`, `proceed_even_if_no_probability = True`,
- `nb_of_missing_to_fill` large enough that **hollows never trigger**,
- `avoid_aggregation = TRUE`,
- a fixed integer area (monkeypatch `_sample` to return the scenario's area and
  eccentricity, as `run_allocation.py` already does),
- a proba layer with **distinct** values along the intended growth direction so
  the argmax is unique.
Export, per scenario: `map_i`, `map_f` (with any pre-existing foreign patch),
`pivot`, `area`, `ecc`, `proba` (full grid), and the resulting `J_allocated`
(sorted 0-based cell set).

### R side (`compare_patch_exact.R`)
`sourceCpp` evoland's `src/alloc_clumpy.cpp`; for each scenario build the
row-major vectors and call `grow_patch_cpp(..., target_area = area,
elongation = ecc, avoid_aggregation = TRUE)`; compare `sort(patch)` (converted to
0-based) to the Python `J_allocated`.

### Scenario battery (all hollow-free, unique-argmax)
- a straight line (1×N), pivot at one end;
- an L / staircase via a potential gradient that forces the path;
- a compact square block (area = perfect square) with `ecc_target = 0`;
- a patch near the raster boundary (clipped neighbourhood);
- a pivot adjacent to a pre-existing foreign patch with `avoid_aggregation = TRUE`
  ⇒ both must FAIL (empty result);
- a pivot whose enclosed free area < target ⇒ both FAIL.

Assert exact set equality for each. Scenarios that would hit hollow-fill or a tie
are **out of scope** and should be documented as intentional re-implementation
differences (see list above), not silently skipped.

## Optional: faithful-port reference (stronger, more work)
Port clumpy's `_weighted_neighbors_patcher` verbatim into a standalone Rcpp file
(as `scripts/exploration/allocate.cpp` already did for the original grower),
expose its draws for replay, and diff evoland's grower against it on *random*
hollow-free inputs (not just hand-built ones). This widens coverage but still
only holds where the two algorithms are defined to agree (no hollows).

## Optional: full-map deterministic replay (end-to-end)
With forced potentials (0/1 ⇒ deterministic MuST via the `u=` replay or extreme
values), fixed areas (`area_var = 0`), `shuffle = FALSE`, and
`avoid_aggregation = TRUE`, both a clumpy-scripted pipeline and evoland's
`allocate_clumpy_cpp` become fully deterministic; compare the whole posterior
map. This is the ultimate check but requires the Python allocate path to accept
the same forced inputs; build it only after the single-patch battery passes.

## Acceptance
- `compare_patch_exact.{py,R}` added under `scripts/comparison/`, wired into a
  `.sh` like `compare_must_exact.sh`.
- Every in-scope scenario reports 0 differing cells.
- The README's findings table gains a "patch growth: exact (restricted)" row.
