#!/usr/bin/env python3
"""
Step 3 — Standalone synthetic allocation script.

Exercises the core clumpy allocation pipeline with synthetic data:
  1. Builds synthetic land-use and explanatory-variable rasters.
  2. Fits a Bayes TPE (using ekde KDE) on calibration data.
  3. Computes transition probabilities for allocation pixels.
  4. Runs GART to select pivot pixels.
  5. Grows patches around pivots using GaussianPatcher.
  6. Saves all inputs and outputs as .npz for later comparison with R.

Usage:
    cd /Users/jhartman/github-repos/clumpy
    MPLBACKEND=Agg uv run python scripts/run_allocation.py [--seed 42] [--outdir scripts/output]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# REPO_ROOT is used only for output paths; package imports come from the
# uv-managed virtualenv (which has the Cython extensions built).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("MPLBACKEND", "Agg")


from clumpy.allocation._gart import generalized_allocation_rejection_test
from clumpy.layer._ev_layer import EVLayer, get_bounds
from clumpy.layer._land_use_layer import LandUseLayer
from clumpy.layer._layer import Layer
from clumpy.layer._proba_layer import create_proba_layer
from clumpy.patch._gaussian_patcher import GaussianPatcher
from clumpy.transition_probability_estimation._bayes import Bayes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_data(rows: int, cols: int, seed: int):
    """
    Build a small synthetic landscape.

    Returns
    -------
    luc_initial : ndarray (rows, cols) int — initial land-use map
        Values: 1 = forest (dominant), 2 = urban, 3 = water
    luc_final   : ndarray (rows, cols) int — observed final land-use map
        Some forest→urban transitions baked in.
    ev1         : ndarray (rows, cols) float — explanatory variable (distance-like)
    ev2         : ndarray (rows, cols) float — explanatory variable (slope-like)
    """
    rng = np.random.RandomState(seed)

    # Start with mostly forest (1), some urban (2), a strip of water (3)
    luc = np.ones((rows, cols), dtype=np.int32)
    # urban block top-left
    luc[1:4, 1:4] = 2
    # water strip along bottom
    luc[-2:, :] = 3

    luc_initial = luc.copy()

    # Create a "final" map where some forest pixels near urban became urban
    luc_final = luc_initial.copy()
    # Manually transition a handful of forest→urban near the urban block
    transition_pixels = [
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 1),
        (4, 2),
        (4, 3),
        (1, 5),
        (4, 4),
        (5, 1),
        (5, 2),
    ]
    for r, c in transition_pixels:
        if 0 <= r < rows and 0 <= c < cols and luc_initial[r, c] == 1:
            luc_final[r, c] = 2

    # Explanatory variable 1: distance to urban in initial map
    urban_mask = (luc_initial == 2).astype(int)
    ev1 = ndimage.distance_transform_edt(1 - urban_mask).astype(np.float64)

    # Explanatory variable 2: random "slope" field
    ev2 = rng.rand(rows, cols).astype(np.float64) * 10.0

    return luc_initial, luc_final, ev1, ev2


# ---------------------------------------------------------------------------
# Core pipeline (mirrors what Case.calibrate + Case.allocate would do,
# but assembled manually so we control every intermediate array)
# ---------------------------------------------------------------------------


def run_pipeline(seed: int, rows: int = 20, cols: int = 20, verbose: int = 1):
    """
    Full pipeline: calibrate on synthetic data, then allocate on a fresh copy
    of the initial map.  Returns all intermediate arrays for saving.
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    luc_init_arr, luc_final_arr, ev1_arr, ev2_arr = make_synthetic_data(
        rows, cols, seed
    )

    if verbose:
        n_trans = np.sum((luc_init_arr == 1) & (luc_final_arr == 2))
        n_forest = np.sum(luc_init_arr == 1)
        print(
            f"Synthetic data: {rows}x{cols}, forest pixels={n_forest}, "
            f"forest→urban transitions={n_trans}"
        )

    # Wrap as clumpy Layer objects
    luc_initial = LandUseLayer(luc_init_arr)
    luc_final = LandUseLayer(luc_final_arr)
    ev1_layer = EVLayer(ev1_arr, label="dist_urban", bounded="left")
    ev2_layer = EVLayer(ev2_arr, label="slope", bounded="none")

    evs = [ev1_layer, ev2_layer]
    bounds = get_bounds(evs)

    # ------------------------------------------------------------------
    # 2. Define states and transition matrix
    # ------------------------------------------------------------------
    # We study the transition *from* forest (initial_state = 1).
    initial_state = 1
    # Possible final states for a forest pixel: stay forest (1) or become urban (2).
    # Water (3) is never a target from forest.
    final_states_for_tpe = [1, 2]  # includes staying = initial_state

    # Observed global transition probability forest→urban ~ fraction that transited
    J_calib = luc_initial.get_J(state=initial_state)
    _, V_calib = luc_final.get_V(J_calib, final_states=final_states_for_tpe)
    # Recompute J_calib after filtering
    J_calib = _  # get_V returns (J_filtered, V_filtered) — but it filters, so reassign

    # One-hot encode transitions for TPE fitting
    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(
        categories=[final_states_for_tpe],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=bool,
    )
    # V_calib has values in {1, 2} corresponding to final_states_for_tpe
    W = ohe.fit_transform(V_calib[:, None])

    # Explanatory variable values at calibration pixels
    Z = luc_initial.get_Z(J=J_calib, evs=evs)

    if verbose:
        print(
            f"Calibration pixels: {J_calib.size}, Z.shape={Z.shape}, W.shape={W.shape}"
        )
        print(f"  W column sums (stay, transition): {W.sum(axis=0)}")

    # ------------------------------------------------------------------
    # 3. Fit Bayes TPE  (P(Z|u,v) via ekde, then Bayes rule)
    # ------------------------------------------------------------------
    tpe = Bayes(
        density_estimator="ekde",
        n_corrections_max=1000,
        n_fit_max=10**5,
        log_computations=False,
        verbose=max(0, verbose - 1),
    )

    tpe.fit(Z=Z, W=W, bounds=bounds)

    if verbose:
        print("Bayes TPE fitted.")

    # ------------------------------------------------------------------
    # 4. Compute transition probabilities on allocation pixels
    #    (we allocate on a *fresh copy* of the initial map)
    # ------------------------------------------------------------------
    luc_alloc = luc_initial.copy()
    luc_alloc_origin = luc_initial.copy()

    J_alloc = luc_alloc.get_J(state=initial_state)
    Z_alloc = luc_alloc.get_Z(J=J_alloc, evs=evs)

    # Global transition probability vector P(v) for each final state
    # Compute from the calibration data
    P_v = W.mean(axis=0)
    # Ensure P_v sums to 1 (it should, since W is one-hot)
    P_v = P_v / P_v.sum()

    if verbose:
        print(f"Allocation pixels: {J_alloc.size}")
        print(f"Global P_v (stay, urban): {P_v}")

    # Compute per-pixel P(v|u,Z) via Bayes
    P_v__u_Z, _, _ = tpe.compute(
        Y=Z_alloc,
        P_v=P_v,
        return_P_Y=True,
        return_P_Y__v=True,
    )

    if verbose:
        print(
            f"P(v|u,Z) shape: {P_v__u_Z.shape}, "
            f"min={P_v__u_Z.min():.6f}, max={P_v__u_Z.max():.6f}"
        )

    # Build a ProbaLayer (3D: n_final_states × rows × cols)
    # final_states_for_tpe = [1, 2] — column 0 = stay forest, column 1 = become urban
    proba_layer = create_proba_layer(
        J=J_alloc,
        P=P_v__u_Z,
        final_states=final_states_for_tpe,
        shape=luc_alloc.shape,
        geo_metadata=luc_alloc.geo_metadata,
    )

    # ------------------------------------------------------------------
    # 5. GART — sample pivots (which pixels will initiate a patch?)
    # ------------------------------------------------------------------
    # We need P with a "stay" column so rows sum to ~1.
    # P_v__u_Z already has columns [stay, urban] that should roughly sum to 1.
    # The GART function interprets columns as final classes.
    # We want to find pixels that are *not* staying = initial_state.

    # Clean probabilities: ensure the "stay" column is the complement
    P_clean = P_v__u_Z.copy()
    # Column 0 = stay (initial_state=1), Column 1 = transition to urban (2)
    P_clean[:, 0] = 1.0 - P_clean[:, 1]
    P_clean[P_clean < 0] = 0.0

    np.random.seed(seed)
    V_gart = generalized_allocation_rejection_test(
        P_clean, final_states_for_tpe, seed=seed
    )

    id_pivot = V_gart != initial_state
    V_pivot = V_gart[id_pivot]
    J_pivot = J_alloc[id_pivot]

    # Shuffle pivots
    np.random.seed(seed + 1)
    n_pivot = J_pivot.size
    if n_pivot > 0:
        shuffle_idx = np.random.choice(n_pivot, size=n_pivot, replace=False)
        J_pivot = J_pivot[shuffle_idx]
        V_pivot = V_pivot[shuffle_idx]

    if verbose:
        print(f"GART selected {n_pivot} pivot pixels for transition")

    # ------------------------------------------------------------------
    # 6. Patch growth using GaussianPatcher
    # ------------------------------------------------------------------
    patcher = GaussianPatcher(
        area_mean=3.0,  # small patches for our small grid
        area_cov=1.0,
        eccentricity=0.5,
        neighbors_structure="rook",
        avoid_aggregation=True,
        nb_of_missing_to_fill=1,
        proceed_even_if_no_probability=True,
        n_tries_target_sample=100,
        equi_neighbors_proba=False,
    )
    # Set the state attributes that Patcher.allocate expects
    patcher.initial_state = initial_state  # forest = 1
    patcher.final_state = 2  # urban = 2

    # Get the probability layer for the *target* final state (urban=2)
    proba_urban = np.array(proba_layer.get_proba(2))  # 2D: (rows, cols)

    # Replace NaN with 0 for pixels outside the allocation set
    proba_urban = np.nan_to_num(proba_urban, nan=0.0)

    # Wrap proba_urban as a Layer so .flat works (it's an ndarray subclass)
    proba_urban_layer = Layer(proba_urban, label="proba_urban")

    np.random.seed(seed + 2)
    total_allocated = 0
    patch_log = []  # list of (j_pivot, final_state, n_allocated, j_list)
    areas_sampled = []  # record the exact area drawn for each pivot
    eccentricities_sampled = []  # record the exact eccentricity for each pivot

    # Monkey-patch GaussianPatcher._sample to record draws into a mutable list
    _draw_buf = []  # temporary buffer filled by _recording_sample, read after each allocate()
    _original_sample = patcher._sample.__func__

    def _recording_sample(self, n):
        areas, eccs = _original_sample(self, n)
        _draw_buf.append((areas.copy(), eccs.copy()))
        return areas, eccs

    import types

    patcher._sample = types.MethodType(_recording_sample, patcher)

    for i in range(n_pivot):
        j = J_pivot[i]
        v = V_pivot[i]
        assert v == 2, f"Expected final_state=2, got {v}"

        _draw_buf.clear()

        s, J_used = patcher.allocate(
            lul=luc_alloc,
            lul_origin=luc_alloc_origin,
            j=j,
            proba_layer=proba_urban_layer,
        )

        # Record the area/eccentricity that was actually drawn for this pivot.
        # If _sample was never called (early exit because pixel already transited),
        # record 0.0 as a sentinel so the R replay has exactly n_pivot entries.
        if _draw_buf:
            areas_sampled.append(float(_draw_buf[0][0][0]))
            eccentricities_sampled.append(float(_draw_buf[0][1][0]))
        else:
            areas_sampled.append(0.0)
            eccentricities_sampled.append(0.0)

        patch_log.append((int(j), int(v), int(s), [int(x) for x in J_used]))

        if s > 0:
            total_allocated += s
            if verbose > 1:
                print(f"  Patch {i}: pivot={j}, allocated {s} pixels")

    if verbose:
        n_success = sum(1 for _, _, s, _ in patch_log if s > 0)
        print(
            f"Patch growth: {n_success}/{n_pivot} patches succeeded, "
            f"{total_allocated} pixels allocated total"
        )

    elapsed = time.time() - t0
    if verbose:
        print(f"\nDone in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # 7. Collect results
    # ------------------------------------------------------------------
    results = {
        # Inputs
        "seed": seed,
        "rows": rows,
        "cols": cols,
        "luc_initial": np.array(luc_initial),
        "luc_final_observed": np.array(luc_final),
        "ev1": ev1_arr,
        "ev2": ev2_arr,
        "bounds": np.array(bounds, dtype=object),
        # Calibration
        "J_calib": J_calib,
        "Z_calib": Z,
        "W_calib": W,
        "P_v_global": P_v,
        # Allocation inputs
        "J_alloc": J_alloc,
        "Z_alloc": Z_alloc,
        "P_v__u_Z": P_v__u_Z,
        "proba_urban": proba_urban,
        # GART
        "V_gart": V_gart,
        "J_pivot": J_pivot,
        "V_pivot": V_pivot,
        # Patch results
        "luc_allocated": np.array(luc_alloc),
        "total_allocated": total_allocated,
        # Patcher config
        "patcher_area_mean": patcher.area_mean,
        "patcher_area_cov": patcher.area_cov,
        "patcher_eccentricity": patcher.eccentricity,
        "patcher_neighbors_structure": patcher.neighbors_structure,
        "patcher_avoid_aggregation": patcher.avoid_aggregation,
        # Per-pivot random draws (for deterministic R replay)
        "areas_sampled": np.array(areas_sampled),
        "eccentricities_sampled": np.array(eccentricities_sampled),
    }

    return results, patch_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Synthetic clumpy allocation")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--rows", type=int, default=20, help="Grid rows")
    parser.add_argument("--cols", type=int, default=20, help="Grid cols")
    parser.add_argument(
        "--outdir",
        type=str,
        default="scripts/output",
        help="Output directory (relative to repo root)",
    )
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity")
    args = parser.parse_args()

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Synthetic Allocation Pipeline ===")
    print(f"seed={args.seed}, grid={args.rows}x{args.cols}, outdir={outdir}\n")

    results, patch_log = run_pipeline(
        seed=args.seed,
        rows=args.rows,
        cols=args.cols,
        verbose=args.verbose,
    )

    # Save arrays (numpy format)
    npz_path = outdir / f"allocation_seed{args.seed}.npz"
    np.savez(str(npz_path), **results)
    print(f"\nArrays saved to {npz_path}")

    # Save patch log as human-readable text
    log_path = outdir / f"patch_log_seed{args.seed}.txt"
    with open(log_path, "w") as f:
        f.write("# pivot_j  final_state  n_allocated  pixel_indices\n")
        for j, v, s, jlist in patch_log:
            f.write(f"{j}\t{v}\t{s}\t{jlist}\n")
    print(f"Patch log saved to {log_path}")

    # ------------------------------------------------------------------
    # Save CSV files so R can read everything without reticulate/numpy
    # ------------------------------------------------------------------
    csv_dir = outdir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 2D integer grids as CSV matrices (no header, space-delimited)
    for name in ("luc_initial", "luc_final_observed", "luc_allocated"):
        np.savetxt(csv_dir / f"{name}.csv", results[name], fmt="%d", delimiter=",")

    # 2D float grids
    for name in ("ev1", "ev2", "proba_urban"):
        np.savetxt(csv_dir / f"{name}.csv", results[name], fmt="%.12g", delimiter=",")

    # 1D index / label vectors
    np.savetxt(csv_dir / "J_alloc.csv", results["J_alloc"], fmt="%d", delimiter=",")
    np.savetxt(csv_dir / "J_pivot.csv", results["J_pivot"], fmt="%d", delimiter=",")
    np.savetxt(csv_dir / "V_pivot.csv", results["V_pivot"], fmt="%d", delimiter=",")
    np.savetxt(csv_dir / "V_gart.csv", results["V_gart"], fmt="%d", delimiter=",")

    # Per-pixel transition probabilities (rows = pixels, cols = final states)
    np.savetxt(
        csv_dir / "P_v__u_Z.csv", results["P_v__u_Z"], fmt="%.12g", delimiter=","
    )

    # Calibration arrays
    np.savetxt(csv_dir / "J_calib.csv", results["J_calib"], fmt="%d", delimiter=",")
    np.savetxt(csv_dir / "Z_calib.csv", results["Z_calib"], fmt="%.12g", delimiter=",")
    np.savetxt(csv_dir / "Z_alloc.csv", results["Z_alloc"], fmt="%.12g", delimiter=",")
    np.savetxt(
        csv_dir / "W_calib.csv", results["W_calib"].astype(int), fmt="%d", delimiter=","
    )
    np.savetxt(
        csv_dir / "P_v_global.csv", results["P_v_global"], fmt="%.12g", delimiter=","
    )

    # Scalar parameters as a single key=value file
    with open(csv_dir / "params.txt", "w") as f:
        f.write(f"seed={results['seed']}\n")
        f.write(f"rows={results['rows']}\n")
        f.write(f"cols={results['cols']}\n")
        f.write(f"total_allocated={results['total_allocated']}\n")
        f.write(f"patcher_area_mean={results['patcher_area_mean']}\n")
        f.write(f"patcher_area_cov={results['patcher_area_cov']}\n")
        f.write(f"patcher_eccentricity={results['patcher_eccentricity']}\n")
        f.write(
            f"patcher_neighbors_structure={results['patcher_neighbors_structure']}\n"
        )
        f.write(f"patcher_avoid_aggregation={results['patcher_avoid_aggregation']}\n")
        f.write(f"initial_state=1\n")
        f.write(f"final_state=2\n")
        f.write(f"final_states_for_tpe=1,2\n")

    # Save per-pivot random draws for deterministic R replay
    np.savetxt(
        csv_dir / "areas_sampled.csv",
        results["areas_sampled"],
        fmt="%.15g",
        delimiter=",",
    )
    np.savetxt(
        csv_dir / "eccentricities_sampled.csv",
        results["eccentricities_sampled"],
        fmt="%.15g",
        delimiter=",",
    )

    # Print summary comparison
    luc_init = results["luc_initial"]
    luc_alloc = results["luc_allocated"]
    changed = luc_init != luc_alloc
    print(f"\n=== Summary ===")
    print(f"Pixels changed: {changed.sum()}")
    print(
        f"Forest→Urban transitions in allocated map: "
        f"{np.sum((luc_init == 1) & (luc_alloc == 2))}"
    )

    # Quick visual (ASCII)
    print(f"\nInitial LUC (1=forest, 2=urban, 3=water):")
    _print_grid(luc_init)
    print(f"\nAllocated LUC:")
    _print_grid(luc_alloc)
    print(f"\nDiff (X = changed):")
    _print_diff(luc_init, luc_alloc)


def _print_grid(arr, max_cols=40):
    """Print a small integer grid."""
    rows, cols = arr.shape
    for r in range(min(rows, 30)):
        print("  " + "".join(str(int(arr[r, c])) for c in range(min(cols, max_cols))))


def _print_diff(a, b, max_cols=40):
    """Print diff of two grids."""
    rows, cols = a.shape
    for r in range(min(rows, 30)):
        line = ""
        for c in range(min(cols, max_cols)):
            if a[r, c] != b[r, c]:
                line += "X"
            else:
                line += "."
        print("  " + line)


if __name__ == "__main__":
    main()
