// ---------------------------------------------------------------------------
// allocate.cpp — Rcpp reimplementation of clumpy's core allocation pipeline
//
// Replicates:
//   1. generalized_allocation_rejection_test  (GART)
//   2. GaussianPatcher patch-growth loop
//
// Compiled via Rcpp::sourceCpp() — no package skeleton needed.
// ---------------------------------------------------------------------------

#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace Rcpp;

// =========================================================================
// Helpers
// =========================================================================

// Rook neighbour structure (3x3 kernel):  0 1 0 / 1 1 1 / 0 1 0
static const int ROOK_KERNEL[3][3] = {{0,1,0},{1,1,1},{0,1,0}};
// Queen neighbour structure (3x3 kernel): 1 1 1 / 1 1 1 / 1 1 1
static const int QUEEN_KERNEL[3][3] = {{1,1,1},{1,1,1},{1,1,1}};

// Return pointer to the appropriate 3x3 kernel
static const int (*get_kernel(const std::string& structure))[3] {
    if (structure == "queen") return QUEEN_KERNEL;
    return ROOK_KERNEL;  // default: rook
}

// Sum of a 3x3 kernel
static int kernel_sum(const int (*kern)[3]) {
    int s = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            s += kern[i][j];
    return s;
}

// 2D convolution of matrix A (ar x ac) with 3x3 kernel, mode="constant", cval=0.
// Output written into B (same size as A). Mirrors scipy.ndimage.convolve with those settings.
static void convolve_3x3(const std::vector<double>& A, int ar, int ac,
                         const int (*kern)[3],
                         std::vector<double>& B) {
    B.assign(ar * ac, 0.0);
    for (int r = 0; r < ar; r++) {
        for (int c = 0; c < ac; c++) {
            double val = 0.0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    int rr = r + kr - 1;
                    int cc = c + kc - 1;
                    if (rr >= 0 && rr < ar && cc >= 0 && cc < ac) {
                        val += A[rr * ac + cc] * kern[kr][kc];
                    }
                }
            }
            B[r * ac + c] = val;
        }
    }
}


// =========================================================================
// 1. GART  — Generalized Allocation Rejection Test
// =========================================================================
//
// Python equivalent (clumpy/allocation/_gart.py):
//   P is (n_samples x n_classes), list_v is the class labels.
//   For each pixel draw U~Uniform(0,1), walk the cumsum columns from the
//   *last* class backwards; assign class where U < cumsum.
//
// [[Rcpp::export]]
IntegerVector gart_cpp(NumericMatrix P, IntegerVector list_v, int seed) {
    int n = P.nrow();
    int k = P.ncol();

    // Clean: NaN → 0, negatives → 0
    NumericMatrix Pc = clone(P);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++) {
            if (R_IsNaN(Pc(i,j)) || R_IsNA(Pc(i,j))) Pc(i,j) = 0.0;
            if (Pc(i,j) < 0.0) Pc(i,j) = 0.0;
        }

    // Cumulative sum along columns (axis=1)
    NumericMatrix cs(n, k);
    for (int i = 0; i < n; i++) {
        double cum = 0.0;
        for (int j = 0; j < k; j++) {
            cum += Pc(i, j);
            cs(i, j) = cum;
        }
    }

    // Draw uniform random values — replicate numpy.random.RandomState(seed)
    // We use R's RNG seeded identically to how Python's script does it.
    // The *shell wrapper* will verify numerical equivalence; here we just need
    // the same algorithmic structure.  The R caller will pass the same U vector
    // that Python produced (read from CSV) so the comparison is deterministic.
    // But we also provide a self-contained path using R's set.seed.

    // For reproducibility across languages we accept an optional external U vector.
    // This function uses R's own RNG:
    // (The R wrapper will call set.seed() before calling this.)
    NumericVector x(n);
    for (int i = 0; i < n; i++)
        x[i] = R::runif(0.0, 1.0);

    IntegerVector y(n, 0);
    for (int id_vf = 0; id_vf < k; id_vf++) {
        int inv_id_vf = k - 1 - id_vf;
        for (int i = 0; i < n; i++) {
            if (x[i] < cs(i, inv_id_vf)) {
                y[i] = list_v[inv_id_vf];
            }
        }
    }

    return y;
}

// Overload: caller supplies the uniform draws directly (for cross-language matching).
// [[Rcpp::export]]
IntegerVector gart_with_u_cpp(NumericMatrix P, IntegerVector list_v,
                              NumericVector U) {
    int n = P.nrow();
    int k = P.ncol();

    NumericMatrix Pc = clone(P);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++) {
            if (R_IsNaN(Pc(i,j)) || R_IsNA(Pc(i,j))) Pc(i,j) = 0.0;
            if (Pc(i,j) < 0.0) Pc(i,j) = 0.0;
        }

    NumericMatrix cs(n, k);
    for (int i = 0; i < n; i++) {
        double cum = 0.0;
        for (int j = 0; j < k; j++) {
            cum += Pc(i, j);
            cs(i, j) = cum;
        }
    }

    IntegerVector y(n, 0);
    for (int id_vf = 0; id_vf < k; id_vf++) {
        int inv_id_vf = k - 1 - id_vf;
        for (int i = 0; i < n; i++) {
            if (U[i] < cs(i, inv_id_vf)) {
                y[i] = list_v[inv_id_vf];
            }
        }
    }

    return y;
}


// =========================================================================
// 2. Patch growth  — GaussianPatcher.allocate()
// =========================================================================
//
// Operates on *flat* (1-D ravelled, row-major) indices into a (rows x cols) grid.
//
// Arguments:
//   lul          – integer vector length rows*cols  (current land-use, MODIFIED in-place)
//   lul_origin   – integer vector length rows*cols  (original land-use, read-only)
//   rows, cols   – grid dimensions
//   j            – flat index of the pivot / kernel pixel
//   proba_layer  – double vector length rows*cols (transition probability for target state)
//   area         – sampled patch area (>= 1)
//   eccentricity – target eccentricity in [0,1]
//   initial_state, final_state – integer labels
//   neighbors_structure – "rook" or "queen"
//   avoid_aggregation – bool
//   nb_of_missing_to_fill – int
//   proceed_even_if_no_probability – bool
//   equi_neighbors_proba – bool
//   hollow_seed  – seed used for random choice among hollow neighbours
//
// Returns a List with:
//   $n_allocated  – integer  (0 means the patch was rejected)
//   $J_allocated  – integer vector of flat indices that were allocated
//   $lul          – the (possibly modified) land-use vector
//
// [[Rcpp::export]]
List patch_allocate_cpp(IntegerVector lul,
                        IntegerVector lul_origin,
                        int rows, int cols,
                        int j,
                        NumericVector proba_layer,
                        double area,
                        double eccentricity,
                        int initial_state,
                        int final_state,
                        std::string neighbors_structure,
                        bool avoid_aggregation,
                        int nb_of_missing_to_fill,
                        bool proceed_even_if_no_probability,
                        bool equi_neighbors_proba) {

    const int (*kern)[3] = get_kernel(neighbors_structure);
    int n_neighbors_to_fill = kernel_sum(kern) - 1 - nb_of_missing_to_fill;

    std::vector<int> J_allocated;
    J_allocated.push_back(j);

    // If kernel pixel already transited → fail
    if (lul[j] != initial_state) {
        return List::create(
            Named("n_allocated") = 0,
            Named("J_allocated") = IntegerVector(J_allocated.begin(), J_allocated.end())
        );
    }

    // Python uses: while len(J_allocated) < area  (float comparison)
    // Do NOT round to int — the fractional part matters.
    double target_area = area;
    if (target_area < 1.0) target_area = 1.0;

    while ((double)J_allocated.size() < target_area) {
        int n_alloc = (int)J_allocated.size();

        // Compute row/col of every allocated pixel
        std::vector<int> x_alloc(n_alloc), y_alloc(n_alloc);
        int x_min = rows, x_max = -1, y_min = cols, y_max = -1;
        for (int i = 0; i < n_alloc; i++) {
            x_alloc[i] = J_allocated[i] / cols;
            y_alloc[i] = J_allocated[i] % cols;
            if (x_alloc[i] < x_min) x_min = x_alloc[i];
            if (x_alloc[i] > x_max) x_max = x_alloc[i];
            if (y_alloc[i] < y_min) y_min = y_alloc[i];
            if (y_alloc[i] > y_max) y_max = y_alloc[i];
        }

        // Build bounding box for the local convolution window
        int box_rows = x_max - x_min + 3;
        int box_cols = y_max - y_min + 3;
        int x_offset = x_min - 1;
        int y_offset = y_min - 1;

        // Boundary adjustments (identical to Python)
        if (x_min == 0)          { x_offset += 1; box_rows -= 1; }
        if (y_min == 0)          { y_offset += 1; box_cols -= 1; }
        if (x_max == rows - 1)   { box_rows -= 1; }
        if (y_max == cols - 1)   { box_cols -= 1; }

        // Build binary matrix A of allocated pixels in the local box
        std::vector<double> A(box_rows * box_cols, 0.0);
        for (int i = 0; i < n_alloc; i++) {
            int lr = x_alloc[i] - x_offset;
            int lc = y_alloc[i] - y_offset;
            A[lr * box_cols + lc] = 1.0;
        }

        // Convolve A with the neighbour kernel → B
        std::vector<double> B;
        convolve_3x3(A, box_rows, box_cols, kern, B);

        // Find neighbour positions: B * (1 - A) > 0
        std::vector<int> nb_box_j;   // flat index in the box
        std::vector<int> nb_x_glob;  // row in global grid
        std::vector<int> nb_y_glob;  // col in global grid
        for (int r = 0; r < box_rows; r++) {
            for (int c = 0; c < box_cols; c++) {
                int idx = r * box_cols + c;
                if (B[idx] * (1.0 - A[idx]) > 0.0) {
                    nb_box_j.push_back(idx);
                    nb_x_glob.push_back(r + x_offset);
                    nb_y_glob.push_back(c + y_offset);
                }
            }
        }

        int n_nb = (int)nb_box_j.size();
        if (n_nb == 0) {
            return List::create(
                Named("n_allocated") = 0,
                Named("J_allocated") = IntegerVector(J_allocated.begin(), J_allocated.end())
            );
        }

        // Compute flat global indices of neighbours
        std::vector<int> j_nb(n_nb);
        for (int i = 0; i < n_nb; i++) {
            j_nb[i] = nb_x_glob[i] * cols + nb_y_glob[i];
        }

        // Read initial/final states of neighbours
        std::vector<int> vi_nb(n_nb), vf_nb(n_nb);
        for (int i = 0; i < n_nb; i++) {
            vi_nb[i] = lul_origin[j_nb[i]];
            vf_nb[i] = lul[j_nb[i]];
        }

        // Aggregation avoidance check:
        // If any neighbour was initial_state in origin AND is now final_state → reject
        if (avoid_aggregation) {
            bool agg = false;
            for (int i = 0; i < n_nb; i++) {
                if (vi_nb[i] == initial_state && vf_nb[i] == final_state) {
                    agg = true;
                    break;
                }
            }
            if (agg) {
                return List::create(
                    Named("n_allocated") = 0,
                    Named("J_allocated") = IntegerVector(J_allocated.begin(), J_allocated.end())
                );
            }
        }

        // Keep only neighbours whose original AND current state == initial_state
        std::vector<int> keep_idx;
        for (int i = 0; i < n_nb; i++) {
            if (vi_nb[i] == initial_state && vf_nb[i] == initial_state) {
                keep_idx.push_back(i);
            }
        }

        if (keep_idx.empty()) {
            return List::create(
                Named("n_allocated") = 0,
                Named("J_allocated") = IntegerVector(J_allocated.begin(), J_allocated.end())
            );
        }

        // Subset the neighbour arrays
        int n_keep = (int)keep_idx.size();
        std::vector<int> kj_nb(n_keep), kx(n_keep), ky(n_keep), k_box_j(n_keep);
        std::vector<double> kb(n_keep);
        for (int i = 0; i < n_keep; i++) {
            int ki = keep_idx[i];
            kj_nb[i]  = j_nb[ki];
            kx[i]     = nb_x_glob[ki];
            ky[i]     = nb_y_glob[ki];
            k_box_j[i] = nb_box_j[ki];
            kb[i]     = B[nb_box_j[ki]];
        }

        // Fill hollows: neighbours with convolution count >= n_neighbors_to_fill
        std::vector<int> hollows;
        for (int i = 0; i < n_keep; i++) {
            if (kb[i] >= (double)n_neighbors_to_fill) {
                hollows.push_back(kj_nb[i]);
            }
        }
        if (!hollows.empty()) {
            // np.random.choice(j_hollows) — pick one at random using R's RNG
            int pick = (int)(R::runif(0.0, 1.0) * hollows.size());
            if (pick >= (int)hollows.size()) pick = (int)hollows.size() - 1;
            J_allocated.push_back(hollows[pick]);
            continue;
        }

        // Compute probability weights for each kept neighbour
        std::vector<double> Pvec(n_keep);
        if (equi_neighbors_proba) {
            std::fill(Pvec.begin(), Pvec.end(), 1.0);
        } else {
            for (int i = 0; i < n_keep; i++) {
                Pvec[i] = proba_layer[kj_nb[i]];
            }
        }

        // If all probabilities ≈ 0, fill with 1 or reject
        double psum = 0.0;
        for (int i = 0; i < n_keep; i++) psum += Pvec[i];
        if (std::abs(psum) < 1e-8) {
            if (proceed_even_if_no_probability) {
                std::fill(Pvec.begin(), Pvec.end(), 1.0);
            } else {
                return List::create(
                    Named("n_allocated") = 0,
                    Named("J_allocated") = IntegerVector(J_allocated.begin(), J_allocated.end())
                );
            }
        }

        // Centroid of currently allocated pixels
        double xc = 0.0, yc = 0.0;
        for (int i = 0; i < n_alloc; i++) {
            xc += x_alloc[i];
            yc += y_alloc[i];
        }
        xc /= n_alloc;
        yc /= n_alloc;

        // Inertia moments for eccentricity calculation
        // Python computes mu_20, mu_02, mu_11 *per neighbour candidate*
        // (each candidate is evaluated as if it were added to the patch)
        // Note: the Python code computes vectorised over all neighbours at once:
        //   mu_20 = (sum((x_alloc - xc)^2) + (x_nb - xc)^2) / (n_alloc + 1)
        //   mu_02 = (sum((y_alloc - yc)^2) + (y_nb - yc)^2) / (n_alloc + 1)
        //   mu_11 = (sum((x_alloc-xc)*(y_alloc-yc)) + (x_nb-xc)*(y_nb-yc)) / (n_alloc+1)
        double sum_dx2 = 0.0, sum_dy2 = 0.0, sum_dxdy = 0.0;
        for (int i = 0; i < n_alloc; i++) {
            double dx = x_alloc[i] - xc;
            double dy = y_alloc[i] - yc;
            sum_dx2  += dx * dx;
            sum_dy2  += dy * dy;
            sum_dxdy += dx * dy;
        }

        // For each candidate neighbour, compute eccentricity if it were added
        std::vector<double> de_vec(n_keep);
        for (int i = 0; i < n_keep; i++) {
            double dx_nb = kx[i] - xc;
            double dy_nb = ky[i] - yc;
            double mu_20 = (sum_dx2 + dx_nb * dx_nb) / (n_alloc + 1);
            double mu_02 = (sum_dy2 + dy_nb * dy_nb) / (n_alloc + 1);
            double mu_11 = (sum_dxdy + dx_nb * dy_nb) / (n_alloc + 1);

            double diff = mu_20 - mu_02;
            double delta = diff * diff + 4.0 * mu_11 * mu_11;
            double sq_delta = std::sqrt(delta);
            double denom = mu_20 + mu_02 + sq_delta;
            double numer = mu_20 + mu_02 - sq_delta;

            double e;
            if (denom <= 0.0) {
                e = 0.0;
            } else {
                double ratio = numer / denom;
                if (ratio < 0.0) ratio = 0.0;
                e = 1.0 - std::sqrt(ratio);
            }

            de_vec[i] = std::abs(eccentricity - e);
        }

        // Final score: P / |eccentricity - e|
        // Pick the neighbour with the maximum score (same as np.argmax)
        double best_score = -1.0;
        int best_idx = 0;
        for (int i = 0; i < n_keep; i++) {
            double score = Pvec[i] / de_vec[i];
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        J_allocated.push_back(kj_nb[best_idx]);
    }

    // ---- Final aggregation check on the last added pixel ----
    if (avoid_aggregation && J_allocated.size() > 1) {
        int last_j = J_allocated.back();
        int last_r = last_j / cols;
        int last_c = last_j % cols;

        // Get rook/queen neighbours of the last pixel
        // Rook: up, right, down, left
        // Queen: all 8 surrounding
        std::vector<int> last_nb;
        if (neighbors_structure == "queen") {
            int dr[] = {-1, -1, -1, 0, 0, 1, 1, 1};
            int dc[] = {-1,  0,  1,-1, 1,-1, 0, 1};
            for (int d = 0; d < 8; d++) {
                int nr = last_r + dr[d];
                int nc = last_c + dc[d];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                    last_nb.push_back(nr * cols + nc);
                }
            }
        } else {
            // rook
            int dr[] = {-1, 0, 1, 0};
            int dc[] = { 0, 1, 0,-1};
            for (int d = 0; d < 4; d++) {
                int nr = last_r + dr[d];
                int nc = last_c + dc[d];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                    last_nb.push_back(nr * cols + nc);
                }
            }
        }

        for (int nb_j : last_nb) {
            if (lul_origin[nb_j] == initial_state && lul[nb_j] == final_state) {
                return List::create(
                    Named("n_allocated") = 0,
                    Named("J_allocated") = IntegerVector(J_allocated.begin(), J_allocated.end())
                );
            }
        }
    }

    // ---- Commit allocation: write final_state into lul ----
    for (int jj : J_allocated) {
        lul[jj] = final_state;
    }

    return List::create(
        Named("n_allocated") = (int)J_allocated.size(),
        Named("J_allocated") = IntegerVector(J_allocated.begin(), J_allocated.end())
    );
}


// =========================================================================
// 3. run_allocation — orchestrate the full pipeline from R-side data
// =========================================================================
//
// This mirrors the loop in run_pipeline() steps 5-6 from run_allocation.py:
//   - Accepts the pre-computed P(v|u,Z), the pivot selection, and patcher config.
//   - Runs GART (or accepts pre-computed GART result for cross-language comparison).
//   - Iterates over pivots and calls patch_allocate_cpp for each.
//
// [[Rcpp::export]]
List run_allocation_cpp(IntegerVector luc_initial_flat,
                        int rows, int cols,
                        NumericVector proba_urban_flat,
                        IntegerVector J_pivot,
                        IntegerVector V_pivot,
                        NumericVector areas,
                        NumericVector eccentricities,
                        int initial_state,
                        int final_state,
                        std::string neighbors_structure,
                        bool avoid_aggregation,
                        int nb_of_missing_to_fill,
                        bool proceed_even_if_no_probability,
                        bool equi_neighbors_proba) {

    // Working copies (patch_allocate_cpp modifies lul in place via Rcpp reference semantics,
    // but we clone here so the caller's original is untouched)
    IntegerVector lul = clone(luc_initial_flat);
    IntegerVector lul_origin = clone(luc_initial_flat);

    int n_pivot = J_pivot.size();
    int total_allocated = 0;

    // Patch log: each entry is (pivot_j, final_state, n_allocated, pixel_indices)
    std::vector<int> log_pivot_j;
    std::vector<int> log_final_state;
    std::vector<int> log_n_allocated;
    std::vector<std::string> log_pixels;

    for (int i = 0; i < n_pivot; i++) {
        int j = J_pivot[i];
        int v = V_pivot[i];
        double area_i = areas[i];
        double ecc_i  = eccentricities[i];

        List res = patch_allocate_cpp(lul, lul_origin,
                                      rows, cols, j,
                                      proba_urban_flat,
                                      area_i, ecc_i,
                                      initial_state, final_state,
                                      neighbors_structure,
                                      avoid_aggregation,
                                      nb_of_missing_to_fill,
                                      proceed_even_if_no_probability,
                                      equi_neighbors_proba);

        int s = as<int>(res["n_allocated"]);
        IntegerVector J_used = as<IntegerVector>(res["J_allocated"]);

        log_pivot_j.push_back(j);
        log_final_state.push_back(v);
        log_n_allocated.push_back(s);

        // Build semicolon-separated pixel list (matching Python output format)
        std::string px_str;
        for (int pi = 0; pi < J_used.size(); pi++) {
            if (pi > 0) px_str += ";";
            px_str += std::to_string(J_used[pi]);
        }
        log_pixels.push_back(px_str);

        if (s > 0) total_allocated += s;
    }

    // Reshape lul back to a matrix for convenience
    IntegerMatrix luc_allocated(rows, cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            luc_allocated(r, c) = lul[r * cols + c];

    // Build patch_log DataFrame
    DataFrame patch_log = DataFrame::create(
        Named("pivot_j")        = IntegerVector(log_pivot_j.begin(), log_pivot_j.end()),
        Named("final_state")    = IntegerVector(log_final_state.begin(), log_final_state.end()),
        Named("n_allocated")    = IntegerVector(log_n_allocated.begin(), log_n_allocated.end()),
        Named("pixel_indices")  = CharacterVector(log_pixels.begin(), log_pixels.end())
    );

    return List::create(
        Named("luc_allocated")   = luc_allocated,
        Named("total_allocated") = total_allocated,
        Named("patch_log")       = patch_log
    );
}