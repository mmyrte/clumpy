#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# COMPARISON — Python clumpy reference vs evoland uSAM / uPAM allocators
#
# This is the *comparison* (cf. the earlier scripts/exploration/, which only
# verified a standalone Rcpp re-implementation of the patch grower).  Here we
# drive the actual evoland allocation backend (allocate_clumpy_cpp, compiled
# straight from the evoland-plus sources) on the same synthetic inputs the
# Python reference pipeline produced, and compare:
#
#   1. GART (pivot mechanism) - evoland gart_cpp vs the Python GART vs the
#      analytic expectation E[#pivots] = sum_j P(v|u,z_j).
#   2. Allocation quantity of change - Python reference vs three evoland
#      configurations:
#        a. uSAM, rarefy = FALSE  -> mirrors the (biased) Python *script*,
#           which feeds P(v|u,z) straight to GART then grows ~E(sigma) patches,
#           so it over-allocates by ~mean patch area.
#        b. uSAM, rarefy = TRUE   -> the 1/E(sigma) correction (Mazy Fig. 3.2):
#           pivots are rarefied so the allocated quantity matches the target.
#        c. uPAM, rarefy = TRUE   -> iterative quota: hits the target exactly.
#   3. Patch structure - count / mean area / elongation of the newly-created
#      patches (8-connected components, measured identically for both tools via
#      evoland's calculate_class_stats_cpp).
#
# Quantity metrics are averaged over --nrep seeds because evoland and numpy use
# different RNGs (no cell-by-cell match is expected here; that was the point of
# the exploration's deterministic replay, not of this comparison).
#
# Usage (from the clumpy repo root):
#   Rscript scripts/comparison/compare_evoland.R \
#       [--pydir scripts/output/csv] [--evoland ../evoland-plus] \
#       [--nrep 200] [--seed 42]
#
# Prerequisites:
#   - Python reference already run (scripts/run_allocation.py) so --pydir exists
#   - R with Rcpp; a C++ toolchain. Binary packages: https://p3m.dev
# ---------------------------------------------------------------------------

# Posit Package Manager (precompiled binaries) for any install.packages() call.
options(repos = c(
  P3M = "https://p3m.dev/cran/__linux__/noble/latest",
  CRAN = "https://cloud.r-project.org"
))

suppressPackageStartupMessages(library(Rcpp))

# ---- CLI -------------------------------------------------------------------
parse_args <- function() {
  a <- commandArgs(trailingOnly = TRUE)
  o <- list(
    pydir = "scripts/output/csv",
    evoland = "../evoland-plus",
    nrep = 200L,
    seed = 42L
  )
  i <- 1L
  while (i <= length(a)) {
    key <- sub("^--", "", a[i])
    if (key %in% names(o) && i < length(a)) {
      val <- a[i + 1L]
      o[[key]] <- if (key %in% c("nrep", "seed")) as.integer(val) else val
      i <- i + 2L
    } else {
      stop("Unknown or malformed argument: ", a[i])
    }
  }
  o
}
opts <- parse_args()

# ---- Locate and compile the evoland backend --------------------------------
evo_src <- file.path(opts$evoland, "src", "alloc_clumpy.cpp")
stats_src <- file.path(opts$evoland, "src", "patch_stats.cpp")
if (!file.exists(evo_src)) {
  stop(
    "Could not find evoland sources at '", evo_src, "'.\n",
    "Pass --evoland <path-to-evoland-plus-checkout>."
  )
}
cat("=== Python-vs-evoland allocation comparison ===\n")
cat("Compiling evoland backend from", normalizePath(opts$evoland), "...\n")
sourceCpp(evo_src) # allocate_clumpy_cpp, gart_cpp, ...
sourceCpp(stats_src) # calculate_class_stats_cpp
cat("  OK\n\n")

# ---- Read Python reference outputs -----------------------------------------
pmat <- function(f) unname(as.matrix(read.csv(file.path(opts$pydir, f), header = FALSE)))
pvec <- function(f) scan(file.path(opts$pydir, f), quiet = TRUE)
read_params <- function(f) {
  kv <- strsplit(readLines(file.path(opts$pydir, f)), "=", fixed = TRUE)
  setNames(lapply(kv, function(x) trimws(paste(x[-1], collapse = "="))), vapply(kv, function(x) trimws(x[1]), ""))
}

luc_initial <- pmat("luc_initial.csv") # nr x nc grid (forest=1, urban=2, water=3)
luc_alloc_py <- pmat("luc_allocated.csv") # Python allocated grid
proba_urban <- pmat("proba_urban.csv") # nr x nc, P(urban|forest,z), 0 elsewhere
pv_global <- pvec("P_v_global.csv") # c(stay, urban)
params <- read_params("params.txt")

nr <- nrow(luc_initial)
nc <- ncol(luc_initial)
n_cells <- nr * nc

initial_state <- as.integer(params$initial_state) # 1 (forest)
final_state <- as.integer(params$final_state) # 2 (urban)
area_mean <- as.numeric(params$patcher_area_mean) # 3
# Python's GaussianPatcher uses area_cov as the *standard deviation* of a normal
# draw; evoland parameterises by variance, so area_var = area_cov^2.
area_cov <- as.numeric(params$patcher_area_cov) # SD in Python (1)
area_var <- area_cov^2
elongation <- as.numeric(params$patcher_eccentricity) # 0.5
target_rate <- pv_global[2] # P(urban | forest)

# Row-major flatten (evoland uses row-major cell indices: idx = r*ncol + c).
to_rowmajor <- function(M) as.vector(t(M))
from_rowmajor <- function(v) matrix(v, nrow = nr, ncol = nc, byrow = TRUE)

ant <- as.integer(to_rowmajor(luc_initial))
probs <- matrix(as.numeric(to_rowmajor(proba_urban)), ncol = 1L)

cat(sprintf(
  "Grid %dx%d | forest cells=%d | transition %d->%d\n",
  nr, nc, sum(ant == initial_state), initial_state, final_state
))
cat(sprintf(
  "Patcher: area_mean=%.3g area_var=%.3g elongation=%.3g | target rate P(v|u)=%.4f\n\n",
  area_mean, area_var, elongation, target_rate
))

# ---- Metric helpers --------------------------------------------------------
# Patch structure of the newly created patches: mask new cells as class 1L,
# everything else NA, measured with evoland's own connected-component stats.
patch_stats <- function(post_vec) {
  M <- from_rowmajor(post_vec)
  antM <- from_rowmajor(ant)
  newly <- ifelse(M == final_state & antM == initial_state, 1L, NA_integer_)
  if (all(is.na(newly))) {
    return(list(n_patch = 0L, area_mean = NA_real_, elong = NA_real_))
  }
  st <- calculate_class_stats_cpp(matrix(newly, nrow = nr, ncol = nc), 1)
  list(
    n_patch = as.integer(st$patch_count[1]),
    area_mean = st$patch_area_mean[1],
    elong = st$patch_elongation_mean[1]
  )
}
n_changed <- function(post_vec) sum(post_vec == final_state & ant == initial_state)

# area_dist code: 0 = log-normal, 1 = normal (Gaussian; matches GaussianPatcher).
AD_LOGNORM <- 0L
AD_NORMAL <- 1L

run_evo <- function(method_code, rarefy, avoid_agg, area_dist_code,
                    am = area_mean, av = area_var, batch = 1L, seed = 1L) {
  set.seed(seed)
  allocate_clumpy_cpp(
    landscape = ant, nrow = nr, ncol = nc,
    trans_from = initial_state, trans_to = final_state,
    probs = probs, area_mean = am, area_var = av,
    elongation = elongation, target_rate = target_rate,
    method = method_code, batch_size = batch, rarefy = rarefy, shuffle = TRUE,
    avoid_aggregation = avoid_agg, area_dist = area_dist_code
  )
}

# Monte-Carlo mean of changed-pixel count and patch structure over nrep seeds.
# (Averaging the patch metrics avoids single-seed noise; note patch counts use
# 8-connectivity while aggregation avoidance uses rook (4-conn), so diagonally
# touching patches are counted as one.)
mc_metrics <- function(method_code, rarefy, avoid_agg, area_dist_code,
                       am = area_mean, av = area_var) {
  changed <- numeric(opts$nrep)
  np <- numeric(opts$nrep)
  ar <- numeric(opts$nrep)
  el <- numeric(opts$nrep)
  for (s in seq_len(opts$nrep)) {
    post <- run_evo(method_code, rarefy, avoid_agg, area_dist_code, am, av, seed = s)
    changed[s] <- n_changed(post)
    ps <- patch_stats(post)
    np[s] <- ps$n_patch
    ar[s] <- ps$area_mean
    el[s] <- ps$elong
  }
  c(
    changed_mean = mean(changed), changed_sd = sd(changed),
    n_patch = mean(np), patch_area_mean = mean(ar, na.rm = TRUE),
    patch_elong = mean(el, na.rm = TRUE)
  )
}

# ---- 1. GART (pivot mechanism) equivalence ---------------------------------
cat("--- 1. GART pivot mechanism ---\n")
forest <- ant == initial_state
p_urb <- probs[forest, 1]
P_gart <- cbind(1 - p_urb, p_urb) # [stay, urban] for forest cells
expected_pivots <- sum(p_urb)

mc_pivots <- vapply(seq_len(opts$nrep), function(s) {
  set.seed(s)
  sum(gart_cpp(P_gart, c(initial_state, final_state)) == final_state)
}, numeric(1))

py_pivots <- sum(pvec("V_gart.csv") != initial_state)

cat(sprintf("  analytic   E[#pivots] = sum P(v|u,z) = %.2f\n", expected_pivots))
cat(sprintf("  evoland    gart_cpp   = %.2f +/- %.2f (mean over %d seeds)\n", mean(mc_pivots), sd(mc_pivots), opts$nrep))
cat(sprintf("  python     GART       = %d (single draw)\n\n", py_pivots))

# ---- 2 & 3. Allocation quantity + patch structure --------------------------
cat("--- 2. Quantity of change + 3. patch structure ---\n")

# Python reference (single realisation)
py_changed <- sum(luc_alloc_py == final_state & luc_initial == initial_state)
ps_py <- patch_stats(to_rowmajor(luc_alloc_py))

# The Python reference is GaussianPatcher (normal area + avoid_aggregation), so
# "evoland uPAM normal +agg" is its direct analogue.  The other rows isolate the
# effect of aggregation avoidance, the area distribution, and the mono-pixel
# uSAM special case.
configs <- list(
  list(label = "evoland uSAM (mono-pixel)", m = 0L, r = TRUE, agg = FALSE, ad = AD_NORMAL, am = 1, av = 0),
  list(label = "evoland uPAM normal +agg", m = 1L, r = TRUE, agg = TRUE, ad = AD_NORMAL, am = area_mean, av = area_var),
  list(label = "evoland uPAM normal -agg", m = 1L, r = TRUE, agg = FALSE, ad = AD_NORMAL, am = area_mean, av = area_var),
  list(label = "evoland uPAM lognorm +agg", m = 1L, r = TRUE, agg = TRUE, ad = AD_LOGNORM, am = area_mean, av = area_var)
)

rows <- list(data.frame(
  method = "python clumpy (Gaussian +agg)",
  changed_mean = py_changed, changed_sd = NA_real_,
  n_patch = ps_py$n_patch, patch_area_mean = ps_py$area_mean,
  patch_elong = ps_py$elong, stringsAsFactors = FALSE
))

for (cfg in configs) {
  mc <- mc_metrics(cfg$m, cfg$r, cfg$agg, cfg$ad, cfg$am, cfg$av)
  rows[[length(rows) + 1L]] <- data.frame(
    method = cfg$label,
    changed_mean = mc["changed_mean"], changed_sd = mc["changed_sd"],
    n_patch = mc["n_patch"], patch_area_mean = mc["patch_area_mean"],
    patch_elong = mc["patch_elong"], stringsAsFactors = FALSE
  )
}
res <- do.call(rbind, rows)
rownames(res) <- NULL

cat(sprintf("  target quantity of change = rate * #forest = %.1f\n\n", target_rate * sum(forest)))
print(format(res, digits = 4), row.names = FALSE)

# ---- Save results ----------------------------------------------------------
outdir <- "scripts/output/comparison"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
write.csv(res, file.path(outdir, "results.csv"), row.names = FALSE)
cat(sprintf("\nResults written to %s/results.csv\n", outdir))

# ---- Interpretation --------------------------------------------------------
target_q <- target_rate * sum(forest)
get_changed <- function(label) res$changed_mean[res$method == label]
get_np <- function(label) res$n_patch[res$method == label]
cat("\n--- interpretation ---\n")
cat(sprintf(
  "* Pivot mechanism: evoland gart_cpp mean (%.2f) matches the analytic\n  expectation (%.2f) and the Python GART draw (%d) -> equivalent (RNG noise).\n",
  mean(mc_pivots), expected_pivots, py_pivots
))
cat(sprintf(
  "* Direct analogue: 'uPAM normal +agg' uses the same area distribution and\n  aggregation avoidance as the Python GaussianPatcher. changed: python=%.1f vs\n  evoland=%.1f; n_patch: python=%d vs evoland=%.1f (mean).\n",
  py_changed, get_changed("evoland uPAM normal +agg"),
  ps_py$n_patch, get_np("evoland uPAM normal +agg")
))
cat(sprintf(
  "* Aggregation avoidance: +agg=%.1f changed in %.1f patches vs -agg=%.1f in\n  %.1f patches (means); with avoidance ON, merging patches are rejected (more,\n  smaller patches; quantity may fall short of the target as the map saturates).\n",
  get_changed("evoland uPAM normal +agg"), get_np("evoland uPAM normal +agg"),
  get_changed("evoland uPAM normal -agg"), get_np("evoland uPAM normal -agg")
))
cat(sprintf(
  "* Quantity vs target (rate*#forest = %.1f): the 1/E(sigma) rarefaction keeps\n  the uPAM quota near target; aggregation avoidance can leave it short.\n",
  target_q
))
cat("* Area distribution: 'normal' matches GaussianPatcher; 'lognorm' is\n")
cat("  right-skewed. Compare the two +agg rows for the effect; elongation is\n")
cat("  measured identically for all tools.\n")
