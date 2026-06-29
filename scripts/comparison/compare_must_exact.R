#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Exact (pixel-perfect) cross-language check of the pivot test.
#
# Reads the (P, x, states, V) exported by compare_must_exact.py, replays the SAME
# uniforms x through evoland's `must_cpp`, and asserts the per-row assignment is
# identical to clumpy's GART output V -- a deterministic, RNG-controlled proof
# that the MuST / GART pivot test is re-implemented faithfully.
#
# Usage (clumpy repo root, after running compare_must_exact.py):
#   Rscript scripts/comparison/compare_must_exact.R [--evoland ../evoland-plus]
# ---------------------------------------------------------------------------
options(repos = c(P3M = "https://p3m.dev/cran/__linux__/noble/latest"))
suppressPackageStartupMessages(library(Rcpp))

args <- commandArgs(trailingOnly = TRUE)
evoland <- "../evoland-plus"
if (length(args) >= 2L && args[1] == "--evoland") evoland <- args[2]

evo_src <- file.path(evoland, "src", "alloc_clumpy.cpp")
if (!file.exists(evo_src)) {
  stop("evoland sources not found at '", evo_src, "'; pass --evoland <path>.")
}
sourceCpp(evo_src) # must_cpp (with the optional uniform-replay argument)

indir <- "scripts/output/must_exact"
if (!file.exists(file.path(indir, "P.csv"))) {
  stop("Inputs missing; run scripts/comparison/compare_must_exact.py first.")
}
P <- unname(as.matrix(read.csv(file.path(indir, "P.csv"), header = FALSE)))
x <- scan(file.path(indir, "x.csv"), quiet = TRUE)
states <- as.integer(scan(file.path(indir, "states.csv"), quiet = TRUE))
V_py <- as.integer(scan(file.path(indir, "V.csv"), quiet = TRUE))

V_evo <- as.integer(must_cpp(P, states, u = x))

n_diff <- sum(V_evo != V_py)
cat(sprintf(
  "Pivot test (MuST/GART) exact comparison: %d rows, %d differing\n",
  length(V_py), n_diff
))
if (n_diff == 0L) {
  cat("PERFECT MATCH: evoland must_cpp == clumpy GART on identical (P, x).\n")
} else {
  d <- which(V_evo != V_py)[1:min(10, n_diff)]
  cat("Mismatches at rows:", paste(d, collapse = ", "), "\n")
  for (i in d) cat(sprintf("  row %d: python=%d evoland=%d\n", i, V_py[i], V_evo[i]))
}
quit(status = if (n_diff == 0L) 0L else 1L)
