#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Exact (pixel-perfect) cross-language check of the pivot test.
#
# Reads the (P, x, states, V) exported by compare_must_exact.py, replays the SAME
# uniforms x through evoland's `must_cpp`, and asserts the per-row assignment is
# identical to clumpy's GART output V -- a deterministic, RNG-controlled proof
# that the MuST / GART pivot test is re-implemented faithfully.
#
# The uniforms are replayed through R's user-supplied random number generator
# (replay_rng.c), so evoland's code runs as it is: `must_cpp` draws one uniform
# per row from R's generator, which here hands out clumpy's.
#
# Usage (clumpy repo root, after running compare_must_exact.py):
#   Rscript scripts/comparison/compare_must_exact.R [--evoland ../evoland.plus]
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(Rcpp))

args <- commandArgs(trailingOnly = TRUE)
evoland <- "../evoland.plus"
if (length(args) >= 2L && args[1] == "--evoland") evoland <- args[2]

evo_src <- file.path(evoland, "src", "alloc_clumpy.cpp")
if (!file.exists(evo_src)) {
  stop("evoland sources not found at '", evo_src, "'; pass --evoland <path>.")
}
sourceCpp(evo_src) # must_cpp

# the replaying generator, compiled into a scratch directory
work <- tempfile("replay-rng-")
dir.create(work)
file.copy("scripts/comparison/replay_rng.c", work)
status <- system2(
  file.path(R.home("bin"), "R"),
  c("CMD", "SHLIB", "-o", file.path(work, "replay_rng.so"), file.path(work, "replay_rng.c")),
  stdout = FALSE
)
if (status != 0L) stop("could not compile scripts/comparison/replay_rng.c")
dyn.load(file.path(work, "replay_rng.so"))

indir <- "scripts/output/must_exact"
if (!file.exists(file.path(indir, "P.csv"))) {
  stop("Inputs missing; run scripts/comparison/compare_must_exact.py first.")
}
P <- unname(as.matrix(read.csv(file.path(indir, "P.csv"), header = FALSE)))
x <- scan(file.path(indir, "x.csv"), quiet = TRUE)
states <- as.integer(scan(file.path(indir, "states.csv"), quiet = TRUE))
V_py <- as.integer(scan(file.path(indir, "V.csv"), quiet = TRUE))

old_kind <- RNGkind()[1]
RNGkind("user-supplied")
invisible(.C("replay_set", as.double(x), length(x)))
V_evo <- as.integer(must_cpp(P, states))
replay <- .C("replay_state", used = 0L, overrun = 0L)
RNGkind(old_kind)
# evoland must draw exactly the uniforms clumpy drew: all of them, and no more
if (replay$used != length(x) || replay$overrun != 0L) {
  stop(sprintf(
    "evoland drew %d uniforms, clumpy %d", replay$used + replay$overrun, length(x)
  ))
}

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
