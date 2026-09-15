#!/usr/bin/env bash
# rerun_GC.sh — rerun the FINAL GC analysis WITHOUT zscore normalization
# (--normalize none), to compare raw-GC outputs against the zscore results.
#
# Thin wrapper over the root-level run_gc_final.sh: same six ROI pairs,
# tasks, contrasts, and parallelism — only NORMALIZE differs (and the arm
# set defaults to just "main"; the theta arm exists for the zscore analysis).
#
#   bash exploratory/rerun_GC.sh
#   DRY_RUN=1 bash exploratory/rerun_GC.sh       # print commands, run nothing
#   ARMS="main theta" bash exploratory/rerun_GC.sh
#   TASKS=overtProd STIMS=prodDiff bash exploratory/rerun_GC.sh
#
# Notes for the comparison:
#   - gc_tag() encodes the normalize mode, so outputs land in separate
#     ...order10_win80ms_fs200_none/ directories — nothing zscore is touched.
#     The main arm's baked-in --overwrite just regenerates deterministically
#     if the old normalize A/B left plain-GC npz at that path (adding dtrgc).
#   - Use --edge-guard 30 for granger_stats.py / plot_gc_edge_review.py on
#     the `none` output: the leading-window dip that motivated the 30 ms
#     guard was measured on none/demean data (on zscore it is 5 ms).
#   - Interpretation caveat: with `none` the ERP stays in the data, so
#     between-region evoked-latency differences read as directed influence —
#     exactly the confound zscore removes (TRGC does not protect against it).
#     Expect inflated effects around stimulus/articulation onset.
#
# (The previous life of this script — the MNE state-space FIXPC3/4 batch
# driver writing to GC_source_space_mne/ — is in git history at f4fa242.)
set -u
cd "$(dirname "$0")/.."

export NORMALIZE="${NORMALIZE:-none}"
export ARMS="${ARMS:-main}"
exec bash run_gc_final.sh
