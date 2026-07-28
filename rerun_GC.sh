#!/usr/bin/env bash
# rerun_GC — batch source-space Granger causality with the MNE (state-space, cwt)
# runner, across task x contrast x leakage. Drop-in replacement for the BSMART
# run_granger.py batch.
#
# Writes to derivatives/source_estimation/GC_source_space_mne/ — a SEPARATE tree
# from the BSMART GC_source_space/, so it does NOT clobber those results.
#
#   conda activate mne          # needs: pip install mne-connectivity
#   bash rerun_GC.sh
#
# set -u only (no -e / pipefail): a failure in one config should not abort the
# rest of the batch.
set -u
cd /media/maxlab_sharedrive/SpeechProduction/EEG/code/source_estimation/
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mne

# ── config (canonical = preset A: MO25 / SW250 ms / fs200 Hz) ───────────
METHOD=LCMV
ATLAS=custom                       # all pairs of the atlas's ROIs (no --pairs)
FEATURE_MODE=vertex_selectkbest
GC_N_LAGS=25                       # MNE gc_n_lags  (was --order 25 for BSMART)
WIN_MS=250                         # effective window; cwt cycles = f*WIN_MS/1000.
                                   # NB: 250 ms under-resolves theta (<2 cycles at
                                   # 4 Hz) — bump to 750 for a theta-focused run.
TARGET_FS=200
# The MNE runner is SUBJECT-parallel (each subject = a couple of MNE calls), so
# effective parallelism is capped at the subject count (~20). More does not help.
NJOBS=20

LOGDIR=~/gc_logs
mkdir -p "$LOGDIR"

for task in overtProd perception; do
  for stim in prodDiff percDiff; do
    for leak in "" "--leakage-correction"; do
      echo ">>> MNE state-space GC: task=$task stim=$stim leak=${leak:-none}"
      python run_granger_mne.py --task "$task" --stim-class "$stim" \
        --method "$METHOD" --atlas "$ATLAS" --feature-mode "$FEATURE_MODE" $leak \
        --gc-n-lags "$GC_N_LAGS" --win-ms "$WIN_MS" --target-fs "$TARGET_FS" \
        --trgc --n-jobs "$NJOBS" \
        2>&1 | tee "$LOGDIR/gc_mne_${task}_${stim}${leak:+_leak}.log"
    done
  done
done

echo "=============================================================="
echo "DONE. Outputs under derivatives/source_estimation/GC_source_space_mne/"
echo "  tag: ssgc_cwt_mo${GC_N_LAGS}_sw${WIN_MS}ms_fs${TARGET_FS} / all_rois / <stim>"
echo "  logs: $LOGDIR/gc_mne_*.log"
echo "=============================================================="
