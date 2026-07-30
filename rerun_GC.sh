#!/usr/bin/env bash
# rerun_GC — batch source-space Granger causality with the MNE (state-space, cwt)
# runner, using FIXPC3/4 multi-PC ROI aggregation (Pellegrini 2023), across
# task x contrast x leakage x n_pcs. Writes to GC_source_space_mne/ (a SEPARATE
# tree from the BSMART GC_source_space/, so nothing is clobbered).
#
# ── PREREQUISITE: the full multi-vertex ROI caches must exist ───────────
# FIXPC3/4 derives k PCs per ROI, so it needs the caches that store ALL vertices
# per ROI (all vertex_* modes share `.../{atlas}/vertex/...`). run_granger_mne.py
# reads them; run_source_localize.py writes them. If they don't exist yet on this
# machine, generate them ONCE (heavy: per-subject inverse; it SKIPS subjects whose
# cache already exists). Uncomment and run this block first, or run it separately:
#
#   for task in overtProd perception; do for stim in prodDiff percDiff; do
#     for leak in "" "--leakage-correction"; do
#       python run_source_localize.py --task $task --stim-class $stim \
#         --method LCMV --atlas custom --feature-mode vertex_selectkbest $leak --n-jobs 2
#   done; done; done
#
# ── runtime note ────────────────────────────────────────────────────────
# FIXPC3/4 is MULTIVARIATE (each ROI = a 3-4 channel block), which is markedly
# slower than FIXPC1 — minutes per subject, not seconds. Sixteen configs
# (2 PC counts x 2 tasks x 2 contrasts x 2 leak) can take a while. Trim PCS/loops
# below if you only need a subset.
#
#   conda activate mne          # needs: pip install mne-connectivity
#   bash rerun_GC.sh
#
# set -u only (no -e / pipefail): a failure in one config should not abort the rest.
set -u
cd /media/maxlab_sharedrive/SpeechProduction/EEG/code/source_estimation/
# Portable conda bootstrap — this box uses miniforge3, the workstation
# anaconda3, so do not hardcode a prefix. Fail loudly rather than silently
# running the wrong interpreter.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _p in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        if [ -f "$_p/etc/profile.d/conda.sh" ]; then
            source "$_p/etc/profile.d/conda.sh"; break
        fi
    done
fi
if ! conda activate mne 2>/dev/null; then
    echo "ERROR: could not activate the 'mne' conda env." >&2
    echo "  conda found at: $(command -v conda || echo '<none on PATH>')" >&2
    exit 1
fi

# ── config ──────────────────────────────────────────────────────────────
METHOD=LCMV
ATLAS=custom                       # all pairs of the atlas's ROIs (no --pairs)
FEATURE_MODE=vertex_selectkbest    # locates the shared multi-vertex cache
GC_N_LAGS=20                       # MNE gc_n_lags (MNE example / Pellegrini ~20)
WIN_MS=250                         # 250 ms under-resolves theta; use 750 for theta
TARGET_FS=200
PCS="3 4"                          # FIXPC3 and FIXPC4 (was FIXPC1 = single PC)
# The MNE runner is SUBJECT-parallel, so effective parallelism caps at ~20 subjects.
NJOBS=20

LOGDIR=~/gc_logs
mkdir -p "$LOGDIR"

for npc in $PCS; do
  for task in overtProd perception; do
    for stim in prodDiff percDiff; do
      for leak in "" "--leakage-correction"; do
        echo ">>> MNE FIXPC${npc} GC: task=$task stim=$stim leak=${leak:-none}"
        python run_granger_mne.py --task "$task" --stim-class "$stim" \
          --method "$METHOD" --atlas "$ATLAS" --feature-mode "$FEATURE_MODE" $leak \
          --gc-n-lags "$GC_N_LAGS" --win-ms "$WIN_MS" --target-fs "$TARGET_FS" \
          --n-pcs "$npc" --trgc --n-jobs "$NJOBS" \
          2>&1 | tee "$LOGDIR/gc_mne_pc${npc}_${task}_${stim}${leak:+_leak}.log"
      done
    done
  done
done

echo "=============================================================="
echo "DONE. Outputs under derivatives/source_estimation/GC_source_space_mne/"
echo "  tags: ssgc_cwt_pc{$PCS}_mo${GC_N_LAGS}_sw${WIN_MS}ms_fs${TARGET_FS} / all_rois / <stim>"
echo "  logs: $LOGDIR/gc_mne_pc*.log"
echo "=============================================================="
