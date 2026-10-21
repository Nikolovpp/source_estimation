#!/usr/bin/env bash
# run_translation_arm.sh — the "translation arm": the sensor-space manuscript
# analysis re-done literally in source space.
#
# WHY. The custom-ROI source results (LCMV, 6 localizer spheres) show no
# regional selectivity: every ROI, including the visual word-form controls,
# decodes the same curve, and GC between the spheres sits at chance.  The
# sensor story was built on four widely separated electrode clusters, plain
# BSMART GC (order 10, 40 ms = 20 samples @ 500 Hz, no normalization) and a
# right-tailed task-vs-baseline t test.  This arm reproduces exactly that at
# the same spatial granularity before anything is refined:
#
#   ROIs       atlas 'sensorEquiv' = the four pseudo-channel equivalents
#              (Temporal, Inferior_Frontal, Superior_Frontal, Superior_Parietal,
#              HCPMMP1 composites) + 'Visual' (V1-V3) as the null region
#   inverse    dSPM, eLORETA and sLORETA (minimum-norm family; LCMV optional —
#              it suppresses the correlated bilateral auditory/motor sources
#              and produced the collapse bug), raw (no leakage correction:
#              TRGC is the single leakage control), 500 Hz input files
#   decoding   logistic, tuned, 40 ms / 5 ms, no pseudo-trials (= the custom
#              ROI runs), then exploratory/decode_onset_latency.py (timing vs
#              the Visual control)
#   GC         run_granger.py order 10 / 40 ms / fs 500 / normalize none /
#              1 PC per ROI, all 10 pairs of the 5 ROIs in ONE run, --trgc so
#              plain GC (fxy/fyx) and net TRGC are side by side
#   stats      granger_stats.py: (a) sensor-faithful full-span right-tailed
#              t vs the leading-100-ms baseline (default), (b) one pass per
#              a-priori window of interest (config.GC_WINDOWS_OF_INTEREST)
#              with the clean _bl baselines
#
# STAGES (STEPS="..." selects; default all, in order):
#   localize   run_source_localize.py per task x method, both contrasts
#   decode     run_decode.py per task x contrast x method
#   latency    decode_onset_latency.py per task x method
#   gc         run_granger.py per task x contrast x method (all 10 pairs)
#   stats      granger_stats.py full span + windows of interest
#
# KNOBS (env):
#   METHODS   "dSPM eLORETA sLORETA"   add LCMV to include the beamformer
#   TASKS     "overtProd perception"
#   STIMS     "prodDiff percDiff"
#   FS        500        input EEGLAB export (500 = 4x smaller caches; 2000 native)
#   LOC_JOBS  8          subjects in parallel for source localization
#   DEC_JOBS  64         (ROI x window) workers for run_decode.py
#   GC_JOBS   10         pair workers inside run_granger.py (10 pairs)
#   ORDER / WIN_MS / TARGET_FS   10 / 40 / 500 (the sensor BSMART config)
#   LEAK      ""         set to "--leakage-correction" for the double-corrected arm
#   DRY_RUN   0          1 = print commands only
#
# DISK. sensorEquiv vertex caches at 500 Hz: ~2 GB (overtProd) / ~1 GB
# (perception) per subject per contrast => ~120 GB per method for all four
# cells; the runner prints the free space of ROI_TIMESERIES_SAVE_ROOT first.
#
#   conda activate mne           # the script activates it itself
#   bash exploratory/run_translation_arm.sh
#   STEPS=localize METHODS="dSPM" bash exploratory/run_translation_arm.sh
#   STEPS="gc stats" TASKS=overtProd bash exploratory/run_translation_arm.sh
set -u
cd "$(dirname "$0")/.."

METHODS="${METHODS:-dSPM eLORETA sLORETA}"
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
ATLAS="${ATLAS:-sensorEquiv}"
FEAT="${FEAT:-vertex_selectkbest}"
FS="${FS:-500}"
LEAK="${LEAK:-}"
LOC_JOBS="${LOC_JOBS:-8}"
DEC_JOBS="${DEC_JOBS:-64}"
GC_JOBS="${GC_JOBS:-10}"
ORDER="${ORDER:-10}"
WIN_MS="${WIN_MS:-40}"
TARGET_FS="${TARGET_FS:-500}"
CLASSIFIER="${CLASSIFIER:-logistic}"
STEPS="${STEPS:-localize decode latency gc stats}"
EDGE_GUARD="${EDGE_GUARD:-30}"
TAIL_GUARD="${TAIL_GUARD:-50}"
STATS_JOBS="${STATS_JOBS:-8}"
DRY_RUN="${DRY_RUN:-0}"

LEAK_TAG=raw; [ -n "$LEAK" ] && LEAK_TAG=leakage_corrected
FS_FLAG=""; [ "$FS" != "2000" ] && FS_FLAG="--fs $FS"
LOG_ROOT="logs/translation_arm"
mkdir -p "$LOG_ROOT"

# conda is needed even in DRY_RUN (the stats stage derives paths via python)
if true; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        for _p in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
            [ -f "$_p/etc/profile.d/conda.sh" ] && { source "$_p/etc/profile.d/conda.sh"; break; }
        done
    fi
    conda activate mne 2>/dev/null || { echo "ERROR: cannot activate 'mne'" >&2; exit 1; }
fi

run () {   # run <log> <cmd...>
    local log="$1"; shift
    echo "  $*"
    [ "$DRY_RUN" = "1" ] && return 0
    local s=$(date +%s)
    if "$@" > "$log" 2>&1; then
        echo "    ok   $(( ($(date +%s) - s) / 60 )) min  ($log)"
    else
        echo "    FAIL ($log)"; return 1
    fi
}
has () { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }

echo "translation arm: $ATLAS / methods [$METHODS] / tasks [$TASKS] / stims [$STIMS] / fs $FS / $LEAK_TAG"
echo "  steps: $STEPS"
echo

# ── 1. source localization ──────────────────────────────────────────────
if has localize; then
echo "== localize (subject-parallel, $LOC_JOBS workers)"
for T in $TASKS; do for M in $METHODS; do
    run "$LOG_ROOT/localize_${T}_${M}.log" \
        python run_source_localize.py --task "$T" --stim-class $STIMS --method "$M" \
            --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK $FS_FLAG --n-jobs "$LOC_JOBS"
done; done
echo
fi

# ── 2. decoding ─────────────────────────────────────────────────────────
if has decode; then
echo "== decode (subject-sequential, $DEC_JOBS ROI x window workers)"
for T in $TASKS; do for S in $STIMS; do for M in $METHODS; do
    run "$LOG_ROOT/decode_${T}_${S}_${M}.log" \
        python run_decode.py --task "$T" --stim-class "$S" --method "$M" \
            --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK $FS_FLAG \
            --classifier "$CLASSIFIER" --tune-hyperparams --n-jobs "$DEC_JOBS"
done; done; done
echo
fi

# ── 3. decoding timing vs the Visual control ────────────────────────────
if has latency; then
echo "== latency (decode_onset_latency.py, control = Visual)"
for T in $TASKS; do for M in $METHODS; do
    run "$LOG_ROOT/latency_${T}_${M}.log" \
        python exploratory/decode_onset_latency.py --task "$T" --method "$M" \
            --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK \
            --classifier "$CLASSIFIER" --tune-hyperparams --stim-classes $STIMS \
            --control-roi Visual
done; done
echo
fi

# ── 4. GC: all 10 pairs of the 5 ROIs, plain GC + TRGC ──────────────────
if has gc; then
echo "== gc (order $ORDER, $WIN_MS ms @ $TARGET_FS Hz, normalize none, 1 PC/ROI, --trgc)"
for T in $TASKS; do for S in $STIMS; do for M in $METHODS; do
    run "$LOG_ROOT/gc_${T}_${S}_${M}.log" \
        python run_granger.py --task "$T" --stim-class "$S" --method "$M" \
            --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK $FS_FLAG --gc-mode pairwise \
            --order "$ORDER" --win-ms "$WIN_MS" --target-fs "$TARGET_FS" \
            --normalize none --n-pcs 1 --trgc --n-jobs "$GC_JOBS"
done; done; done
echo
fi

# ── 5. GC stats: sensor-faithful full span + fixed windows of interest ──
if has stats; then
echo "== stats (granger_stats.py)"
GC_ROOT=$(python -c "from run_granger import GC_OUTPUT_ROOT, gc_tag; print(GC_OUTPUT_ROOT)")
TAG=$(python -c "from run_granger import gc_tag; print(gc_tag($ORDER, $WIN_MS, $TARGET_FS, 'none'))")
WINDOWS=$(python - <<'EOF'
from config import GC_WINDOWS_OF_INTEREST
for task, d in GC_WINDOWS_OF_INTEREST.items():
    for name, (lo, hi) in d.items():
        print(task, name, lo, hi)
EOF
)
bl_for () { case "$1" in overtProd) echo "-1.47 -1.37" ;; perception) echo "-0.08 0" ;; esac; }
for T in $TASKS; do for S in $STIMS; do for M in $METHODS; do
    D="$GC_ROOT/$T/$M/$ATLAS/$FEAT/$LEAK_TAG/$TAG/all_rois/$S"
    if [ ! -d "$D" ] && [ "$DRY_RUN" != "1" ]; then echo "  MISSING $D"; continue; fi
    # (a) sensor-faithful: leading-100 ms baseline, whole task span, right-tailed t
    run "$LOG_ROOT/stats_${T}_${S}_${M}_full.log" \
        python granger_stats.py --gc-dir "$D" --out-dir "$D/group_stats" \
            --edge-guard "$EDGE_GUARD" --tail-guard "$TAIL_GUARD" --n-jobs "$STATS_JOBS"
    # (b) clean baseline + each a-priori window of interest
    BL=$(bl_for "$T")
    echo "$WINDOWS" | while read -r WT WN LO HI; do
        [ "$WT" = "$T" ] || continue
        run "$LOG_ROOT/stats_${T}_${S}_${M}_${WN}.log" \
            python granger_stats.py --gc-dir "$D" --out-dir "$D/group_stats_win_${WN}" \
                --baseline-start ${BL% *} --baseline-end ${BL#* } \
                --task-start "$LO" --task-end "$HI" \
                --edge-guard "$EDGE_GUARD" --tail-guard "$TAIL_GUARD" --n-jobs "$STATS_JOBS"
    done
done; done; done
echo
echo "Summaries: $GC_ROOT/{task}/{method}/$ATLAS/$FEAT/$LEAK_TAG/$TAG/all_rois/{stim}/group_stats*/gc_permutation_summary_ttest.log"
fi

echo "done."
