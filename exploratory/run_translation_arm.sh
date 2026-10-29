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
#              TRGC is the single leakage control)
#   input fs   DECODING runs on the native 2000 Hz exports (DEC_FS=2000), GC
#              on the 500 Hz exports (GC_FS=500) — user rule 2026-10-22. The
#              localize stage therefore writes one cache per sampling rate
#              (<arm>/ for 2000 Hz, <arm>_fs500/ for 500 Hz).
#   decoding   logistic, tuned, 40 ms / 5 ms, no pseudo-trials (= the custom
#              ROI runs), then exploratory/decode_onset_latency.py (timing vs
#              the Visual control)
#   GC         run_granger.py order 10 / 40 ms / fs 500 / normalize none /
#              1 PC per ROI, all 10 pairs of the 5 ROIs in ONE run, --trgc so
#              plain GC (fxy/fyx) and net TRGC are side by side.  ONE leakage
#              control per arm (2026-10-29): the vertex arm reads net TRGC;
#              a DeFleCT arm has its nulls as the control and reads the
#              directed GC, with the evoked response removed first:
#                FEAT=deflect... NORMALIZE=demean TRGC=0 N_PCS=3
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
#   DEC_FS    2000       EEGLAB export source-localized for DECODING (native)
#   GC_FS     500        EEGLAB export source-localized for GC (4x smaller caches)
#   LOC_JOBS  4          subjects in parallel for source localization. Each
#                        worker holds its subject's vertex caches in RAM (~9 GB
#                        for a 2048 Hz overtProd subject with the 5 sensorEquiv
#                        ROIs, plus ~0.5 GB epochs + one full-cortex STC); 8
#                        workers took the workstation down on 2026-10-22.
#   DEC_JOBS  64         (ROI x window) workers for run_decode.py
#   GC_JOBS   10         pair workers inside run_granger.py (10 pairs)
#   ORDER / WIN_MS / TARGET_FS   10 / 40 / 500 (the sensor BSMART config)
#   NORMALIZE none      run_granger --normalize (demean = ERP removed; own GC tag/dirs)
#   TRGC      1         0 = directed GC only (DeFleCT arms)
#   LEAK      ""         set to "--leakage-correction" for the double-corrected arm
#   DRY_RUN   0          1 = print commands only
#
# DISK. sensorEquiv vertex caches: ~8 GB (overtProd) / ~4 GB (perception) per
# subject per contrast at 2000 Hz, a quarter of that at 500 Hz => ~600 GB per
# method for both rates and all four cells; the runner prints the free space
# of ROI_TIMESERIES_SAVE_ROOT first.
#
#   conda activate mne           # the script activates it itself
#   # run it inside a tmux session on the workstation:
#   bash exploratory/run_translation_arm.sh 2>&1 | tee logs/translation_arm_driver.log
#   tail -f logs/translation_arm_driver.log
#   STEPS=localize METHODS="dSPM" bash exploratory/run_translation_arm.sh
#   STEPS="gc stats" TASKS=overtProd bash exploratory/run_translation_arm.sh
#
# RESUMING. Every stage skips work that is already on disk: localize skips
# subjects whose cache exists, decode skips subjects whose result CSV exists
# (run_decode.py --overwrite to redo), gc skips subjects whose npz exists,
# stats simply recompute (cheap). Re-running the same command after a crash
# or a kill continues at the first unfinished subject.
#
# DEFLECT ARM (2026-10-26). FEAT=deflect (or a full design string such as
# deflect_k3_null2_other_rois_sm) replaces the vertex caches with k DeFleCT
# virtual channels per ROI (deflect.py): no inverse, filters designed from the
# leadfield + the subject's noise covariance at the SNR of METHODS
# (dSPM_snr10 -> 10).  Caches are ~1 % of the vertex caches and the decode
# stage is correspondingly faster (3 features instead of 200).  Logs carry the
# feature mode in their name.  GC uses N_PCS components per ROI (<= k).
#   FEAT=deflect METHODS="dSPM_snr10" N_PCS=3 bash exploratory/run_translation_arm.sh 2>&1 | tee logs/translation_arm_deflect.log
#
# REGULARIZATION VARIANTS (2026-10-26, from the resolution sweep). A method
# may carry an _snr<N> suffix: METHODS="dSPM_snr10" runs the same pipeline
# with lambda2 = 1/100 instead of the production 1/9 (config.parse_method).
# The full string is the {method} path segment everywhere, so the variant
# gets its own caches / results / GC dirs and never touches the SNR-3 ones:
#   METHODS="dSPM_snr10" bash exploratory/run_translation_arm.sh 2>&1 \
#       | tee logs/translation_arm_dSPM_snr10.log
#
# DEC_METHODS (default = METHODS) restricts the decode + latency stages only.
# dSPM and sLORETA give the same decoding curves (r = 1.000 on overtProd /
# prodDiff, n = 20: the pipeline standardizes every vertex, which removes
# the per-vertex noise normalization that is the only difference between
# them), so decoding sLORETA is redundant; GC (PC1 per ROI, no
# standardization) still differs between the two, so METHODS keeps all
# three for localize / gc / stats:
#   DEC_METHODS="dSPM eLORETA" bash exploratory/run_translation_arm.sh
set -u
cd "$(dirname "$0")/.."

METHODS="${METHODS:-dSPM eLORETA sLORETA}"
DEC_METHODS="${DEC_METHODS:-$METHODS}"   # decode + latency stages only
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
ATLAS="${ATLAS:-sensorEquiv}"
FEAT="${FEAT:-vertex_selectkbest}"
DEC_FS="${DEC_FS:-2000}"
GC_FS="${GC_FS:-500}"
LEAK="${LEAK:-}"
LOC_JOBS="${LOC_JOBS:-4}"
DEC_JOBS="${DEC_JOBS:-64}"
GC_JOBS="${GC_JOBS:-10}"
ORDER="${ORDER:-10}"
WIN_MS="${WIN_MS:-40}"
TARGET_FS="${TARGET_FS:-500}"
CLASSIFIER="${CLASSIFIER:-logistic}"
N_PCS="${N_PCS:-1}"           # components per ROI for GC (<= k for a DeFleCT arm)
NORMALIZE="${NORMALIZE:-none}"   # run_granger --normalize: none (sensor-faithful) | demean (ERP removed)
TRGC="${TRGC:-1}"             # 1 = also compute net TRGC (the vertex arm's leakage control); 0 = directed GC only (DeFleCT arm)
STEPS="${STEPS:-localize decode latency gc stats}"
EDGE_GUARD="${EDGE_GUARD:-30}"
TAIL_GUARD="${TAIL_GUARD:-50}"
STATS_JOBS="${STATS_JOBS:-8}"
DRY_RUN="${DRY_RUN:-0}"

LEAK_TAG=raw; [ -n "$LEAK" ] && LEAK_TAG=leakage_corrected
fs_flag () { [ "$1" != "2000" ] && echo "--fs $1"; return 0; }
# sampling rates to localize: the decoding one and the GC one (deduplicated)
LOC_FS="$DEC_FS"; [ "$GC_FS" != "$DEC_FS" ] && LOC_FS="$DEC_FS $GC_FS"
GC_CACHE_TAG="$LEAK_TAG"; [ "$GC_FS" != "2000" ] && GC_CACHE_TAG="${LEAK_TAG}_fs${GC_FS}"
LOG_ROOT="logs/translation_arm"
mkdir -p "$LOG_ROOT"
# non-default feature modes (e.g. FEAT=deflect) get their own log names so a
# DeFleCT arm never overwrites the vertex arm's logs for the same method
# (FT is set after conda activation, once FEAT is canonical)

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

# the python runners normalize the feature mode (deflect -> deflect_k3_null2_other_rois)
# and use the full string as a path segment; do the same here so the stats
# stage and the log names agree with what the runners write
FEAT=$(python -c "from config import canonical_feature_mode; print(canonical_feature_mode('$FEAT'))") || { echo "ERROR: bad FEAT=$FEAT" >&2; exit 1; }
FT=""; [ "$FEAT" != "vertex_selectkbest" ] && FT="_${FEAT}"

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

echo "translation arm: $ATLAS / methods [$METHODS] (decode: [$DEC_METHODS]) / tasks [$TASKS] / stims [$STIMS] / decode @ ${DEC_FS} Hz, GC @ ${GC_FS} Hz / $LEAK_TAG"
echo "  steps: $STEPS"
echo

# ── 1. source localization ──────────────────────────────────────────────
if has localize; then
echo "== localize (subject-parallel, $LOC_JOBS workers; rates: $LOC_FS Hz)"
for FS in $LOC_FS; do for T in $TASKS; do for M in $METHODS; do
    run "$LOG_ROOT/localize_${T}_${M}${FT}_fs${FS}.log" \
        python run_source_localize.py --task "$T" --stim-class $STIMS --method "$M" \
            --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK $(fs_flag "$FS") --n-jobs "$LOC_JOBS"
done; done; done
echo
fi

# ── 2. decoding ─────────────────────────────────────────────────────────
if has decode; then
echo "== decode (subject-sequential, $DEC_JOBS ROI x window workers, ${DEC_FS} Hz caches)"
for T in $TASKS; do for S in $STIMS; do for M in $DEC_METHODS; do
    run "$LOG_ROOT/decode_${T}_${S}_${M}${FT}.log" \
        python run_decode.py --task "$T" --stim-class "$S" --method "$M" \
            --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK $(fs_flag "$DEC_FS") \
            --classifier "$CLASSIFIER" --tune-hyperparams --n-jobs "$DEC_JOBS"
done; done; done
echo
fi

# ── 3. decoding timing vs the Visual control ────────────────────────────
if has latency; then
echo "== latency (decode_onset_latency.py, control = Visual)"
for T in $TASKS; do for M in $DEC_METHODS; do
    run "$LOG_ROOT/latency_${T}_${M}${FT}.log" \
        python exploratory/decode_onset_latency.py --task "$T" --method "$M" \
            --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK \
            --classifier "$CLASSIFIER" --tune-hyperparams --stim-classes $STIMS \
            --control-roi Visual
done; done
echo
fi

# ── 4. GC: all 10 pairs of the 5 ROIs, plain GC + TRGC ──────────────────
if has gc; then
TRGC_FLAG=""; [ "$TRGC" = "1" ] && TRGC_FLAG="--trgc"
NT=""; [ "$NORMALIZE" != "none" ] && NT="_${NORMALIZE}"
echo "== gc (order $ORDER, $WIN_MS ms @ $TARGET_FS Hz, normalize $NORMALIZE, $N_PCS PC/ROI, trgc=$TRGC; ${GC_FS} Hz caches)"
for T in $TASKS; do for S in $STIMS; do for M in $METHODS; do
    run "$LOG_ROOT/gc_${T}_${S}_${M}${FT}${NT}.log" \
        python run_granger.py --task "$T" --stim-class "$S" --method "$M" \
            --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK $(fs_flag "$GC_FS") --gc-mode pairwise \
            --order "$ORDER" --win-ms "$WIN_MS" --target-fs "$TARGET_FS" \
            --normalize "$NORMALIZE" --n-pcs "$N_PCS" $TRGC_FLAG --n-jobs "$GC_JOBS"
done; done; done
echo
fi

# ── 5. GC stats: sensor-faithful full span + fixed windows of interest ──
if has stats; then
echo "== stats (granger_stats.py)"
GC_ROOT=$(python -c "from run_granger import GC_OUTPUT_ROOT, gc_tag; print(GC_OUTPUT_ROOT)")
NT=""; [ "$NORMALIZE" != "none" ] && NT="_${NORMALIZE}"
TAG=$(python -c "from run_granger import gc_tag; print(gc_tag($ORDER, $WIN_MS, $TARGET_FS, '$NORMALIZE', n_pcs=$N_PCS))")
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
    run "$LOG_ROOT/stats_${T}_${S}_${M}${FT}${NT}_full.log" \
        python granger_stats.py --gc-dir "$D" --out-dir "$D/group_stats" \
            --edge-guard "$EDGE_GUARD" --tail-guard "$TAIL_GUARD" --n-jobs "$STATS_JOBS"
    # (b) clean baseline + each a-priori window of interest
    BL=$(bl_for "$T")
    echo "$WINDOWS" | while read -r WT WN LO HI; do
        [ "$WT" = "$T" ] || continue
        run "$LOG_ROOT/stats_${T}_${S}_${M}${FT}${NT}_${WN}.log" \
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
