#!/usr/bin/env bash
#
# Granger causality — full run script (sensor-space + source-space).
# Copy to the production workstation and run:  bash run_gc_all.sh
#
# Runs are grouped so you can comment out sections you don't want.
# Each runner processes ALL 20 subjects internally (parallel within subject),
# so one command per (task, contrast, mode) covers the whole group.
#
set -euo pipefail
cd "$(dirname "$0")"

# ── environment ──────────────────────────────────────────────────────
# Activate the mne conda env (portable across machines).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mne

# ── shared parameters ────────────────────────────────────────────────
METHOD=dSPM                 # inverse method (matches your source cache)
ATLAS=HCPMMP1               # 16 speech ROIs defined; finest parcellation
FEATURE_MODE=vertex_selectkbest   # locates the shared vertex cache (any vertex_* mode works)
ORDER=10                    # AR model order (BSMART: 10)
WIN_MS=40                   # moving-window length in ms (40 ms @ 500 Hz = 20 samples)
TARGET_FS=500               # resample virtual channels to this rate (matches MATLAB GC)
NJOBS=64                    # physical cores on the workstation
SENSOR_NJOBS=8              # sensor GC has few pairs; 8 is plenty

echo "=============================================================="
echo " SECTION A — SENSOR-SPACE GC  (reproduces the MATLAB PWGC)"
echo "=============================================================="
# The MATLAB analysis combined all trials ('all'); pairwise (bivariate) GC
# between the 4 sensor pseudo-channels (Temporal, Inferior_Frontal,
# Superior_Frontal, Superior_Parietal).  --trgc adds the time-reversed control.

python run_granger_sensor.py --task overtProd  --stim-class all \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --trgc --n-jobs $SENSOR_NJOBS

python run_granger_sensor.py --task perception --stim-class all \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --trgc --n-jobs $SENSOR_NJOBS

# Optional: contrast-split (prodDiff / percDiff) instead of all-trials-combined
# python run_granger_sensor.py --task overtProd  --stim-class prodDiff --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --trgc --n-jobs $SENSOR_NJOBS
# python run_granger_sensor.py --task perception --stim-class percDiff --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --trgc --n-jobs $SENSOR_NJOBS

echo "=============================================================="
echo " SECTION B — SOURCE-SPACE PREREQUISITE"
echo "=============================================================="
# Source GC reads the per-subject VERTEX ROI cache written by the decoding
# pipeline.  If that cache exists (external drives mounted, or produced by a
# prior vertex-mode decoding run) these steps are NOT needed.  Otherwise run
# source localization ONCE per (task, contrast) to generate the cache:
#
# python run_source_localize.py --task overtProd  --stim-class prodDiff --method $METHOD --atlas $ATLAS --feature-mode $FEATURE_MODE --n-jobs 2
# python run_source_localize.py --task perception --stim-class percDiff --method $METHOD --atlas $ATLAS --feature-mode $FEATURE_MODE --n-jobs 2

echo "=============================================================="
echo " SECTION C — SOURCE-SPACE PAIRWISE GC  (BSMART, all 16 ROIs)"
echo "=============================================================="
# Pairwise bivariate Geweke GC between every pair of the 16 speech ROIs,
# each ROI reduced to one virtual channel (fixed first-PC filter).  + TRGC.

python run_granger.py --task overtProd  --stim-class prodDiff --method $METHOD \
    --atlas $ATLAS --feature-mode $FEATURE_MODE \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --trgc --n-jobs $NJOBS

python run_granger.py --task perception --stim-class percDiff --method $METHOD \
    --atlas $ATLAS --feature-mode $FEATURE_MODE \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --trgc --n-jobs $NJOBS

# Optional: the other contrast per task
# python run_granger.py --task overtProd  --stim-class percDiff --method $METHOD --atlas $ATLAS --feature-mode $FEATURE_MODE --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --trgc --n-jobs $NJOBS
# python run_granger.py --task perception --stim-class prodDiff --method $METHOD --atlas $ATLAS --feature-mode $FEATURE_MODE --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --trgc --n-jobs $NJOBS

echo "=============================================================="
echo " SECTION D — SOURCE-SPACE CONDITIONAL GC  (state-space, direct edges)"
echo "=============================================================="
# Conditional GC: each directed edge conditioned on all OTHER ROIs (Barnett &
# Seth 2015 state-space method).  Isolates DIRECT influence (removes indirect /
# common-input paths).  Note: fits the joint 16-ROI MVAR per window — heavier
# and can be ill-conditioned on short windows.  If unstable, restrict to a hub
# set with --roi-subset (see the commented variant below).

python run_granger.py --task overtProd  --stim-class prodDiff --method $METHOD \
    --atlas $ATLAS --feature-mode $FEATURE_MODE --gc-mode conditional \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --n-jobs $NJOBS

python run_granger.py --task perception --stim-class percDiff --method $METHOD \
    --atlas $ATLAS --feature-mode $FEATURE_MODE --gc-mode conditional \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --n-jobs $NJOBS

# Optional: conditional GC on a focused hub subset (more stable, faster)
# python run_granger.py --task overtProd --stim-class prodDiff --method $METHOD \
#     --atlas $ATLAS --feature-mode $FEATURE_MODE --gc-mode conditional \
#     --roi-subset Temporal Inferior_Frontal vSMC Supramarginal Planum_Temporale \
#     --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --n-jobs $NJOBS

echo "=============================================================="
echo " SECTION E — GROUP STATS + FIGURES  (task-vs-baseline t-test)"
echo "=============================================================="
# Aggregates the per-subject .npz, runs the right-tailed task-vs-baseline
# t-test (matches production_pwgc_data_to_python.m), writes per-edge figures
# + a stats CSV under a group_stats/ subdir of each run.

# Sensor-space
python granger_stats.py --space sensor --task overtProd  --stim-class all \
    --method sensor --atlas sensor --feature-mode pseudochan \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS
python granger_stats.py --space sensor --task perception --stim-class all \
    --method sensor --atlas sensor --feature-mode pseudochan \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS

# Source-space pairwise
python granger_stats.py --space source --task overtProd  --stim-class prodDiff \
    --method $METHOD --atlas $ATLAS --feature-mode $FEATURE_MODE \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS
python granger_stats.py --space source --task perception --stim-class percDiff \
    --method $METHOD --atlas $ATLAS --feature-mode $FEATURE_MODE \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS

# Source-space conditional
python granger_stats.py --space source --task overtProd  --stim-class prodDiff \
    --method $METHOD --atlas $ATLAS --feature-mode $FEATURE_MODE --gc-mode conditional \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS
python granger_stats.py --space source --task perception --stim-class percDiff \
    --method $METHOD --atlas $ATLAS --feature-mode $FEATURE_MODE --gc-mode conditional \
    --order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS

echo "=============================================================="
echo " DONE.  Outputs:"
echo "   sensor : derivatives/source_estimation/GC_sensor_space/..."
echo "   source : derivatives/source_estimation/GC_source_space/..."
echo "   stats  : group_stats/ subdir under each run (figures + CSV)"
echo "=============================================================="
