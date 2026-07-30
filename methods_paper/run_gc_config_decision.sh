#!/usr/bin/env bash
# run_gc_config_decision — sweep the two switchable MNE-GC config choices across
# every task x contrast, BEFORE committing to a full re-run.
#
# For each (task, contrast) it crosses
#     normalize : none (legacy)  vs  demean (new default; ERP removed within y)
#     cwt grid  : 4-30 Hz (shipped)  vs  ~DC-Nyquist
# and reports, per pair x band, the group-mean net GC in each cell plus whether
# the sign agrees with the legacy cell. One CSV per (task, contrast).
#
# Reads the multi-vertex ROI caches (same inputs as run_granger_mne.py), so it
# gets `y` and can remove the ERP per condition. It writes NO GC outputs and
# touches nothing under GC_source_space_mne/ — it is a read-only decision aid.
#
# ── on --win-ms ─────────────────────────────────────────────────────────
# It is NOT a literal window: it sets the Morlet cycle count,
# n_cycles(f) = f * win_ms/1000, floored at --ncycle-floor (default 1). At
# win_ms 250 on a 4-30 Hz grid that is 1 cycle at 4 Hz (exactly on the floor)
# up to 7.5 at 30 Hz, giving sigma_f = 500/win_ms = 2 Hz at every frequency.
# Nothing has to fit inside the crop, so both tasks use the SAME window.
# What IS crop-dependent: mne_connectivity returns only frequencies at/above
# 5*fs/n_times (duration-based), so perception's 570 ms crop yields >= ~8.8 Hz
# and its theta cells come back NaN. The Python script prints the returned bin
# count per cell, so that is visible rather than silent.
#
#   conda activate mne
#   bash methods_paper/run_gc_config_decision.sh            # all 20 subjects
#   QUICK=1 bash methods_paper/run_gc_config_decision.sh    # first 6, smoke test
#
# set -u only (no -e / pipefail): one failing config must not abort the rest.
set -u
cd /media/maxlab_sharedrive/SpeechProduction/EEG/code/source_estimation/
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mne

# ── config ──────────────────────────────────────────────────────────────
METHOD=LCMV
ATLAS=custom
FEATURE_MODE=vertex_selectkbest
LEAK="--leakage-correction"          # set to "" for the raw caches
GC_N_LAGS=15
TARGET_FS=200
N_PCS=1                              # the pc1 config the existing outputs use
PAIRS="awfa-lh:ifc-lh tpc-lh:ifc-lh tpc-lh:pmc-lh ifc-lh:pmc-lh"
NJOBS=20
QUICK="${QUICK:-}"                   # QUICK=1 -> --quick (first 6 subjects)

WIN_MS="${WIN_MS:-250}"              # sigma_f = 500/WIN_MS = 2 Hz; the practical floor

LOGDIR=~/gc_logs
OUTDIR=methods_paper/config_decision
mkdir -p "$LOGDIR" "$OUTDIR"

[ -n "$QUICK" ] && QFLAG="--quick" || QFLAG=""

echo "=============================================================="
echo "GC config decision sweep"
echo "  method=$METHOD atlas=$ATLAS feature=$FEATURE_MODE leak=${LEAK:-none}"
echo "  MO=$GC_N_LAGS fs=$TARGET_FS n_pcs=$N_PCS n_jobs=$NJOBS ${QFLAG:+(QUICK)}"
echo "  pairs: $PAIRS"
echo "  win_ms=$WIN_MS (n_cycles = f*win_ms/1000, floor 1)"
echo "  outputs: $OUTDIR/   logs: $LOGDIR/"
echo "=============================================================="

for task in overtProd perception; do
  for stim in prodDiff percDiff; do
    echo
    echo ">>> task=$task  stim=$stim"
    log="$LOGDIR/gc_config_decision_${task}_${stim}.log"
    python methods_paper/gc_config_decision.py \
      --task "$task" --stim-class "$stim" \
      --method "$METHOD" --atlas "$ATLAS" --feature-mode "$FEATURE_MODE" $LEAK \
      --pairs $PAIRS \
      --gc-n-lags "$GC_N_LAGS" --win-ms "$WIN_MS" --target-fs "$TARGET_FS" \
      --n-pcs "$N_PCS" --n-jobs "$NJOBS" --out-dir "$OUTDIR" $QFLAG \
      2>&1 | tee "$log"
  done
done

echo
echo "=============================================================="
echo "DONE."
echo "  CSVs : $OUTDIR/gc_config_decision_<task>_<stim>_mo${GC_N_LAGS}_sw${WIN_MS}.csv"
echo "  logs : $LOGDIR/gc_config_decision_<task>_<stim>.log"
echo
echo "How to read a row: four net-GC values in the order"
echo "  none/4-30 (legacy) | demean/4-30 | none/full | demean/full"
echo "then a 3-character agreement string vs legacy ('.' same sign, 'X' flipped)."
echo "'...' means that pair x band is robust to both changes."
echo "=============================================================="
