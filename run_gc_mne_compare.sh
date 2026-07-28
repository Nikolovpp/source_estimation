#!/usr/bin/env bash
# Compare two MNE state-space GC parameter configs on all 20 subjects.
#   Config A: MO 25 / SW 250 ms / fs 200 Hz   (canonical)
#   Config B: MO 15 / SW  60 ms / fs 500 Hz   (fast / short window)
#
# Runs the two heavy per-subject GC jobs, then the light comparison/plots.
# Intended for the remote workstation (subject-parallel; set N_JOBS to cores).
#
#   conda activate mne
#   bash run_gc_mne_compare.sh
#
# Override any of the variables below inline, e.g.:
#   TASK=perception STIM=percDiff N_JOBS=32 bash run_gc_mne_compare.sh
set -euo pipefail

TASK="${TASK:-overtProd}"
STIM="${STIM:-prodDiff}"
METHOD="${METHOD:-LCMV}"
ATLAS="${ATLAS:-custom}"
FEAT="${FEAT:-vertex_selectkbest}"
LEAKAGE="${LEAKAGE:---leakage-correction}"   # set to "" for raw
N_JOBS="${N_JOBS:-8}"

# Specific ROI-to-ROI pairs (GC computed both ways for each). Edit freely.
# Custom-atlas speech ROIs: awfa-lh ifc-lh owfa-lh pmc-lh tpc-lh vwfa-lh
PAIRS="${PAIRS:-awfa-lh:ifc-lh tpc-lh:ifc-lh tpc-lh:pmc-lh ifc-lh:pmc-lh vwfa-lh:ifc-lh}"

# Bands to plot the comparison for (each makes its own pair of figures).
BANDS="${BANDS:-theta alpha low_beta high_beta}"
# Event-onset reference line: 0 for both tasks (production / stimulus onset).
ONSET="${ONSET:-0.0}"

echo "=== Config A: MO25/SW250/fs200 ==="
python run_granger_mne.py --task "$TASK" --stim-class "$STIM" --method "$METHOD" \
    --atlas "$ATLAS" --feature-mode "$FEAT" $LEAKAGE \
    --preset A --pairs $PAIRS --trgc --n-jobs "$N_JOBS"

echo "=== Config B: MO15/SW60/fs500 ==="
python run_granger_mne.py --task "$TASK" --stim-class "$STIM" --method "$METHOD" \
    --atlas "$ATLAS" --feature-mode "$FEAT" $LEAKAGE \
    --preset B --pairs $PAIRS --trgc --n-jobs "$N_JOBS"

echo "=== Comparison figures (per band) ==="
for BAND in $BANDS; do
    python compare_gc_mne_configs.py --task "$TASK" --stim-class "$STIM" \
        --method "$METHOD" --atlas "$ATLAS" --feature-mode "$FEAT" $LEAKAGE \
        --pairs $PAIRS --band "$BAND" --onset "$ONSET"
done

echo "Done. Figures in derivatives/source_estimation/GC_source_space_mne/_config_comparison/"
