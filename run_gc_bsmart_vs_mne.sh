#!/usr/bin/env bash
# Head-to-head: home-brewed BSMART GC vs MNE state-space (cwt) GC, same config,
# same ROI pairs, all 20 subjects. Runs both estimators, then the comparison.
#
#   conda activate mne
#   bash run_gc_bsmart_vs_mne.sh
#
# Uses config A (MO25/SW250/fs200) for BOTH so the only difference is the
# estimator. Override inline: ORDER=15 WIN=60 FS=500 bash run_gc_bsmart_vs_mne.sh
set -euo pipefail

TASK="${TASK:-overtProd}"
STIM="${STIM:-prodDiff}"
METHOD="${METHOD:-LCMV}"
ATLAS="${ATLAS:-custom}"
FEAT="${FEAT:-vertex_selectkbest}"
LEAKAGE="${LEAKAGE:---leakage-correction}"   # "" for raw
N_JOBS="${N_JOBS:-8}"

ORDER="${ORDER:-25}"; WIN="${WIN:-250}"; FS="${FS:-200}"
PAIRS="${PAIRS:-awfa-lh:ifc-lh tpc-lh:ifc-lh tpc-lh:pmc-lh ifc-lh:pmc-lh}"
BANDS="${BANDS:-theta alpha low_beta high_beta}"

# BSMART is run on the ROI subset (it computes all pairs among them); derive it
# from the requested pairs so the comparison can locate its output folder.
SUBSET=$(echo "$PAIRS" | tr ' :' '\n\n' | sed '/^$/d' | sort -u | tr '\n' ' ')
echo "ROI subset for BSMART: $SUBSET"

echo "=== BSMART (run_granger.py, order=$ORDER win=${WIN}ms fs=$FS) ==="
python run_granger.py --task "$TASK" --stim-class "$STIM" --method "$METHOD" \
    --atlas "$ATLAS" --feature-mode "$FEAT" $LEAKAGE \
    --order "$ORDER" --win-ms "$WIN" --target-fs "$FS" \
    --roi-subset $SUBSET --trgc --n-jobs 64

echo "=== MNE state-space cwt (run_granger_mne.py, MO=$ORDER SW=${WIN}ms fs=$FS) ==="
python run_granger_mne.py --task "$TASK" --stim-class "$STIM" --method "$METHOD" \
    --atlas "$ATLAS" --feature-mode "$FEAT" $LEAKAGE \
    --gc-n-lags "$ORDER" --win-ms "$WIN" --target-fs "$FS" \
    --pairs $PAIRS --trgc --n-jobs "$N_JOBS"

echo "=== Compare (per band) ==="
for BAND in $BANDS; do
    python compare_bsmart_vs_mne.py --task "$TASK" --stim-class "$STIM" \
        --method "$METHOD" --atlas "$ATLAS" --feature-mode "$FEAT" $LEAKAGE \
        --order "$ORDER" --win-ms "$WIN" --target-fs "$FS" \
        --bsmart-roi-subset $SUBSET --pairs $PAIRS --band "$BAND"
done

echo "Done. Figures in derivatives/source_estimation/GC_source_space_mne/_bsmart_vs_mne/"
