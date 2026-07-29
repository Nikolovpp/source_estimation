#!/usr/bin/env bash
# compare_damera_bsmart_vs_mne.sh
# FOUR GC runs on the same data/pairs — two parametric (BSMART) + two state-space (MNE),
# all at fs 200 Hz:
#
#   1. BSMART, short window  : order 6,  60 ms,  fs 200   (parametric)
#   2. BSMART, medium window : order 10, 120 ms, fs 200   (parametric)
#   3. MNE state-space (cwt) : gc_n_lags 15, 250 ms, fs 200   (moderate — alpha/beta)
#   4. MNE state-space (cwt) : gc_n_lags 15, 750 ms, fs 200   (long — theta-appropriate)
#
# WHY fs 200 everywhere (see short_window_parametric_gc.md): at 200 Hz each AR lag buys 5 ms
# instead of 2 ms, so a lower order reaches theta (fewer params, lower floor); the ~30 ms
# delay is 6 samples (the 2-10 sweet spot); and 5 ms sampling is far better-conditioned than
# 500 Hz's 2 ms. fs 200 is the native rate of the founding AMVAR papers (Ding 2000; Brovelli
# 2004). Damera/Martin used 500 only because it was their EGI acquisition rate.
#
# NOTE on the "Damera" short pass: Damera used order 15 / 60 ms at fs 500. At fs 200 a 60 ms
# window is only 12 samples, which CANNOT hold order 15 (an AR fit needs window > order). So we
# match Damera's MEMORY instead: order 15 @ 500 Hz = 30 ms = order 6 @ 200 Hz — which also fits
# 12 samples comfortably and is exactly Ding's founding short-window config.
#
# WHY the MNE windows are longer: parametric MVAR is not bound by 1/T and works at 60-120 ms;
# MNE's cwt/state-space IS bound by ~1/T, needs >=250 ms for beta and ~750 ms for theta. The
# comparison holds order/bands/pairs comparable and gives each estimator a window it can use,
# then asks whether they AGREE on direction/band.
#
#   conda activate mne          # needs: pip install mne-connectivity
#   bash compare_damera_bsmart_vs_mne.sh
set -u
cd /media/maxlab_sharedrive/SpeechProduction/EEG/code/source_estimation/
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mne

# ── shared descriptors ──────────────────────────────────────────────────
TASK="${TASK:-overtProd}"
STIM="${STIM:-prodDiff}"
METHOD="${METHOD:-LCMV}"
ATLAS="${ATLAS:-custom}"
FEAT="${FEAT:-vertex_selectkbest}"
LEAK="${LEAK:---leakage-correction}"           # "" for raw
PAIRS="${PAIRS:-awfa-lh:ifc-lh tpc-lh:ifc-lh tpc-lh:pmc-lh ifc-lh:pmc-lh}"
BANDS="${BANDS:-theta low_beta high_beta}"     # Damera's theta + beta (split lo/hi)
BS_NJOBS="${BS_NJOBS:-64}"                      # BSMART parallelises over pairs
MNE_NJOBS="${MNE_NJOBS:-20}"                    # MNE parallelises over subjects (~20)

# per-run parameters (override any inline)
BS1_ORDER="${BS1_ORDER:-6}";  BS1_WIN="${BS1_WIN:-60}";  BS1_FS="${BS1_FS:-200}"   # short
BS2_ORDER="${BS2_ORDER:-10}"; BS2_WIN="${BS2_WIN:-120}"; BS2_FS="${BS2_FS:-200}"   # medium
MNE_LAGS="${MNE_LAGS:-15}";   MNE_FS="${MNE_FS:-200}"
MNE1_WIN="${MNE1_WIN:-250}"                     # moderate (alpha/beta; theta marginal)
MNE2_WIN="${MNE2_WIN:-750}"                     # long (theta-appropriate)

SUBSET=$(echo "$PAIRS" | tr ' :' '\n\n' | sed '/^$/d' | sort -u | tr '\n' ' ')
echo "ROI subset (BSMART): $SUBSET"
LOGDIR=~/gc_logs; mkdir -p "$LOGDIR"
OUTROOT=$(python -c "from config import DECODE_OUTPUT_ROOT; print(DECODE_OUTPUT_ROOT.parent/'GC_source_space_mne'/'_bsmart_vs_mne')")

bsmart () {   # $1=order $2=win $3=fs $4=logtag
    echo "===== BSMART (parametric): order $1 / ${2} ms / fs $3 ====="
    python run_granger.py --task "$TASK" --stim-class "$STIM" --method "$METHOD" \
        --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK \
        --order "$1" --win-ms "$2" --target-fs "$3" \
        --roi-subset $SUBSET --trgc --n-jobs "$BS_NJOBS" \
        2>&1 | tee "$LOGDIR/bsmart_${4}_${TASK}_${STIM}.log"
}
mne_run () {  # $1=win $2=logtag
    echo "===== MNE state-space (cwt): gc_n_lags $MNE_LAGS / ${1} ms / fs $MNE_FS ====="
    python run_granger_mne.py --task "$TASK" --stim-class "$STIM" --method "$METHOD" \
        --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK \
        --gc-n-lags "$MNE_LAGS" --win-ms "$1" --target-fs "$MNE_FS" \
        --pairs $PAIRS --trgc --n-jobs "$MNE_NJOBS" \
        2>&1 | tee "$LOGDIR/mne_sw${1}_${TASK}_${STIM}.log"
}
compare () {  # $1=bs_order $2=bs_win $3=bs_fs  $4=mne_win  $5=out-subdir
    for BAND in $BANDS; do
        python compare_bsmart_vs_mne.py --task "$TASK" --stim-class "$STIM" \
            --method "$METHOD" --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK \
            --order "$1" --win-ms "$2" --target-fs "$3" --bsmart-roi-subset $SUBSET \
            --bsmart-win-ms "$2" \
            --mne-gc-n-lags "$MNE_LAGS" --mne-win-ms "$4" --mne-target-fs "$MNE_FS" \
            --pairs $PAIRS --band "$BAND" --out-dir "$OUTROOT/$5"
    done
}

# ── the four runs ───────────────────────────────────────────────────────
bsmart  "$BS1_ORDER" "$BS1_WIN" "$BS1_FS" "short"
bsmart  "$BS2_ORDER" "$BS2_WIN" "$BS2_FS" "medium"
mne_run "$MNE1_WIN" "$MNE1_WIN"
mne_run "$MNE2_WIN" "$MNE2_WIN"

# ── comparisons: each parametric window × each state-space window ────────
echo "===== Compare (per band) ====="
compare "$BS1_ORDER" "$BS1_WIN" "$BS1_FS" "$MNE1_WIN" "short60_vs_mne${MNE1_WIN}"
compare "$BS1_ORDER" "$BS1_WIN" "$BS1_FS" "$MNE2_WIN" "short60_vs_mne${MNE2_WIN}"
compare "$BS2_ORDER" "$BS2_WIN" "$BS2_FS" "$MNE1_WIN" "med120_vs_mne${MNE1_WIN}"
compare "$BS2_ORDER" "$BS2_WIN" "$BS2_FS" "$MNE2_WIN" "med120_vs_mne${MNE2_WIN}"

echo "=============================================================="
echo "DONE. Four runs (all fs 200):"
echo "  BSMART short  : GC_source_space/.../order${BS1_ORDER}_win${BS1_WIN}ms_fs${BS1_FS}/"
echo "  BSMART medium : GC_source_space/.../order${BS2_ORDER}_win${BS2_WIN}ms_fs${BS2_FS}/"
echo "  MNE ${MNE1_WIN}ms    : GC_source_space_mne/.../ssgc_cwt_pc1_mo${MNE_LAGS}_sw${MNE1_WIN}ms_fs${MNE_FS}/"
echo "  MNE ${MNE2_WIN}ms    : GC_source_space_mne/.../ssgc_cwt_pc1_mo${MNE_LAGS}_sw${MNE2_WIN}ms_fs${MNE_FS}/"
echo "  Figures       : $OUTROOT/{short60,med120}_vs_mne{${MNE1_WIN},${MNE2_WIN}}/"
echo "=============================================================="
