#!/usr/bin/env bash
# compare_damera_bsmart_vs_mne.sh
# GC runs on the same data/pairs — two parametric (BSMART) + one state-space (MNE)
# per window in MNE_WINS, all at fs 200 Hz:
#
#   1. BSMART, short window  : order 6,  60 ms,  fs 200   (parametric)
#   2. BSMART, medium window : order 10, 120 ms, fs 200   (parametric)
#   3. MNE state-space (cwt) : gc_n_lags 15, 250 ms, fs 200   (see MNE_WINS)
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
# WHY the MNE window is longer than the parametric ones: win_ms sets the Morlet cycle count,
# and sigma_f = 500/win_ms Hz, so 250 ms gives 2 Hz — half the width of the narrowest band
# reported here. A 60 ms MNE window would give sigma_f = 8.3 Hz and smear theta/alpha/low-beta
# into one another. The comparison holds order/bands/pairs comparable and gives each estimator
# a window it can use, then asks whether they AGREE on direction/band.
#
# ── loops over task x contrast ──────────────────────────────────────────
# All four runs + all four comparisons are repeated for every
#   task in {overtProd, perception}  x  stim in {prodDiff, percDiff}.
# Set TASKS / STIMS to trim, e.g.  TASKS=overtProd bash compare_damera_bsmart_vs_mne.sh
#
# --win-ms is NOT a literal window: it sets the Morlet cycle count,
# n_cycles(f) = f * win_ms/1000, floored at --ncycle-floor (default 1). At
# win_ms 750 on a 4-30 Hz grid that is 3 cycles at 4 Hz up to 22.5 at 30 Hz, and
# the 1-cycle floor never engages. Nothing has to "fit" inside the crop, so both
# tasks use the SAME window; drop win_ms below ~33 ms if you want the floor to
# pin every frequency to a single cycle.
# What IS crop-dependent: mne_connectivity returns only frequencies at/above
# 5*fs/n_times, which is duration-based, so perception's 570 ms crop yields
# >= ~8.8 Hz and its theta band comes back NaN. The BSMART (parametric) side is
# not bound this way, which is precisely the contrast being tested.
#
# Both runners now DEFAULT to --normalize demean (ERP removal), and the mode is
# part of the output path, so these runs land in *_demean directories and cannot
# overwrite legacy runs. NORMALIZE below is passed to the runners AND to
# compare_bsmart_vs_mne.py so the reader looks where the writer wrote.
#
#   conda activate mne          # needs: pip install mne-connectivity
#   bash compare_damera_bsmart_vs_mne.sh
set -u
cd /media/maxlab_sharedrive/SpeechProduction/EEG/code/source_estimation/
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mne

# ── shared descriptors ──────────────────────────────────────────────────
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
NORMALIZE="${NORMALIZE:-demean}"                # ERP removal; part of the output path
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
# MNE window(s), ms. n_cycles(f) = f*win/1000 (floor NCYCLE_FLOOR), so the
# Morlet resolutions are frequency-independent: sigma_f = 500/win Hz,
# sigma_t = win/(2000*pi) s. At 250 ms that is sigma_f = 2 Hz — half the width
# of the narrowest reported band (theta and alpha are 4 Hz wide), which is the
# practical floor; below it adjacent bands blur into each other. Space-separate
# to sweep several, e.g. MNE_WINS="250 750".
MNE_WINS="${MNE_WINS:-250}"
NCYCLE_FLOOR="${NCYCLE_FLOOR:-1}"               # min Morlet cycles at any freq

SUBSET=$(echo "$PAIRS" | tr ' :' '\n\n' | sed '/^$/d' | sort -u | tr '\n' ' ')
echo "ROI subset (BSMART): $SUBSET"
LOGDIR=~/gc_logs; mkdir -p "$LOGDIR"
OUTROOT=$(python -c "from config import DECODE_OUTPUT_ROOT; print(DECODE_OUTPUT_ROOT.parent/'GC_source_space_mne'/'_bsmart_vs_mne')")

bsmart () {   # $1=task $2=stim $3=order $4=win $5=fs $6=logtag
    echo "===== [$1/$2] BSMART (parametric): order $3 / ${4} ms / fs $5 ====="
    python run_granger.py --task "$1" --stim-class "$2" --method "$METHOD" \
        --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK \
        --order "$3" --win-ms "$4" --target-fs "$5" --normalize "$NORMALIZE" \
        --roi-subset $SUBSET --trgc --n-jobs "$BS_NJOBS" \
        2>&1 | tee "$LOGDIR/bsmart_${6}_${1}_${2}.log"
}
mne_run () {  # $1=task $2=stim $3=win
    echo "===== [$1/$2] MNE state-space (cwt): gc_n_lags $MNE_LAGS / ${3} ms / fs $MNE_FS ====="
    python run_granger_mne.py --task "$1" --stim-class "$2" --method "$METHOD" \
        --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK \
        --gc-n-lags "$MNE_LAGS" --win-ms "$3" --target-fs "$MNE_FS" \
        --normalize "$NORMALIZE" --ncycle-floor "$NCYCLE_FLOOR" \
        --pairs $PAIRS --trgc --n-jobs "$MNE_NJOBS" \
        2>&1 | tee "$LOGDIR/mne_sw${3}_${1}_${2}.log"
}
compare () {  # $1=task $2=stim $3=bs_order $4=bs_win $5=bs_fs $6=mne_win $7=out-subdir
    for BAND in $BANDS; do
        python compare_bsmart_vs_mne.py --task "$1" --stim-class "$2" \
            --method "$METHOD" --atlas "$ATLAS" --feature-mode "$FEAT" $LEAK \
            --order "$3" --win-ms "$4" --target-fs "$5" --bsmart-roi-subset $SUBSET \
            --bsmart-win-ms "$4" \
            --bsmart-normalize "$NORMALIZE" --mne-normalize "$NORMALIZE" \
            --mne-gc-n-lags "$MNE_LAGS" --mne-win-ms "$6" --mne-target-fs "$MNE_FS" \
            --pairs $PAIRS --band "$BAND" --out-dir "$OUTROOT/$1_$2/$7"
    done
}

# ── task x contrast ─────────────────────────────────────────────────────
for TASK in $TASKS; do
  for STIM in $STIMS; do
    echo
    echo "##############################################################"
    echo "## task=$TASK  stim=$STIM"
    echo "##############################################################"

    # ── the parametric runs ─────────────────────────────────────────────
    bsmart  "$TASK" "$STIM" "$BS1_ORDER" "$BS1_WIN" "$BS1_FS" "short"
    bsmart  "$TASK" "$STIM" "$BS2_ORDER" "$BS2_WIN" "$BS2_FS" "medium"

    # ── the state-space runs, one per MNE window ─────────────────────────
    for MW in $MNE_WINS; do
      mne_run "$TASK" "$STIM" "$MW"
    done

    # ── comparisons: each parametric config × each state-space window ────
    echo "===== [$TASK/$STIM] Compare (per band) ====="
    for MW in $MNE_WINS; do
      compare "$TASK" "$STIM" "$BS1_ORDER" "$BS1_WIN" "$BS1_FS" "$MW" "short${BS1_WIN}_vs_mne${MW}"
      compare "$TASK" "$STIM" "$BS2_ORDER" "$BS2_WIN" "$BS2_FS" "$MW" "med${BS2_WIN}_vs_mne${MW}"
    done
  done
done

NSUF=""; [ "$NORMALIZE" != "none" ] && NSUF="_$NORMALIZE"
echo
echo "=============================================================="
echo "DONE. Per (task x contrast), all fs 200, normalize=$NORMALIZE:"
echo "  tasks    : $TASKS"
echo "  contrasts: $STIMS"
echo "  BSMART short  : GC_source_space/.../order${BS1_ORDER}_win${BS1_WIN}ms_fs${BS1_FS}${NSUF}/"
echo "  BSMART medium : GC_source_space/.../order${BS2_ORDER}_win${BS2_WIN}ms_fs${BS2_FS}${NSUF}/"
for MW in $MNE_WINS; do
echo "  MNE ${MW}ms    : GC_source_space_mne/.../ssgc_cwt_pc1_mo${MNE_LAGS}_sw${MW}ms_fs${MNE_FS}${NSUF}/"
done
echo "  Figures       : $OUTROOT/<task>_<stim>/{short${BS1_WIN},med${BS2_WIN}}_vs_mne<win>/"
echo "  logs          : $LOGDIR/{bsmart_*,mne_sw*}_<task>_<stim>.log"
echo "=============================================================="
