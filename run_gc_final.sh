#!/usr/bin/env bash
# run_gc_final.sh
# The FINAL (non-sweep) GC analysis: pairwise TRGC over the six speech-route
# ROI pairs of {awfa, ifc, pmc, tpc}, in two configurations:
#
#   main   MO10 / 80 ms / fs200   beta-band analysis. Model memory = 50 ms:
#          covers cortico-cortical conduction+synaptic delays (5-20 ms) with
#          margin and spans 1.5 cycles of high beta. 80 ms is the shortest
#          window that fits order 10 with room (16 samples against the Morf
#          recursion's > order+1 = 11). Sits inside the completed zscore
#          sweep grid, so the 352-config sweep is its robustness supplement.
#
#   theta  MO25 / 200 ms / fs200  theta-reach supplement. Model memory =
#          125 ms: a full cycle at 8 Hz, half at 4 Hz. The 200 ms window is
#          only there to satisfy samples > order+1 (40 > 26) — it is the
#          ORDER-AS-DURATION, not the window, that buys theta reach.
#
# Both arms: LCMV / custom atlas / vertex_selectkbest / leakage correction,
# --normalize zscore (ERP removal + ensemble-SD), per-trial demean on,
# --trgc. TRGC is pairwise-only in run_granger.py, and the sweep showed
# conditioning on a third ROI changes nothing — so no triple-wise runs here.
#
# *** THE MAIN ARM OVERWRITES THE SWEEP'S win80/order10 CELL. ***
# gc_tag() does not encode TRGC, and the completed zscore sweep already wrote
# plain-GC npz at order10_win80ms_fs200_zscore for all six pairs. Without
# --overwrite the runner would skip every subject and never compute dtrgc.
# granger.py / run_granger.py are unchanged since the sweep (verified
# 2026-08-26) and the pipeline is deterministic, so fxy/fyx regenerate
# identically — the overwrite only ADDS the dtrgc_* arrays. Set
# MAIN_OVERWRITE=0 only if you know every existing npz already has dtrgc.
#
# *** PERCEPTION HAS NO BASELINE AT THE THETA WINDOW. ***
# A 200 ms window is fully pre-stimulus only if it starts at or before
# -200 ms; the source axis starts at -100 ms (LCMV covariance segment), so
# perception has ZERO fully pre-stimulus windows in the theta arm. Its time
# courses and directional asymmetries (TRGC sign, forward vs reverse) are
# valid; task-vs-baseline statistics on perception@theta are NOT. overtProd
# (t=0 = articulation onset) is unaffected. The main arm (80 ms) is fine for
# both tasks.
#
#   conda activate mne          # the script activates it itself
#   bash run_gc_final.sh
#   DRY_RUN=1 bash run_gc_final.sh          # print commands, run nothing
#   ARMS=theta bash run_gc_final.sh         # one arm only
#   TASKS=overtProd bash run_gc_final.sh    # skip perception
#
# 2 arms x 6 pairs x 2 tasks x 2 contrasts = 48 configs, 20 subjects each,
# PARALLEL at a time (8 by default — the shared-drive I/O ceiling; 56
# crashed the workstation during the sweep).
set -u

cd "$(dirname "$0")"

ARMS="${ARMS:-main theta}"
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
METHOD="${METHOD:-LCMV}"
ATLAS="${ATLAS:-custom}"
FEAT="${FEAT:-vertex_selectkbest}"
LEAK="${LEAK:---leakage-correction}"
NORMALIZE="${NORMALIZE:-zscore}"
PARALLEL="${PARALLEL:-8}"
INNER_JOBS="${INNER_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAIN_OVERWRITE="${MAIN_OVERWRITE:-1}"

# arm parameters: order, window (ms), target fs (Hz), extra flags
arm_params () {
    case "$1" in
        main)  echo "10 80 200" ;;
        theta) echo "25 200 200" ;;
        *)     echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

PAIRS="\
awfa-lh ifc-lh; \
awfa-lh pmc-lh; \
awfa-lh tpc-lh; \
ifc-lh pmc-lh; \
ifc-lh tpc-lh; \
pmc-lh tpc-lh"
IFS=';' read -ra PAIR_ARR <<< "$PAIRS"

trim () { echo "$1" | sed 's/^ *//;s/ *$//'; }
label () { trim "$1" | sed 's/-lh//g;s/ /+/g'; }

if [ "$DRY_RUN" != "1" ]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        for _p in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
            [ -f "$_p/etc/profile.d/conda.sh" ] && { source "$_p/etc/profile.d/conda.sh"; break; }
        done
    fi
    conda activate mne 2>/dev/null || { echo "ERROR: cannot activate 'mne'" >&2; exit 1; }
fi

n_total=0
for ARM in $ARMS; do for _ in "${PAIR_ARR[@]}"; do for _ in $TASKS; do for _ in $STIMS; do
    n_total=$(( n_total + 1 ))
done; done; done; done

echo "final GC analysis — pairwise TRGC, $NORMALIZE, $METHOD/$ATLAS/$FEAT $LEAK"
for ARM in $ARMS; do
    read -r ORDER WIN_MS TARGET_FS <<< "$(arm_params "$ARM")"
    samp=$(( WIN_MS * TARGET_FS / 1000 ))
    echo "  $ARM: order $ORDER, ${WIN_MS} ms @ ${TARGET_FS} Hz = ${samp} samples" \
         "(needs > $(( ORDER + 1 ))), memory $(( 1000 * ORDER / TARGET_FS )) ms"
done
echo "  $n_total configs, 20 subjects each, $PARALLEL at a time"
case " $ARMS " in *" main "*) [ "$MAIN_OVERWRITE" = "1" ] && \
    echo "  main arm runs with --overwrite (adds dtrgc to the sweep's win80/order10 cell)" ;; esac
case " $ARMS " in *" theta "*) case " $TASKS " in *" perception "*)
    echo
    echo "  *** perception has NO fully pre-stimulus window at 200 ms."
    echo "      Time courses and TRGC asymmetries are valid; do not run"
    echo "      task-vs-baseline statistics on perception in the theta arm." ;;
esac; esac
echo

CMD_FILE=$(mktemp); DONE_FILE=$(mktemp)
trap 'rm -f "$CMD_FILE" "$DONE_FILE"' EXIT
t0=$(date +%s); n=0

for ARM in $ARMS; do
read -r ORDER WIN_MS TARGET_FS <<< "$(arm_params "$ARM")"
OVR=""
[ "$ARM" = "main" ] && [ "$MAIN_OVERWRITE" = "1" ] && OVR="--overwrite"
LOG_DIR="logs/gc_final_${ARM}"
mkdir -p "$LOG_DIR"
for PP in "${PAIR_ARR[@]}"; do
PP=$(trim "$PP")
for T in $TASKS; do for S in $STIMS; do
    tag="${T}_${S}_$(label "$PP")_trgc_win${WIN_MS}ms_order${ORDER}"
    log="$LOG_DIR/${tag}.log"
    n=$(( n + 1 ))
    echo "[$n/$n_total] $ARM  $tag"
    CMD="python run_granger.py --task $T --stim-class $S --method $METHOD \
--atlas $ATLAS --feature-mode $FEAT $LEAK --gc-mode pairwise \
--win-ms $WIN_MS --order $ORDER --target-fs $TARGET_FS \
--normalize $NORMALIZE --trgc --roi-subset $PP --n-jobs $INNER_JOBS $OVR"
    if [ "$DRY_RUN" = "1" ]; then echo "    $CMD"; continue; fi
    printf '%s\0' "
s=\$(date +%s)
$CMD > '$log' 2>&1
rc=\$?
e=\$(( (\$(date +%s) - s + 30) / 60 ))
echo x >> '$DONE_FILE'
d=\$(wc -l < '$DONE_FILE')
if [ \$rc -eq 0 ]; then
    u=\$(grep -c '\[unstable\]' '$log' 2>/dev/null || true)
    if [ \"\${u:-0}\" -gt 0 ]; then
        printf '  [%s/%s] ok    %s  %s min  (%s subj had unstable windows)\\n' \\
            \"\$d\" '$n_total' '$tag' \"\$e\" \"\$u\" >&2
    else
        printf '  [%s/%s] ok    %s  %s min\\n' \"\$d\" '$n_total' '$tag' \"\$e\" >&2
    fi
else
    printf '  [%s/%s] FAIL  %s  (see %s)\\n' \"\$d\" '$n_total' '$tag' '$log' >&2
fi
" >> "$CMD_FILE"
done; done; done
done

[ "$DRY_RUN" = "1" ] && exit 0

echo
echo "running $n_total configs, $PARALLEL at a time; each reports as it finishes."
echo "Per-subject progress:  tail -f logs/gc_final_*/*.log"
echo
xargs -0 -P "$PARALLEL" -n1 bash -c 'eval "$0"' < "$CMD_FILE"

echo
echo "$n_total configs attempted in $(( ($(date +%s) - t0) / 60 )) min"
echo
echo "Then:"
echo "  python granger_stats.py ...                 # baseline-referenced TRGC stats (main arm; overtProd only for theta)"
echo "  python plot_gc_pathway_timecourses.py       # theta-arm pathway figure"
grep -h 'NON-MINIMUM-PHASE\|consistency:' logs/gc_final_*/*.log 2>/dev/null | head -20
