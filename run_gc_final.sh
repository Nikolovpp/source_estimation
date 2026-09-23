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
# per-trial demean on, --trgc. TRGC is pairwise-only in run_granger.py, and
# the sweep showed conditioning on a third ROI changes nothing — so no
# triple-wise runs here.
#
# ROI AGGREGATION AND NORMALIZATION (the two knobs this script exposes):
#
#   NPCS       components kept per ROI (run_granger.py --n-pcs). Default 4 =
#              FIXPC4: each ROI is a block of its top-4 fixed-filter PCs and
#              every pair is an 8-channel VAR scored with block (multivariate)
#              Geweke GC / TRGC — Pellegrini et al. 2023's recommended
#              source-space pipeline. 1 = the earlier single virtual-channel
#              bivariate analysis. The block size is capped at each ROI's
#              numerical rank (a collapsed LCMV ROI stays one component; the
#              log reports it as [fixpc]).
#   NORMALIZE  across-trial ensemble normalization. Default none = raw
#              BSMART-faithful signals (per-trial DC removal only; no ERP
#              removal, no ensemble-SD equalization). 'demean' removes the
#              ERP; 'zscore' also divides by the ensemble SD per time point.
#
#   Output path: .../order{MO}_win{SW}ms_fs200[_{NORMALIZE}][_pc{NPCS}]/...
#   The normalize name and the PC suffix are both part of the path (the
#   suffix only for NPCS > 1), so runs with different settings never collide.
#
#   The earlier final arm (zscore, FIXPC1) is reproduced with
#       NORMALIZE=zscore NPCS=1 MAIN_OVERWRITE=1 bash run_gc_final.sh
#   MAIN_OVERWRITE matters ONLY for that cell: gc_tag() does not encode TRGC,
#   and the completed zscore sweep wrote plain-GC npz at
#   order10_win80ms_fs200_zscore for all six pairs, so without --overwrite
#   the runner would skip every subject there and never compute dtrgc.
#   Every NPCS > 1 path is new, so the default here is 0 (skip-if-exists,
#   which lets an interrupted run resume).
#
#   conda activate mne          # the script activates it itself
#   bash run_gc_final.sh
#   DRY_RUN=1 bash run_gc_final.sh          # print commands, run nothing
#   ARMS=theta bash run_gc_final.sh         # one arm only
#   TASKS=overtProd bash run_gc_final.sh    # skip perception
#   NORMALIZE=demean bash run_gc_final.sh   # ERP removed, still FIXPC4
#
# 2 arms x 6 pairs x 2 tasks x 2 contrasts = 48 configs, 20 subjects each,
# PARALLEL at a time (8 by default — the shared-drive I/O ceiling; 56
# crashed the workstation during the sweep). An 8-channel block VAR costs
# more per window than the 2-channel one (order x 64 coefficients instead of
# order x 4), so expect each config to take longer than the FIXPC1 runs.
set -u

cd "$(dirname "$0")"

ARMS="${ARMS:-main theta}"
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
METHOD="${METHOD:-LCMV}"
ATLAS="${ATLAS:-custom}"
FEAT="${FEAT:-vertex_selectkbest}"
LEAK="${LEAK:---leakage-correction}"
NORMALIZE="${NORMALIZE:-none}"
NPCS="${NPCS:-4}"
PARALLEL="${PARALLEL:-8}"
INNER_JOBS="${INNER_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAIN_OVERWRITE="${MAIN_OVERWRITE:-0}"

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

# Run label: normalize name plus the PC suffix (only for NPCS > 1, mirroring
# gc_tag), used for the log directory and the per-config tags.
RUN_LABEL="$NORMALIZE"
[ "$NPCS" -gt 1 ] && RUN_LABEL="${NORMALIZE}_pc${NPCS}"

echo "final GC analysis — pairwise TRGC, FIXPC${NPCS}, normalize=$NORMALIZE, $METHOD/$ATLAS/$FEAT $LEAK"
for ARM in $ARMS; do
    read -r ORDER WIN_MS TARGET_FS <<< "$(arm_params "$ARM")"
    samp=$(( WIN_MS * TARGET_FS / 1000 ))
    echo "  $ARM: order $ORDER, ${WIN_MS} ms @ ${TARGET_FS} Hz = ${samp} samples" \
         "(needs > $(( ORDER + 1 ))), memory $(( 1000 * ORDER / TARGET_FS )) ms," \
         "$(( 2 * NPCS ))-channel block VAR per pair"
done
echo "  $n_total configs, 20 subjects each, $PARALLEL at a time"
case " $ARMS " in *" main "*) [ "$MAIN_OVERWRITE" = "1" ] && \
    echo "  main arm runs with --overwrite" ;; esac
echo

CMD_FILE=$(mktemp); DONE_FILE=$(mktemp)
trap 'rm -f "$CMD_FILE" "$DONE_FILE"' EXIT
t0=$(date +%s); n=0

for ARM in $ARMS; do
read -r ORDER WIN_MS TARGET_FS <<< "$(arm_params "$ARM")"
OVR=""
[ "$ARM" = "main" ] && [ "$MAIN_OVERWRITE" = "1" ] && OVR="--overwrite"
LOG_DIR="logs/gc_final_${ARM}_${RUN_LABEL}"
mkdir -p "$LOG_DIR"
for PP in "${PAIR_ARR[@]}"; do
PP=$(trim "$PP")
for T in $TASKS; do for S in $STIMS; do
    tag="${T}_${S}_$(label "$PP")_trgc_win${WIN_MS}ms_order${ORDER}_pc${NPCS}"
    log="$LOG_DIR/${tag}.log"
    n=$(( n + 1 ))
    echo "[$n/$n_total] $ARM  $tag"
    CMD="python run_granger.py --task $T --stim-class $S --method $METHOD \
--atlas $ATLAS --feature-mode $FEAT $LEAK --gc-mode pairwise \
--win-ms $WIN_MS --order $ORDER --target-fs $TARGET_FS \
--normalize $NORMALIZE --n-pcs $NPCS --trgc --roi-subset $PP --n-jobs $INNER_JOBS $OVR"
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
echo "Then (pass the same --normalize and --n-pcs so the derived path matches,"
echo "or point --gc-dir at the results directory):"
echo "  python granger_stats.py --normalize $NORMALIZE --n-pcs $NPCS ...   # baseline-referenced TRGC stats (main arm; overtProd only for theta)"
echo "  python exploratory/plot_gc_pathway_timecourses.py       # theta-arm pathway figure"
grep -h 'NON-MINIMUM-PHASE\|consistency:\|\[fixpc\]' logs/gc_final_*/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -20
