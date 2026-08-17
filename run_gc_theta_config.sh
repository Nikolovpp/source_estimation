#!/usr/bin/env bash
# run_gc_theta_config.sh
# The theta-sized config — MO25 / SW200 ms / fs200 — for the pairwise and
# triple-wise ROI subsets.
#
# WHY THIS CONFIG. Model order is a duration: p/fs seconds. The sweep's grid
# tops out at order 10 @ 200 Hz = 50 ms of memory, which spans 1.5 cycles of
# high beta but only 0.40 of a theta cycle — theta was under-covered at every
# cell of it (see gc_build_log.md §5). Order 25 at 200 Hz spans 125 ms, i.e. a
# full cycle at 8 Hz and half a cycle at 4 Hz.
#
# fs200 RATHER THAN THE fs250 OF THE ORIGINAL. The GC_qc investigation settled
# on MO25/SW200/fs250 (100 ms memory). At 200 Hz the same order buys 125 ms
# instead, and it keeps the analysis rate identical to the completed sweep, so
# the two are directly comparable. 200 ms at 200 Hz is 40 samples against a
# Morf recursion needing > 26 — feasible with room to spare.
#
# *** PERCEPTION HAS NO BASELINE AT THIS WINDOW. ***
# A window starting at t covers [t, t+200], so it is fully pre-stimulus only if
# it starts at or before -200 ms. The perception GC axis begins at -100 ms
# (the epoch starts at -200 and LCMV consumes the first 100 ms for its
# covariance), so ZERO windows are fully pre-stimulus. Perception is still run
# here — the time courses and the sweep-style comparisons are valid — but any
# task-vs-baseline statistic on it at this window is meaningless. overtProd
# has 261 fully pre-articulation windows and is unaffected.
#
#   conda activate mne
#   bash run_gc_theta_config.sh
#   DRY_RUN=1 bash run_gc_theta_config.sh
#   TASKS=overtProd bash run_gc_theta_config.sh      # skip perception
set -u

cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _p in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_p/etc/profile.d/conda.sh" ] && { source "$_p/etc/profile.d/conda.sh"; break; }
    done
fi
conda activate mne 2>/dev/null || { echo "ERROR: cannot activate 'mne'" >&2; exit 1; }

ORDER="${ORDER:-25}"
WIN_MS="${WIN_MS:-200}"
TARGET_FS="${TARGET_FS:-200}"
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
LOG_DIR="${LOG_DIR:-logs/gc_theta_config}"
mkdir -p "$LOG_DIR"

# Same eight subsets as the main sweep: two triples, all six pairs.
SUBSETS="${SUBSETS:-\
awfa-lh ifc-lh tpc-lh; \
awfa-lh pmc-lh tpc-lh; \
awfa-lh ifc-lh; \
ifc-lh tpc-lh; \
awfa-lh tpc-lh; \
awfa-lh pmc-lh; \
pmc-lh tpc-lh; \
ifc-lh pmc-lh}"
IFS=';' read -ra SUBSET_ARR <<< "$SUBSETS"

trim () { echo "$1" | sed 's/^ *//;s/ *$//'; }
label () { trim "$1" | sed 's/-lh//g;s/ /+/g'; }
mode_for () { case $(trim "$1" | wc -w) in 2) echo pairwise ;; *) echo conditional ;; esac; }

n_total=0
for SS in "${SUBSET_ARR[@]}"; do for T in $TASKS; do for S in $STIMS; do
    n_total=$(( n_total + 1 ))
done; done; done

samp=$(( WIN_MS * TARGET_FS / 1000 ))
echo "theta-sized GC config"
echo "  order $ORDER, ${WIN_MS} ms @ ${TARGET_FS} Hz = ${samp} samples "\
"(needs > $(( ORDER + 1 )))"
echo "  model memory: $(( 1000 * ORDER / TARGET_FS )) ms"
echo "  normalize: $NORMALIZE   method: $METHOD/$ATLAS/$FEAT $LEAK"
echo "  $n_total configs, 20 subjects each, $PARALLEL at a time"
case " $TASKS " in *" perception "*)
    echo
    echo "  *** perception has NO fully pre-stimulus window at a ${WIN_MS} ms"
    echo "      window (axis starts -100 ms). Time courses are fine; do not"
    echo "      run task-vs-baseline statistics on it at this config." ;;
esac
echo

CMD_FILE=$(mktemp); DONE_FILE=$(mktemp)
trap 'rm -f "$CMD_FILE" "$DONE_FILE"' EXIT
t0=$(date +%s); n=0

for SS in "${SUBSET_ARR[@]}"; do
SS=$(trim "$SS")
for T in $TASKS; do for S in $STIMS; do
    MODE=$(mode_for "$SS")
    tag="${T}_${S}_$(label "$SS")_${MODE}_win${WIN_MS}ms_order${ORDER}"
    log="$LOG_DIR/${tag}.log"
    n=$(( n + 1 ))
    echo "[$n/$n_total] $tag"
    CMD="python run_granger.py --task $T --stim-class $S --method $METHOD \
--atlas $ATLAS --feature-mode $FEAT $LEAK --gc-mode $MODE \
--win-ms $WIN_MS --order $ORDER --target-fs $TARGET_FS \
--normalize $NORMALIZE --roi-subset $SS --n-jobs $INNER_JOBS"
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

[ "$DRY_RUN" = "1" ] && exit 0

echo
echo "running $n_total configs, $PARALLEL at a time; each reports as it finishes."
echo "Per-subject progress:  tail -f $LOG_DIR/*.log"
echo
xargs -0 -P "$PARALLEL" -n1 bash -c 'eval "$0"' < "$CMD_FILE"

echo
echo "$n_total configs attempted in $(( ($(date +%s) - t0) / 60 )) min"
echo
echo "Then:  python plot_gc_pathway_timecourses.py"
grep -h 'NON-MINIMUM-PHASE\|consistency:' "$LOG_DIR"/*.log 2>/dev/null | head -20
