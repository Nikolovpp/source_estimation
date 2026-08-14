#!/usr/bin/env bash
# run_gc_window_order_sweep.sh
# State-space spectral GC over an INDEPENDENT window x order grid.
#
#   windows : 40, 60, 80 ms          (fs 200 Hz)
#   orders  : 2, 4, 6, 10
#   tasks   : overtProd, perception
#   stims   : prodDiff, percDiff
#   estimator: state-space (Barnett & Seth 2015) throughout. The analysis is
#             set by SUBSET SIZE: 2 ROIs -> pairwise (bivariate, identical to
#             state-space at 1.67e-16), 3 ROIs -> conditional A->B|C. Spectral
#             in both cases — this project does not compute time-domain GC.
#   subsets : two triples plus all six bivariate pairs of the four ROIs, so
#             every edge appears both conditioned and unconditioned.
#             8 subsets x 11 feasible cells x 2 tasks x 2 contrasts = 352 runs;
#             trim with SUBSETS=..., WINDOWS=..., ORDERS=... or TASKS=... .
#
# WHY THIS AND NOT THE GC_routes SWEEPS: run_granger_routes.py computes
# conditional spectral, time-domain and block GC. It stores band-resolved
# PAIRWISE spectral GC only when --triples is set. Pairwise spectral GC is the
# quantity this project reports, so the sweep has to come from run_granger.py,
# the production pipeline, which writes fxy_{band} / fyx_{band} directly.
#
# WHY WINDOW AND ORDER VARY INDEPENDENTLY: the existing configs confound them
# (order6_win60ms vs order10_win120ms changes both at once), so they cannot say
# which of the two drives a difference. This grid can.
#
# WHY NO 20 ms WINDOW: at fs 200 that is 4 samples, which cannot support order 4
# or above. Running it would mean fs 500 for that column alone, introducing a
# sampling-rate confound into a sweep built to isolate window and order. Run it
# separately if you want it.
#
# NOTE ON PERCEPTION: those epochs span -200 to +600 ms (config.PERCEPTION_TMIN/
# TMAX), which gives a genuine pre-stimulus baseline at every window size here:
# 36/34/32 pre-stimulus window centres at 40/60/80 ms, and still 33/31/29 after
# dropping the first three edge windows. A [-200,-100] ms baseline sits well
# inside that. Both tasks support baseline-referenced statistics.
#
#   conda activate mne
#   bash run_gc_window_order_sweep.sh
#   DRY_RUN=1 bash run_gc_window_order_sweep.sh          # print commands only
#   TASKS=overtProd STIMS=prodDiff bash run_gc_window_order_sweep.sh
#   WINDOWS="60" ORDERS="6" bash run_gc_window_order_sweep.sh   # time one cell
#   SUBSETS="awfa-lh ifc-lh tpc-lh" bash run_gc_window_order_sweep.sh  # one subset
#   GC_MODE=pairwise bash run_gc_window_order_sweep.sh    # parametric, to check M0
#   PARALLEL=32 bash run_gc_window_order_sweep.sh         # 32 configs at once
#   SCOPE=bivariate bash run_gc_window_order_sweep.sh     # only the 2-ROI pairs
set -u

cd "$(dirname "$0")"

# Portable conda bootstrap — this box uses miniforge3, the workstation
# anaconda3, so do not hardcode a prefix. Fail loudly rather than silently
# running the wrong interpreter.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _p in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        if [ -f "$_p/etc/profile.d/conda.sh" ]; then
            source "$_p/etc/profile.d/conda.sh"; break
        fi
    done
fi
if ! conda activate mne 2>/dev/null; then
    echo "ERROR: could not activate the 'mne' conda env." >&2
    echo "  conda found at: $(command -v conda || echo '<none on PATH>')" >&2
    exit 1
fi

# ── knobs ───────────────────────────────────────────────────────────────
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
WINDOWS="${WINDOWS:-40 60 80}"
ORDERS="${ORDERS:-2 4 6 10}"
METHOD="${METHOD:-LCMV}"
ATLAS="${ATLAS:-custom}"
FEAT="${FEAT:-vertex_selectkbest}"
LEAK="${LEAK:---leakage-correction}"          # "" for raw
NORMALIZE="${NORMALIZE:-demean}"              # ERP removal; part of the path
# One estimator for everything: state-space (Barnett & Seth 2015). The ANALYSIS
# is then chosen by how many ROIs each subset holds, because --gc-mode
# conditional conditions each edge on the OTHER ROIs in the subset:
#
#   2 ROIs -> --gc-mode pairwise, the bivariate quantity.
#   3 ROIs -> conditioned on the single remaining ROI -> A->B|C, the
#             hypothesis-based triple-wise form.
#   4+     -> conditioned on everything else, which is the all-ROI conditioning
#             this project deliberately moved away from.
#
# AUTO is the default and picks the mode per subset. It is a labelling choice,
# not a correctness one: --gc-mode conditional on a 2-ROI subset gives the SAME
# numbers (verified, 4.4e-16), because ss_conditional_gc branches on an empty
# conditioning set exactly as MVGC's autocov_to_smvgc.m does ("if isempty(z)
# % unconditional"). auto keeps the output path and the log honest about which
# quantity was computed.
#
# An earlier comment here blamed a 2-ROI NaN wipeout on that empty conditioning
# set. That was wrong: the cause was the --normalize axis bug in
# compute_subject_gc (ff2cdd6, fixed 95bd838), which made the pair rank 1 and
# every covariance singular. See validate_granger_normalize.py.
GC_MODE="${GC_MODE:-auto}"
mode_for () {  # mode_for <subset>
    if [ "$GC_MODE" != "auto" ]; then echo "$GC_MODE"; return; fi
    case $(trim "$1" | wc -w) in
        2) echo pairwise ;;
        *) echo conditional ;;
    esac
}
TARGET_FS="${TARGET_FS:-200}"
# Semicolon-separated ROI subsets; each is run as its own sweep. The three
# named pathways are temporal<->frontal, frontal<->parietal, temporal<->parietal.
# The two triples give every one of those pairs conditioned on the third ROI;
# the pairs cover ALL SIX bivariate combinations of the four ROIs, so every
# edge that appears in a triple also has an unconditioned counterpart.
# pmc-lh is the second frontal node,
# so it needs its own triple rather than being added to the first (that would
# make the conditioning set two ROIs).
# Keep BOTH arms listed here and select with SCOPE at call time — commenting a
# line out breaks SCOPE=triplewise, and a trailing "\" inside a "#" comment does
# not continue the line anyway, so the entry silently vanishes rather than being
# preserved for later.
#   SCOPE=bivariate   -> the six pairs
#   SCOPE=triplewise  -> the two triples
#   SCOPE=all         -> everything (default; finished cells are skipped)
SUBSETS="${SUBSETS:-\
awfa-lh ifc-lh tpc-lh; \
awfa-lh pmc-lh tpc-lh; \
awfa-lh ifc-lh; \
ifc-lh tpc-lh; \
awfa-lh tpc-lh; \
awfa-lh pmc-lh; \
pmc-lh tpc-lh; \
ifc-lh pmc-lh}"
# Back-compat: ROIS=... still works and collapses the run to that one subset.
if [ -n "${ROIS:-}" ]; then SUBSETS="$ROIS"; fi
IFS=';' read -ra SUBSET_ARR <<< "$SUBSETS"
# SCOPE filters the list by subset size without retyping it. The triple-wise
# cells are already on disk, so SCOPE=bivariate is the fast way to run only the
# arm being recomputed. (Leaving it at 'all' is also fine — finished configs are
# skipped — but it still stats 20 files per triple cell.)
SCOPE="${SCOPE:-all}"
if [ "$SCOPE" != "all" ]; then
    _keep=()
    for _s in "${SUBSET_ARR[@]}"; do
        _n=$(echo "$_s" | wc -w)
        case "$SCOPE" in
            bivariate)  [ "$_n" -eq 2 ] && _keep+=("$_s") ;;
            triplewise) [ "$_n" -eq 3 ] && _keep+=("$_s") ;;
            *) echo "ERROR: SCOPE must be all, bivariate or triplewise" >&2; exit 2 ;;
        esac
    done
    if [ ${#_keep[@]} -eq 0 ]; then
        echo "ERROR: SCOPE=$SCOPE left no subsets" >&2; exit 2
    fi
    SUBSET_ARR=("${_keep[@]}")
fi
# PARALLELISM. run_granger.py loops subjects sequentially and parallelizes only
# over WINDOWS inside a subject. Each window is a 3-variable VAR on ~12 samples
# — milliseconds — while joblib's process backend pickles the data array per
# batch, so the overhead swamps the work and only 1-4 cores stay busy no matter
# what --n-jobs says.
#
# Configs, by contrast, are big and completely independent. So run PARALLEL of
# them at once, each with INNER_JOBS internal workers.
#
# START LOW AND MEASURE. Core count is NOT the ceiling. Each worker reads a
# multi-GB vertex cache, and when those caches live on a shared/network drive
# the I/O contention stalls the whole box long before RAM or cores run out —
# PARALLEL=8 on a 64-core, 256 GB machine took the system down. Step up from
# the default and watch, rather than reasoning from free memory:
#
#   ps -o rss=,comm= -C python | awk '{s+=$1; n++} END \
#       {print n" workers, "s/1048576" GB, "s/n/1048576" GB each"}'
#   iostat -x 5      # %util near 100 on the cache volume = I/O bound, back off
#
# Double it only once the box is comfortably idle at the current value.
PARALLEL="${PARALLEL:-8}"
INNER_JOBS="${INNER_JOBS:-1}"
NJOBS="$INNER_JOBS"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-./logs/gc_window_order_sweep}"
mkdir -p "$LOG_DIR"

# ── feasibility: an AR fit needs more samples than parameters ───────────
# Infeasible cells are SKIPPED and listed, not silently dropped and not fatal:
# the grid is deliberately not rectangular, because a short window cannot carry
# a high order. At fs 200, 40 ms is 8 samples and so tops out around order 6.
trim  () { echo "$1" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'; }
label () { trim "$1" | sed 's/-lh//g; s/[[:space:]][[:space:]]*/+/g'; }
scope () {  # what the conditioning set makes this subset mean
    case $(trim "$1" | wc -w) in
        2) echo bivariate ;;
        3) echo triplewise ;;
        *) echo multi ;;
    esac
}

feasible () {  # feasible <win_ms> <order>
    local samp=$(( $1 * TARGET_FS / 1000 ))
    [ "$samp" -gt $(( $2 + 1 )) ]
}

n_total=0; skipped=""
for SS in "${SUBSET_ARR[@]}"; do
for T in $TASKS; do for S in $STIMS; do for W in $WINDOWS; do for O in $ORDERS; do
    if feasible "$W" "$O"; then
        n_total=$(( n_total + 1 ))
    else
        case " $skipped " in *" ${W}/${O} "*) ;; *) skipped="$skipped ${W}/${O}";; esac
    fi
done; done; done; done; done
if [ -n "$skipped" ]; then
    echo "SKIPPING infeasible window/order cells (window must exceed order):"
    for c in $skipped; do
        w=${c%%/*}; o=${c##*/}
        echo "   ${w} ms = $(( w * TARGET_FS / 1000 )) samples at ${TARGET_FS} Hz, cannot hold order ${o}"
    done
    echo
fi

echo "tasks:   $TASKS"
echo "stims:   $STIMS"
echo "windows: $WINDOWS ms      orders: $ORDERS"
echo "method:  $METHOD   normalize: $NORMALIZE"
echo "gc-mode: $GC_MODE   (2 ROIs -> pairwise, 3+ -> conditional)"
echo "subsets:"
for SS in "${SUBSET_ARR[@]}"; do
    printf '   %-28s %-11s %s\n' "$(label "$SS")" "$(scope "$SS")" "$(mode_for "$SS")"
done
echo "$n_total configurations, 20 subjects each"
echo

CMD_FILE=$(mktemp)
DONE_FILE=$(mktemp)
trap 'rm -f "$CMD_FILE" "$DONE_FILE"' EXIT

t0=$(date +%s)
n_done=0; n_fail=0
for SS in "${SUBSET_ARR[@]}"; do
SS=$(trim "$SS")
for T in $TASKS; do
for S in $STIMS; do
for W in $WINDOWS; do
for O in $ORDERS; do
    feasible "$W" "$O" || continue
    MODE=$(mode_for "$SS")
    tag="${T}_${S}_$(label "$SS")_$(scope "$SS")_${MODE}_win${W}ms_order${O}"
    log="$LOG_DIR/${tag}.log"
    n_done=$(( n_done + 1 ))
    echo "[$n_done/$n_total] $tag"

    # shellcheck disable=SC2086
    CMD="python run_granger.py --task $T --stim-class $S --method $METHOD \
        --atlas $ATLAS --feature-mode $FEAT $LEAK --gc-mode $MODE \
        --win-ms $W --order $O --target-fs $TARGET_FS --normalize $NORMALIZE \
        --roi-subset $SS --n-jobs $NJOBS"

    if [ "$DRY_RUN" = "1" ]; then
        echo "    $CMD"
        continue
    fi
    # Queue it; xargs runs PARALLEL of these at once. Skipping a config whose
    # output is already complete keeps a restart cheap after a partial run.
    #
    # Each config reports to STDERR as it finishes. Without this the terminal
    # is silent for the whole run — every config's stdout goes to its log, so
    # a 352-config sweep looks identical to a hung one for hours.
    # NUL-delimited and eval'd from $0, because `xargs -I{}` substitutes
    # without shell quoting and mangles any nested quote below.
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
done; done; done; done; done

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

echo "running $n_total configs, $PARALLEL at a time, $INNER_JOBS worker(s) each"
echo "Each reports as it finishes. Per-subject progress:  tail -f $LOG_DIR/*.log"
echo
xargs -0 -P "$PARALLEL" -n1 bash -c 'eval "$0"' < "$CMD_FILE"

echo
echo "$n_total configs attempted in $(( ($(date +%s) - t0) / 60 )) min"
echo "failures: grep -l Error $LOG_DIR/*.log"
echo "logs:    $LOG_DIR"
echo "outputs: derivatives/source_estimation/GC_source_space/{task}/${METHOD}/${ATLAS}/${FEAT}/..."
