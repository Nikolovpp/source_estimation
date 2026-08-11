#!/usr/bin/env bash
# run_gc_window_order_sweep.sh
# State-space spectral GC over an INDEPENDENT window x order grid.
#
#   windows : 40, 60, 80 ms          (fs 200 Hz)
#   orders  : 2, 4, 6, 10
#   tasks   : overtProd, perception
#   stims   : prodDiff, percDiff
#   estimator: state-space (Barnett & Seth 2015) throughout. The analysis is
#             set by SUBSET SIZE, not by a second estimator: 2 ROIs give
#             bivariate GC (identical to parametric BSMART pairwise, 1.67e-16),
#             3 ROIs give A->B|C. Spectral in both cases — this project does
#             not compute time-domain GC.
#   subsets : two triples plus the three named pathways bivariately, so every
#             pathway appears both conditioned and unconditioned. 5 subsets x 11
#             feasible cells x 2 tasks x 2 contrasts = 220 runs; trim with
#             SUBSETS=..., WINDOWS=..., ORDERS=... or TASKS=... .
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
#   2 ROIs -> conditioning set empty -> bivariate GC. Provably identical to
#             parametric BSMART pairwise; measured at 1.67e-16 by the M0 check
#             in run_granger_routes.py --self-test.
#   3 ROIs -> conditioned on the single remaining ROI -> A->B|C, the
#             hypothesis-based triple-wise form.
#   4+     -> conditioned on everything else, which is the all-ROI conditioning
#             this project deliberately moved away from.
#
# So there is no parametric arm here. Run GC_MODE=pairwise if you want to
# confirm the M0 identity on a specific result.
GC_MODE="${GC_MODE:-conditional}"
TARGET_FS="${TARGET_FS:-200}"
# Semicolon-separated ROI subsets; each is run as its own sweep. The three
# named pathways are temporal<->frontal, frontal<->parietal, temporal<->parietal.
# The two triples give every one of those pairs conditioned on the third ROI;
# the pairs give the same edges bivariately. pmc-lh is the second frontal node,
# so it needs its own triple rather than being added to the first (that would
# make the conditioning set two ROIs).
SUBSETS="${SUBSETS:-\
awfa-lh ifc-lh tpc-lh; \
awfa-lh pmc-lh tpc-lh; \
awfa-lh ifc-lh; \
ifc-lh tpc-lh; \
awfa-lh tpc-lh}"
# Back-compat: ROIS=... still works and collapses the run to that one subset.
if [ -n "${ROIS:-}" ]; then SUBSETS="$ROIS"; fi
IFS=';' read -ra SUBSET_ARR <<< "$SUBSETS"
NJOBS="${NJOBS:-64}"
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
echo "gc-mode: $GC_MODE   (analysis set by subset size)"
echo "subsets:"
for SS in "${SUBSET_ARR[@]}"; do
    printf '   %-28s %s\n' "$(label "$SS")" "$(scope "$SS")"
done
echo "$n_total configurations, 20 subjects each"
echo

t0=$(date +%s)
n_done=0; n_fail=0
for SS in "${SUBSET_ARR[@]}"; do
SS=$(trim "$SS")
for T in $TASKS; do
for S in $STIMS; do
for W in $WINDOWS; do
for O in $ORDERS; do
    feasible "$W" "$O" || continue
    tag="${T}_${S}_$(label "$SS")_$(scope "$SS")_win${W}ms_order${O}"
    log="$LOG_DIR/${tag}.log"
    n_done=$(( n_done + 1 ))
    echo "[$n_done/$n_total] $tag"

    # shellcheck disable=SC2086
    CMD="python run_granger.py --task $T --stim-class $S --method $METHOD \
        --atlas $ATLAS --feature-mode $FEAT $LEAK --gc-mode $GC_MODE \
        --win-ms $W --order $O --target-fs $TARGET_FS --normalize $NORMALIZE \
        --roi-subset $SS --n-jobs $NJOBS"

    if [ "$DRY_RUN" = "1" ]; then
        echo "    $CMD"
        continue
    fi

    c0=$(date +%s)
    if $CMD >"$log" 2>&1; then
        echo "    done in $(( $(date +%s) - c0 ))s"
        # after the first cell, project the total
        if [ "$n_done" = "1" ]; then
            echo "    -> ~$(( ($(date +%s) - c0) * n_total / 60 )) min projected for all $n_total"
        fi
    else
        n_fail=$(( n_fail + 1 ))
        echo "    FAILED — see $log" >&2
        tail -5 "$log" | sed 's/^/      /' >&2
    fi
done; done; done; done; done

echo
echo "$(( n_done - n_fail ))/$n_total succeeded in $(( ($(date +%s) - t0) / 60 )) min"
echo "logs:    $LOG_DIR"
echo "outputs: derivatives/source_estimation/GC_source_space/{task}/${METHOD}/${ATLAS}/${FEAT}/..."
