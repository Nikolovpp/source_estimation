#!/usr/bin/env bash
# run_gc_window_order_sweep.sh
# Pairwise spectral GC over an INDEPENDENT window x order grid.
#
#   windows : 40, 60, 80 ms          (fs 200 Hz)
#   orders  : 2, 4, 6, 10
#   tasks   : overtProd, perception
#   stims   : prodDiff, percDiff
#   gc-mode : pairwise (parametric BSMART) by default; GC_MODE=conditional
#             runs the state-space arm. Spectral in both cases — this project
#             does not use time-domain GC.
#   pairs   : the three dual-stream pathways, via a 4-ROI subset
#             temporal<->frontal (awfa<->ifc), frontal<->parietal (ifc<->tpc),
#             temporal<->parietal (awfa<->tpc); pmc included as the second
#             frontal node.
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
#   GC_MODE=conditional bash run_gc_window_order_sweep.sh       # state-space arm
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
# pairwise    = bivariate parametric (BSMART) spectral GC — what the manuscript
#               reports, and the default here.
# conditional = state-space conditional spectral GC (Barnett & Seth 2015), each
#               edge conditioned on the other ROIs in --roi-subset.
# These are DIFFERENT QUANTITIES, not two estimators of one quantity. For
# pairwise spectral GC the parametric and state-space estimators are provably
# identical (measured 8e-16 here), so there is no separate estimator arm to run
# — the estimator choice only becomes live once you condition.
GC_MODE="${GC_MODE:-pairwise}"
TARGET_FS="${TARGET_FS:-200}"
ROIS="${ROIS:-awfa-lh tpc-lh ifc-lh pmc-lh}"
NJOBS="${NJOBS:-64}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-./logs/gc_window_order_sweep}"
mkdir -p "$LOG_DIR"

# ── feasibility: an AR fit needs more samples than parameters ───────────
# Infeasible cells are SKIPPED and listed, not silently dropped and not fatal:
# the grid is deliberately not rectangular, because a short window cannot carry
# a high order. At fs 200, 40 ms is 8 samples and so tops out around order 6.
feasible () {  # feasible <win_ms> <order>
    local samp=$(( $1 * TARGET_FS / 1000 ))
    [ "$samp" -gt $(( $2 + 1 )) ]
}

n_total=0; skipped=""
for T in $TASKS; do for S in $STIMS; do for W in $WINDOWS; do for O in $ORDERS; do
    if feasible "$W" "$O"; then
        n_total=$(( n_total + 1 ))
    else
        case " $skipped " in *" ${W}/${O} "*) ;; *) skipped="$skipped ${W}/${O}";; esac
    fi
done; done; done; done
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
echo "method:  $METHOD   ROIs: $ROIS   normalize: $NORMALIZE"
echo "gc-mode: $GC_MODE"
echo "$n_total configurations, 20 subjects each"
echo

t0=$(date +%s)
n_done=0; n_fail=0
for T in $TASKS; do
for S in $STIMS; do
for W in $WINDOWS; do
for O in $ORDERS; do
    feasible "$W" "$O" || continue
    tag="${T}_${S}_${GC_MODE}_win${W}ms_order${O}"
    log="$LOG_DIR/${tag}.log"
    n_done=$(( n_done + 1 ))
    echo "[$n_done/$n_total] $tag"

    # shellcheck disable=SC2086
    CMD="python run_granger.py --task $T --stim-class $S --method $METHOD \
        --atlas $ATLAS --feature-mode $FEAT $LEAK --gc-mode $GC_MODE \
        --win-ms $W --order $O --target-fs $TARGET_FS --normalize $NORMALIZE \
        --roi-subset $ROIS --n-jobs $NJOBS"

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
done; done; done; done

echo
echo "$(( n_done - n_fail ))/$n_total succeeded in $(( ($(date +%s) - t0) / 60 )) min"
echo "logs:    $LOG_DIR"
echo "outputs: derivatives/source_estimation/GC_source_space/{task}/${METHOD}/${ATLAS}/${FEAT}/..."
