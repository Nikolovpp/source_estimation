#!/usr/bin/env bash
# run_gc_routes_batch.sh
# Parametric vs state-space Granger causality, five configurations.
#
#   A  canonical (Damera)   60 ms / order 6 / fs 200, step 5 ms
#   B  window sweep         40, 60, 80, 120 ms @ order 6
#   C  order sweep          2,4,6,8 @ 60 ms   and   4,8,12,16,20 @ 120 ms
#   E  FIXPC1 vs FIXPC4     60 ms / order 6
#   F  triple-wise cond GC  every ordered triple, primaries flagged
#
# WHY 60 ms / order 6 / fs 200 is canonical: it is the lab's fs-200 translation
# of Damera & Martin (order 15 / 60 ms / fs 500). Order 15 at 500 Hz is 30 ms of
# memory; order 6 at 200 Hz is the same 30 ms, and 12 samples can hold order 6
# where it could never hold order 15.
#
# WHY the order sweep needs two windows: window bounds order (an AR fit needs
# window > order). 12 samples at 60 ms cannot carry order 10, so a single sweep
# would confound "higher order" with "infeasible fit". Orders 2-8 at 60 ms and
# 4-20 at 120 ms separate them, and the overlap at 4-8 cross-checks.
#
# WHY the nonparametric (Wilson) route is absent: its wrap-around limit is set
# by the data window, not by zero-padding. A 60 ms window at 200 Hz gives a
# circular lag domain of 12 samples, so the autocovariance must have decayed by
# lag 6 (30 ms); theta at 6 Hz has a 167 ms period. It is implemented and
# verified in granger_wilson.py, and excluded on measurement, not by omission.
#
# WHY step 25 ms for the sweeps: GC cannot resolve anything finer than its own
# window, so a 25 ms step on a 60 ms window still overlaps 58% and runs 5x
# faster. Config A keeps 5 ms because that is the grid the decoding results use.
#
# SCOPE: A and B run the full grid (both inverses x both tasks x both classes).
# C, E and F are methods diagnostics and run on dSPM only, for
# overtProd/prodDiff and perception/percDiff.
#
#   conda activate mne
#   bash run_gc_routes_batch.sh
#   CONFIGS="A F" bash run_gc_routes_batch.sh        # subset
#   DRY_RUN=1 bash run_gc_routes_batch.sh            # print commands only
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
CONFIGS="${CONFIGS:-A B C E F}"
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
METHODS="${METHODS:-dSPM LCMV}"
DIAG_METHOD="${DIAG_METHOD:-dSPM}"          # C/E/F run on this inverse only
ATLAS="${ATLAS:-custom}"
ROIS="${ROIS:-awfa-lh ifc-lh owfa-lh pmc-lh tpc-lh vwfa-lh}"
NJOBS="${NJOBS:-64}"
SWEEP_STEP="${SWEEP_STEP:-25}"              # ms, for B/C/E/F
CANON_STEP="${CANON_STEP:-5}"               # ms, for A (matches decoding)
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-./logs/gc_routes}"
mkdir -p "$LOG_DIR"

RUN="python run_granger_routes.py --atlas $ATLAS --leakage-correction \
     --rois $ROIS --n-jobs $NJOBS"

# The diagnostic configs (C/E/F) pair each task with its own contrast, and have
# their own scope independent of TASKS/STIMS — override DIAG_PAIRS to trim them.
DIAG_PAIRS="${DIAG_PAIRS:-overtProd:prodDiff perception:percDiff}"

have() { case " $CONFIGS " in *" $1 "*) return 0;; *) return 1;; esac; }

go () {  # go <tag> <logname> <extra args...>
    local tag="$1"; shift
    local logname="$1"; shift
    local log="$LOG_DIR/${logname}.log"
    echo "── $tag → $log"
    if [ "$DRY_RUN" = "1" ]; then
        echo "   $RUN $* --config-tag $tag"
        return 0
    fi
    # shellcheck disable=SC2086
    if $RUN "$@" --config-tag "$tag" >"$log" 2>&1; then
        tail -2 "$log" | sed 's/^/   /'
    else
        echo "   FAILED — see $log" >&2
        tail -5 "$log" | sed 's/^/   /' >&2
    fi
}

# Preflight: resolve one config's caches before launching anything. Cheap, and
# it turns "wrong paths" from a wall of tracebacks into one line, up front.
if [ "$DRY_RUN" != "1" ]; then
    _t=${TASKS%% *}; _s=${STIMS%% *}; _m=${METHODS%% *}
    if ! $RUN --task "$_t" --stim-class "$_s" --method "$_m" --check >/dev/null 2>&1; then
        echo "PREFLIGHT FAILED — no ROI timeseries caches resolved. Details:" >&2
        $RUN --task "$_t" --stim-class "$_s" --method "$_m" --check >&2
        exit 2
    fi
fi

t0=$(date +%s)
echo "configs: $CONFIGS   tasks: $TASKS   stims: $STIMS   methods: $METHODS"
echo "ROIs: $ROIS"
echo

# ── A · canonical (Damera), full grid, 5 ms step ────────────────────────
if have A; then
  echo "=== A · canonical: 60 ms / order 6 / fs 200 / step ${CANON_STEP} ms ==="
  for m in $METHODS; do for t in $TASKS; do for s in $STIMS; do
    go "A_canonical" "A_${m}_${t}_${s}" \
       --task "$t" --stim-class "$s" --method "$m" \
       --win-ms 60 --order 6 --step-ms "$CANON_STEP"
  done; done; done
fi

# ── B · window sweep, full grid ─────────────────────────────────────────
if have B; then
  echo "=== B · window sweep: 40 60 80 120 ms @ order 6 ==="
  for m in $METHODS; do for t in $TASKS; do for s in $STIMS; do
    go "B_winsweep" "B_${m}_${t}_${s}" \
       --task "$t" --stim-class "$s" --method "$m" \
       --win-ms 40 60 80 120 --order 6 --step-ms "$SWEEP_STEP"
  done; done; done
fi

# ── C · order sweep, two windows, diagnostics scope ─────────────────────
if have C; then
  echo "=== C · order sweep: 2-8 @ 60 ms, 4-20 @ 120 ms ==="
  for pair in $DIAG_PAIRS; do
    t="${pair%%:*}"; s="${pair##*:}"
    go "C_ordersweep_win60"  "C60_${t}_${s}" \
       --task "$t" --stim-class "$s" --method "$DIAG_METHOD" \
       --win-ms 60  --order 2 4 6 8       --step-ms "$SWEEP_STEP"
    go "C_ordersweep_win120" "C120_${t}_${s}" \
       --task "$t" --stim-class "$s" --method "$DIAG_METHOD" \
       --win-ms 120 --order 4 8 12 16 20  --step-ms "$SWEEP_STEP"
  done
fi

# ── E · FIXPC1 vs FIXPC4, diagnostics scope ─────────────────────────────
if have E; then
  echo "=== E · FIXPC1 vs FIXPC4 @ 60 ms / order 6 ==="
  for pair in $DIAG_PAIRS; do
    t="${pair%%:*}"; s="${pair##*:}"
    go "E_fixpc" "E_${t}_${s}" \
       --task "$t" --stim-class "$s" --method "$DIAG_METHOD" \
       --win-ms 60 --order 6 --n-pcs 1 4 --step-ms "$SWEEP_STEP"
  done
fi

# ── F · triple-wise conditional GC, exhaustive ──────────────────────────
# Every ordered triple over the 6 ROIs (120 of them); the pre-registered
# mediation hypotheses are flagged primary in the output so the remainder stay
# exploratory. This is the slowest config — it is the one to trim first.
if have F; then
  echo "=== F · exhaustive triple-wise conditional GC @ 60 ms / order 6 ==="
  for pair in $DIAG_PAIRS; do
    t="${pair%%:*}"; s="${pair##*:}"
    go "F_triples" "F_${t}_${s}" \
       --task "$t" --stim-class "$s" --method "$DIAG_METHOD" \
       --win-ms 60 --order 6 --triples exhaustive --step-ms "$SWEEP_STEP"
  done
fi

echo
echo "done in $(( ($(date +%s) - t0) / 60 )) min — logs in $LOG_DIR"
echo "outputs under: derivatives/source_estimation/GC_routes/"
grep -l "PC cap\|collapse?" "$LOG_DIR"/*.log 2>/dev/null | while read -r f; do
    echo "  ⚠ diagnostics flagged in $(basename "$f")"
done
