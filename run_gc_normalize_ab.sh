#!/usr/bin/env bash
# run_gc_normalize_ab.sh
# A/B the two open preprocessing decisions BEFORE committing the full sweep.
#
# Both change stored output, so deciding either one after the 264-config sweep
# means running the sweep again. Both are cheap to settle here.
#
#   arm 0  none              NO ERP removal. BSMART-faithful and what the
#                            legacy runs used. Included so the cost of the
#                            correction is measured, not assumed: Ding et al.
#                            (2000) Fig. 7 show the fit going non-minimum-phase
#                            after stimulus onset without it, and the model
#                            validation block in each log now tests that
#                            directly on this data (rho>=1 count).
#   arm A  demean            ensemble-mean (ERP) removal only. Current default.
#                            = Ding, Bressler, Yang & Liang (2000) step 2.
#   arm B  zscore            + point-by-point ensemble-SD normalisation.
#                            = Ding et al. steps 2+3. They call step 3 "crucial
#                            for allowing the dynamical changes in model-derived
#                            spectral quantities to be compared at each stage of
#                            task processing" — exactly the across-window
#                            comparison this project reports.
#   arm C  demean, no        drops the per-trial whole-epoch temporal demean.
#          per-trial demean  MVGC's stats/demean.m warns that per-trial
#                            demeaning "can introduce large bias in VAR model
#                            estimation"; Ding et al. do it anyway, over the
#                            whole trial. Untested here, one flag to check.
#   arm D  demean per class  removes the ERP WITHIN each stimulus class.
#                            GC pools both classes, so the pooled removal in
#                            arms A-C leaves the between-class evoked
#                            difference in: class A keeps +delta/2, class B
#                            -delta/2, a deterministic class-locked waveform.
#                            On a synthetic 25 ms ROI-to-ROI latency
#                            difference, pooled removal left a theta peak of
#                            1.80 where per-class gave 0.93.
#
# ARM B'S COST IS SMALLER THAN IT LOOKED. The ensemble SD is estimated from N
# epochs and jitters by ~1/sqrt(2N), which whitens. Measured on this data
# (report_gc_diagnostics.py): N is 234 mean for perception and 202 for
# production, so the jitter is 4.6-5.0% against Ding et al.'s 2.4% at 888
# trials — about 2x, entering as ~0.2% of added variance. Not enough to reject
# step 3 on noise grounds. Arm B now turns on one question only: does the
# high-beta production effect survive removing the power modulation?
#
# WHY ARM B MIGHT ALSO CHANGE THE ANSWER. The surviving effect in this project
# is high-beta suppression during production, and movement-related beta
# desynchronisation IS a power change. Ensemble-SD normalisation erases exactly
# that modulation. So this is a robustness check on the headline result, not a
# methods formality: if high beta survives arm B it is coupling; if it does not,
# it was riding on the power change.
#
# WHY TWO ORDERS. Plain GC is order-fragile in this data (the dominant band has
# been seen to invert with order), so a single-order verdict would not
# generalise. 6 and 10 bracket the range the sweep uses.
#
# WHY BOTH TASKS. perception/percDiff and overtProd/prodDiff differ in epoch
# length and in how strong the evoked response is, which is precisely what
# these two arms interact with. A decision taken on production alone would not
# transfer.
#
# Arms write to DIFFERENT directories and nothing needs deleting first:
# gc_tag appends _demean / _zscore, and (as of this change) _notrialdemean when
# --no-demean-trials is set. That last one was missing, so arm C would have
# landed on arm A's path and silently overwritten its own control.
#
#   conda activate mne
#   bash run_gc_normalize_ab.sh
#   DRY_RUN=1 bash run_gc_normalize_ab.sh        # print the commands only
#   PARALLEL=6 bash run_gc_normalize_ab.sh
set -u

TASKS_STIMS="${TASKS_STIMS:-perception:percDiff overtProd:prodDiff}"
ORDERS="${ORDERS:-6 10}"
WIN_MS="${WIN_MS:-60}"
ROIS="${ROIS:-awfa-lh ifc-lh}"        # the pathway carrying the headline effect
NORMALIZE_ARMS="${NORMALIZE_ARMS:-none demean zscore}"
RUN_ARM_C="${RUN_ARM_C:-1}"           # --no-demean-trials probe
RUN_ARM_D="${RUN_ARM_D:-1}"           # --normalize-per-class probe

METHOD="${METHOD:-LCMV}"
ATLAS="${ATLAS:-custom}"
FEAT="${FEAT:-vertex_selectkbest}"
LEAK="${LEAK:---leakage-correction}"
TARGET_FS="${TARGET_FS:-200}"
GC_MODE="${GC_MODE:-pairwise}"        # 2 ROIs -> bivariate

PARALLEL="${PARALLEL:-4}"
INNER_JOBS="${INNER_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${LOG_DIR:-logs/gc_normalize_ab}"
mkdir -p "$LOG_DIR"

n_total=0
for TS in $TASKS_STIMS; do for A in $NORMALIZE_ARMS; do for O in $ORDERS; do
    n_total=$(( n_total + 1 ))
done; done; done
for ARM in C D; do
    eval "on=\$RUN_ARM_$ARM"
    [ "$on" = "1" ] || continue
    for TS in $TASKS_STIMS; do for O in $ORDERS; do
        n_total=$(( n_total + 1 ))
    done; done
done

echo "GC preprocessing A/B"
echo "  task:stim   : $TASKS_STIMS"
echo "  window      : ${WIN_MS} ms @ ${TARGET_FS} Hz"
echo "  orders      : $ORDERS"
echo "  ROIs        : $ROIS   (gc-mode $GC_MODE)"
echo "  arms        : $NORMALIZE_ARMS$([ "$RUN_ARM_C" = 1 ] && echo ' + no-demean-trials')$([ "$RUN_ARM_D" = 1 ] && echo ' + per-class')"
echo "  method      : $METHOD / $ATLAS / $FEAT $LEAK"
echo "  $n_total configs, 20 subjects each, $PARALLEL at a time"
echo
echo "Run report_gc_diagnostics.py first if you have not — arm B's cost is set"
echo "by N, and it reads N from the vertex caches on the shared drive."
echo

CMD_FILE=$(mktemp)
trap 'rm -f "$CMD_FILE"' EXIT
t0=$(date +%s)
n=0

queue () {   # queue <tag> <extra-args...>
    local tag="$1"; shift
    local log="$LOG_DIR/${tag}.log"
    n=$(( n + 1 ))
    echo "[$n/$n_total] $tag"
    local CMD="python run_granger.py --task $TASK --stim-class $STIM \
--method $METHOD --atlas $ATLAS --feature-mode $FEAT $LEAK \
--gc-mode $GC_MODE --win-ms $WIN_MS --order $O --target-fs $TARGET_FS \
--roi-subset $ROIS --n-jobs $INNER_JOBS $*"
    if [ "$DRY_RUN" = "1" ]; then
        echo "    $CMD"
        return
    fi
    printf '%s > %s 2>&1 || echo "FAILED %s" >&2\n' "$CMD" "$log" "$tag" \
        >> "$CMD_FILE"
}

for TS in $TASKS_STIMS; do
    TASK="${TS%%:*}"; STIM="${TS##*:}"
    for ARM in $NORMALIZE_ARMS; do
        for O in $ORDERS; do
            queue "${TASK}_${STIM}_${ARM}_win${WIN_MS}ms_order${O}" \
                  "--normalize $ARM"
        done
    done
done

if [ "$RUN_ARM_C" = "1" ]; then
    for TS in $TASKS_STIMS; do
        TASK="${TS%%:*}"; STIM="${TS##*:}"
        for O in $ORDERS; do
            queue "${TASK}_${STIM}_demean_noTrialDemean_win${WIN_MS}ms_order${O}" \
                  "--normalize demean --no-demean-trials"
        done
    done
fi

if [ "$RUN_ARM_D" = "1" ]; then
    for TS in $TASKS_STIMS; do
        TASK="${TS%%:*}"; STIM="${TS##*:}"
        for O in $ORDERS; do
            queue "${TASK}_${STIM}_demean_perClass_win${WIN_MS}ms_order${O}" \
                  "--normalize demean --normalize-per-class"
        done
    done
fi

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

xargs -P "$PARALLEL" -I{} bash -c '{}' < "$CMD_FILE"

echo
echo "$n_total configs attempted in $(( ($(date +%s) - t0) / 60 )) min"
echo
echo "Each log ends with a MODEL VALIDATION block: N, ensemble-SD jitter,"
echo "spectral radius, the non-minimum-phase count, and consistency."
echo "Compare the arms on:"
echo "  1. sign and rank of awfa->ifc          (does the effect survive?)"
echo "  2. the band carrying it                (does high beta hold under zscore?)"
echo "  3. rho>=1 count                        (arm 0 vs A is Ding Fig. 7:"
echo "                                          does ERP removal buy stability?)"
echo "  5. arm D vs arm A                      (how much of the effect was the"
echo "                                          between-class evoked difference?)"
echo "  4. consistency median                  (relative only — see"
echo "                                          report_gc_diagnostics.py)"
echo
grep -h 'NON-MINIMUM-PHASE\|ensemble-SD\|consistency:' "$LOG_DIR"/*.log 2>/dev/null \
    | head -40
