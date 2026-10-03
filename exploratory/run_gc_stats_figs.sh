#!/usr/bin/env bash
# run_gc_stats_figs.sh
# Stats + review figures for run_gc_final.sh results. Takes the SAME knobs
# with the same defaults (ORDER, WINS, TARGET_FS, NORMALIZE, NPCS, NPCS_ROI,
# TASKS, STIMS), so the env line that produced a run also analyses it:
#
#   bash run_gc_final.sh                       # order 15, 40/60/80 ms, fs 500
#   bash exploratory/run_gc_stats_figs.sh      # stats + figures for all three
#
# One pass per window in WINS; each pass covers both tasks, both contrasts
# and the six pairs = 24 result dirs under
#     order{ORDER}_win{W}ms_fs{TARGET_FS}[_{NORMALIZE}][_pc{NPCS}][_{roi}{K}]
# (the tag comes from run_granger.gc_tag, so it always matches the runner).
# Result dirs that do not exist yet are reported as MISSING and skipped, so
# the script can be run while some windows are still computing.
#
# Four steps per window, each skippable with STEPS="...":
#   extent   leading-edge inflation figure (exploratory/plot_gc_edge_extent.py)
#            -> _figures_final_review/edge_extent_{tag}.{png,csv}
#            LOOK AT THIS FIRST for a new window length: its settle_epoch_ms
#            column says where the edge inflation ends, and baseline_ratio
#            is the GC level inside the _bl baselines below relative to the
#            plateau (1 = clean). To move a baseline, set BL_OVERTPROD /
#            BL_PERCEPTION and rerun with STEPS="stats review summary".
#   stats    granger_stats.py on every result dir, twice:
#              group_stats     default leading baseline: epoch start +
#                              [EDGE_GUARD, 100] ms (70 ms long at guard 30)
#              group_stats_bl  explicit baseline (window-start times, s):
#                overtProd   BL_OVERTPROD, default "auto" = BL_DUR_MS (100) ms
#                            starting right after the edge guard, i.e. epoch
#                            start + [EDGE_GUARD, EDGE_GUARD + BL_DUR_MS] =
#                            -1.47..-1.37 s at guard 30. The epoch start is
#                            read from the results (window_ms[0]).
#                perception  BL_PERCEPTION, default -0.08..0 s
#            Reruns overwrite. SKIP_EXISTING=1 keeps dirs whose stats CSV is
#            already there (resume after an interruption).
#   review   per-pair review figure (exploratory/plot_gc_edge_review.py),
#            default-baseline and _bl variant -> 48 png per window
#   summary  one row per task x contrast x pair x edge x band with the
#            cluster / TFCE minima and the number of significant windows
#            -> _figures_final_review/permutation_summary_{tag}{,_bl}.csv,
#            and a one-line count per window and baseline at the end.
#
# NOT run: plot_gc_fixpc_compare.py. It needs a FIXPC1 arm at the same
# order/window/fs in the per-pair (rois_*) layout.
#
#   conda activate mne          # the script activates it itself
#   bash exploratory/run_gc_stats_figs.sh
#   DRY_RUN=1 bash exploratory/run_gc_stats_figs.sh           # print only
#   WINS=60 bash exploratory/run_gc_stats_figs.sh             # one window
#   ORDER=10 WINS=40 bash exploratory/run_gc_stats_figs.sh    # the MO10 run
#   STEPS="stats review summary" bash exploratory/run_gc_stats_figs.sh
#   BL_PERCEPTION="-0.12 -0.04" bash exploratory/run_gc_stats_figs.sh
#   BL_OVERTPROD="-1.3 -1.1" bash exploratory/run_gc_stats_figs.sh   # explicit
#   EDGE_GUARD=50 BL_DUR_MS=100 bash exploratory/run_gc_stats_figs.sh
#   TASKS=overtProd STIMS=prodDiff bash exploratory/run_gc_stats_figs.sh
set -u
cd "$(dirname "$0")/.."

ORDER="${ORDER:-15}"
WINS="${WINS:-40 60 80}"
TARGET_FS="${TARGET_FS:-500}"
NORMALIZE="${NORMALIZE:-none}"
NPCS="${NPCS:-2}"
NPCS_ROI="${NPCS_ROI-pmc-lh=3}"
METHOD="${METHOD:-LCMV}"
ATLAS="${ATLAS:-custom}"
FEAT="${FEAT:-vertex_selectkbest}"
LEAK=leakage_corrected                    # plot_gc_edge_review.py reads only this

STEPS="${STEPS:-extent stats review summary}"
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
PAIRS="${PAIRS:-awfa-lh,ifc-lh awfa-lh,pmc-lh awfa-lh,tpc-lh ifc-lh,pmc-lh ifc-lh,tpc-lh pmc-lh,tpc-lh}"
EDGE_GUARD="${EDGE_GUARD:-30}"            # ms, leading windows kept out of both baselines
BL_OVERTPROD="${BL_OVERTPROD:-auto}"      # "auto" or "start end" (s, window start)
BL_DUR_MS="${BL_DUR_MS:-100}"             # ms, length of the auto overtProd baseline
BL_PERCEPTION="${BL_PERCEPTION:--0.08 0}" # s, window start
PARALLEL="${PARALLEL:-6}"                 # concurrent granger_stats / figure jobs
STATS_JOBS="${STATS_JOBS:-4}"             # --n-jobs inside each granger_stats
SKIP_EXISTING="${SKIP_EXISTING:-0}"       # 1 = keep stats dirs that have a CSV
DRY_RUN="${DRY_RUN:-0}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _p in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_p/etc/profile.d/conda.sh" ] && { source "$_p/etc/profile.d/conda.sh"; break; }
    done
fi
conda activate mne 2>/dev/null || { echo "ERROR: cannot activate 'mne'" >&2; exit 1; }

PC_FLAGS="--n-pcs $NPCS"
[ -n "$NPCS_ROI" ] && PC_FLAGS="$PC_FLAGS --n-pcs-roi $NPCS_ROI"

# results root, then per window: the directory tag (straight from the runner)
# and the overtProd epoch start in s (first window start of any result there;
# -1.5, the overtProd epoch start, while nothing has been written yet)
INFO=$(python - "$ORDER" "$TARGET_FS" "$NORMALIZE" "$NPCS" "$NPCS_ROI" \
              "$METHOD/$ATLAS/$FEAT/$LEAK" $WINS <<'EOF'
import sys, glob
import numpy as np
from run_granger import GC_OUTPUT_ROOT, gc_tag
order, fs, norm, npcs, over, mid = sys.argv[1:7]
print(GC_OUTPUT_ROOT)
for w in sys.argv[7:]:
    tag = gc_tag(int(order), float(w), float(fs), norm, n_pcs=int(npcs),
                 n_pcs_roi=over.split() or None)
    f = sorted(glob.glob(f'{GC_OUTPUT_ROOT}/overtProd/{mid}/{tag}/rois_*/*/*.npz'))
    t0 = float(np.load(f[0], allow_pickle=True)['window_ms'][0]) / 1000.0 if f else -1.5
    print(tag, f'{t0:g}')
EOF
) || { echo "ERROR: cannot derive the results paths (run_granger import failed)" >&2; exit 1; }
GC_ROOT=$(echo "$INFO" | sed -n 1p)
mapfile -t TAGS < <(echo "$INFO" | sed 1d | cut -d' ' -f1)
mapfile -t T0S  < <(echo "$INFO" | sed 1d | cut -d' ' -f2)
BL_OVERTPROD_ARG="$BL_OVERTPROD"
FIG_DIR="$GC_ROOT/_figures_final_review"
[ "$DRY_RUN" = "1" ] || mkdir -p "$FIG_DIR"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

bl_for () { case "$1" in overtProd) echo "$BL_OVERTPROD" ;; perception) echo "$BL_PERCEPTION" ;; esac; }
run () {   # run <logfile> <cmd...>   (prints in DRY_RUN)
    local log="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then echo "  $*"; return; fi
    "$@" > "$log" 2>&1 && echo "  ok    $(basename "$log" .log)" \
                        || echo "  FAIL  $(basename "$log" .log)  (see $log)"
}
export -f run; export DRY_RUN

echo "GC stats + figures: order $ORDER, windows [$WINS] ms, fs $TARGET_FS, normalize=$NORMALIZE, ${NPCS} PC(s) per ROI${NPCS_ROI:+ (except $NPCS_ROI)}"
echo "  results: $GC_ROOT"
echo "  steps: $STEPS"
echo "  edge guard ${EDGE_GUARD} ms; _bl baselines: overtProd [$BL_OVERTPROD_ARG]$([ "$BL_OVERTPROD_ARG" = auto ] && echo " = ${BL_DUR_MS} ms right after the edge guard"), perception [$BL_PERCEPTION] s"

wi=0
for WIN_MS in $WINS; do
TAG="${TAGS[$wi]}"; T0="${T0S[$wi]}"; wi=$(( wi + 1 ))
if [ "$BL_OVERTPROD_ARG" = "auto" ]; then
    BL_OVERTPROD=$(awk -v t="$T0" -v g="$EDGE_GUARD" -v d="$BL_DUR_MS" \
        'BEGIN { printf "%g %g", t + g / 1000, t + (g + d) / 1000 }')
fi
LOG_DIR="logs/gc_stats_figs_${TAG}"
[ "$DRY_RUN" = "1" ] || mkdir -p "$LOG_DIR"
pair_dir () { echo "$GC_ROOT/$1/$METHOD/$ATLAS/$FEAT/$LEAK/$TAG/rois_${2/,/-}/$3"; }

n_have=0; n_want=0
for T in $TASKS; do for PR in $PAIRS; do for S in $STIMS; do
    n_want=$(( n_want + 1 ))
    [ -d "$(pair_dir "$T" "$PR" "$S")" ] && n_have=$(( n_have + 1 ))
done; done; done
echo
echo "== $TAG   ($n_have of $n_want result dirs present)"
echo "   overtProd _bl baseline [$BL_OVERTPROD] s (epoch start $T0 s)"
if [ "$n_have" -eq 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "   nothing to do — run_gc_final.sh has not written this window yet"
    continue
fi

# ── extent ──────────────────────────────────────────────────────────────
case " $STEPS " in *" extent "*)
echo "-- edge extent"
run "$LOG_DIR/edge_extent.log" python exploratory/plot_gc_edge_extent.py \
    --method $METHOD --atlas $ATLAS --feature-mode $FEAT \
    --order $ORDER --win-ms $WIN_MS --fs $TARGET_FS --normalize $NORMALIZE \
    $PC_FLAGS --edge-guard $EDGE_GUARD --tasks $TASKS \
    --baseline-overtprod $BL_OVERTPROD --baseline-perception $BL_PERCEPTION
[ "$DRY_RUN" = "1" ] || { echo; grep -A40 "task " "$LOG_DIR/edge_extent.log" | head -30; }
;; esac

# ── stats ───────────────────────────────────────────────────────────────
case " $STEPS " in *" stats "*)
echo "-- stats (default baseline -> group_stats, explicit -> group_stats_bl), $PARALLEL at a time"
JOBS="$TMP/stats_$TAG"; : > "$JOBS"
for T in $TASKS; do for PR in $PAIRS; do for S in $STIMS; do
    D=$(pair_dir "$T" "$PR" "$S")
    [ -d "$D" ] || { echo "  MISSING $D"; continue; }
    base="${T}_${S}_${PR/,/-}"
    csv=gc_task_vs_baseline_stats_ttest.csv
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$D/group_stats/$csv" ]; then
        echo "  kept  stats_${base}"
    else
        printf '%s\0' "run '$LOG_DIR/stats_${base}.log' python granger_stats.py --gc-dir '$D' --task $T \
--edge-guard $EDGE_GUARD --n-jobs $STATS_JOBS" >> "$JOBS"
    fi
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$D/group_stats_bl/$csv" ]; then
        echo "  kept  stats_bl_${base}"
    else
        printf '%s\0' "run '$LOG_DIR/stats_bl_${base}.log' python granger_stats.py --gc-dir '$D' --task $T \
--baseline-start $(bl_for $T | cut -d' ' -f1) --baseline-end $(bl_for $T | cut -d' ' -f2) \
--out-dir '$D/group_stats_bl' --n-jobs $STATS_JOBS" >> "$JOBS"
    fi
done; done; done
xargs -0 -P "$PARALLEL" -n1 bash -c 'eval "$0"' < "$JOBS"
;; esac

# ── review figures ──────────────────────────────────────────────────────
case " $STEPS " in *" review "*)
echo "-- review figures, $PARALLEL at a time"
JOBS="$TMP/review_$TAG"; : > "$JOBS"
for T in $TASKS; do for PR in $PAIRS; do for S in $STIMS; do
    [ -d "$(pair_dir "$T" "$PR" "$S")" ] || continue
    base="${T}_${S}_${PR/,/-}"
    COMMON="python exploratory/plot_gc_edge_review.py --task $T --stim-class $S --roi-subset ${PR/,/ } \
--method $METHOD --atlas $ATLAS --feature-mode $FEAT \
--order $ORDER --win-ms $WIN_MS --target-fs $TARGET_FS --normalize $NORMALIZE $PC_FLAGS"
    printf '%s\0' "run '$LOG_DIR/review_${base}.log' $COMMON --edge-guard $EDGE_GUARD" >> "$JOBS"
    printf '%s\0' "run '$LOG_DIR/review_bl_${base}.log' $COMMON --stats-subdir group_stats_bl \
--baseline-start $(bl_for $T | cut -d' ' -f1) --baseline-end $(bl_for $T | cut -d' ' -f2)" >> "$JOBS"
done; done; done
xargs -0 -P "$PARALLEL" -n1 bash -c 'eval "$0"' < "$JOBS"
;; esac

# ── summary tables ──────────────────────────────────────────────────────
case " $STEPS " in *" summary "*)
echo "-- permutation summary CSVs"
if [ "$DRY_RUN" = "1" ]; then echo "  (python: aggregate */group_stats{,_bl}/gc_task_vs_baseline_stats_ttest.csv)"; else
python - "$GC_ROOT" "$TAG" "$FIG_DIR" "$METHOD/$ATLAS/$FEAT/$LEAK" "$TMP/recap" <<'EOF'
import sys, glob, os, pandas as pd
root, tag, fig_dir, mid, recap = sys.argv[1:]
for sub, sfx in (('group_stats', ''), ('group_stats_bl', '_bl')):
    rows = []
    for f in sorted(glob.glob(f'{root}/*/{mid}/{tag}/rois_*/*Diff/{sub}/gc_task_vs_baseline_stats_ttest.csv')):
        parts = f.split(os.sep)
        task, pair, stim = parts[-10], parts[-4].replace('rois_', ''), parts[-3]
        df = pd.read_csv(f)
        g = df.groupby(['measure', 'src', 'tgt', 'band'])
        out = g.agg(n_sig_cluster=('sig_cluster', 'sum'), n_sig_tfce=('sig_tfce', 'sum'),
                    n_sig_pointwise=('sig', 'sum'), p_cluster_min=('p_cluster_min', 'first'),
                    p_tfce_min=('p_tfce_min', 'first'), n_windows=('window_ms', 'size')).reset_index()
        out.insert(0, 'pair', pair); out.insert(0, 'stim', stim); out.insert(0, 'task', task)
        rows.append(out)
    if not rows:
        print(f'  no {sub} CSVs found'); continue
    S = pd.concat(rows, ignore_index=True)
    p = os.path.join(fig_dir, f'permutation_summary_{tag}{sfx}.csv'); S.to_csv(p, index=False)
    n = len(S); k = int((S.n_sig_cluster > 0).sum()); t = int((S.n_sig_tfce > 0).sum())
    line = f'{sub}: {n} edge x band cells, {k} with a significant cluster ({0.05*n:.0f} expected by chance), {t} with TFCE'
    print(f'  {line} -> {p}')
    with open(recap, 'a') as fh:
        fh.write(f'  {tag}  {line}\n')
    hit = S[S.n_sig_cluster > 0]
    if len(hit): print(hit[['task','stim','measure','src','tgt','band','n_sig_cluster','p_cluster_min']].to_string(index=False))
EOF
fi
;; esac
done

echo
if [ -s "$TMP/recap" ]; then
    echo "cluster-significant cells per window and baseline (no cross-cell correction):"
    cat "$TMP/recap"
    echo
fi
echo "figures: $FIG_DIR/   logs: logs/gc_stats_figs_order${ORDER}_win*ms_fs${TARGET_FS}*/"
