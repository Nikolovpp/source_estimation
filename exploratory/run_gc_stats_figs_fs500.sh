#!/usr/bin/env bash
# run_gc_stats_figs_fs500.sh
# Stats + review figures for the run_gc_final.sh arm
#     order 10 / 40 ms / fs 500 / normalize none / 2 PCs per ROI, 3 for pmc
# (results dir tag order10_win40ms_fs500_pc2_pmc-lh3), both tasks, both
# contrasts, six pairs = 24 result dirs.
#
# Four steps, each skippable with STEPS="...":
#   extent   leading-edge inflation figure (exploratory/plot_gc_edge_extent.py)
#            -> _figures_final_review/edge_extent_order10_win40ms_fs500_pc2_pmc-lh3.{png,csv}
#            LOOK AT THIS FIRST: its settle_epoch_ms column says where the
#            edge inflation ends for 40 ms windows; the corrected baselines
#            below (BL_OVERTPROD / BL_PERCEPTION) were carried over from the
#            80 ms arm and should be checked against it.
#   stats    granger_stats.py on every result dir, twice:
#              group_stats     default leading baseline, --edge-guard 30
#              group_stats_bl  corrected baseline (window-start times, s):
#                              overtProd  $BL_OVERTPROD   perception $BL_PERCEPTION
#   review   per-pair review figure (exploratory/plot_gc_edge_review.py),
#            default-baseline and _bl variant -> 48 png
#   summary  one row per task x contrast x pair x edge x band with the
#            cluster / TFCE minima and the number of significant windows
#            -> _figures_final_review/permutation_summary_order10_win40ms_fs500_pc2_pmc-lh3{,_bl}.csv
#
# NOT run: plot_gc_fixpc_compare.py. It needs a FIXPC1 arm at the same
# order/window/fs in the per-pair (rois_*) layout; the only PC1 results at
# 40 ms / 500 Hz are the 2026-07-17 all_rois runs, which the script does
# not read.
#
#   conda activate mne
#   bash exploratory/run_gc_stats_figs_fs500.sh
#   DRY_RUN=1 bash exploratory/run_gc_stats_figs_fs500.sh      # print only
#   STEPS="stats review summary" bash exploratory/run_gc_stats_figs_fs500.sh
#   BL_PERCEPTION="-0.12 -0.04" bash exploratory/run_gc_stats_figs_fs500.sh
set -u
cd "$(dirname "$0")/.."

ORDER=10; WIN_MS=40; FS=500; NORMALIZE=none
NPCS=2; NPCS_ROI="pmc-lh=3"
TAG="order${ORDER}_win${WIN_MS}ms_fs${FS}_pc${NPCS}_pmc-lh3"
METHOD=LCMV; ATLAS=custom; FEAT=vertex_selectkbest; LEAK=leakage_corrected

STEPS="${STEPS:-extent stats review summary}"
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
PAIRS="${PAIRS:-awfa-lh,ifc-lh awfa-lh,pmc-lh awfa-lh,tpc-lh ifc-lh,pmc-lh ifc-lh,tpc-lh pmc-lh,tpc-lh}"
EDGE_GUARD="${EDGE_GUARD:-30}"           # ms, default-baseline stats
BL_OVERTPROD="${BL_OVERTPROD:--1.3 -1.1}" # s, window start; same as the 80 ms arm
BL_PERCEPTION="${BL_PERCEPTION:--0.08 0}" # s, window start; same as the 80 ms arm
PARALLEL="${PARALLEL:-6}"                 # concurrent granger_stats / figure jobs
STATS_JOBS="${STATS_JOBS:-4}"             # --n-jobs inside each granger_stats
DRY_RUN="${DRY_RUN:-0}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate mne 2>/dev/null || { echo "ERROR: cannot activate 'mne'" >&2; exit 1; }

GC_ROOT=$(python -c "from run_granger import GC_OUTPUT_ROOT; print(GC_OUTPUT_ROOT)")
FIG_DIR="$GC_ROOT/_figures_final_review"
LOG_DIR="logs/gc_stats_figs_${TAG}"
mkdir -p "$LOG_DIR" "$FIG_DIR"

bl_for () { case "$1" in overtProd) echo "$BL_OVERTPROD" ;; perception) echo "$BL_PERCEPTION" ;; esac; }
run () {   # run <logfile> <cmd...>   (prints in DRY_RUN)
    local log="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then echo "  $*"; return; fi
    "$@" > "$log" 2>&1 && echo "  ok    $(basename "$log" .log)" \
                        || echo "  FAIL  $(basename "$log" .log)  (see $log)"
}
export -f run; export DRY_RUN

echo "== $TAG   steps: $STEPS"
echo "   edge guard ${EDGE_GUARD} ms; corrected baselines overtProd [$BL_OVERTPROD] s, perception [$BL_PERCEPTION] s"

# ── extent ──────────────────────────────────────────────────────────────
case " $STEPS " in *" extent "*)
echo "-- edge extent"
run "$LOG_DIR/edge_extent.log" python exploratory/plot_gc_edge_extent.py \
    --order $ORDER --win-ms $WIN_MS --fs $FS --normalize $NORMALIZE \
    --n-pcs $NPCS --n-pcs-roi $NPCS_ROI --edge-guard $EDGE_GUARD \
    --baseline-overtprod $BL_OVERTPROD --baseline-perception $BL_PERCEPTION
[ "$DRY_RUN" = "1" ] || { echo; grep -A40 "task " "$LOG_DIR/edge_extent.log" | head -30; }
;; esac

# ── stats ───────────────────────────────────────────────────────────────
case " $STEPS " in *" stats "*)
echo "-- stats (default baseline -> group_stats, corrected -> group_stats_bl), $PARALLEL at a time"
JOBS=$(mktemp); trap 'rm -f "$JOBS"' EXIT
for T in $TASKS; do for PR in $PAIRS; do for S in $STIMS; do
    D="$GC_ROOT/$T/$METHOD/$ATLAS/$FEAT/$LEAK/$TAG/rois_${PR/,/-}/$S"
    [ -d "$D" ] || { echo "  MISSING $D"; continue; }
    base="${T}_${S}_${PR/,/-}"
    printf '%s\0' "run '$LOG_DIR/stats_${base}.log' python granger_stats.py --gc-dir '$D' --task $T \
--edge-guard $EDGE_GUARD --n-jobs $STATS_JOBS" >> "$JOBS"
    printf '%s\0' "run '$LOG_DIR/stats_bl_${base}.log' python granger_stats.py --gc-dir '$D' --task $T \
--baseline-start $(bl_for $T | cut -d' ' -f1) --baseline-end $(bl_for $T | cut -d' ' -f2) \
--out-dir '$D/group_stats_bl' --n-jobs $STATS_JOBS" >> "$JOBS"
done; done; done
xargs -0 -P "$PARALLEL" -n1 bash -c 'eval "$0"' < "$JOBS"
;; esac

# ── review figures ──────────────────────────────────────────────────────
case " $STEPS " in *" review "*)
echo "-- review figures, $PARALLEL at a time"
JOBS=$(mktemp); trap 'rm -f "$JOBS"' EXIT
for T in $TASKS; do for PR in $PAIRS; do for S in $STIMS; do
    base="${T}_${S}_${PR/,/-}"
    COMMON="python exploratory/plot_gc_edge_review.py --task $T --stim-class $S --roi-subset ${PR/,/ } \
--order $ORDER --win-ms $WIN_MS --target-fs $FS --normalize $NORMALIZE --n-pcs $NPCS --n-pcs-roi $NPCS_ROI"
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
python - "$GC_ROOT" "$TAG" "$FIG_DIR" "$METHOD/$ATLAS/$FEAT/$LEAK" <<'EOF'
import sys, glob, os, pandas as pd
root, tag, fig_dir, mid = sys.argv[1:]
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
    print(f'  {sub}: {n} edge x band cells, {k} with a significant cluster ({0.05*n:.0f} expected by chance), {t} with TFCE -> {p}')
    hit = S[S.n_sig_cluster > 0]
    if len(hit): print(hit[['task','stim','measure','src','tgt','band','n_sig_cluster','p_cluster_min']].to_string(index=False))
EOF
fi
;; esac

echo; echo "figures: $FIG_DIR/*${TAG}*   logs: $LOG_DIR/"
