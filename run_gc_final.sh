#!/usr/bin/env bash
# run_gc_final.sh
# The FINAL (non-sweep) GC analysis: pairwise TRGC over the six speech-route
# ROI pairs of {awfa, ifc, pmc, tpc}, at one model order and sampling rate,
# swept over the sliding-window length:
#
#   ORDER      MVAR order. Default 15 = 30 ms of model memory at 500 Hz
#              (order 10 = 20 ms was the sensor-space BSMART config
#              SW20_MO10_fs500; order 15 is no longer that config).
#   WINS       window lengths in ms, one full run per window. Default
#              "40 60 80" = 20 / 30 / 40 samples at 500 Hz. The Morf
#              recursion needs > order+1 = 16 samples, so 40 ms is close to
#              the shortest window that fits order 15: it leaves 5 samples
#              per trial beyond the lags, 60 ms leaves 15, 80 ms leaves 25.
#              The script refuses a window that is too short for the order.
#   TARGET_FS  rate the virtual channels are resampled to. Default 500.
#
#   The former theta arm (MO25 / 200 ms / fs200) is no longer run here; see
#   exploratory/run_gc_theta_config.sh if it is needed again.
#
# Settings: LCMV / custom atlas / vertex_selectkbest / leakage correction,
# per-trial demean on, --trgc. TRGC is pairwise-only in run_granger.py, and
# the sweep showed conditioning on a third ROI changes nothing — so no
# triple-wise runs here.
#
# ROI AGGREGATION AND NORMALIZATION (the two knobs this script exposes):
#
#   NPCS       components kept per ROI (run_granger.py --n-pcs). Default 2:
#              each ROI is a block of its top-2 fixed-filter PCs and every
#              pair is scored with block (multivariate) Geweke GC / TRGC
#              (Pellegrini et al. 2023). 1 = the earlier single
#              virtual-channel bivariate analysis. The block size is capped
#              at each ROI's numerical rank (a collapsed LCMV ROI stays one
#              component; the log reports it as [fixpc]).
#   NPCS_ROI   per-ROI overrides of NPCS, space-separated ROI=K
#              (run_granger.py --n-pcs-roi). Default "pmc-lh=3". Set
#              NPCS_ROI="" for a uniform run.
#
#              WHY 2, AND 3 FOR PMC. Chosen from the variance each component
#              captures (fixpc_variance_explained_*.csv): two components
#              reach 95% of ROI variance in most subjects for awfa, ifc and
#              tpc; pmc needs three. FIXPC4 (the 2026-09-24 run) was the
#              Pellegrini default but cost more than it bought here: the
#              finite-sample GC floor scales with order x k_src x k_tgt, so
#              16x the FIXPC1 floor at k=4 against 4x at k=2 (6x for a pair
#              with pmc), and it produced fewer significant clusters.
#   NORMALIZE  across-trial ensemble normalization. Default none = raw
#              BSMART-faithful signals (per-trial DC removal only; no ERP
#              removal, no ensemble-SD equalization). 'demean' removes the
#              ERP; 'zscore' also divides by the ensemble SD per time point.
#
#   Output path:
#     .../order{MO}_win{SW}ms_fs{FS}[_{NORMALIZE}][_pc{NPCS}][_{roi}{K}]/...
#   e.g. order15_win40ms_fs500_pc2_pmc-lh3. The order, window, rate,
#   normalize name, PC suffix and overrides are all part of the path, so
#   runs with different settings never collide (the three windows land in
#   three sibling directories, and the earlier order10_win40ms_fs500 run is
#   untouched). The path names the POLICY: all six pairs of a run share it,
#   including pairs without an overridden ROI.
#
#   OVERWRITE=1 forces --overwrite (MAIN_OVERWRITE is still accepted).
#   gc_tag() does not encode TRGC, so if a plain-GC run already wrote npz at
#   the same path the runner would skip every subject there and never
#   compute dtrgc; set it for such cells. The default order-15 paths are
#   new, so the default is 0 (skip-if-exists, which lets an interrupted run
#   resume).
#
#   conda activate mne          # the script activates it itself
#   bash run_gc_final.sh
#   DRY_RUN=1 bash run_gc_final.sh          # print commands, run nothing
#   TASKS=overtProd bash run_gc_final.sh    # skip perception
#   NORMALIZE=demean bash run_gc_final.sh   # ERP removed, same PC policy
#   WINS=60 bash run_gc_final.sh            # one window only
#   ORDER=10 WINS=40 bash run_gc_final.sh   # the earlier MO10 / 40 ms run
#
# 3 windows x 6 pairs x 2 tasks x 2 contrasts = 72 configs, 20 subjects
# each, PARALLEL at a time (8 by default — the shared-drive I/O ceiling; 56
# crashed the workstation during the sweep). A block VAR costs more per
# window than the 2-channel one (order x (k_i+k_j)^2 coefficients instead of
# order x 4), and order 15 has 1.5x the coefficients of order 10, so expect
# each config to take longer than the MO10 run at the same window.
set -u

cd "$(dirname "$0")"

ORDER="${ORDER:-15}"
WINS="${WINS:-40 60 80}"
TARGET_FS="${TARGET_FS:-500}"
TASKS="${TASKS:-overtProd perception}"
STIMS="${STIMS:-prodDiff percDiff}"
METHOD="${METHOD:-LCMV}"
ATLAS="${ATLAS:-custom}"
FEAT="${FEAT:-vertex_selectkbest}"
LEAK="${LEAK:---leakage-correction}"
NORMALIZE="${NORMALIZE:-none}"
NPCS="${NPCS:-2}"
NPCS_ROI="${NPCS_ROI-pmc-lh=3}"
PARALLEL="${PARALLEL:-8}"
INNER_JOBS="${INNER_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-${MAIN_OVERWRITE:-0}}"

# every window must fit the order: the Morf recursion needs > order+1 samples
for WIN_MS in $WINS; do
    samp=$(( WIN_MS * TARGET_FS / 1000 ))
    if [ "$samp" -le $(( ORDER + 1 )) ]; then
        echo "ERROR: ${WIN_MS} ms @ ${TARGET_FS} Hz = ${samp} samples, order $ORDER needs > $(( ORDER + 1 ))" >&2
        exit 1
    fi
done

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
for _ in $WINS; do for _ in "${PAIR_ARR[@]}"; do for _ in $TASKS; do for _ in $STIMS; do
    n_total=$(( n_total + 1 ))
done; done; done; done

# Run label: normalize name plus the PC suffix (only for NPCS > 1, mirroring
# gc_tag), used for the log directory and the per-config tags.
# Overrides are appended as {roi}{K} in name order, again as gc_tag does.
PC_LABEL=""
{ [ "$NPCS" -gt 1 ] || [ -n "$NPCS_ROI" ]; } && PC_LABEL="pc${NPCS}"
for ov in $(echo $NPCS_ROI | tr ' ' '\n' | tr 'A-Z' 'a-z' | sort); do
    PC_LABEL="${PC_LABEL}_${ov/=/}"
done
RUN_LABEL="$NORMALIZE"
[ -n "$PC_LABEL" ] && RUN_LABEL="${NORMALIZE}_${PC_LABEL}"
PC_FLAGS="--n-pcs $NPCS"
[ -n "$NPCS_ROI" ] && PC_FLAGS="$PC_FLAGS --n-pcs-roi $NPCS_ROI"

echo "final GC analysis — pairwise TRGC, ${NPCS} PC(s) per ROI${NPCS_ROI:+ (except $NPCS_ROI)}, normalize=$NORMALIZE, $METHOD/$ATLAS/$FEAT $LEAK"
for WIN_MS in $WINS; do
    samp=$(( WIN_MS * TARGET_FS / 1000 ))
    echo "  order $ORDER, ${WIN_MS} ms @ ${TARGET_FS} Hz = ${samp} samples" \
         "(needs > $(( ORDER + 1 ))), memory $(( 1000 * ORDER / TARGET_FS )) ms"
done
echo "  $n_total configs, 20 subjects each, $PARALLEL at a time"
[ "$OVERWRITE" = "1" ] && echo "  running with --overwrite"
echo

CMD_FILE=$(mktemp); DONE_FILE=$(mktemp)
trap 'rm -f "$CMD_FILE" "$DONE_FILE"' EXIT
t0=$(date +%s); n=0

OVR=""
[ "$OVERWRITE" = "1" ] && OVR="--overwrite"
LOG_GLOB="logs/gc_final_order${ORDER}_win*ms_fs${TARGET_FS}_${RUN_LABEL}"

for WIN_MS in $WINS; do
LOG_DIR="logs/gc_final_order${ORDER}_win${WIN_MS}ms_fs${TARGET_FS}_${RUN_LABEL}"
[ "$DRY_RUN" = "1" ] || mkdir -p "$LOG_DIR"
for PP in "${PAIR_ARR[@]}"; do
PP=$(trim "$PP")
for T in $TASKS; do for S in $STIMS; do
    tag="${T}_${S}_$(label "$PP")_trgc_win${WIN_MS}ms_order${ORDER}_${PC_LABEL:-pc1}"
    log="$LOG_DIR/${tag}.log"
    n=$(( n + 1 ))
    echo "[$n/$n_total] $tag"
    CMD="python run_granger.py --task $T --stim-class $S --method $METHOD \
--atlas $ATLAS --feature-mode $FEAT $LEAK --gc-mode pairwise \
--win-ms $WIN_MS --order $ORDER --target-fs $TARGET_FS \
--normalize $NORMALIZE $PC_FLAGS --trgc --roi-subset $PP --n-jobs $INNER_JOBS $OVR"
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
echo "Per-subject progress:  tail -f $LOG_GLOB/*.log"
echo
xargs -0 -P "$PARALLEL" -n1 bash -c 'eval "$0"' < "$CMD_FILE"

echo
echo "$n_total configs attempted in $(( ($(date +%s) - t0) / 60 )) min"
echo
echo "Then stats + review figures for these runs (same knobs, so the paths match):"
echo "  ORDER=$ORDER WINS=\"$WINS\" TARGET_FS=$TARGET_FS NORMALIZE=$NORMALIZE NPCS=$NPCS NPCS_ROI=\"$NPCS_ROI\" \\"
echo "  TASKS=\"$TASKS\" STIMS=\"$STIMS\" bash exploratory/run_gc_stats_figs.sh"
grep -h 'NON-MINIMUM-PHASE\|consistency:\|\[fixpc\]' $LOG_GLOB/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -20
