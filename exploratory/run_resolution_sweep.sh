#!/usr/bin/env bash
# Inverse-model optimization: resolution baseline + settings sweep + report,
# for both tasks.  Light (no decoding, no GC); runs alongside the decode
# queue.  See exploratory/INVERSE_OPTIMIZATION_PLAN.md.
#
#   nohup bash exploratory/run_resolution_sweep.sh > logs/resolution_sweep_driver.log 2>&1 &
#
# Knobs (env):
#   TASKS="overtProd perception"   STIM is derived: overtProd->prodDiff, perception->percDiff
#   N_JOBS=8                       subject-parallel workers (each ~2 GB)
#   GRID=star|full                 resolution_sweep.py --grid
#   ATLASES="custom sensorEquiv"
#   STEPS="baseline sweep report"
#   BRAIN_MAPS=1                   render CTF brain maps (PyVista; set 0 if headless rendering fails)
#   DRY_RUN=1                      print commands only
# Every step resumes (per-subject caches are skipped when present).
# PROGRESS: this log only gets a line when a step finishes; the live output of
# the running step is in logs/resolution_sweep/<step>_<task>.log (one line per
# subject as it completes), and finished subjects appear as files in
# derivatives/source_estimation/RESOLUTION_ANALYSIS/{task}_{stim}/[sweep/]per_subject/.
set -u
cd "$(dirname "$0")/.."

TASKS="${TASKS:-overtProd perception}"
N_JOBS="${N_JOBS:-8}"
GRID="${GRID:-star}"
ATLASES="${ATLASES:-custom sensorEquiv}"
STEPS="${STEPS:-baseline sweep report}"
BRAIN_MAPS="${BRAIN_MAPS:-1}"
DRY_RUN="${DRY_RUN:-0}"
LOG_ROOT="logs/resolution_sweep"
mkdir -p "$LOG_ROOT"

# activate the mne env (same probing as run_translation_arm.sh)
if ! command -v python >/dev/null 2>&1 || [ "${CONDA_DEFAULT_ENV:-}" != mne ]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        for _p in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
            # shellcheck disable=SC1091
            [ -f "$_p/etc/profile.d/conda.sh" ] && { source "$_p/etc/profile.d/conda.sh"; break; }
        done
    fi
    conda activate mne 2>/dev/null || { echo "ERROR: cannot activate conda env 'mne'" >&2; exit 1; }
fi
echo "python: $(command -v python)  ($(python -c 'import mne; print("mne", mne.__version__)' 2>/dev/null || echo 'mne import FAILED'))"

has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
stim_of() { [ "$1" = overtProd ] && echo prodDiff || echo percDiff; }
run() {
    local log="$1"; shift
    echo "  $*"
    [ "$DRY_RUN" = 1 ] && return 0
    local t0=$SECONDS
    echo "    running; live output: $log"
    if "$@" > "$log" 2>&1; then
        printf '    ok   %d min  (%s)\n' $(( (SECONDS - t0) / 60 )) "$log"
    else
        printf '    FAIL %d min  (%s)\n' $(( (SECONDS - t0) / 60 )) "$log"
    fi
}
BM=""; [ "$BRAIN_MAPS" = 1 ] && BM="--brain-maps"

echo "resolution sweep: tasks [$TASKS] / grid $GRID / atlases [$ATLASES] / $N_JOBS workers"
echo "  steps: $STEPS"
echo

if has baseline; then
echo "== baseline resolution analysis (production inverses, all four methods)"
for T in $TASKS; do S=$(stim_of "$T")
    run "$LOG_ROOT/baseline_${T}.log" \
        python exploratory/resolution_analysis.py --task "$T" --stim-class "$S" \
            --methods dSPM sLORETA eLORETA LCMV --atlases $ATLASES --n-jobs "$N_JOBS" $BM
done
echo
fi

if has sweep; then
echo "== settings sweep (mn / lcmv / deflect, grid=$GRID)"
for T in $TASKS; do S=$(stim_of "$T")
    run "$LOG_ROOT/sweep_${T}.log" \
        python exploratory/resolution_sweep.py --task "$T" --stim-class "$S" \
            --atlases $ATLASES --grid "$GRID" --n-jobs "$N_JOBS"
done
echo
fi

if has report; then
echo "== report"
for T in $TASKS; do S=$(stim_of "$T")
    run "$LOG_ROOT/report_${T}.log" \
        python exploratory/resolution_sweep_report.py --task "$T" --stim-class "$S" $BM
done
echo
fi
echo "done.  Read: derivatives/source_estimation/RESOLUTION_ANALYSIS/{task}_{stim}/summary.txt"
echo "            derivatives/source_estimation/RESOLUTION_ANALYSIS/{task}_{stim}/sweep/report/summary.txt"
