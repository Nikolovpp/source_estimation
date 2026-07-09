# methods_paper/

Self-contained scripts for a **potential separate methods paper** on
source-space Granger-causality preprocessing choices. **Not part of the
current manuscript** — nothing in the core pipeline imports these, so this
whole directory can be deleted without affecting `run_source_localize.py` /
`run_decode.py` / `run_granger.py` / `granger_stats.py`.

These scripts *import from* the core modules one level up (`config`,
`run_granger`, `data_loader`, `forward_model`, `inverse_pipelines`,
`granger`, `decoding_io`) via
`sys.path.insert(0, dirname(dirname(__file__)))`. That dependency is
one-way: core → nothing here. Run them from anywhere; paths derive from
`config.env`.

## Scripts

### `resample_edge_artifact.py`
Robust before/after of the per-epoch **resample edge artifact** (naïve scipy
`resample_poly` zero-pad = pre-fix #1) vs **fix #1** (`resample_channels`,
edge-pad+crop). Same real LCMV source TCs through the exact production GC
code (`run_granger.compute_subject_gc`) twice — only the resample differs —
across all subjects × 4 bands × all directed edges.

- Subject-parallel (`multiprocessing.Pool`, `--subject-jobs`); GC sequential
  inside each worker (never nested). `--subject-jobs 1` runs subjects
  sequentially with GC pair-parallelism (`--gc-jobs`).
- Resumable: `per_subject/{subj}.npz` (skip done) + `source_tc_cache/{subj}.npz`
  (LCMV once). `--plot-only` rebuilds figures from caches.
- Output: `derivatives/.../methods_paper_analyses/resample_edge_artifact/{atlas}/{task}_{stim}/`
  (5 figures + caches).

```bash
conda activate mne
python resample_edge_artifact.py --atlas custom  --task overtProd  --stim-class prodDiff --subject-jobs 20
python resample_edge_artifact.py --atlas custom  --task perception --stim-class percDiff --subject-jobs 20
python resample_edge_artifact.py --atlas HCPMMP1 --task overtProd  --stim-class prodDiff --subject-jobs 16
python resample_edge_artifact.py --atlas custom  --task overtProd  --stim-class prodDiff --plot-only
```

### `compare_downsample_order.py`
LCMV **downsampling-order** A/B: source-estimate on fs_500 data
(downsample-before) vs source-estimate on fs_2048 then resample the ROI
courses to 500 (source-then-downsample). Quantifies how the beamformer's
data-adaptive filter makes the two orders diverge (bimodal at n=8).

- Output: `derivatives/.../GC_downsample_order_check/` (its `OUT_ROOT`; left
  at the original location so the existing n=8 results still match).

```bash
conda activate mne
python compare_downsample_order.py --task overtProd --stim-class prodDiff --subjects EEGPROD4001 EEGPROD4003
```

See `docs/2026-07-07_gc_edges_downsampling_baseline.md` and
`derivatives/.../methods_paper_analyses/README.md` for the full framing.
