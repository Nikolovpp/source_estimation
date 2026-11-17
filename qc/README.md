# qc/

Quality-control stages of the toolbox: checks that say how far a result can
be trusted, as opposed to the pipeline that produces it (repo root) or the
one-off analyses built on it (`exploratory/`).

Conventions are the same as `exploratory/`: every script inserts the repo
root into `sys.path` and the shell driver `cd`s to the repo root, so
everything runs from anywhere as `python qc/<script>.py`. Nothing at the repo
root imports from `qc/`; `qc/` never imports from `exploratory/`.

## Spatial resolution of the inverse

- `resolution_analysis.py` — ROI-level cross-talk (CTF) and point-spread
  (PSF) of the pipeline's own inverse operators (dSPM, sLORETA, eLORETA,
  LCMV) for any ROI set, with the sensor pseudo-channels on the same footing.
  Answers "can this montage separate these ROIs at all?". Per-ROI and
  per-pair CSVs, overlap matrices, optional brain maps. Light: no decoding,
  no GC.
- `resolution_sweep.py` — the same metrics across inverse settings (DeFleCT
  ROI filters, λ²/depth, noise covariance, LCMV conditioning, loose
  orientation), with `noise_gain` so configurations are comparable on the
  noise axis. Workstation compute.
- `resolution_sweep_report.py` — reads the sweep CSVs: factor curves, Pareto
  (noise vs cross-talk), rankings, recommendation. Runs locally.
- `resolution_subject_qc.py` — per-subject QC from the sweep CSVs: robust-z
  heatmap of the production metrics, noise gain per ROI, LCMV collapse
  detector, every subject's overlap matrix. The forward model is shared, so
  subject spread = that subject's noise / data covariance. Runs locally.
- `run_resolution_sweep.sh` — driver: baseline + sweep + report, both tasks.

## Regional specificity of decoding

- `decode_onset_latency.py` — decoding timing per ROI against a null ROI:
  jackknife onset/peak latencies, ROI-vs-control cluster test over time,
  ROI × time RM-ANOVA, fixed-window accuracy. Reads `run_decode.py` CSVs.
  Needs an atlas with a genuine control region (`sensorEquiv`,
  `sensorEquivSM`: `Visual`); the custom localizer spheres have none.
