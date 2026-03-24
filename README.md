# Source Estimation + SVM Decoding Pipeline

Projects 64-channel BIOSEMI EEG data into source space using the fsaverage template anatomy, then runs sliding-window SVM decoding within cortical ROIs.

## Requirements

- Python 3.9+
- MNE-Python 1.x (`pip install mne`)
- scikit-learn
- mat73 (`pip install mat73`)
- numpy, scipy, pandas, matplotlib

## Quick Start

```bash
# Validate on one subject first
python validate_pipeline.py

# Basic run (backward-compatible, aparc atlas, 8 ROIs)
python run_source_svm.py --task overtProd --stim-class prodDiff --method dSPM

# Advanced: Schaefer-200 atlas with leakage correction and pseudo-trials
python run_source_svm.py --task overtProd --stim-class prodDiff --method dSPM \
    --atlas Schaefer200 --leakage-correction --pseudo-trial-size 5

# Parallel (low-RAM, 2 workers)
python run_parallel_lowram.py --task overtProd --stim-class prodDiff --method dSPM \
    --atlas Schaefer200 --leakage-correction --n-jobs 2
```

## Output

Results are saved as CSV files under:

```
derivatives/SVM_source/{task}/{method}/{atlas}/{feature_mode}/{sw_dur}_{sw_step}/{stim_class}/
    {subj}_{task}_{stim_class}_{sw_dur}_{sw_step}.csv
```

CSV columns:

| Column | Description |
|--------|-------------|
| `key` | ROI name (e.g., `Temporal`, `17Networks_LH_AudA_1-lh`) |
| `ms` | Center of the sliding window in milliseconds |
| `mean_list` | Per-repeat CV accuracies (list of 5 values) |
| `SVM_acc` | Mean accuracy across all CV repeats |
| `best_params` | Classifier configuration (e.g., `C=1.0`) |

## Pipeline Steps

1. **Load data** — EEGLAB `.mat` files → MNE Epochs
2. **Forward model** — fsaverage ico-5 source space + 3-layer BEM
3. **Inverse solution** — dSPM or LCMV → source estimates per epoch
4. **ROI extraction** — extract time courses from cortical ROIs (multi-atlas)
5. **Leakage correction** *(optional)* — remove spatial cross-talk between ROIs
6. **SVM decoding** — sliding-window LinearSVC with repeated CV (optional pseudo-trial averaging)
7. **Save CSV** — one file per subject

## Atlas Selection

Four cortical parcellation options are supported, selectable via `--atlas`:

| Atlas | CLI name | Parcels | Source |
|-------|----------|---------|--------|
| Desikan-Killiany | `aparc` | 16 composite ROIs | MNE built-in (default, backward compatible) |
| Schaefer 2018 | `Schaefer200` | 200 (100/hemi) | Yeo lab, 17-network functional parcellation |
| HCP-MMP 1.0 | `HCPMMP1` | 360 (180/hemi) | Glasser et al. multi-modal parcellation |
| Custom functional localizer | `custom` | 6 (LH only) | Lab-derived language ROIs (Chang et al.) |

### Why finer-grained atlases?

The Desikan-Killiany (aparc) atlas is structurally too coarse for multivariate
investigation of the dual-stream speech model. It merges functionally distinct
regions — Area Spt, discrete auditory cortex parcellations, and IFG sub-regions
(BA44/45) — into broad macro-regions that average their distinct temporal dynamics
and obscure the sensorimotor interactions the study seeks to investigate.

However, adopting an ultra-high-resolution atlas (e.g., 360-region HCP-MMP) without
dimensionality reduction introduces instability: a 64-channel EEG array cannot
resolve 360 independent spatial sources, and the cross-talk from distributed inverse
solutions injects massive multicollinearity that triggers the curse of dimensionality
in the SVM.

The optimal feature space for 64-channel EEG lies between **100 and 250 parcels**
(Tait et al., 2021). The **Schaefer-200** atlas (derived from functional connectivity
gradients) provides an excellent dimensional match while naturally separating
cognitive and sensorimotor networks. The **HCPMMP1** atlas provides the finest
available functional boundaries and can be used effectively with PCA or SelectKBest
dimensionality reduction.

## Leakage Correction

Source-space inverse solutions are inherently spatially blurred: the point-spread
function of a 64-channel array means that activity from one cortical region leaks
into neighboring regions. When ROI time courses are fed to an SVM, this cross-talk
inflates the apparent multivariate pattern with redundant, algorithmically blurred
signals rather than true neurophysiological communication.

Enable with `--leakage-correction`. Two methods are applied automatically based on
the feature mode:

### Symmetric Orthogonalization (pca_flip mode)

*Reference: Colclough et al. (2015), NeuroImage.*

Per-epoch, computes D_orth = C^{-1/2} @ D where C = D @ D^T. This removes all
zero-lag (instantaneous) correlations between ROI summary signals — correlations
that are almost certainly artifacts of the inverse solution rather than true neural
interactions. Lagged correlations (reflecting genuine communication with propagation
delays) are preserved.

### Regression-Based Vertex Correction (vertex_pca / vertex_selectkbest modes)

*Reference: Hipp et al. (2012), Nature Neuroscience.*

For each target ROI's vertices, regresses out the PCA-flip summary signals of all
*other* ROIs, removing the linear contribution of cross-ROI leakage while preserving
within-ROI spatial patterns needed by the SVM.

## Pseudo-Trial Averaging

Single-trial EEG source estimates have low SNR due to spontaneous background brain
activity. Pseudo-trial averaging groups N same-class trials and averages them,
effectively averaging out random noise while preserving time-locked signals. This
provides the SVM with cleaner, more separable decision boundaries.

Enable with `--pseudo-trial-size N` (e.g., `--pseudo-trial-size 5`). Applied only to
training data within each CV fold (test set remains individual trials). With ~80–100
trials per class and group_size=5, yields ~16–20 pseudo-trials per class. Group sizes
of 5–10 are recommended.

## Files

### `config.py`

Central configuration. All paths, subject IDs, ROI definitions, atlas maps, and pipeline parameters.

- `PROJECT_ROOT`, `EEGLAB_DIR`, `SVM_OUTPUT_ROOT` — directory paths
- `SUBJECT_IDS` — list of 20 subject IDs
- `PROD_DIFF_TH`, `PROD_DIFF_F`, `PERC_DIFF_S`, `PERC_DIFF_T` — word classes for each stimulus contrast
- `SPEECH_ROIS` — 16 speech-network ROIs mapped per atlas: `SPEECH_ROIS[atlas][roi_name]` → parcel list
- `SPEECH_ROI_NAMES` — ordered list of the 16 ROI names
- `ATLAS_PARC_MAP` — maps CLI atlas names to MNE parcellation strings
- `CUSTOM_ROI_DIR` — path to custom volumetric NIfTI ROI masks (functional localizers)
- `CUSTOM_ROI_NAMES` — ordered list of the 6 custom ROI names
- `SVM_C`, `PSEUDO_TRIAL_SIZE`, `LEAKAGE_CORRECTION` — advanced pipeline defaults
- `SW_DUR`, `SW_STEP_SIZE` — sliding window parameters (40 ms, 5 ms)
- `N_CV_FOLDS`, `N_CV_REPEATS` — cross-validation settings (5 folds, 5 repeats)
- `LAMBDA2` — regularization parameter for dSPM (SNR=3)
- `BASELINE_WINDOWS` — per-task noise covariance baselines
- `DECODE_TMIN` — per-task SVM decode start time (after baseline)

### `data_loader.py`

Loads EEGLAB-preprocessed `.mat` files and converts them to MNE Epochs.

- **`load_subject_epochs(subj_id, task_cond, stim_class)`** — Main entry point. Loads one subject's data, applies artifact rejection, filters to the requested stimulus contrast, converts units (uV → V), sets the biosemi64 montage and average reference. Returns `(epochs, y_labels, sfreq)`.

### `forward_model.py`

Builds the forward model using the fsaverage template anatomy. Supports multiple atlases.

- **`setup_fsaverage()`** — Fetches fsaverage files, reads ico-5 source space and 3-layer BEM. Returns `(subjects_dir, fs_dir, src, bem)`.
- **`make_forward(epochs_info, src, bem)`** — Computes the forward solution. Built once and shared across all subjects.
- **`load_custom_volumetric_rois(roi_dir, threshold, subjects_dir, surf)`** — Projects volumetric NIfTI ROI masks onto the fsaverage surface via nilearn `vol_to_surf`. Returns `{roi_name: mne.Label}`.
- **`build_roi_labels(subjects_dir, atlas, composite_rois, custom_roi_dir)`** — Reads cortical labels from the specified atlas. For `atlas='aparc'` with `composite_rois`, merges labels into composite ROIs. For `atlas='custom'`, dispatches to `load_custom_volumetric_rois()`. For other atlases, returns all parcels (excluding `???` labels).

### `inverse_pipelines.py`

Two inverse methods for projecting sensor-space EEG into source space.

- **`run_dspm(...)`** / **`run_lcmv(...)`** — Standard inverse solutions returning full SourceEstimate lists.
- **`run_dspm_lowram(...)`** / **`run_lcmv_lowram(...)`** — Generator-based variants that extract ROI data during inverse computation and discard full STCs to save memory.

### `leakage_correction.py`

Spatial leakage correction for source-estimated ROI time courses.

- **`symmetric_orthogonalize(data)`** — Lowdin orthogonalization for one epoch (pca_flip mode).
- **`apply_leakage_correction(X_roi)`** — Per-epoch symmetric orthogonalization for all epochs.
- **`apply_vertex_leakage_correction(roi_data, X_all_pca, roi_names)`** — Regression-based correction for vertex modes. Regresses out other ROIs' PCA-flip signals from each target ROI's vertices.
- **`compute_pca_summaries_from_vertices(roi_data_list, n_times)`** — Computes PCA-like summaries from vertex data (for low-RAM path where stcs are unavailable).

### `pseudo_trials.py`

Pseudo-trial averaging for SNR improvement.

- **`create_pseudo_trials(X, y, group_size, rng)`** — Groups same-class trials, averages within groups, returns pseudo-trial features and labels. Applied only to training data within CV folds.

### `svm_decoding.py`

Sliding-window SVM classification in source space.

- **`sliding_window_svm_decode(X_roi, y, ..., svm_c, pseudo_trial_size, random_state)`** — Core decoding function. Crops baseline, applies sliding window averaging, runs LinearSVC(C=svm_c) with manual CV loop supporting optional pseudo-trial averaging.
- **`extract_roi_data_pca_flip(stcs, roi_labels, src)`** — One summary time course per ROI via PCA-flip. Shape: `(n_epochs, n_rois, n_times)`.
- **`extract_roi_data_vertices(stcs, roi_label)`** — All vertex time courses within one ROI. Shape: `(n_epochs, n_vertices, n_times)`.

### `run_source_svm.py`

Main batch runner (sequential). Accepts `--atlas`, `--leakage-correction`, `--pseudo-trial-size`, `--svm-c`.

### `run_parallel.py` / `run_parallel_lowram.py`

Parallel batch runners. Same CLI arguments as `run_source_svm.py` plus `--n-jobs`.

### `validate_pipeline.py`

End-to-end validation on a single subject.

### `visualize_rois.py`

Interactive and publication-ready ROI visualization on fsaverage cortical surface.

- **`build_speech_roi_labels(atlas, subjects_dir)`** — Build composite `mne.Label` for each speech ROI.
- **`plot_roi_brain(..., atlas, fmt)`** — All ROIs on one brain (any atlas or speech-ROI subset).
- **`plot_single_roi(roi_name, ..., atlas, fmt)`** — Single named ROI or substring filter. Accepts `SPEECH_ROIS` names (e.g., `Anterior_STS`, `vSMC`).
- **`plot_compare_modes(..., atlas, fmt)`** — Full-resolution vs ico-5 side-by-side.
- Supports `--format svg` for publication figures (vector labels/axes, embedded raster brain).
- CLI flags: `--speech-rois` (all 16 ROIs), `--list-rois` (print names), `--roi NAME`.

### `run_pipeline_notebook.py`

Interactive percent-format notebook for step-by-step exploration.

## Feature Modes

| Mode | Features per ROI | Description |
|------|-----------------|-------------|
| `pca_flip` | 1 | PCA-flipped summary time course (MNE `extract_label_time_course`) |
| `vertex_pca` | All vertices, PCA-reduced | All vertex time courses with PCA (95% variance) in the sklearn pipeline |
| `vertex_selectkbest` | All vertices, top-k selected | All vertex time courses with supervised feature selection (ANOVA F-test, k=200) |

## ROIs

### Legacy ROIs (aparc, 8 composite ROIs)

| ROI | Aparc labels |
|-----|-------------|
| Temporal | superiortemporal, middletemporal, bankssts |
| Inferior_Frontal | parsopercularis, parstriangularis |
| Superior_Frontal | superiorfrontal |
| Superior_Parietal | superiorparietal, precuneus |
| (+ right-hemisphere analogues for each) | |

These are defined in `config.SPEECH_ROIS['aparc']` and used when `--atlas aparc` is
specified without other ROI options (backward-compatible default).

### Speech Network ROIs (16 regions, all atlases)

Defined in `config.SPEECH_ROIS`, these 16 ROIs span the dual-stream speech processing
network (left hemisphere). Each ROI is mapped to atlas-specific parcels for aparc,
Schaefer-200, and HCP-MMP1.

| # | ROI | Stream | Motivation |
|---|-----|--------|------------|
| 1 | `Temporal` | Both | Sensor ROI (FT7/T7/TP7) — STG, MTG, STS |
| 2 | `Inferior_Frontal` | Both | Sensor ROI (F5/FC5/FC3) — IFG/Broca's, BA44/45 |
| 3 | `Superior_Frontal` | Dorsal | Sensor ROI (F1/FC1/FCz) — SMA, pre-SMA |
| 4 | `Superior_Parietal` | Dorsal | Sensor ROI (CPz/CP1/P1) — SPL, precuneus |
| 5 | `vSMC` | Motor | Chang — ventral sensorimotor cortex (articulatory representations) |
| 6 | `Supramarginal` | Dorsal | Hickok, Poeppel, Flinker — phonological processing, area Spt |
| 7 | `Angular_Gyrus` | Ventral | Poeppel, Tian — semantic integration |
| 8 | `Insula` | Planning | Flinker, Dronkers — articulatory planning |
| 9 | `TPOJ` | Hub | Glasser, Poeppel — multimodal integration |
| 10 | `Cingulate_Motor` | Dorsal | Tian — speech initiation, efference copy / forward predictions |
| 11 | `Planum_Temporale` | Hub | Rauschecker & Scott — computational hub between streams |
| 12 | `Anterior_STS` | Ventral | Scott et al. — intelligible speech processing |
| 13 | `Temporal_Pole` | Ventral | Rauschecker — ventral stream semantic terminus |
| 14 | `Pars_Orbitalis` | Ventral | Scott & Eisner — ventral stream prefrontal terminus (BA47) |
| 15 | `Posterior_STS` | Entry | Rauschecker & Scott — acoustic-phonetic analysis |
| 16 | `DLPFC` | Dorsal | Rauschecker — dorsal stream learned sequence storage |

**Key references:**
- Rauschecker & Scott (2009), *Nat Neurosci* — dual-stream model, anterior STS hierarchy
- DeWitt & Rauschecker (2012), *PNAS* — phoneme/word ventral stream gradient
- Scott et al. (2000), *Brain* — anterior STS pathway for intelligible speech
- Hickok & Poeppel (2007), *Nat Rev Neurosci* — dorsal/ventral stream cortical organization
- Chang lab (Bouchard et al., 2013) — vSMC articulatory somatotopy
- Flinker et al. (2015) — sequential STG → IFG → motor cortex activation
- Tian & Poeppel (2010, 2013) — efference copy / forward predictions in speech
- Glasser et al. (2016) — HCP-MMP1 multi-modal parcellation

**Atlas resolution notes:** The aparc atlas cannot separate anterior from posterior
STS, or isolate planum temporale — several ROIs map to the same coarse labels.
Use HCPMMP1 or Schaefer-200 for the full benefit of the 16-ROI speech network.

### Full atlas parcellations

When running without `--speech-rois` or `--roi`, all parcels from the atlas are used:

| Atlas | Parcels | Example names |
|-------|---------|---------------|
| Schaefer-200 | 200 (100/hemi) | `17Networks_LH_SomMotB_Aud_1-lh`, `17Networks_LH_TempPar_1-lh` |
| HCPMMP1 | 360 (180/hemi) | `L_A1_ROI-lh`, `L_44_ROI-lh`, `L_STSda_ROI-lh` |

### Custom Functional-Localizer ROIs (`--atlas custom`)

Six language-network ROIs derived from functional localizers (Chang et al.), defined
as 15 mm spheres with anatomical restriction in MNI volumetric space.

| # | ROI | Full Name | Localizer | Vertices (full) | Vertices (ico-5) |
|---|-----|-----------|-----------|-----------------|------------------|
| 1 | `awfa` | Auditory word form area | audioLoc | ~1,466 | ~95 |
| 2 | `ifc` | Inferior frontal cortex | bothLoc | ~2,521 | ~159 |
| 3 | `owfa` | Orthographic word form area | vwfaLoc | ~1,645 | ~105 |
| 4 | `pmc` | Premotor cortex | audioLoc | ~2,731 | ~173 |
| 5 | `tpc` | Temporo-parietal cortex | bothLoc | ~2,649 | ~159 |
| 6 | `vwfa` | Visual word form area | vwfaLoc | ~2,081 | ~130 |

All six ROIs are left-hemisphere only. Source masks are binary NIfTI volumes
(91 x 109 x 91, 2 mm isotropic, MNI152 space).

#### Volume-to-Surface Projection Method

The atlas-based ROIs (aparc, Schaefer200, HCPMMP1) are natively defined on the
FreeSurfer surface and can be read directly as vertex labels. The custom ROIs, by
contrast, are defined as volumetric binary masks in MNI space and must be projected
onto the fsaverage cortical surface. The projection is performed by
`forward_model.load_custom_volumetric_rois()` using the following procedure:

1. **Input**: Each ROI is a binary NIfTI mask (voxel values 0 or 1) in MNI152
   space at 2 mm isotropic resolution, with an affine matrix mapping voxel
   indices to MNI coordinates (origin at [-90, -126, -72] mm).

2. **Surface mesh**: The fsaverage pial surface (`lh.pial`, `rh.pial`) is loaded
   from the MNE-Python fsaverage dataset. This mesh has ~163,842 vertices per
   hemisphere and is registered to MNI305 space, which is related to MNI152 by a
   known affine transform that nilearn handles internally.

3. **Projection via `nilearn.surface.vol_to_surf()`**: For each vertex on the
   fsaverage pial surface, nilearn samples the volumetric mask along the surface
   normal within a 3.0 mm radius ball. The `nearest_most_frequent` interpolation
   method is used: within the sampling sphere, all voxels are evaluated and the
   most frequently occurring non-NaN value is assigned to the vertex. For binary
   masks this is equivalent to a majority-vote: a vertex is labelled positive only
   if the majority of nearby voxels within 3 mm of the surface are inside the ROI.

   - **Why `nearest_most_frequent`**: This interpolation is designed for
     deterministic atlases and categorical labels. Unlike linear interpolation
     (which would produce fractional values at ROI boundaries and blur the edges)
     or simple nearest-neighbor (which takes only the single closest voxel and is
     sensitive to partial-volume effects), `nearest_most_frequent` aggregates all
     voxels within the search radius and assigns the mode. This produces clean
     binary boundaries on the surface that respect the volumetric ROI shape without
     artificial smoothing.

   - **Why 3.0 mm radius**: The search radius must be large enough to bridge the
     gap between the cortical surface mesh and nearby voxels (especially in sulcal
     folds where the pial surface may not pass exactly through voxel centers), but
     small enough to avoid pulling in voxels from adjacent gyri. At 2 mm voxel
     resolution, a 3 mm radius captures the immediate 1–2 voxel neighborhood
     around each surface vertex. This is the nilearn default and provides reliable
     coverage without spatial blurring.

4. **Thresholding**: After projection, each vertex has a projected value between
   0.0 and 1.0. Vertices with projected values >= 25% of the maximum projected
   value are included in the surface label. For binary input masks where the
   maximum is 1.0, this threshold effectively keeps all vertices where the
   majority-vote projection returned a positive value, while discarding vertices
   at the extreme periphery where only a small fraction of neighboring voxels
   fell inside the ROI.

5. **Hemisphere selection**: Each volumetric mask is projected onto both the left
   and right hemisphere surfaces independently. Hemispheres where no vertices
   survive thresholding are discarded. For these language-specific ROIs, only left-
   hemisphere labels are produced (all 6 ROIs yield `*-lh` labels; no vertices
   survive on `rh`).

6. **Output**: Standard `mne.Label` objects with the same interface as atlas-derived
   labels. These are directly consumed by `mne.extract_label_time_course()` (for
   `pca_flip` mode) and `stc.in_label()` (for vertex modes), ensuring full
   compatibility with the existing pipeline infrastructure including leakage
   correction and pseudo-trial averaging.

#### Accuracy Considerations

- **Spatial precision**: The projection preserves the volumetric ROI boundaries
  to within the resolution of the surface mesh (~1 mm vertex spacing on fsaverage
  pial) and the voxel grid (2 mm). The 3 mm search radius introduces at most one
  voxel of spatial uncertainty at ROI boundaries.

- **Coordinate system alignment**: The NIfTI masks use MNI152 coordinates
  (sform/qform with `aligned` code). The fsaverage surfaces are in MNI305 space.
  nilearn's `vol_to_surf` handles the MNI152-to-MNI305 affine transformation
  internally, so no manual registration is required.

- **Source-space restriction**: When the pipeline runs with the default ico-5
  source space (~10,242 vertices per hemisphere), the full-resolution surface
  labels (~1,400–2,700 vertices) are automatically restricted to the ico-5 vertex
  subset via `label.restrict(src)`, yielding ~95–173 source-space vertices per
  ROI. This is the same restriction applied to all atlas-based labels.

```bash
# Run SVM decoding with custom functional-localizer ROIs
python run_source_svm.py --task overtProd --stim-class prodDiff --method dSPM \
    --atlas custom

# Visualize custom ROIs on fsaverage (full resolution)
python visualize_rois.py --atlas custom --save --mode full --hemi lh

# Visualize with ico-5 restriction (what the pipeline actually uses)
python visualize_rois.py --atlas custom --save --mode ico5 --hemi lh
```

## Command-Line Reference

```bash
# Basic (backward compatible)
python run_source_svm.py \
    --task {perception,overtProd} \
    --stim-class {prodDiff,percDiff} \
    --method {dSPM,LCMV} \
    [--feature-mode {pca_flip,vertex_pca,vertex_selectkbest}] \
    [--subjects EEGPROD4001 EEGPROD4003 ...] \
    [--sw-dur 40] [--sw-step 5]

# Advanced options
python run_source_svm.py \
    --task overtProd --stim-class prodDiff --method dSPM \
    --atlas Schaefer200 \
    --leakage-correction \
    --pseudo-trial-size 5 \
    --svm-c 1.0

# Parallel (same args, plus --n-jobs)
python run_parallel_lowram.py \
    --task overtProd --stim-class prodDiff --method dSPM \
    --atlas HCPMMP1 --leakage-correction \
    --n-jobs 2

# Validate
python validate_pipeline.py \
    [--subject EEGPROD4001] [--task overtProd] \
    [--stim-class prodDiff] [--skip-lcmv]

# ── Visualization ──

# List available speech-network ROI names
python visualize_rois.py --list-rois --atlas HCPMMP1

# All 16 speech ROIs on HCP-MMP1, save as SVG
python visualize_rois.py --speech-rois --atlas HCPMMP1 --save --format svg

# Single named ROI (parcels auto-selected from config.SPEECH_ROIS)
python visualize_rois.py --roi Anterior_STS --atlas HCPMMP1 --save --format svg
python visualize_rois.py --roi vSMC --atlas Schaefer200 --save

# Full-resolution vs ico-5 comparison
python visualize_rois.py --mode compare --atlas Schaefer200 --save --format svg
```
