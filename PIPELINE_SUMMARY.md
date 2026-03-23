# Source Estimation Pipeline — Summary

## Overview

This pipeline performs source-space SVM decoding of EEG data from a speech
production/perception experiment. It projects sensor-level EEG onto cortical
source space using an fsaverage template, extracts activity from predefined
cortical ROIs, and runs sliding-window linear SVM classification to identify
when brain regions discriminate between stimulus classes.

---

## Directory Structure

| File | Purpose |
|------|---------|
| `config.env.example` | Template for machine-specific `EEG_PROJECT_ROOT` path |
| `config.py` | Central configuration: paths, subjects, ROIs, atlas maps, parameters |
| `data_loader.py` | EEGLAB `.mat` → MNE `Epochs` conversion |
| `forward_model.py` | fsaverage BEM, source space, multi-atlas cortical ROI labels |
| `inverse_pipelines.py` | dSPM and LCMV inverse solutions (standard + low-RAM) |
| `leakage_correction.py` | Spatial leakage correction (orthogonalization + regression) |
| `pseudo_trials.py` | Pseudo-trial averaging within CV folds |
| `svm_decoding.py` | ROI feature extraction + sliding-window SVM |
| `plotting.py` | Figure generation for sensor ERPs, ROI time courses, and SVM accuracy |
| `run_source_svm.py` | Sequential main runner (CLI) |
| `run_parallel.py` | Multiprocessing parallel runner (CLI) |
| `run_parallel_lowram.py` | Low-RAM parallel runner (generator-based) |
| `run_pipeline_notebook.py` | Interactive notebook-style runner (standard memory) |
| `run_pipeline_notebook_lowram.py` | Interactive notebook-style runner (low-RAM) |
| `source_stats_viz.py` | Group-level statistics and visualization |
| `validate_pipeline.py` | End-to-end validation / smoke test |
| `visualize_rois.py` | Interactive and publication-ready ROI visualization on fsaverage surface |

---

## Configuration Parameters (`config.py`)

### Paths

- **EEGLAB_DIR**: `$EEG_PROJECT_ROOT/derivatives/EEGLAB/`
  - Perception data: `perception/{subj}/average_ref/eeglab_standard/fs_2000/*popthresh120.mat`
  - Production data: `overtProd/{subj}/average_ref/fs_2000/*ProdOnset.mat`
- **SVM_OUTPUT_ROOT**: `$EEG_PROJECT_ROOT/derivatives/SVM_source/`
- **ROI_TIMESERIES_ROOT**: `$EEG_PROJECT_ROOT/derivatives/SVM_source_timeseries/`
- **FIGURES_ROOT**: `$EEG_PROJECT_ROOT/derivatives/SOURCE_ESTIMATION/`
- **CODE_DIR**: `$EEG_PROJECT_ROOT/code/source_estimation/`

#### Output directory structure

Results are saved as CSV files under a hierarchical path that encodes all run parameters:

```
derivatives/SVM_source/{task}/{method}/{atlas}/{feature_mode}/{sw_dur}_{sw_step}/{stim_class}/
    {subj}_{task}_{stim_class}_{sw_dur}_{sw_step}.csv
```

### Subjects

20 participants: `EEGPROD4001`–`EEGPROD4023` (some IDs missing).

### Stimulus Classes and Word Lists

Two binary contrasts are defined:

| Contrast | Class 0 | Class 1 | Description |
|----------|---------|---------|-------------|
| **prodDiff** | F-pseudowords (e.g., "FEEP", "FOPE") | TH-pseudowords (e.g., "THEEP", "THOPE") | Production difference |
| **percDiff** | S-pseudowords (e.g., "SAFF", "SOOG") | T-pseudowords (e.g., "TAFF", "TOOG") | Perceptual difference |

### Cortical ROIs and Atlas Selection

The pipeline supports three atlas parcellations, selectable via `--atlas`:

| Atlas | Parcels | Description |
|-------|---------|-------------|
| `aparc` (default) | 8 composite ROIs | Desikan-Killiany labels merged into 4 bilateral regions (backward compatible) |
| `Schaefer200` | 200 (100/hemi) | Schaefer et al. (2018) functional parcellation, 17-network variant |
| `HCPMMP1` | 360 (180/hemi) | Glasser et al. (2016) Human Connectome Project multi-modal parcellation |

**Speech Network ROIs** (16 regions, left hemisphere, all atlases):

The pipeline defines 16 cortical ROIs spanning the dual-stream speech processing
network. Each ROI is mapped to atlas-specific parcels in `config.SPEECH_ROIS[atlas]`.
Right-hemisphere analogues are obtained by flipping label suffixes
(`-lh`→`-rh`, `L_`→`R_`).

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

**Aparc atlas labels** (first 4 ROIs, backward-compatible defaults):

| ROI Name | aparc Labels |
|----------|-------------|
| Temporal | superiortemporal, middletemporal, bankssts |
| Inferior_Frontal | parsopercularis, parstriangularis |
| Superior_Frontal | superiorfrontal |
| Superior_Parietal | superiorparietal, precuneus |

**Atlas resolution notes:** The aparc atlas is too coarse to separate anterior from
posterior STS, or to isolate planum temporale — several ROIs map to the same broad
labels. Use HCPMMP1 or Schaefer-200 for the full benefit of the 16-ROI speech network.

**Rationale for finer-grained atlases**: The Desikan-Killiany atlas is structurally
too coarse for multivariate investigation of the dual-stream speech model. It merges
functionally distinct regions such as Area Spt, discrete auditory parcellations, and
IFG sub-regions into broad macro-regions that obscure sensorimotor interactions.
However, adopting an ultra-high-resolution atlas (e.g., 360-region HCP-MMP) without
dimensionality reduction introduces instability: a 64-channel EEG array cannot
resolve 360 independent spatial sources, and the cross-talk from distributed inverse
solutions injects massive multicollinearity that triggers the curse of dimensionality
in the SVM.

The optimal feature space for 64-channel EEG lies between **100 and 250 parcels**
(Tait et al., 2021). The Schaefer-200 atlas (200 parcels derived from functional
connectivity gradients) provides an excellent dimensional match while naturally
separating cognitive and sensorimotor networks. The HCPMMP1 atlas (360 parcels with
fine functional boundaries) can be used effectively with PCA or supervised feature
selection (`vertex_pca` or `vertex_selectkbest` modes) for dimensionality reduction.

**Full atlas parcellations** — when running without `--speech-rois` or `--roi`, all
parcels from the selected atlas are used:

| Atlas | Parcels | Example names |
|-------|---------|---------------|
| Schaefer-200 | 200 (100/hemi) | `17Networks_LH_SomMotB_Aud_1-lh`, `17Networks_LH_TempPar_1-lh` |
| HCPMMP1 | 360 (180/hemi) | `L_A1_ROI-lh`, `L_44_ROI-lh`, `L_STSda_ROI-lh` |

### SVM Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SW_DUR` | 40 (ms) | Sliding window duration |
| `SW_STEP_SIZE` | 5 (ms) | Sliding window step |
| `N_SPLITS` | 5 | K-fold CV splits |
| `N_REPEATS` | 5 | Number of CV repeats |
| `SVM_C` | 1.0 | LinearSVC regularization (explicitly set; C >= 1 recommended for EEG decoding) |

### Advanced Pipeline Options

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `LEAKAGE_CORRECTION` | `--leakage-correction` | Off | Spatial leakage correction after source projection |
| `PSEUDO_TRIAL_SIZE` | `--pseudo-trial-size N` | 0 (off) | Average N same-class trials into pseudo-trials within CV folds |
| `SVM_C` | `--svm-c` | 1.0 | LinearSVC regularization parameter C |

### Task-Specific Timing

| Parameter | Perception | Production |
|-----------|-----------|------------|
| Baseline window | −200 to −100 ms | −1600 to −1500 ms |
| Decoding start (`DECODE_TMIN`) | −100 ms | −1500 ms |

---

## Pipeline Steps

### Step 1: Data Loading (`data_loader.py`)

1. Load EEGLAB `.mat` file (HDF5 format via `mat73`)
2. Extract `data` array (channels × timepoints × trials)
3. Extract behavioral word labels with task-specific handling:
   - **Production**: flatten nested list structure
   - **Perception**: filter out MEEV (missing) trials
4. Apply artifact rejection using `trialInfo` (0 = bad trial)
5. Build binary class labels (0/1) from word lists
6. Transpose to MNE format (trials × channels × timepoints)
7. Build MNE `EpochsArray` with BIOSEMI 64-channel montage + average reference

### Step 2: Forward Model (`forward_model.py`)

1. Fetch fsaverage template (`ico-5` source space: ~10,242 vertices/hemisphere)
2. Load 3-layer BEM solution (skin, skull, brain)
3. Create forward solution using the EEG montage from the loaded data
4. Build ROI labels from the selected atlas parcellation:
   - **`aparc`**: Reads Desikan-Killiany labels; merges them into composite speech-network ROIs via `config.SPEECH_ROIS['aparc']` (backward compatible)
   - **`Schaefer200`** / **`HCPMMP1`**: Reads all parcels from the atlas; when `--speech-rois` is used, selects only the 16 speech-network ROIs defined in `config.SPEECH_ROIS`

The forward model is built once and reused across all subjects (template approach).

### Step 3: Inverse Solution (`inverse_pipelines.py`)

The "inverse problem" is estimating which cortical sources produced the pattern of
voltages measured at the scalp. This is fundamentally ill-posed (many more source
locations than sensors), so additional constraints are needed. Two families of
solutions are implemented.

#### dSPM (Dynamic Statistical Parametric Mapping)

**Reference**: Dale et al. (2000) *Neuron*. dSPM is the standard noise-normalized
minimum-norm estimate (MNE) used widely in MEG/EEG source imaging.

**Steps and parameters:**

1. **Noise covariance** (`method='shrunk'`): Estimated from the pre-stimulus
   baseline window. The Ledoit-Wolf shrinkage estimator regularizes the 64×64
   covariance matrix by pulling it toward a diagonal, which stabilizes the
   estimate when you have relatively few baseline time samples. This is the
   recommended approach in MNE-Python for EEG.
   - *Alternative*: `method='empirical'` (no regularization — fine if you have a
     long baseline) or `method='auto'` (tries multiple estimators, picks best
     by log-likelihood).

2. **Inverse operator** — the spatial filter that maps 64 sensors → ~20,484 vertices:
   - **`loose=0.0`**: Source orientations are **fixed** perpendicular to the cortical
     surface. Each vertex has exactly one dipole pointing along the surface normal.
     This halves the unknowns (vs. free orientation) and is well-motivated because
     cortical pyramidal neurons are oriented perpendicular to the cortical sheet.
     - *Alternative*: `loose=0.2` allows orientations to deviate up to ~20% from
       normal (a compromise), `loose=1.0` is fully free orientation (3 components
       per vertex — rarely used for cortical sources).
   - **`depth=0.8`**: Depth weighting exponent. Without depth weighting, MNE
     solutions are biased toward superficial (gyral) sources because they produce
     larger scalp signals. Depth weighting compensates by penalizing shallow
     sources less. The value 0.8 is the MNE-Python default based on Lin et al.
     (2006). Range is 0.0 (no depth weighting) to 1.0 (full compensation).
     - *Typical values*: 0.8 is standard. Values of 0.0–0.5 give more superficial
       solutions; 1.0 may over-correct.
   - **`fixed=True`**: Enforces exactly normal orientation (redundant with `loose=0`
     but explicit). The result is a scalar (signed amplitude) per vertex per time
     point rather than a 3D vector.

3. **Applying the inverse** — projects each epoch from sensor space to source space:
   - **`lambda2 = 1/SNR²`** where **`SNR=3.0`** → **`lambda2 ≈ 0.111`**: This is
     the regularization parameter. It controls the trade-off between fitting the
     data exactly and keeping the solution smooth/small. SNR=3 is the standard
     MNE-Python default for single-trial (evoked) analysis. Higher SNR → less
     regularization (trusts the data more); lower SNR → smoother solution.
     - *Guidance*: SNR=3 (λ²≈0.11) for averaged evoked responses is standard
       (Hämäläinen & Ilmoniemi 1994). For noisier single-trial data you could use
       SNR=1 (λ²=1.0). In practice, results are not very sensitive to this value
       within the 1–3 range.
   - **`method='dSPM'`**: After computing the raw MNE solution, each vertex's time
     course is divided by the expected noise at that vertex (derived from the noise
     covariance and the inverse operator). The result is a **noise-normalized
     F-statistic** — unitless values where the baseline should hover around 1.0,
     and deviations indicate signal that exceeds the noise floor. This makes values
     comparable across vertices with different noise levels.
     - *Alternative*: `method='MNE'` (raw minimum-norm, in Am), `method='sLORETA'`
       (standardized LORETA — normalizes by resolution matrix, tends to give more
       focal activation).
   - **`nave=1`**: Number of averages. Tells MNE this is single-trial data (not a
     grand average), so it scales the regularization appropriately.

**What you could change:**
- `SNR` / `lambda2`: Try SNR=1 for more regularization on noisy single trials
- `depth`: 0.0 to see superficial bias, or leave at 0.8
- `method`: 'sLORETA' for potentially more focal estimates (better if you have subject specific anatomy though..)
- `loose`: 0.2 if you suspect sources aren't perfectly perpendicular

#### LCMV (Linearly Constrained Minimum Variance Beamformer)

**Reference**: Van Veen et al. (1997) *IEEE Trans. Biomed. Eng.* Beamformers take a
fundamentally different approach from MNE: instead of inverting the whole forward
model at once, they design an optimal spatial filter for each vertex independently.

**Steps and parameters:**

1. **Data covariance** (`method='empirical'`, from post-baseline window): Captures
   the spatial structure of the signal you want to localize. Estimated from the
   active window (post-baseline to end of epoch) because the beamformer needs to
   "see" the signal of interest in this matrix to form a good filter.
   - *Why empirical?* The active window is typically long enough that the empirical
     estimator is stable. Shrinkage could also be used but is less common here.

2. **Noise covariance** (`method='shrunk'`, from baseline): Used to whiten the data
   — equalize noise across channels so the beamformer isn't distorted by channels
   with different noise levels.

3. **Spatial filter** (`make_lcmv`) — builds one filter per vertex:
   - **`reg=0.05`**: Tikhonov regularization (5% of the trace of the data
     covariance is added to its diagonal). Prevents the filter from becoming
     unstable when the data covariance is near-singular. Standard range is 0.01–0.1.
     - *Guidance*: Lower values (0.01) give sharper but potentially noisier
       estimates. Higher values (0.1) are more stable but more blurred.
   - **`pick_ori='max-power'`**: At each vertex, the forward model defines a
     3D dipole orientation, but we want a scalar output. 'max-power' selects the
     orientation that maximizes the output power of the beamformer at that vertex.
     This is data-driven and typically gives better SNR than fixing orientation
     to the surface normal.
     - *Alternative*: `pick_ori='normal'` (fix to cortical normal, like dSPM),
       `pick_ori=None` (keep 3D vector output).
   - **`weight_norm='unit-noise-gain'`**: Normalizes each vertex's spatial filter
     so that it passes unit noise. Without this, deeper sources would have larger
     filter weights (to compensate for signal attenuation), amplifying noise.
     This makes the output a **pseudo-Z-score** (neural activity index) analogous
     to dSPM's F-statistic.
     - *Alternative*: `weight_norm='nai'` (neural activity index — similar concept),
       `weight_norm=None` (no normalization — output in Am but biased by depth).

4. **Applying the filter**: Each epoch is multiplied by the precomputed spatial
   filter. This is fast (just matrix multiplication) compared to building the filter.

**What you could change:**
- `reg`: 0.01 for sharper, 0.1 for more stable
- `pick_ori`: 'normal' for consistency with dSPM
- `weight_norm`: 'nai' is similar; None gives raw beamformer output

#### dSPM vs. LCMV: When to prefer which

| | dSPM | LCMV |
|--|------|------|
| **Approach** | Global inverse (all vertices at once) | Local filter (one vertex at a time) |
| **Correlated sources** | Handles well | Suppresses them (known limitation) |
| **Focal sources** | Tends to be spatially smooth | Better at localizing focal activity |
| **Assumptions** | Distributed source model | Point source model |
| **Computation** | Inverse operator built once | Filter built once, fast to apply |
| **Common use** | General-purpose, exploratory | When focal sources are expected |

For SVM decoding, the choice matters less than for localization per se, because the
SVM only cares whether the ROI time courses carry discriminative information, not
whether they are perfectly localized.

**Low-RAM variants** (`run_dspm_lowram`, `run_lcmv_lowram`): Identical math, but
process epochs one at a time via generators, extract ROI data immediately, and
discard full vertex-level STCs. This avoids holding ~20,484 vertices × n_times ×
n_epochs float64 arrays in memory.

---

### Step 4: ROI Feature Extraction — PCA-flip and Vertex PCA explained

After the inverse solution, you have ~20,484 vertex time courses per epoch. Each ROI
contains hundreds of vertices (e.g., the Temporal ROI spans superiortemporal +
middletemporal + transversetemporal cortex). The feature extraction step reduces
these to a manageable representation.

#### Mode 1: `pca_flip` — MNE's standard virtual sensor

**What it does** (implemented in `mne.extract_label_time_course(mode='pca_flip')`):

1. **Collect** all vertex time courses within the ROI label for one epoch:
   a matrix V of shape (n_vertices_in_roi × n_times).

2. **Compute PCA** across vertices (spatial PCA): find the direction of maximum
   variance in vertex space. The first principal component (PC1) captures the
   dominant spatial pattern — the one mode of activation shared by most vertices.

3. **Project** all vertex time courses onto PC1 to get a single scalar time course.
   This is the "virtual sensor" — a weighted average of all vertices where the
   weights come from the PCA loading vector.

4. **Sign flip**: PCA is sign-ambiguous (PC1 and −PC1 explain the same variance).
   MNE resolves this by checking the PCA loading vector against the vertex normals
   in the forward model. If the majority of vertices have loadings that point
   opposite to their cortical normal (which would mean the dominant signal component
   is "inverted"), MNE flips the sign. This ensures the virtual sensor has a
   physiologically consistent polarity — positive values mean net outward current
   flow (the conventional direction for cortical pyramidal cell activation).

   *Why this matters*: Without the flip, averaging across subjects or conditions
   could cancel out real effects if some subjects/epochs happened to get the
   arbitrary negative sign.

**Result**: 1 value per ROI per time point. This is a dimensionality reduction
from hundreds of vertices to 1, preserving only the dominant spatial pattern.

**Trade-off**: If the ROI contains two distinct sub-regions with independent
activation patterns, PC1 captures only the dominant one. The minority pattern
is discarded.

#### Mode 2: `vertex_pca` — All vertices, PCA inside the classifier

This takes a fundamentally different approach:

1. **Extract** all vertex time courses within the ROI: (n_epochs × n_vertices × n_times).
   For the Temporal ROI this might be ~500 vertices.

2. **At each sliding window**, the features for one time point are all n_vertices
   values (after window averaging). So the feature vector is ~500-dimensional.

3. **Inside the sklearn pipeline**, before the SVM, PCA is applied with
   `PCA(n_components=0.95)`. This means: keep the minimum number of components
   that together explain ≥95% of the total variance.

   The number of components **varies** because it depends on the data structure
   at that particular time window:
   - If vertex activations are highly correlated (e.g., during a strong evoked
     response where the whole ROI activates together), most variance is in PC1–PC3,
     so you might get only 3–5 components.
   - If activations are diverse (e.g., during baseline noise), variance is spread
     across many components, so you might get 20–50 components.

   This is recalculated **at each time window** and **within each CV fold** (because
   PCA is fit only on training data to avoid data leakage).

4. The SVM then classifies based on these PCA-reduced features.

**Why variable components?** The `n_components=0.95` threshold is adaptive. Rather
than choosing a fixed number (which might be too few for complex windows or too many
for simple ones), it adjusts to the intrinsic dimensionality of the data at each
point. This is a standard sklearn approach.

**Trade-off vs. pca_flip**: `vertex_pca` preserves more spatial information (multiple
components rather than just one), giving the SVM more to work with. But it also has
more features, which increases the risk of overfitting — hence the PCA reduction and
cross-validation.

#### Mode 3: `vertex_selectkbest` — Supervised feature selection

1. Same vertex extraction as `vertex_pca`.
2. Instead of PCA, uses `SelectKBest(f_classif, k=200)`: at each time window,
   within each CV fold, it runs a one-way ANOVA F-test for each vertex (does this
   vertex differ between class 0 and class 1?) and keeps the top 200 most
   discriminative vertices.
3. SVM classifies on these 200 selected features.

**Trade-off**: This is supervised (uses labels during feature selection), which can
leak information if not done inside CV — but here it is inside the pipeline, so
it is applied per fold correctly.

### Step 4b: Spatial Leakage Correction (`leakage_correction.py`)

Source-space inverse solutions (dSPM, LCMV) are inherently spatially blurred: the
point-spread function of a 64-channel EEG array means that activity from one cortical
region leaks into neighboring regions in the source estimate. When these ROI time
courses are fed to an SVM, cross-talk inflates the apparent multivariate pattern by
introducing correlated features that reflect algorithmic blurring rather than true
neurophysiological communication. Leakage correction mathematically removes these
zero-lag cross-talk artifacts, ensuring the SVM decodes genuine functional differences.

Two correction methods are implemented, matched to the feature extraction mode:

#### Symmetric (Lowdin) Orthogonalization — for `pca_flip` mode

**Reference**: Colclough et al. (2015), *NeuroImage*.

Applied per-epoch to the ROI summary matrix D of shape (n_rois × n_timepoints):

1. Compute covariance: C = D @ D^T
2. Eigendecompose: C = V @ diag(w) @ V^T
3. Compute C^{-1/2} = V @ diag(w^{-1/2}) @ V^T
4. Orthogonalized data: D_orth = C^{-1/2} @ D

This is the symmetric solution that minimizes the sum of squared differences between
original and orthogonalized signals. It removes all instantaneous (zero-phase-lag)
correlations between ROIs — correlations that are almost certainly artifacts of the
inverse solution's spatial smoothing rather than true neural interactions. Crucially,
lagged correlations (reflecting genuine neural communication with propagation delays)
are preserved.

#### Regression-Based Vertex Correction — for `vertex_pca` / `vertex_selectkbest` modes

**Reference**: Hipp et al. (2012), *Nature Neuroscience*.

For vertex-level features, direct orthogonalization across all vertices of all ROIs
would create an impractically large matrix. Instead, a targeted regression approach is
used:

1. Compute PCA-flip summary signals for all ROIs (one time series per ROI)
2. For each target ROI's vertices, regress out the summary signals of all *other* ROIs:
   - beta = X_vertices @ X_others^T @ (X_others @ X_others^T)^{-1}
   - X_clean = X_vertices - beta @ X_others
3. The residual retains within-ROI spatial patterns while removing the linear
   contribution of other ROIs' leaked signals

This preserves the vertex-level spatial structure needed by PCA and SelectKBest
feature modes while still eliminating cross-ROI cross-talk.

### Step 4c: Pseudo-Trial Averaging (`pseudo_trials.py`)

Single-trial EEG source estimates have low signal-to-noise ratio due to spontaneous
background brain activity. Pseudo-trial averaging groups N individual trials of the
same experimental class and averages them, effectively averaging out random background
noise while preserving the time-locked stimulus-related signal. This provides the SVM
with a much cleaner, more separable decision boundary.

**Implementation**: Applied *only* to training data within each CV fold to prevent
data leakage to the test set:

1. Within each CV fold's training split, separate trials by class label
2. Randomly shuffle each class's trial order (seeded per CV repeat for reproducibility)
3. Form groups of `group_size` consecutive trials; discard incomplete remainders
4. Average the feature vectors within each group → one pseudo-trial per group
5. Test set remains individual (unaveraged) trials for unbiased evaluation

**Practical guidance**: With ~80–100 trials per class and group_size=5, this yields
~16–20 pseudo-trials per class for training. Group sizes of 5–10 are recommended;
larger groups give higher SNR but fewer training samples, which can destabilize the
SVM margin. The group size is configurable via `--pseudo-trial-size`.

### Step 5: Sliding-Window SVM Decoding (`svm_decoding.py`)

For each ROI independently:

1. Apply sliding window average (40 ms window, 5 ms step) to the ROI time series
2. At each time window center:
   - Extract features for that window across all epochs
   - Run `RepeatedStratifiedKFold` (5 splits × 5 repeats = 25 fits)
   - Classifier: `LinearSVC` (for `pca_flip`) or `Pipeline[PCA/SelectKBest → LinearSVC]` (for vertex modes)
   - Record mean cross-validated accuracy
3. Save results as CSV: columns `key` (ROI), `ms` (time), `SVM_acc` (accuracy)

### Step 6: Group-Level Statistics (`source_stats_viz.py`)

Per ROI, per stimulus class:

1. **Load** per-subject CSVs and stack into accuracy matrix (n_timepoints × n_subjects)
2. **Mean & SEM** across subjects at each time point
3. **Pointwise one-sample t-tests** against chance (0.5), one-tailed
4. **Multiple comparisons correction** (per ROI):
   - Bonferroni
   - FDR (Benjamini-Hochberg)
5. **Cluster-based permutation test** (`mne.stats.permutation_cluster_1samp_test`):
   - Clusters accuracy values centered on chance (subtract 0.5)
   - 1024 permutations, one-tailed
   - Significant clusters: p < 0.05
6. **TFCE** (Threshold-Free Cluster Enhancement):
   - Same permutation framework with `threshold=dict(start=0, step=0.2)`
   - Provides a continuous significance score at each time point
   - Significant: p < 0.05

### Step 7: Visualization (`source_stats_viz.py`)

Five plot types are generated:

1. **Per-ROI cluster plots**: SVM accuracy ± SEM with significant cluster time windows shaded
2. **Per-ROI TFCE plots**: Two subplots — accuracy (top) and TFCE scores (bottom)
3. **Multi-ROI cluster panel**: All ROIs in a 2×4 grid with cluster shading
4. **Multi-ROI TFCE panel**: Same grid with TFCE shading
5. **Source-space ERPs**: Class-averaged ROI time courses (PCA-flip) with SEM

All figures saved as `.svg` and `.png`.

---

## Source-Space vs. Sensor-Space SVM: Key Differences

| Aspect | Sensor-Space | Source-Space |
|--------|-------------|--------------|
| **Feature space** | Raw EEG channels (3-sensor custom ROIs) | Cortical source estimates within anatomical ROIs |
| **ROI definition** | Hand-picked sensor triplets (e.g., FT7/T7/TP7) | Anatomically defined cortical regions from multi-atlas parcellations (aparc, Schaefer200, HCPMMP1) |
| **Spatial resolution** | Limited by sensor spacing (~3 cm) | Vertex-level (~5 mm on cortex), though constrained by inverse accuracy |
| **Volume conduction** | Present — each sensor mixes multiple sources | Mitigated by inverse modeling (dSPM/LCMV) |
| **Forward model** | Not needed | Required (fsaverage BEM + source space) |
| **Inverse method** | None | dSPM or LCMV |
| **Feature extraction** | Average across 3 sensors per ROI | PCA-flip (1 feature), vertex PCA (~95% var), or SelectKBest (200 features) |
| **Normalization** | Per-channel `StandardScaler` before SVM | Handled by inverse normalization (dSPM: F-stat; LCMV: unit-noise-gain) |
| **CV structure** | 10-fold × 10 repeats | 5-fold × 5 repeats |
| **Classifier** | `LinearSVC` (C=1.0 default) | `LinearSVC` (same) |
| **Sliding window** | 40 ms / 5 ms step (identical) | 40 ms / 5 ms step (identical) |
| **Number of ROIs** | ~14+ custom sensor groups per hemisphere | 16 speech-network ROIs (LH; expandable to bilateral), or full atlas parcellation |
| **Anatomical interpretability** | Indirect — sensor location ≠ source location | Direct — decoding in defined cortical regions |

### Summary of Differences

The **sensor-space** approach classifies directly from scalp EEG voltages at small
hand-picked sensor clusters. It is simple and fast but subject to volume conduction
(each sensor captures a weighted mixture of many cortical sources), making anatomical
interpretation indirect.

The **source-space** approach first projects sensor data onto ~20,000 cortical
vertices using a forward model and inverse solver (dSPM or LCMV), then extracts
features from anatomically defined cortical ROIs. This provides more direct
anatomical interpretability — decoding accuracy in the "Temporal" ROI reflects
information genuinely localized to temporal cortex, not just proximity of a sensor
to that region. The trade-off is added complexity (forward model, inverse solver
choice, feature extraction strategy) and the assumptions inherent in the inverse
solution.

Both approaches use the same classification framework (LinearSVC with sliding
window), making their results directly comparable in terms of temporal dynamics.
The source-space pipeline uses slightly fewer CV folds/repeats (5×5 vs 10×10) and
offers richer feature extraction options (vertex PCA, SelectKBest) beyond simple
channel averaging.

---

## ROI Visualization (`visualize_rois.py`)

Interactive and publication-ready visualization of cortical ROIs on the fsaverage
surface. Supports all three atlases and the 16 speech-network ROIs.

- **`plot_roi_brain(atlas, fmt)`** — All ROIs on one brain (any atlas or speech-ROI subset)
- **`plot_single_roi(roi_name, atlas, fmt)`** — Single named ROI or substring filter; accepts `SPEECH_ROIS` names (e.g., `Anterior_STS`, `vSMC`)
- **`plot_compare_modes(atlas, fmt)`** — Full-resolution vs ico-5 side-by-side comparison
- Supports `--format svg` for publication figures (vector labels/axes, embedded raster brain)
- CLI flags: `--speech-rois` (all 16 ROIs), `--list-rois` (print names), `--roi NAME`

---

## Command-Line Reference

```bash
# Basic (backward compatible, aparc atlas, 8 composite ROIs)
python run_source_svm.py \
    --task {perception,overtProd} \
    --stim-class {prodDiff,percDiff} \
    --method {dSPM,LCMV} \
    [--feature-mode {pca_flip,vertex_pca,vertex_selectkbest}] \
    [--subjects EEGPROD4001 EEGPROD4003 ...] \
    [--sw-dur 40] [--sw-step 5]

# Advanced: Schaefer-200 atlas with leakage correction and pseudo-trials
python run_source_svm.py \
    --task overtProd --stim-class prodDiff --method dSPM \
    --atlas Schaefer200 \
    --leakage-correction \
    --pseudo-trial-size 5 \
    --svm-c 1.0

# Parallel (low-RAM, same args plus --n-jobs)
python run_parallel_lowram.py \
    --task overtProd --stim-class prodDiff --method dSPM \
    --atlas HCPMMP1 --leakage-correction \
    --n-jobs 2

# Validate on one subject
python validate_pipeline.py \
    [--subject EEGPROD4001] [--task overtProd] \
    [--stim-class prodDiff] [--skip-lcmv]

# ── ROI Visualization ──

# List available speech-network ROI names
python visualize_rois.py --list-rois --atlas HCPMMP1

# All 16 speech ROIs on HCP-MMP1, save as SVG
python visualize_rois.py --speech-rois --atlas HCPMMP1 --save --format svg

# Single named ROI
python visualize_rois.py --roi Anterior_STS --atlas HCPMMP1 --save --format svg
python visualize_rois.py --roi vSMC --atlas Schaefer200 --save

# Full-resolution vs ico-5 comparison
python visualize_rois.py --mode compare --atlas Schaefer200 --save --format svg
```
