"""
Configuration for EEG source estimation + SVM decoding pipeline.

All paths, subject IDs, ROI definitions, and pipeline parameters are
centralized here so that the main scripts stay clean.

Machine-specific paths are read from ``config.env`` (not tracked by git).
Copy ``config.env.example`` → ``config.env`` and set EEG_PROJECT_ROOT.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load config.env from the same directory as this file
load_dotenv(Path(__file__).resolve().parent / 'config.env')

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ['EEG_PROJECT_ROOT'])
EEGLAB_DIR = PROJECT_ROOT / 'derivatives' / 'EEGLAB'
DECODE_OUTPUT_ROOT = PROJECT_ROOT / 'derivatives' / 'source_estimation' / 'DECODE_source_space'
ROI_TIMESERIES_ROOT = PROJECT_ROOT / 'derivatives' / 'source_estimation' / 'DECODE_source_space_timeseries'
FIGURES_ROOT = PROJECT_ROOT / 'derivatives' / 'source_estimation' / 'SOURCE_ESTIMATION'
CODE_DIR = PROJECT_ROOT / 'code' / 'source_estimation'

# Optional external locations for ROI timeseries (e.g. external HDDs).
# Set ROI_TIMESERIES_EXTERNAL in config.env to one or more colon-
# separated paths.  When loading cached .npz files, the pipeline checks
# ROI_TIMESERIES_ROOT first, then each external root in order.  New
# files are always saved to ROI_TIMESERIES_ROOT.
_ext = os.environ.get('ROI_TIMESERIES_EXTERNAL', '')
ROI_TIMESERIES_EXTERNAL = [Path(p) for p in _ext.split(':') if p.strip()]

# Destination for NEW ROI-timeseries caches.  The vertex caches are large
# (~2.5 GB/subject); a small project drive overflows and writes TRUNCATE
# silently.  Set ROI_TIMESERIES_SAVE_ROOT in config.env to a roomy volume
# (e.g. the big external drive) to send new caches there instead.  Defaults to
# ROI_TIMESERIES_ROOT (previous behavior).  find_cached_npz searches this
# location too, so redirected caches are still found on read.
_save_root = os.environ.get('ROI_TIMESERIES_SAVE_ROOT', '').strip()
ROI_TIMESERIES_SAVE_ROOT = Path(_save_root) if _save_root else ROI_TIMESERIES_ROOT

# ─────────────────────────────────────────────────────────────────────
# Subjects
# ─────────────────────────────────────────────────────────────────────
SUBJECT_IDS = [
    'EEGPROD4001', 'EEGPROD4003', 'EEGPROD4004',
    'EEGPROD4005', 'EEGPROD4006', 'EEGPROD4007',
    'EEGPROD4008', 'EEGPROD4009', 'EEGPROD4010',
    'EEGPROD4011', 'EEGPROD4013', 'EEGPROD4014',
    'EEGPROD4015', 'EEGPROD4016', 'EEGPROD4018',
    'EEGPROD4019', 'EEGPROD4020', 'EEGPROD4021',
    'EEGPROD4022', 'EEGPROD4023', ]

# SUBJECT_IDS = ['EEGPROD4001']
# ─────────────────────────────────────────────────────────────────────
# Task / stimulus class definitions
# ─────────────────────────────────────────────────────────────────────
# Word classes grouped by initial phoneme
PROD_DIFF_TH = ['THEEP', 'THIPE', 'THEEN', 'THOPE', 'THUP']
PROD_DIFF_F = ['FEEP', 'FIPE', 'FEEN', 'FOPE', 'FUP']
PERC_DIFF_S = ['SARG', 'SOOG', 'SAFF', 'SILP', 'SEEB']
PERC_DIFF_T = ['TARG', 'TOOG', 'TAFF', 'TILP', 'TEEB']
COMPLETE_WORD_LIST = PROD_DIFF_TH + PROD_DIFF_F + PERC_DIFF_S + PERC_DIFF_T

# ─────────────────────────────────────────────────────────────────────
# Data paths per task condition
# ─────────────────────────────────────────────────────────────────────
def get_perception_data_path(subj_id, fs=2000):
    """Return path to the perception .mat file (average_ref, popthresh120).

    ``fs`` selects the sampling-rate directory: 2000 (native, default) or
    500 (continuous-resampled, ``*_500Hz_reSample_*``).  The fs=2000 branch
    is unchanged from the original single-argument behavior.
    """
    base = EEGLAB_DIR / 'perception' / subj_id / 'average_ref' / 'eeglab_standard' / f'fs_{fs}'
    for f in base.iterdir():
        if not (f.name.endswith('popthresh120.mat') and 'good_trials' not in f.name):
            continue
        if fs != 2000 and '500Hz_reSample' not in f.name:
            continue
        return f
    raise FileNotFoundError(f'No popthresh120.mat (fs={fs}) found for {subj_id} in {base}')


def get_production_data_path(subj_id, fs=2000):
    """Return path to the production .mat file (average_ref, ProdOnset).

    ``fs`` selects the sampling-rate directory: 2000 (native, default) or
    500 (continuous-resampled).  The fs_500 directory holds several epoch
    variants, so for fs=500 we additionally require the reSample tag and the
    same -1.5–0.4 s epoch as the 2000 Hz default.  The fs=2000 branch is
    unchanged from the original single-argument behavior.
    """
    base = EEGLAB_DIR / 'overtProd' / subj_id / 'average_ref' / f'fs_{fs}'
    if fs == 2000:
        for f in base.iterdir():
            if f.name.endswith('ProdOnset.mat'):
                return f
    else:
        for f in base.iterdir():
            if (f.name.endswith('ProdOnset.mat') and 'good_trials' not in f.name
                    and '500Hz_reSample' in f.name and '-1.5_0.4' in f.name):
                return f
    raise FileNotFoundError(f'No ProdOnset.mat (fs={fs}) found for {subj_id} in {base}')


# ─────────────────────────────────────────────────────────────────────
# Atlas configuration
# ─────────────────────────────────────────────────────────────────────
ATLAS = 'aparc'  # backward-compatible default
# Supported: 'aparc', 'HCPMMP1', 'Schaefer200', 'custom'
# 'aparc' with no other flags uses composite ROIs (backward compat)
# 'custom' loads volumetric NIfTI masks from CUSTOM_ROI_DIR

ATLAS_PARC_MAP = {
    'aparc': 'aparc',
    'HCPMMP1': 'HCPMMP1',
    'Schaefer200': 'Schaefer2018_200Parcels_17Networks_order',
    # 'custom' is handled separately (volumetric NIfTI → surface projection)
}

# ─────────────────────────────────────────────────────────────────────
# Custom volumetric ROIs (functional-localizer based)
# ─────────────────────────────────────────────────────────────────────
# Lab-derived ROIs from Phil & Lillian's language functional localizers,
# 15mm spheres with anatomical restriction.  Binary NIfTI masks in MNI
# space, projected onto fsaverage surface at pipeline runtime.
CUSTOM_ROI_DIR = (
    PROJECT_ROOT / 'derivatives' / 'ROIs'
    / 'functional_localizer_language_ROIs' / 'rois_15mm_anatrestrict_final'
)

# Readable names for the 6 custom ROIs
CUSTOM_ROI_NAMES = [
    'awfa',   # auditory word form area (audioLoc)
    'ifc',    # inferior frontal cortex (bothLoc)
    'owfa',   # orthographic word form area (vwfaLoc)
    'pmc',    # premotor cortex (audioLoc)
    'tpc',    # temporo-parietal cortex (bothLoc)
    'vwfa',   # visual word form area (vwfaLoc)
]

# ─────────────────────────────────────────────────────────────────────
# Speech-network ROIs — atlas-specific parcel definitions
# ─────────────────────────────────────────────────────────────────────
# 16 ROIs spanning the dual-stream speech processing network (LH only).
# Motivated by: sensor-space SVM ROIs (FT7/T7/TP7, F5/FC5/FC3,
# F1/FC1/FCz, CPz/CP1/P1), Chang, Hickok & Poeppel, Flinker, Tian,
# Rauschecker & Scott, Glasser.
#
# Access: SPEECH_ROIS[atlas][roi_name] → list of native parcel names.
# Parcels may overlap between ROIs (each ROI is decoded independently).
# For RH analogues: 'L_'→'R_', '-lh'→'-rh' (HCPMMP1/Schaefer);
#                   '-lh'→'-rh' (aparc).
SPEECH_ROIS = {
    # ── Desikan-Killiany (aparc) ──────────────────────────────────────
    # NOTE: aparc is too coarse to separate several sub-regions (e.g.
    # anterior vs posterior STS, planum temporale).  Overlaps are
    # unavoidable; use HCPMMP1 or Schaefer200 for finer resolution.
    'aparc': {
        # Sensor-space ROI equivalents
        'Temporal': [
            'superiortemporal-lh', 'middletemporal-lh', 'bankssts-lh',
        ],
        'Inferior_Frontal': [
            'parsopercularis-lh', 'parstriangularis-lh',
        ],
        'Superior_Frontal': [
            'superiorfrontal-lh',
        ],
        'Superior_Parietal': [
            'superiorparietal-lh', 'precuneus-lh',
        ],
        # Additional speech-network ROIs
        'vSMC': [
            'precentral-lh', 'postcentral-lh',
        ],
        'Supramarginal': [
            'supramarginal-lh',
        ],
        'Angular_Gyrus': [
            'inferiorparietal-lh',
        ],
        'Insula': [
            'insula-lh',
        ],
        'TPOJ': [
            'bankssts-lh', 'supramarginal-lh', 'inferiorparietal-lh',
        ],
        'Cingulate_Motor': [
            'caudalanteriorcingulate-lh', 'posteriorcingulate-lh',
        ],
        # Rauschecker & Scott
        'Planum_Temporale': [
            'superiortemporal-lh', 'transversetemporal-lh',
        ],
        # 'Anterior_STS': [
        #     'superiortemporal-lh', 'bankssts-lh',
        # ],
        # 'Temporal_Pole': [
        #     'temporalpole-lh',
        # ],
        'Pars_Orbitalis': [
            'parsorbitalis-lh',
        ],
        # 'Posterior_STS': [
        #     'superiortemporal-lh', 'bankssts-lh',
        # ],
        'DLPFC': [
            'caudalmiddlefrontal-lh', 'rostralmiddlefrontal-lh',
        ],
    },

    # ── HCP Multi-Modal Parcellation (Glasser et al. 2016) ───────────
    'HCPMMP1': {
        # Sensor-space ROI equivalents
        'Temporal': [
            'L_A1_ROI-lh', 'L_LBelt_ROI-lh', 'L_MBelt_ROI-lh',
            'L_PBelt_ROI-lh', 'L_A4_ROI-lh', 'L_A5_ROI-lh',
            'L_STGa_ROI-lh', 'L_TA2_ROI-lh',
            'L_STSda_ROI-lh', 'L_STSdp_ROI-lh',
            'L_STSva_ROI-lh', 'L_STSvp_ROI-lh', 'L_PSL_ROI-lh',
        ],
        'Inferior_Frontal': [
            'L_44_ROI-lh', 'L_45_ROI-lh',
            'L_IFJa_ROI-lh', 'L_IFJp_ROI-lh',
            'L_IFSa_ROI-lh', 'L_IFSp_ROI-lh',
            'L_FOP4_ROI-lh', 'L_FOP5_ROI-lh',
        ],
        'Superior_Frontal': [
            'L_6ma_ROI-lh', 'L_6mp_ROI-lh',
            'L_6a_ROI-lh', 'L_6d_ROI-lh',
            'L_SFL_ROI-lh', 'L_8BM_ROI-lh', 'L_8Ad_ROI-lh',
        ],
        'Superior_Parietal': [
            'L_7AL_ROI-lh', 'L_7Am_ROI-lh', 'L_7PC_ROI-lh',
            'L_7PL_ROI-lh', 'L_7Pm_ROI-lh', 'L_7m_ROI-lh',
            'L_5L_ROI-lh', 'L_5m_ROI-lh', 'L_5mv_ROI-lh',
            'L_MIP_ROI-lh', 'L_AIP_ROI-lh', 'L_PCV_ROI-lh',
        ],
        # Chang — ventral sensorimotor cortex
        'vSMC': [
            'L_4_ROI-lh', 'L_3a_ROI-lh', 'L_3b_ROI-lh',
            'L_1_ROI-lh', 'L_2_ROI-lh',
            'L_6v_ROI-lh', 'L_43_ROI-lh', 'L_OP4_ROI-lh',
        ],
        # Hickok, Poeppel, Flinker — supramarginal / phonological
        'Supramarginal': [
            'L_PFm_ROI-lh', 'L_PF_ROI-lh', 'L_PFop_ROI-lh',
            'L_PFt_ROI-lh', 'L_PFcm_ROI-lh',
        ],
        # Poeppel, Tian — angular gyrus / semantic integration
        'Angular_Gyrus': [
            'L_PGi_ROI-lh', 'L_PGs_ROI-lh', 'L_PGp_ROI-lh',
        ],
        # Flinker, Dronkers — insula / articulatory planning
        'Insula': [
            'L_Ig_ROI-lh', 'L_MI_ROI-lh', 'L_AVI_ROI-lh',
            'L_AAIC_ROI-lh', 'L_PoI1_ROI-lh', 'L_PoI2_ROI-lh',
            'L_FOP1_ROI-lh', 'L_FOP2_ROI-lh', 'L_FOP3_ROI-lh',
        ],
        # Glasser, Poeppel — temporo-parieto-occipital junction
        'TPOJ': [
            'L_TPOJ1_ROI-lh', 'L_TPOJ2_ROI-lh', 'L_TPOJ3_ROI-lh',
        ],
        # Tian — cingulate motor / speech initiation
        'Cingulate_Motor': [
            'L_24dd_ROI-lh', 'L_24dv_ROI-lh',
            'L_a24pr_ROI-lh', 'L_p24pr_ROI-lh',
            'L_SCEF_ROI-lh',
        ],
        # Rauschecker & Scott — planum temporale (stream hub)
        'Planum_Temporale': [
            'L_A1_ROI-lh', 'L_LBelt_ROI-lh', 'L_MBelt_ROI-lh',
            'L_RI_ROI-lh', 'L_PBelt_ROI-lh',
            'L_52_ROI-lh', 'L_A4_ROI-lh',
        ],
        # # Scott et al. — anterior STS (intelligible speech)
        # 'Anterior_STS': [
        #     'L_STSda_ROI-lh', 'L_STSva_ROI-lh',
        #     'L_STGa_ROI-lh', 'L_TA2_ROI-lh',
        # ],
        # # Rauschecker — temporal pole (ventral stream terminus)
        # 'Temporal_Pole': [
        #     'L_TGd_ROI-lh', 'L_TGv_ROI-lh',
        # ],
        # Scott & Eisner — pars orbitalis / BA 47
        'Pars_Orbitalis': [
            'L_47l_ROI-lh', 'L_a47r_ROI-lh', 'L_p47r_ROI-lh',
        ],
        # Rauschecker & Scott — posterior STS (acoustic-phonetic)
        # 'Posterior_STS': [
        #     'L_STSdp_ROI-lh', 'L_STSvp_ROI-lh',
        #     'L_PHT_ROI-lh', 'L_PSL_ROI-lh',
        # ],
        # Rauschecker — dorsolateral PFC (sequence storage)
        'DLPFC': [
            'L_8C_ROI-lh', 'L_8Av_ROI-lh',
            'L_i6-8_ROI-lh', 'L_s6-8_ROI-lh',
            'L_9-46d_ROI-lh', 'L_46_ROI-lh',
        ],
    },

    # ── Schaefer 2018, 200 parcels, 17 networks ─────────────────────
    # Parcels are defined by functional connectivity, not anatomy.
    # Some overlap with broad ROIs (e.g. Temporal) is expected.
    'Schaefer200': {
        # Sensor-space ROI equivalents
        'Temporal': [
            '17Networks_LH_SomMotB_Aud_1-lh',
            '17Networks_LH_SomMotB_Aud_2-lh',
            '17Networks_LH_SomMotB_Aud_3-lh',
            '17Networks_LH_TempPar_1-lh',
            '17Networks_LH_TempPar_2-lh',
            # '17Networks_LH_DefaultB_Temp_1-lh',
            # '17Networks_LH_DefaultB_Temp_2-lh',
            '17Networks_LH_DefaultB_Temp_3-lh',
            '17Networks_LH_DefaultB_Temp_4-lh',
            # '17Networks_LH_ContA_Temp_1-lh',
            # '17Networks_LH_ContB_Temp_1-lh',
        ],
        'Inferior_Frontal': [
            '17Networks_LH_SalVentAttnA_FrOper_1-lh',
            '17Networks_LH_SalVentAttnA_FrOper_2-lh',
            '17Networks_LH_ContA_PFClv_1-lh',
            '17Networks_LH_ContB_PFClv_1-lh',
            '17Networks_LH_ContB_PFClv_2-lh',
        ],
        'Superior_Frontal': [
            '17Networks_LH_SomMotA_1-lh',
            '17Networks_LH_SomMotA_2-lh',
            '17Networks_LH_SomMotA_3-lh',
            '17Networks_LH_SomMotA_4-lh',
            '17Networks_LH_SalVentAttnA_FrMed_1-lh',
            '17Networks_LH_SalVentAttnA_FrMed_2-lh',
            '17Networks_LH_ContA_PFCd_1-lh',
        ],
        'Superior_Parietal': [
            '17Networks_LH_DorsAttnA_SPL_1-lh',
            '17Networks_LH_DorsAttnA_SPL_2-lh',
            '17Networks_LH_DorsAttnA_SPL_3-lh',
            '17Networks_LH_DorsAttnA_ParOcc_1-lh',
            '17Networks_LH_DorsAttnB_PostC_1-lh',
            '17Networks_LH_DorsAttnB_PostC_2-lh',
            '17Networks_LH_DorsAttnB_PostC_3-lh',
            '17Networks_LH_DorsAttnB_PostC_4-lh',
            '17Networks_LH_ContC_pCun_1-lh',
            '17Networks_LH_ContC_pCun_2-lh',
        ],
        # Chang — ventral sensorimotor cortex
        'vSMC': [
            '17Networks_LH_SomMotB_Cent_1-lh',
            '17Networks_LH_SomMotB_Cent_2-lh',
            '17Networks_LH_SomMotB_S2_1-lh',
            '17Networks_LH_SomMotB_S2_2-lh',
            '17Networks_LH_SomMotB_S2_3-lh',
        ],
        # Hickok, Poeppel, Flinker — supramarginal
        'Supramarginal': [
            '17Networks_LH_SalVentAttnA_ParOper_1-lh',
            '17Networks_LH_SalVentAttnB_IPL_1-lh',
        ],
        # Poeppel, Tian — angular gyrus
        'Angular_Gyrus': [
            '17Networks_LH_DefaultA_IPL_1-lh',
            '17Networks_LH_DefaultB_IPL_1-lh',
            '17Networks_LH_DefaultC_IPL_1-lh',
        ],
        # Flinker, Dronkers — insula
        'Insula': [
            '17Networks_LH_SalVentAttnA_Ins_1-lh',
            '17Networks_LH_SalVentAttnB_Ins_1-lh',
        ],
        # Glasser, Poeppel — TPOJ
        'TPOJ': [
            '17Networks_LH_TempPar_1-lh',
            '17Networks_LH_TempPar_2-lh',
        ],
        # Tian — cingulate motor
        'Cingulate_Motor': [
            '17Networks_LH_SalVentAttnA_FrMed_1-lh',
            '17Networks_LH_SalVentAttnA_FrMed_2-lh',
            '17Networks_LH_SalVentAttnA_ParMed_1-lh',
            '17Networks_LH_ContA_Cingm_1-lh',
        ],
        # Rauschecker & Scott — planum temporale
        'Planum_Temporale': [
            '17Networks_LH_SomMotB_Aud_1-lh',
            '17Networks_LH_SomMotB_Aud_2-lh',
            '17Networks_LH_SomMotB_Aud_3-lh',
        ],
        # # Scott et al. — anterior STS
        # 'Anterior_STS': [
        #     '17Networks_LH_DefaultB_Temp_1-lh',
        #     '17Networks_LH_DefaultB_Temp_2-lh',
        #     '17Networks_LH_DefaultB_Temp_3-lh',
        #     '17Networks_LH_DefaultB_Temp_4-lh',
        # ],
        # Rauschecker — temporal pole
        # 'Temporal_Pole': [
        #     '17Networks_LH_LimbicA_TempPole_1-lh',
        #     '17Networks_LH_LimbicA_TempPole_2-lh',
        #     '17Networks_LH_LimbicA_TempPole_3-lh',
        #     '17Networks_LH_LimbicA_TempPole_4-lh',
        # ],
        # Scott & Eisner — pars orbitalis / ventral PFC
        'Pars_Orbitalis': [
            '17Networks_LH_DefaultB_PFCv_1-lh',
            '17Networks_LH_DefaultB_PFCv_2-lh',
            '17Networks_LH_DefaultB_PFCv_3-lh',
            '17Networks_LH_DefaultB_PFCv_4-lh',
        ],
        # Rauschecker & Scott — posterior STS
        # 'Posterior_STS': [
        #     '17Networks_LH_DorsAttnA_TempOcc_1-lh',
        #     # '17Networks_LH_DorsAttnA_TempOcc_2-lh',
        # ],
        # Rauschecker — DLPFC
        'DLPFC': [
            '17Networks_LH_ContA_PFCl_1-lh',
            '17Networks_LH_ContA_PFCl_2-lh',
            '17Networks_LH_ContA_PFCl_3-lh',
            '17Networks_LH_ContA_PFCd_1-lh',
            '17Networks_LH_ContB_PFCl_1-lh',
            '17Networks_LH_DefaultA_PFCd_1-lh',
        ],
    },
}

# Ordered list of speech ROI names (for consistent iteration)
SPEECH_ROI_NAMES = list(SPEECH_ROIS['aparc'].keys())

# ─────────────────────────────────────────────────────────────────────
# Decoding enhancement parameters
# ─────────────────────────────────────────────────────────────────────
# Per-classifier default C, selected as the modal pick across the full
# explore_decoding hyperparameter sweep (6 ROIs × 20 subjects × 3 sw_durs
# × 2 contrasts, HCPMMP1 / vertex_selectkbest, May 2026):
#   - svm:      C=0.01  (51% modal share, 2.6× the next contender)
#   - logistic: C=0.1   (35% pooled, narrowly beats C=10 at 33%)
DEFAULT_C = {'svm': 0.01, 'logistic': 0.1}
PSEUDO_TRIAL_SIZE = 0      # 0 = disabled; 5 or 10 recommended when enabled
LEAKAGE_CORRECTION = False # orthogonalization (pca_flip) or regression (vertex modes)

# ─────────────────────────────────────────────────────────────────────
# Sliding-window SVM parameters (matching existing sensor-space pipeline)
# ─────────────────────────────────────────────────────────────────────
SW_DUR = 40          # sliding window duration in ms
SW_STEP_SIZE = 5     # step size in ms
N_CV_FOLDS = 5       # cross-validation folds
N_CV_REPEATS = 5     # number of CV repeats (matches existing pipeline)

# ─────────────────────────────────────────────────────────────────────
# Source estimation parameters
# ─────────────────────────────────────────────────────────────────────
SNR = 3.0
LAMBDA2 = 1.0 / SNR ** 2  # regularization for MNE/dSPM

# Epoch parameters (in seconds)
PERCEPTION_TMIN = -0.2
PERCEPTION_TMAX = 0.6
PRODUCTION_TMIN = -1.6
PRODUCTION_TMAX = 0.4

# ─────────────────────────────────────────────────────────────────────
# Pre-stimulus baselines for the inverse NOISE COVARIANCE (in seconds).
# The window MUST lie inside the loaded epoch — otherwise mne.compute_covariance
# collapses onto ~1 sample and the noise covariance is degenerate, which
# destabilises the beamformer.  The exports load as:
#   perception [-0.2, 0.6] s -> baseline -0.2..-0.1 (pre-stimulus)
#   overtProd  [-1.5, 0.4] s -> baseline -1.5..-1.4 (earliest pre-production quiet)
# There is NO -1.6 s overtProd epoch, so the old (-1.6, -1.5) sat entirely before
# the epoch start (1 sample) — fixed below.  resolve_noise_baseline() re-validates
# this against each subject's ACTUAL epoch at runtime and logs the window used.
# ─────────────────────────────────────────────────────────────────────
BASELINE_WINDOWS = {
    'perception': (-0.200, -0.100),   # -200 to -100 ms (epoch -0.2..0.6)
    'overtProd':  (-1.500, -1.400),   # -1500 to -1400 ms (epoch -1.5..0.4)
}

# SVM decoding starts AFTER the baseline ends (in seconds)
DECODE_TMIN = {
    'perception': -0.100,   # start decoding at -100 ms
    'overtProd':  -1.500,   # start decoding at -1500 ms
}


def resolve_noise_baseline(task_cond, epoch_tmin, epoch_tmax, tol=1e-6):
    """Noise-covariance baseline guaranteed to lie inside the LOADED epoch.

    Returns ``(tmin, tmax, warning)``.  ``BASELINE_WINDOWS[task_cond]`` is the
    intended window, but if it falls outside a subject's actual epoch, MNE's
    ``compute_covariance`` collapses onto ~1 sample and the noise covariance is
    degenerate (which destabilises the beamformer).  This clamps the window into
    ``[epoch_tmin, epoch_tmax]`` (preserving its width where possible) and returns
    a non-None ``warning`` string when it had to adjust, so the caller can log it.
    """
    lo, hi = BASELINE_WINDOWS[task_cond]
    if lo >= epoch_tmin - tol and hi <= epoch_tmax + tol and lo < hi:
        return lo, hi, None
    width = (hi - lo) if hi > lo else 0.1
    nlo = min(max(lo, epoch_tmin), epoch_tmax)
    nhi = min(max(hi, epoch_tmin), epoch_tmax)
    if nhi - nlo < 0.5 * width:                      # window collapsed at an edge
        nlo = epoch_tmin
        nhi = min(epoch_tmin + width, epoch_tmax)
    warning = (f"BASELINE_WINDOWS[{task_cond!r}]=({lo:g}, {hi:g}) lies outside the "
               f"loaded epoch [{epoch_tmin:.3f}, {epoch_tmax:.3f}] s; clamped to "
               f"[{nlo:.3f}, {nhi:.3f}] s to avoid a degenerate noise covariance. "
               f"Update BASELINE_WINDOWS in config.py.")
    return nlo, nhi, warning

# ─────────────────────────────────────────────────────────────────────
# Granger-causality task-vs-baseline windows (in seconds).
# These are SEPARATE from BASELINE_WINDOWS above.  BASELINE_WINDOWS sets the
# inverse noise-covariance window; it is the WRONG window for the GC
# task-vs-baseline test because both baselines sit at the very START of the epoch,
# inside the moving-window MVAR leading-edge ramp (overtProd -1.5..-1.4 begins at
# the epoch edge; perception -0.2..-0.1 sits at the -0.2 s edge) — which would give
# spurious edge-dominated GC.
# The GC windows below sit INSIDE the epoch and past that edge ramp; GC
# task windows begin at GC_TASK_START.  The leading segment between the
# epoch start and the baseline is left out of both baseline and task.
# Override per run with granger_stats.py --baseline-start/--baseline-end/
# --task-start.
#   perception epoch -0.2..0.6 s (stimulus onset at 0)
#   overtProd  epoch -1.5..0.4 s (production onset at 0)
# ─────────────────────────────────────────────────────────────────────
GC_BASELINE_WINDOWS = {
    'perception': (-0.150, -0.050),
    'overtProd':  (-1.450, -1.350),
}
GC_TASK_START = {
    'perception': -0.050,
    'overtProd':  -1.350,
}
# GC task windows also END here (in seconds), to drop the trailing
# moving-window MVAR boundary artifact: the last window straddles the
# epoch end, so windows whose data reaches into the final win_ms are
# excluded.  Set to tmax - 2*win (one window length before the last
# window start).  Override with granger_stats.py --task-end.
#   perception epoch end +0.6 s (last window start +0.56)
#   overtProd  epoch end +0.4 s (last window start +0.36)
GC_TASK_END = {
    'perception': 0.520,
    'overtProd':  0.320,
}


def cache_feat_mode(feat_mode):
    """Normalize a feature_mode to its cache-directory name.

    All ``vertex_*`` modes share the same extracted payload
    (``(n_epochs, n_vertices, n_times)`` per ROI), so they share one
    cache directory. ``pca_flip`` remains separate.
    """
    if feat_mode.startswith('vertex'):
        return 'vertex'
    return feat_mode


def find_cached_npz(task, method, atlas, feat_mode, leakage_correction,
                    subj, stim_class):
    """Return the path to a cached .npz ROI timeseries file, or None.

    Checks ROI_TIMESERIES_ROOT first, then each path in
    ROI_TIMESERIES_EXTERNAL in order.  Also falls back to the legacy
    per-feat_mode directory name so existing ``vertex_pca`` /
    ``vertex_selectkbest`` caches keep working.
    """
    leakage_tag = 'leakage_corrected' if leakage_correction else 'raw'
    filename = f'{subj}_{task}_{stim_class}.npz'
    candidates = [cache_feat_mode(feat_mode)]
    if feat_mode not in candidates:
        candidates.append(feat_mode)

    roots = []
    for r in (ROI_TIMESERIES_ROOT, ROI_TIMESERIES_SAVE_ROOT, *ROI_TIMESERIES_EXTERNAL):
        if r not in roots:
            roots.append(r)

    for root in roots:
        for name in candidates:
            path = root / task / method / atlas / name / leakage_tag / filename
            if path.exists():
                return path
    return None


def classifier_path_segment(classifier, c, tune_hyperparams=False):
    """Path segment encoding the classifier and its hyperparameter choice.

    Inserted between the sliding-window dir (e.g., ``40_5``) and
    ``stim_class`` so swapping classifier/C does not silently overwrite
    the previous run's CSVs.

    Format
    ------
    - ``lda``                              — LDA has no tunable C.
    - ``{svm,logistic}_{c_str}``           — fixed C; ``c_str`` formats the
                                             value with ``:g`` and replaces
                                             ``.`` with ``_`` (e.g. 0.01 →
                                             ``0_01``, 1.0 → ``1``, 10 →
                                             ``10``).
    - ``{svm,logistic}_tuned``             — ``--tune-hyperparams``: C is
                                             selected per fold from the
                                             grid, so the input ``c`` does
                                             not pin the result.
    """
    if classifier == 'lda':
        return 'lda'
    if tune_hyperparams:
        return f'{classifier}_tuned'
    c_str = f'{c:g}'.replace('.', '_')
    return f'{classifier}_{c_str}'


def explore_run_segment(leakage_correction, pseudo_trial_size, c):
    """Path segment encoding the run-time params that change accuracies
    but aren't otherwise represented in the explore_decoding output path.

    Without this, switching --leakage-correction, --pseudo-trial-size,
    or --c silently overwrites previous results in the same CSV.

    Format: ``lc{0,1}_pt{N}_C{c}`` — e.g. ``lc0_pt0_C0.01`` for an
    explicit value, ``lc1_pt5_Cdef`` when ``c`` is None and per-classifier
    defaults from DEFAULT_C are in effect.  ``:g`` formatting on C strips
    trailing zeros (1.0 → C1, 0.01 → C0.01).
    """
    c_tag = 'def' if c is None else f'{c:g}'
    return f'lc{int(bool(leakage_correction))}_pt{int(pseudo_trial_size)}_C{c_tag}'
