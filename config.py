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
SVM_OUTPUT_ROOT = PROJECT_ROOT / 'derivatives' / 'SVM_source'
ROI_TIMESERIES_ROOT = PROJECT_ROOT / 'derivatives' / 'SVM_source_timeseries'
FIGURES_ROOT = PROJECT_ROOT / 'derivatives' / 'SOURCE_ESTIMATION'
CODE_DIR = PROJECT_ROOT / 'code' / 'source_estimation'

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
def get_perception_data_path(subj_id):
    """Return path to the perception .mat file (average_ref, popthresh120)."""
    base = EEGLAB_DIR / 'perception' / subj_id / 'average_ref' / 'eeglab_standard' / 'fs_2000'
    for f in base.iterdir():
        if f.name.endswith('popthresh120.mat') and 'good_trials' not in f.name:
            return f
    raise FileNotFoundError(f'No popthresh120.mat found for {subj_id} in {base}')


def get_production_data_path(subj_id):
    """Return path to the production .mat file (average_ref, ProdOnset)."""
    base = EEGLAB_DIR / 'overtProd' / subj_id / 'average_ref' / 'fs_2000'
    for f in base.iterdir():
        if f.name.endswith('ProdOnset.mat'):
            return f
    raise FileNotFoundError(f'No ProdOnset.mat found for {subj_id} in {base}')


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
        'Anterior_STS': [
            'superiortemporal-lh', 'bankssts-lh',
        ],
        'Temporal_Pole': [
            'temporalpole-lh',
        ],
        'Pars_Orbitalis': [
            'parsorbitalis-lh',
        ],
        'Posterior_STS': [
            'superiortemporal-lh', 'bankssts-lh',
        ],
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
        # Scott et al. — anterior STS (intelligible speech)
        'Anterior_STS': [
            'L_STSda_ROI-lh', 'L_STSva_ROI-lh',
            'L_STGa_ROI-lh', 'L_TA2_ROI-lh',
        ],
        # Rauschecker — temporal pole (ventral stream terminus)
        'Temporal_Pole': [
            'L_TGd_ROI-lh', 'L_TGv_ROI-lh',
        ],
        # Scott & Eisner — pars orbitalis / BA 47
        'Pars_Orbitalis': [
            'L_47l_ROI-lh', 'L_a47r_ROI-lh', 'L_p47r_ROI-lh',
        ],
        # Rauschecker & Scott — posterior STS (acoustic-phonetic)
        'Posterior_STS': [
            'L_STSdp_ROI-lh', 'L_STSvp_ROI-lh',
            'L_PHT_ROI-lh', 'L_PSL_ROI-lh',
        ],
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
        # Scott et al. — anterior STS
        'Anterior_STS': [
            '17Networks_LH_DefaultB_Temp_1-lh',
            '17Networks_LH_DefaultB_Temp_2-lh',
            '17Networks_LH_DefaultB_Temp_3-lh',
            '17Networks_LH_DefaultB_Temp_4-lh',
        ],
        # Rauschecker — temporal pole
        'Temporal_Pole': [
            '17Networks_LH_LimbicA_TempPole_1-lh',
            '17Networks_LH_LimbicA_TempPole_2-lh',
            '17Networks_LH_LimbicA_TempPole_3-lh',
            '17Networks_LH_LimbicA_TempPole_4-lh',
        ],
        # Scott & Eisner — pars orbitalis / ventral PFC
        'Pars_Orbitalis': [
            '17Networks_LH_DefaultB_PFCv_1-lh',
            '17Networks_LH_DefaultB_PFCv_2-lh',
            '17Networks_LH_DefaultB_PFCv_3-lh',
            '17Networks_LH_DefaultB_PFCv_4-lh',
        ],
        # Rauschecker & Scott — posterior STS
        'Posterior_STS': [
            '17Networks_LH_DorsAttnA_TempOcc_1-lh',
            '17Networks_LH_DorsAttnA_TempOcc_2-lh',
        ],
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
# SVM enhancement parameters
# ─────────────────────────────────────────────────────────────────────
SVM_C = 1.0                # regularization parameter for LinearSVC
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
# Pre-stimulus baselines for noise covariance (in seconds)
# These are the reference periods used ONLY for estimating noise —
# they are excluded from the SVM decoding time range.
# NOTE: These are conceptual boundaries.  MNE snaps them to the
# nearest actual sample when computing covariances.  For SVM decoding,
# the exact time vector (epochs.times / eeg_dict['times']) is used to
# determine crop indices and window-center timestamps.
# ─────────────────────────────────────────────────────────────────────
BASELINE_WINDOWS = {
    'perception': (-0.200, -0.100),   # -200 ms to -100 ms
    'overtProd':  (-1.600, -1.500),   # -1600 ms to -1500 ms
}

# SVM decoding starts AFTER the baseline ends (in seconds)
DECODE_TMIN = {
    'perception': -0.100,   # start decoding at -100 ms
    'overtProd':  -1.500,   # start decoding at -1500 ms
}
