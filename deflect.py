"""
DeFleCT ROI virtual channels (Hauk & Stenroos 2014, "Design of Flexible
Cross-Talk functions").

Instead of a whole-brain inverse followed by ROI extraction, each ROI gets a
small bank of sensor-space spatial filters designed directly for it:

    W_i (k x n_ch)  with   W_i @ [U_i, N] = [I_k, 0]

* U_i (n_ch x k)   the ROI's own k leading leadfield topographies (left
                   singular vectors of the ROI's block of the fixed-orientation
                   leadfield G).  Channel c of the ROI passes topography c with
                   unit gain and gives zero output to the other k-1.
* N  (n_ch x m)    the nulled topographies: the leading ``null_comp``
                   topographies of every null region (the other ROIs, and/or
                   the HCPMMP1 sensorimotor strip), to which every channel of
                   the ROI gives zero output.
* elsewhere the output power w^T (G G^T + C/snr^2) w is minimal, which keeps
  the cross-talk function compact where it is not constrained and bounds the
  sensor-noise gain (the minimum-norm prior; C = baseline noise covariance,
  scaled to the leadfield's trace).

Closed form (one solve per ROI):  W_i^T = M^-1 F (F^T M^-1 F)^-1 D,
F = [U_i, N], D = [I_k; 0].

The bank is applied to the (average-referenced) sensor epochs, so the cache
holds k time courses per ROI: ``(n_epochs, k, n_times)`` — the same layout as
the vertex caches with k instead of n_vertices, which is why the decoder and
the GC stage read it unchanged (``feature_mode`` deflect = no feature
selection; GC n_pcs <= k).

What it answers that the vertex features cannot: the vertex block of an ROI
passes 10-16 sensor directions that overlap 50-80 % with every other ROI's,
so vertex decoding is inverse- and ROI-agnostic.  A DeFleCT bank passes the
ROI's own directions with the other ROIs' directions removed; a class
difference that survives it is the ROI's own.  For GC the nulls remove the
zero-lag shared component that produces spurious directed estimates.

Design parameters come from the feature-mode string (config.parse_feature_mode)
and the assumed SNR from the --method string (config.parse_method).
"""
import numpy as np
import mne

from config import DEFLECT_SM_PARCELS


# ─────────────────────────────────────────────────────────────────────
# Leadfield helpers
# ─────────────────────────────────────────────────────────────────────
def leadfield_fixed(fwd, info):
    """Fixed-orientation leadfield (n_ch x n_src) in ``info`` channel order,
    with the average-reference projector applied; plus the source vertices
    per hemisphere and the projector P itself (n_ch x n_ch)."""
    fwd_f = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                         use_cps=True, verbose=False)
    fwd_f = mne.pick_channels_forward(fwd_f, include=info['ch_names'],
                                      ordered=True, verbose=False)
    G = np.asarray(fwd_f['sol']['data'], float)
    try:                                   # MNE >= 1.6
        from mne._fiff.proj import make_projector
    except ImportError:                    # older MNE
        from mne.io.proj import make_projector
    P, _, _ = make_projector(info['projs'], info['ch_names'])
    G = P @ G
    verts = [s['vertno'].copy() for s in fwd_f['src']]
    return G, verts, P


def label_rows(label, vertices_lh, vertices_rh):
    """Row indices (stacked lh+rh source list) of a label's vertices."""
    if label.hemi == 'lh':
        return np.where(np.isin(vertices_lh, label.vertices))[0]
    return len(vertices_lh) + np.where(np.isin(vertices_rh, label.vertices))[0]


def patch_topographies(G, rows, k):
    """First k left singular vectors of the patch leadfield G[:, rows]
    (dominant field patterns of the patch), the first sign-aligned with the
    mean column.  k is capped at the patch's numerical rank."""
    U, s, _ = np.linalg.svd(G[:, rows], full_matrices=False)
    k_eff = int(min(k, (s > s[0] * 1e-8).sum()))
    U = U[:, :k_eff].copy()
    if U[:, 0] @ G[:, rows].mean(axis=1) < 0:
        U[:, 0] *= -1
    return U


def deflect_Minv(G, C, snr):
    """pinv(G G^T + C_s / snr^2), C_s = C scaled to the trace of G G^T."""
    GG = G @ G.T
    Cs = C * (np.trace(GG) / np.trace(C))
    return np.linalg.pinv(GG + Cs / snr ** 2, rcond=1e-12, hermitian=True)


def deflect_filter_bank(G, Minv, rows_pass, null_rows, k, null_comp):
    """Filter bank W (k_eff x n_ch) for one ROI.

    rows_pass  source rows of the ROI
    null_rows  list of row arrays, one per null region (may be empty)
    Returns (W, U_pass, n_null_patterns)."""
    U = patch_topographies(G, rows_pass, k)
    k_eff = U.shape[1]
    N = [patch_topographies(G, r, null_comp) for r in null_rows if len(r)]
    F = np.hstack([U] + N) if N else U
    n_null = F.shape[1] - k_eff
    D = np.zeros((F.shape[1], k_eff))
    D[:k_eff, :k_eff] = np.eye(k_eff)
    A = F.T @ Minv @ F
    A = A + 1e-10 * np.trace(A) / len(A) * np.eye(len(A))
    W = (Minv @ F @ np.linalg.solve(A, D)).T          # (k_eff, n_ch)
    return W, U, n_null


# ─────────────────────────────────────────────────────────────────────
# Design for one subject
# ─────────────────────────────────────────────────────────────────────
def sensorimotor_label(subjects_dir):
    """HCPMMP1 sensorimotor strip (areas 4, 3a, 3b, 1, 2, left) as one Label."""
    from forward_model import build_roi_labels
    comp = {'Sensorimotor': [f'L_{p}_ROI-lh' for p in DEFLECT_SM_PARCELS]}
    return build_roi_labels(subjects_dir, atlas='HCPMMP1',
                            composite_rois=comp)['Sensorimotor']


def design_filters(G, verts, C, snr, roi_labels, roi_names, k, null_comp,
                   nulls, sm_label=None, verbose=True):
    """Design the DeFleCT bank for every ROI.

    Returns dict with
      W        {roi: (k_eff, n_ch)}
      diag     {roi: dict(k=.., n_null=.., se_own=[per channel], noise_gain=[..],
                          max_xtalk_cos=.. )}
    se_own    share of each channel's CTF energy on the ROI's own vertices
    noise_gain  sqrt(w^T C w) / mean |CTF| over own vertices (sensor noise SD
                reaching the channel per unit own-source gain)
    max_xtalk_cos  largest |cos| between this ROI's channel-1 CTF and any other
                   ROI's channel-1 CTF (the expected zero-lag correlation).
    """
    rows = {n: label_rows(l, verts[0], verts[1]) for n, l in zip(roi_names, roi_labels)}
    sm_rows = label_rows(sm_label, verts[0], verts[1]) if sm_label is not None else None
    Minv = deflect_Minv(G, C, snr)
    n_ch = G.shape[0]
    W, diag, ctf1 = {}, {}, {}
    for name in roi_names:
        own = rows[name]
        null_rows = []
        if nulls in ('other_rois', 'other_rois_sm'):
            for o in roi_names:
                if o == name:
                    continue
                r = np.setdiff1d(rows[o], own)
                if len(r) < len(rows[o]) and verbose:
                    print(f'    {name}: null region {o} shares '
                          f'{len(rows[o]) - len(r)} vertices with it; '
                          f'those are left out of the null')
                null_rows.append(r)
        if nulls in ('sm', 'other_rois_sm') and sm_rows is not None:
            r = np.setdiff1d(sm_rows, own)
            if len(r) < len(sm_rows) and verbose:
                print(f'    {name}: sensorimotor null shares '
                      f'{len(sm_rows) - len(r)} vertices with it; left out')
            null_rows.append(r)
        Wi, U, n_null = deflect_filter_bank(G, Minv, own, null_rows, k, null_comp)
        if Wi.shape[0] + n_null >= n_ch - 1:
            raise ValueError(f'{name}: {Wi.shape[0]} pass + {n_null} null '
                             f'constraints for {n_ch} channels is over-determined')
        ctf = Wi @ G                                       # (k_eff, n_src)
        e = ctf ** 2
        se_own = e[:, own].sum(axis=1) / e.sum(axis=1)
        own_gain = np.abs(ctf[:, own]).mean(axis=1)
        ng = np.sqrt(np.einsum('ij,jk,ik->i', Wi, C, Wi).clip(0)) / own_gain
        W[name] = Wi
        ctf1[name] = ctf[0]
        diag[name] = dict(k=int(Wi.shape[0]), n_null=int(n_null),
                          se_own=se_own, noise_gain=ng)
    for name in roi_names:
        a = ctf1[name]
        cos = [abs(a @ ctf1[o]) / (np.linalg.norm(a) * np.linalg.norm(ctf1[o]))
               for o in roi_names if o != name]
        diag[name]['max_xtalk_cos'] = float(max(cos)) if cos else 0.0
    if verbose:
        print(f'  DeFleCT bank: snr {snr:g}, k {k}, {null_comp} pattern(s) per '
              f'null region, nulls = {nulls}')
        for name in roi_names:
            d = diag[name]
            print(f'    {name:18s} k={d["k"]} nulls={d["n_null"]:2d}  '
                  f'se_own ch1 {d["se_own"][0]:.2f} (mean {d["se_own"].mean():.2f})  '
                  f'noise gain ch1 {d["noise_gain"][0]:.3g}  '
                  f'max |cos| vs other ROIs {d["max_xtalk_cos"]:.2f}')
    return dict(W=W, diag=diag)


def apply_filter_bank(epochs, W, P):
    """Apply the bank to the average-referenced sensor data.

    Returns {roi: (n_epochs, k, n_times) float32}.  P is the projector the
    leadfield was built with; applying it on the filter side (P w) is the
    same as projecting the data and keeps the data untouched."""
    X = epochs.get_data(copy=False)                     # (n_ep, n_ch, n_t)
    out = {}
    for name, Wi in W.items():
        Wp = Wi @ P                                     # P symmetric
        out[name] = np.einsum('kc,ect->ekt', Wp, X).astype(np.float32)
    return out


def run_deflect(epochs, fwd, baseline_tmin, baseline_tmax, roi_labels,
                roi_names, snr, k, null_comp, nulls, sm_label=None):
    """Full per-subject path: noise covariance -> design -> apply.

    Returns (roi_data {roi: (n_ep, k, n_t)}, times, design dict with the
    filters, diagnostics and channel names)."""
    from inverse_pipelines import make_noise_cov, _compute_data_rank
    rank = _compute_data_rank(epochs)
    noise_cov = make_noise_cov(epochs, baseline_tmin, baseline_tmax, rank)
    ch = list(epochs.ch_names)
    idx = [noise_cov.ch_names.index(c) for c in ch]
    C = np.asarray(noise_cov.data, float)[np.ix_(idx, idx)]
    G, verts, P = leadfield_fixed(fwd, epochs.info)
    design = design_filters(G, verts, C, snr, roi_labels, roi_names, k,
                            null_comp, nulls, sm_label=sm_label)
    roi_data = apply_filter_bank(epochs, design['W'], P)
    design['ch_names'] = ch
    design['snr'] = snr
    return roi_data, epochs.times.copy(), design
