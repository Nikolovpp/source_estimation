"""State-space Granger causality in source space, via MNE-Connectivity.

Drop-in alternative to the BSMART-parametric AR estimator in ``granger.py``.
Where ``granger.py`` fits a parametric MVAR per window and reads off Geweke's
spectral GC, this module calls
``mne_connectivity.spectral_connectivity_epochs(method=["gc","gc_tr"])`` — the
state-space GC of Barnett & Seth (2015), which derives the reduced model from
the full one instead of fitting it twice, and is the citable, order-stable
estimator we are migrating to (see ``GC_fundamentals/mne_vs_bsmart_findings.md``).

Why continuous-wavelet mode
---------------------------
MNE's GC estimates the cross-spectral density first, then factorises it. In its
FFT modes (``multitaper`` / ``fourier``) the frequency grid — and the hard
constraint ``gc_n_lags < 2 * n_frequencies`` — is set by the segment length, so
short moving windows are impossible: a 50-sample window (250 ms @ 200 Hz) has
~12 frequency bins and rejects ``gc_n_lags=25``; a 30-sample window (60 ms @
500 Hz) has ~0. ``mode="cwt_morlet"`` decouples the frequency grid from the
window: you request a fixed grid (here 4-30 Hz) and it returns *time-resolved*
GC directly, ``(n_edges, n_freqs, n_times)``. Both configs then live on the
same grid and are directly comparable band-by-band.

Mapping the study's (MO, SW, fs) knobs onto MNE
-----------------------------------------------
* ``fs``  -> resample the virtual channels to this rate (``target_fs``).
* ``MO``  -> ``gc_n_lags`` (the state-space autocovariance depth).
* ``SW``  -> the Morlet time window: ``n_cycles(f) = max(f * SW/1000, floor)``,
  so the effective analysis window is ~``SW`` ms at every frequency. The floor
  (default 1 cycle) protects the estimate where ``SW`` is shorter than one
  cycle — which is exactly where a short window *cannot* resolve a low-frequency
  band. ``under_resolved_bands`` reports those (band, config) cells so the caller
  can flag them rather than silently trusting them.

Output schema matches ``granger.compute_subject_gc`` (``fxy``/``fyx``/``dtrgc``
dicts, band -> ``(n_pairs, n_times)``), so the stats/plotting layer is shared.
Here the time axis is per-sample (cwt is time-resolved), not a moving-window
index; ``window_ms`` therefore carries the real time axis in ms.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from granger import band_average, DEFAULT_BANDS
from run_granger import resample_channels

# This estimator is ALWAYS continuous-wavelet (Morlet). MNE's FFT modes
# (multitaper/fourier) cannot run state-space GC at these window sizes — see
# ``min_fft_window_samples`` — so cwt is not a tuning choice here, it is the
# only mode that works. Everything downstream is tagged 'cwt' to make that
# explicit (``run_granger_mne.gc_tag_mne`` -> ``ssgc_cwt_...``).
MODE = 'cwt_morlet'

# Common frequency grid for both configs (Hz). Starts at 4 Hz (theta floor);
# 1 Hz spacing is plenty given band-averaging.
DEFAULT_CWT_FREQS = np.arange(4.0, 30.0 + 1e-9, 1.0)


def sw_to_ncycles(freqs, sw_ms, floor=1.0):
    """Morlet cycle count that makes the effective time window ~``sw_ms``."""
    return np.maximum(np.asarray(freqs, float) * sw_ms / 1000.0, floor)


def min_fft_window_samples(gc_n_lags, fs, fmin=4.0, fmax=30.0):
    """Smallest window (samples) an FFT-mode MNE GC would accept for these lags.

    MNE's multitaper/fourier GC enforces ``gc_n_lags < 2 * n_freq_bins``, where
    ``n_freq_bins`` is the count of rfft bins inside ``[fmin, fmax]`` for the
    window. This returns the smallest ``n_times`` satisfying it (or None up to a
    generous cap). ``cwt_morlet`` — the mode this module uses — has **no** such
    limit and runs to ~4 samples, so this is here only to explain *why* cwt is
    mandatory for short windows.
    """
    for n in range(2, 20000):
        fbins = np.fft.rfftfreq(n, d=1.0 / fs)
        n_in = int(((fbins >= fmin) & (fbins <= fmax)).sum())
        if gc_n_lags < 2 * n_in:
            return n
    return None


def min_trustworthy_window_ms(freq_lo, min_cycles=2.0):
    """Window (ms) needed to hold ``min_cycles`` cycles of ``freq_lo`` Hz.

    The practical floor for cwt GC: below this the lowest frequency in a band is
    edge-dominated. e.g. theta (4 Hz) needs ~500 ms; alpha (8 Hz) ~250 ms.
    """
    return 1000.0 * min_cycles / float(freq_lo)


def under_resolved_bands(sw_ms, bands=None, min_cycles=2.0):
    """Bands whose lowest frequency gets < ``min_cycles`` cycles in ``sw_ms``.

    A wavelet with fewer than ~2 cycles cannot cleanly isolate a rhythm, so GC
    in these bands is edge-dominated and should be read with suspicion (this is
    how the short-window config loses theta/alpha).
    """
    if bands is None:
        bands = DEFAULT_BANDS
    out = []
    for name, (lo, _hi) in bands.items():
        if lo * sw_ms / 1000.0 < min_cycles:
            out.append(name)
    return out


def reduce_roi_top_pcs(vertex_data, n_pcs):
    """Collapse an ROI's vertices to its top-``n_pcs`` PC time courses (FIXPC-k).

    Pellegrini et al. (2023): estimate the ROI's first ``k`` principal
    components ONCE from the trial ensemble (fixed spatial filters) and apply
    them to every trial, so the ROI becomes a **k-channel block** for the
    block/group-to-group GC.  ``k`` is capped at the vertex count.  This is the
    multivariate generalisation of ``granger.reduce_roi_first_pc`` (which is the
    ``k = 1`` case); FIXPC3/4 is what the source-FC literature recommends over
    FIXPC1 because a few fixed PCs + TRGC is more robust to source leakage.

    Parameters
    ----------
    vertex_data : (n_epochs, n_vertices, n_times), or (n_epochs, n_times) if the
        ROI is already a single channel (then only k = 1 is possible).
    n_pcs : int

    Returns
    -------
    (n_epochs, k, n_times), with k = min(n_pcs, n_vertices).
    """
    X = np.asarray(vertex_data, float)
    if X.ndim == 2:                                 # already a single channel
        return X[:, None, :]
    n_ep, n_v, n_t = X.shape
    k = min(int(n_pcs), n_v)
    M = np.transpose(X, (1, 0, 2)).reshape(n_v, n_ep * n_t)
    M = M - M.mean(axis=1, keepdims=True)           # PCA = SVD of mean-centred
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    W = U[:, :k]                                    # (n_v, k) fixed spatial filters
    for j in range(k):                              # deterministic sign per PC
        if W[np.argmax(np.abs(W[:, j])), j] < 0:
            W[:, j] = -W[:, j]
    return np.einsum('vk,evt->ekt', W, X)           # (n_ep, k, n_t)


def timefreq_gc_mne(data, fs, cwt_freqs, n_cycles, gc_n_lags, seeds, targets,
                    method='gc', rank=None):
    """One MNE state-space GC call for a set of directed edges (cwt mode).

    Parameters
    ----------
    data : (n_epochs, n_signals, n_times)
    seeds, targets : lists of channel-index lists (one group per directed edge).
        A group with >1 channel makes the edge multivariate (block GC).
    rank : (seed_ranks, target_ranks) — one rank per edge (defaults to full,
        i.e. the group size). For FIXPC-k blocks, rank = k.
    method : 'gc' or 'gc_tr' (time-reversed).

    Returns
    -------
    (n_edges, n_freqs, n_times) real array — one GC value per directed edge.
    """
    from mne_connectivity import spectral_connectivity_epochs
    import mne
    mne.set_log_level('ERROR')
    if rank is None:
        rank = ([len(s) for s in seeds], [len(t) for t in targets])
    con = spectral_connectivity_epochs(
        data, method=method, indices=(seeds, targets), sfreq=fs,
        mode='cwt_morlet', cwt_freqs=np.asarray(cwt_freqs, float),
        cwt_n_cycles=np.asarray(n_cycles, float), gc_n_lags=gc_n_lags,
        rank=rank, verbose=False)
    return np.asarray(con.get_data())            # (n_edges, n_freqs, n_times)


def compute_subject_gc_mne(roi_data, times, sfreq, *, gc_n_lags=20, win_ms=250.0,
                           target_fs=200.0, cwt_freqs=None, bands=None, pairs=None,
                           trgc=True, tmin=None, tmax=None, ncycle_floor=1.0,
                           n_pcs=1):
    """Time-resolved MNE state-space GC for one subject, all requested ROI pairs.

    ``roi_data`` : dict ``{roi_name: array}`` — vertex data
    ``(n_epochs, n_vertices, n_times)`` (reduced here to the ROI's top-``n_pcs``
    PC block) or already-reduced ``(n_epochs, n_times)`` (a single channel; then
    only ``n_pcs = 1`` is possible).

    ``n_pcs`` : PCs kept per ROI (FIXPC-k). 1 = one virtual channel + bivariate
    GC (the original behaviour); 3 or 4 = multivariate block / group-to-group GC.

    ``pairs`` : list of (roi_i, roi_j) *index* pairs (into ``roi_data`` order),
    or None for all pairs. GC is computed both ways for each.

    Returns a dict with ``roi_names``, ``pair_i``/``pair_j``, ``window_ms``
    (the real time axis in ms), ``freqs``, ``fs``, ``n_pcs``, ``fxy``/``fyx``
    (band -> ``(n_pairs, n_times)``; ``fxy`` = i->j, ``fyx`` = j->i) and, if
    ``trgc``, ``dtrgc`` (band -> ``(n_pairs, n_times)``, Diff-TRGC for i->j).
    """
    if cwt_freqs is None:
        cwt_freqs = DEFAULT_CWT_FREQS
    if bands is None:
        bands = DEFAULT_BANDS
    cwt_freqs = np.asarray(cwt_freqs, float)
    band_names = list(bands)
    roi_names = list(roi_data)

    # Reduce each ROI to its top-n_pcs PC block (n_pcs=1 -> one channel). Each
    # ROI occupies a contiguous block of channels in the stacked array.
    roi_blocks, roi_channels, offset = [], {}, 0
    for ri, r in enumerate(roi_names):
        blk = reduce_roi_top_pcs(roi_data[r], n_pcs)   # (n_ep, k_r, n_t)
        roi_blocks.append(blk)
        roi_channels[ri] = list(range(offset, offset + blk.shape[1]))
        offset += blk.shape[1]
    V = np.concatenate(roi_blocks, axis=1)             # (n_ep, total_ch, n_t)

    V, fs = resample_channels(V, sfreq, target_fs)     # resamples the time axis
    n_t = V.shape[2]
    new_times = times[0] + np.arange(n_t) / fs
    lo = 0 if tmin is None else int(np.searchsorted(new_times, tmin))
    hi = n_t if tmax is None else int(np.searchsorted(new_times, tmax, 'right'))
    V = V[:, :, lo:hi]
    win_times = new_times[lo:hi]
    V = V - V.mean(axis=2, keepdims=True)              # per-trial demean

    n_cyc = sw_to_ncycles(cwt_freqs, win_ms, ncycle_floor)
    n_roi = len(roi_names)
    if pairs is None:
        pairs = list(combinations(range(n_roi), 2))
    pair_i = np.array([p[0] for p in pairs])
    pair_j = np.array([p[1] for p in pairs])

    # Directed edges -> block seeds/targets + per-edge rank (= block size).
    edges = []
    for (i, j) in pairs:
        edges += [(i, j), (j, i)]
    seeds = [roi_channels[e[0]] for e in edges]
    targets = [roi_channels[e[1]] for e in edges]
    rank = ([len(roi_channels[e[0]]) for e in edges],
            [len(roi_channels[e[1]]) for e in edges])
    edge_idx = {e: k for k, e in enumerate(edges)}

    data = V                                           # (n_ep, total_ch, n_t)
    gc = timefreq_gc_mne(data, fs, cwt_freqs, n_cyc, gc_n_lags,
                         seeds, targets, 'gc', rank)

    def bavg(edge):                               # (n_freq,n_time) -> {band:(n_time,)}
        return band_average(gc[edge_idx[edge]], cwt_freqs, bands)

    fxy = {b: np.full((len(pairs), win_times.size), np.nan) for b in band_names}
    fyx = {b: np.full((len(pairs), win_times.size), np.nan) for b in band_names}
    bxy_all, byx_all = [], []
    for k, (i, j) in enumerate(pairs):
        bxy = bavg((i, j)); byx = bavg((j, i))
        bxy_all.append(bxy); byx_all.append(byx)
        for b in band_names:
            fxy[b][k] = bxy[b]
            fyx[b][k] = byx[b]

    result = {
        'roi_names': roi_names, 'pair_i': pair_i, 'pair_j': pair_j,
        'window_ms': win_times * 1000.0, 'freqs': cwt_freqs, 'bands': dict(bands),
        'fxy': fxy, 'fyx': fyx, 'fs': fs, 'mode': MODE, 'n_pcs': n_pcs,
        'under_resolved': under_resolved_bands(win_ms, bands),
    }

    if trgc:
        gct = timefreq_gc_mne(data, fs, cwt_freqs, n_cyc, gc_n_lags,
                              seeds, targets, 'gc_tr', rank)

        def bavg_tr(edge):
            return band_average(gct[edge_idx[edge]], cwt_freqs, bands)

        dtr = {b: np.full((len(pairs), win_times.size), np.nan) for b in band_names}
        for k, (i, j) in enumerate(pairs):
            txy = bavg_tr((i, j)); tyx = bavg_tr((j, i))
            for b in band_names:
                # Diff-TRGC i->j = net(gc) - net(gc_tr)
                dtr[b][k] = (bxy_all[k][b] - byx_all[k][b]) - (txy[b] - tyx[b])
        result['dtrgc'] = dtr

    return result
