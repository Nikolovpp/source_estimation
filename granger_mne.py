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


def mne_cwt_freq_floor(fs, n_times):
    """Lowest frequency ``mne_connectivity`` will RETURN for a cwt run, in Hz.

    Not a rule of thumb — a hard property of the library. It discards every
    requested frequency below ``5 * fs / n_times``, applied to the length of the
    array passed in (not ``win_ms``, not ``cwt_n_cycles``), and reports nothing.
    Measured at fs=200 requesting 4-30 Hz (27 bins): n_times 60 -> 14 bins
    (>=17 Hz), 118 -> 22 (>=9 Hz), 150 -> 24 (>=7 Hz), 250 -> all 27 (>=4 Hz).

    Provided so callers can predict the returned grid; nothing in this module
    warns or errors on it. Band values with no surviving bins come back NaN.
    """
    return 5.0 * float(fs) / float(n_times)


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
    gc : (n_edges, n_freqs_out, n_times) real array — one GC value per edge.
    freqs_out : (n_freqs_out,) the frequencies MNE ACTUALLY returned.

    ``freqs_out`` is **not** necessarily ``cwt_freqs``: MNE silently drops every
    requested frequency below ``5 * fs / n_times`` (see
    :func:`mne_cwt_freq_floor`). Callers must band-average against ``freqs_out``,
    never against the requested grid — doing the latter raises
    ``IndexError: boolean index did not match indexed array`` downstream, or
    worse, silently averages the wrong bins when the counts happen to match.
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
    return (np.asarray(con.get_data()),          # (n_edges, n_freqs_out, n_times)
            np.asarray(con.freqs, dtype=float))


def compute_subject_gc_mne(roi_data, times, sfreq, *, gc_n_lags=20, win_ms=250.0,
                           target_fs=200.0, cwt_freqs=None, bands=None, pairs=None,
                           trgc=True, tmin=None, tmax=None, ncycle_floor=1.0,
                           n_pcs=1, normalize='demean', labels=None):
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
    V = V - V.mean(axis=2, keepdims=True)              # per-trial temporal demean

    # ── Ensemble (ERP) removal — ON BY DEFAULT ────────────────────────────
    # The step above is a per-trial TEMPORAL demean; it does nothing about the
    # phase-locked evoked response, which is a deterministic component shared by
    # every trial. MNE's cwt CSD is the across-trial average of wavelet outer
    # products, so an ERP enters it directly, and any latency difference between
    # two ROIs' evoked responses is read as directed influence from the earlier
    # one. Measured on two INDEPENDENT AR(2) processes (true GC ~= 0) plus one
    # identical evoked burst arriving 25 ms later in ch1: theta 0->1 = 0.409 vs
    # 0.005 truth, and dTRGC = +0.378 vs +0.003 — i.e. TRGC does NOT catch it,
    # because an ERP latency difference genuinely IS a time lag.
    #
    # Removed WITHIN each level of ``labels`` when given: the cache holds both
    # classes of a contrast, so a single grand mean would leave the
    # between-condition evoked difference in as a two-level deterministic driver.
    if normalize != 'none':
        if labels is None:
            groups = [np.ones(V.shape[0], bool)]
        else:
            lab = np.asarray(labels)
            groups = [lab == u for u in np.unique(lab)]
        for g in groups:
            if not g.any():
                continue
            mu = V[g].mean(axis=0, keepdims=True)       # ERP per (chan, time)
            if normalize == 'demean':
                V[g] = V[g] - mu
            elif normalize == 'zscore':
                sd = V[g].std(axis=0, keepdims=True)
                sd[sd == 0] = 1.0
                V[g] = (V[g] - mu) / sd
            else:
                raise ValueError(f'Unknown normalize mode: {normalize!r}')

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
    gc, freqs_out = timefreq_gc_mne(data, fs, cwt_freqs, n_cyc, gc_n_lags,
                                    seeds, targets, 'gc', rank)

    # MNE drops requested frequencies below 5*fs/n_times without raising, so
    # band-average against what it RETURNED, and record what was lost.
    def bavg(edge):                               # (n_freq,n_time) -> {band:(n_time,)}
        """Band-average on the RETURNED grid, via granger.band_average so the
        half-open/disjoint band definition has exactly one implementation."""
        return band_average(gc[edge_idx[edge]], freqs_out, bands, on_empty='nan')

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
        'window_ms': win_times * 1000.0, 'freqs': freqs_out, 'bands': dict(bands),
        'fxy': fxy, 'fyx': fyx, 'fs': fs, 'mode': MODE, 'n_pcs': n_pcs,
        'under_resolved': under_resolved_bands(win_ms, bands),
        # ``freqs`` above is what MNE returned, which can be a subset of what was
        # requested (it applies its own low-frequency floor). Keep the request
        # alongside it so the two are reconstructable from the npz alone; a band
        # with no surviving bins comes back NaN, which is self-evident in fxy/fyx.
        'freqs_requested': np.asarray(cwt_freqs, float),
        'normalize': normalize,
        # Frequency-resolved GC, so band definitions can be changed WITHOUT
        # recomputing. Only band means were persisted historically, which is why
        # the band-overlap fix forced a full re-run.
        'gc_freq': gc, 'gc_freq_edges': np.array(edges, dtype=int),
    }

    if trgc:
        gct, freqs_tr = timefreq_gc_mne(data, fs, cwt_freqs, n_cyc, gc_n_lags,
                                        seeds, targets, 'gc_tr', rank)

        def bavg_tr(edge):
            return band_average(gct[edge_idx[edge]], freqs_tr, bands,
                                on_empty='nan')

        dtr = {b: np.full((len(pairs), win_times.size), np.nan) for b in band_names}
        for k, (i, j) in enumerate(pairs):
            txy = bavg_tr((i, j)); tyx = bavg_tr((j, i))
            for b in band_names:
                # Diff-TRGC i->j = net(gc) - net(gc_tr)
                dtr[b][k] = (bxy_all[k][b] - byx_all[k][b]) - (txy[b] - tyx[b])
        result['dtrgc'] = dtr

    return result
