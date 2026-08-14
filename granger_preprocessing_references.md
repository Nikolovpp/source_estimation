# Is ERP removal before Granger causality defensible?

Why this file: `run_granger.py` defaults to `--normalize demean`, which subtracts
the ensemble mean (the ERP) point-by-point from each trial. The MVGC toolbox has
no such step, which raised the question of whether the pipeline is doing
something unsupported that a reviewer could object to.

**Answer: ERP removal is the better-supported choice, and the support includes
the MVGC authors themselves.** Doing only what the MVGC *code* does would be the
harder position to defend for event-related data, not the easier one.

---

## 1. What the MVGC code actually does

`matlab_source_code_ref/MVGC1-1.3`:

- `stats/demean.m` reshapes `X` `(n_vars, n_obs, n_trials)` to `(n, m*N)` and
  subtracts `mean(Y,2)` — **one scalar per variable**, over time and trials
  pooled. Never across variables.
- Called unconditionally by `core/tsdata_to_var.m:116` and
  `core/tsdata_to_autocov.m:72`.
- Its header warns: *"For multi-trial data we don't demean on a 'per-trial'
  basis, since this really doesn't make sense... demeaning trials separately can
  introduce large bias in VAR model estimation... If you feel you absolutely
  have to demean per-trial, then call this function for each trial series
  X(:,:,r) and then call it with X."*
- The strings `ERP`, `evoked` and `event-related` appear **exactly once** in the
  whole toolbox: in the reference list of `stats/consistency.m`, citing Ding et
  al. (2000) — the very paper that establishes ensemble-mean removal as
  essential. MVGC imports that paper's model-validation statistic (percent
  consistency) while implementing none of its preprocessing.

So MVGC's silence on ERPs is scope, not a position. The toolbox targets
stationary data where a time-varying ensemble mean cannot exist by construction.

## 2. What the MVGC authors say in print

Seth, Barrett & Barnett (2015), *J. Neurosci.* 35(8):3293–3297 — Barnett and
Seth are the toolbox authors:

> "For event-related or induced data, nonstationarity is likely to be a common
> issue which can be tackled either by (1) a 'vertical' regression in which VAR
> models are estimated for very short windows across trials, rather than for
> each trial separately, or (2) removing the ensemble average ERP. The former
> assumes that each trial is an independent realization of the same underlying
> stochastic process; the latter assumes minimal intertrial variation in the
> ERP."

This pipeline does **both**. That is the citable sentence for the methods
section, from the authors of the toolbox being cited for the estimator.

## 3. The origin: Ding, Bressler, Yang & Liang (2000)

*Biol. Cybern.* 83:35–45. This is the same Ding/Bressler lineage as BSMART,
which `granger.py` ports (`armorf`), so it is the natural citation here.

Abstract/introduction:

> "we identify a set of preprocessing steps, including removal of the averaged
> event-related potentials (ensemble means) from the ensemble of single-trial
> ERPs, that prove to be essential for the meaningful estimation of the time
> series models."

§4.4, stating the exact trade-off MVGC's `demean.m` header raises, and
resolving it the other way:

> "A common procedure in AR modeling is to subtract the temporal mean within a
> given window from the single trial time series before fitting. Although this
> is acceptable in the case of long stationary time series, the results of
> Sect. 3 above show that this is a bad practice when the window is short. In
> the short-window case, provided that there exists an ensemble of trials, a
> much more important procedure is to subtract the ensemble mean from each
> trial. By doing so, we (1) remove the first-order nonstationarity from the
> data; and (2) make the ensemble mean equal to zero, which is a requirement
> for model fitting."

§4.2, on the axis — the same point as the `ff2cdd6` regression:

> "The nonstationarity embodied in the mean and standard deviation can be easily
> removed by subtracting the ensemble mean, point-by-point, from each trial and
> then dividing the result, again point-by-point, by the standard deviation. In
> fact, these two procedures constitute our second and third preprocessing
> steps. **They are applied separately to the data from each channel.**"

§4.4/Fig. 7 — the strongest practical argument, and one this pipeline can check
directly:

> "removal of the ensemble mean is also crucial for achieving stability of the
> fitted models... In the case of not removing the ensemble mean (dashed line),
> the values of SI became positive for a period after the onset of stimulus (at
> 0 ms), rendering the fitted models invalid during this time period. After the
> removal of the ensemble mean, all the values of SI were negative (solid line),
> indicating that all of the fitted models throughout the task were stable."

Footnote 2 gives the reductio: without ensemble-mean removal you find
significant coherence between channels recorded from **two different subjects**,
at the frequency of the ensemble mean. It also notes the mitigating caveat —
*"For scalp electroencephalogram (or magnetoencephalogram) this factor may not
be as significant because the ensemble means are small relative to the raw
signals."* This is source-reconstructed EEG from an overt production task, so
the evoked component is not small.

**Note the parameters.** Ding et al. sample at 200 Hz and argue for windows of
"40–80 ms (or possibly even shorter)", i.e. "a data string of only 8–16 points",
settling on 10 points (50 ms). The sweep's 40/60/80 ms at 200 Hz = 8/12/16
samples is exactly their regime, which makes their preprocessing prescription
directly applicable rather than merely analogous.

## 4. The counter-caution

Wang, Chen & Ding (2008), *NeuroImage* 41(3):767–776, "Estimating Granger
causality after stimulus onset: a cautionary note". Subtracting the *average*
ERP leaves residual evoked activity, because single-trial ERPs vary in latency
and amplitude; that residual is itself nonstationary and can produce artifactual
GC. Their remedy is single-trial ERP estimation (ASEO) and per-trial
subtraction, not abandoning the correction.

Practitioner-level framing, Brainstorm's Granger causality documentation:

> "Check this box to remove the averaged evoked. It is also recommended by some
> as it meets the zero-mean stationarity requirement (improves stationarity of
> the system). However, the problem with this approach is that it does not
> account for trial-to-trial variability. For a discussion see (Wang et al.,
> 2008)."

So the literature's disagreement is about *how well* to remove the evoked
response, not *whether* to. No source argues for leaving it in.

## 5. What this pipeline does, against Ding's recipe

| Ding et al. (2000) step | This pipeline |
|---|---|
| detrend each single-trial series | not done |
| subtract temporal mean of the **entire trial** | `demean_trials=True`, `run_granger.py:222` (whole cropped epoch: 160 samples perception, 400 production — not the 8–16-sample window) |
| divide by temporal SD | not done |
| subtract ensemble mean point-by-point, **per channel** | `--normalize demean`, `run_granger.py:236` |
| divide by ensemble SD point-by-point, per channel | not done; `--normalize zscore` would do it |

The per-trial temporal demean is not in conflict with MVGC's warning: both MVGC
and Ding §3 are objecting to demeaning over a *short* segment, and this one runs
over the whole epoch before windowing. Ding et al. do the same thing.

The one substantive gap is the ensemble-SD normalization (Ding's third step),
which they call *"crucial for allowing the dynamical changes in model-derived
spectral quantities to be compared at each stage of task processing"* — i.e.
precisely the comparison across sliding windows this project makes.
`--normalize zscore` implements steps 2+3 together. Untested here; worth an A/B
before the manuscript.

## 6. Bottom line

Keep `--normalize demean`. It is (a) named by the MVGC authors as one of the two
accepted treatments for event-related data, (b) called essential by the paper
MVGC itself cites for model validation, and (c) the fix for a failure mode
measured on this pipeline — an evoked response with an ROI-to-ROI latency
difference gave theta GC 0.409 against a truth of 0.005, with TRGC offering no
protection.

If a reviewer objects, the objection to expect is Wang et al. (2008)'s — that
average-ERP subtraction ignores trial-to-trial variability — not that the step
is unwarranted.

## References

- Barnett L, Seth AK (2014). The MVGC multivariate Granger causality toolbox: a
  new approach to Granger-causal inference. *J. Neurosci. Methods* 223:50–68.
- Seth AK, Barrett AB, Barnett L (2015). Granger causality analysis in
  neuroscience and neuroimaging. *J. Neurosci.* 35(8):3293–3297.
- Ding M, Bressler SL, Yang W, Liang H (2000). Short-window spectral analysis of
  cortical event-related potentials by adaptive multivariate autoregressive
  modeling: data preprocessing, model validation, and variability assessment.
  *Biol. Cybern.* 83:35–45.
- Wang X, Chen Y, Ding M (2008). Estimating Granger causality after stimulus
  onset: a cautionary note. *NeuroImage* 41(3):767–776.
- Brovelli A, Ding M, Ledberg A, Chen Y, Nakamura R, Bressler SL (2004). Beta
  oscillations in a large-scale sensorimotor cortical network: directional
  influences revealed by Granger causality. *PNAS* 101(26):9849–9854.
