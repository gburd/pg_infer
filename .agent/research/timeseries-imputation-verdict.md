# Timeseries foundation models for probabilistic column imputation in pg_infer

Research report, 2026-11-18.  Investigates whether a pretrained timeseries
model could drive a BayesStore-like "fill the missing column value with a
calibrated probability" feature in pg_infer.

**Verdict up front: no.  Do not take this on.**  The reasoning is in the
final section; the four questions that produce it are answered first.
Section 4 is the one that decides it, and it does not depend on any of the
model facts in section 1.

Claims are marked **[verified]** when I ran it or read the primary source,
**[reproduced]** when I am relaying a number a paper reports, and
**[unverified]** when I could not confirm it.  Measurements I took myself
ran on floki (48 cores, 185 GiB, torch 2.15.0+cpu, CPU only) and are in
`/scratch/pg_infer/scratch-ts-research/`.

---

## Question 1: the models

Parameter counts below are **[verified]** from the HuggingFace API
(`safetensors.total`, the actual tensor element count in the published
checkpoint), not from the papers' claims, because the two disagree
in a few places.  Licences are **[verified]** from `cardData.license`
plus the LICENSE file where the licence is non-standard.

| Model | Params | Licence | Weights open? | Output form | Max context | CPU realistic? |
|---|---|---|---|---|---|---|
| chronos-t5-tiny | 8.39M | Apache-2.0 | yes | **sample paths** (categorical over quantised bins) | 512 | yes |
| chronos-t5-mini | 20.5M | Apache-2.0 | yes | sample paths | 512 | yes |
| chronos-t5-small | 46.2M | Apache-2.0 | yes | sample paths | 512 | yes |
| chronos-t5-base | 201M | Apache-2.0 | yes | sample paths | 512 | marginal |
| chronos-t5-large | 709M | Apache-2.0 | yes | sample paths | 512 | no |
| chronos-bolt-tiny | 8.65M | Apache-2.0 | yes | **quantiles** (9 levels, direct multi-step) | 2048 | yes |
| chronos-bolt-mini | 21.2M | Apache-2.0 | yes | quantiles | 2048 | yes |
| chronos-bolt-small | 47.7M | Apache-2.0 | yes | quantiles | 2048 | yes (15.5 ms, measured) |
| chronos-bolt-base | 205M | Apache-2.0 | yes | quantiles | 2048 | yes |
| **chronos-2** | 120M | Apache-2.0 | yes | quantiles | 8192 | yes (43 ms, measured) |
| timesfm-1.0-200m | ~200M | Apache-2.0 | yes | point + quantiles | 512 | yes |
| timesfm-2.0-500m | 499M | Apache-2.0 | yes | point + quantiles | 2048 | marginal |
| timesfm-2.5-200m | 231M | Apache-2.0 | yes | point + 9 quantiles (optional 30M head) | 16384 | yes |
| **timesfm-3.0** | 331M | **timesfm-non-commercial-v1.0** | yes but **non-commercial** | point + quantiles | ~15360 | yes |
| Nixtla TimeGPT | n/a | **closed** | **no** | quantiles via API | n/a | **no, API only** |
| granite-ttm-r2 (512-96) | **0.805M** | Apache-2.0 | yes | **point forecast only** | 512 | yes (3.5 ms, measured) |
| granite-ttm-r2 (512-192) | 0.904M | Apache-2.0 | yes | point only | 512 | yes |
| granite-patchtst | 0.616M | Apache-2.0 | yes | point only | n/a | yes |
| granite-patchtsmixer | 0.196M | Apache-2.0 | yes | point only | n/a | yes |
| moirai-1.1-R-small | 13.8M | **CC-BY-NC-4.0** | yes but **non-commercial** | **mixture distribution** | flexible (~5000) | yes |
| moirai-1.1-R-large | 311M | CC-BY-NC-4.0 | non-commercial | mixture distribution | flexible | marginal |
| moirai-2.0-R-small | 11.4M | CC-BY-NC-4.0 | non-commercial | quantiles | flexible | yes |
| MOMENT-1-small | 37.9M | MIT | yes | **reconstruction** (point) | 512 | yes |
| MOMENT-1-large | 346M | MIT | yes | reconstruction (point) | 512 | marginal |
| Lag-Llama | 2.45M | Apache-2.0 | yes | Student-t parametric | ~1024 | yes |
| StatsForecast (AutoARIMA/AutoETS) | n/a | Apache-2.0 | n/a (fitted per series) | prediction intervals | unbounded | yes |
| TabPFN-v2 (as TabPFN-TS) | ~11M/ckpt | **PriorLabs-1.1** (Apache-2.0 + attribution) | yes, **commercial OK** | full predictive distribution | in-context | **no, ~1 s/window on H100** |

Several of these facts cut against the brief's framing and are worth
stating plainly.

**TTM is even smaller than advertised, and gives you no distribution.**
The model card says "starting from 1M params"; the published `main`
checkpoint is 805,280 parameters **[verified]**, which is 3.1 MiB at fp32.
I loaded it and ran it: 3.5 ms per forward pass on CPU **[verified]**.  That
is by far the cheapest thing in this table.  But its output dict is
`['prediction_outputs', 'backbone_hidden_state', 'decoder_hidden_state',
'loc', 'scale', 'input_data']` **[verified]**, there is no quantile and no
distribution head anywhere in it.  The model card confirms this in words:
"The TTM r2 releases support point forecasting use-cases specifically."
For a feature whose entire premise is returning a calibrated probability,
TTM is disqualified at the first hurdle, and its tiny size is irrelevant.

**TimesFM's newest and best weights are non-commercial.**  Weights up to
2.5 are Apache-2.0, but TimesFM 3.0 ships under a bespoke
`timesfm-non-commercial-license-v1.0` **[verified]** that forbids
"any revenue-generating activity", "direct or indirect interactions with
end users or production systems", and distribution of the weights or any
derivative.  Google's own README says commercial use is permitted only
"through authorized Google Cloud services, including BigQuery ML".  So the
model that currently ranks #1 on fev-bench, GIFT-Eval, and TIME is the one
pg_infer could never ship.  TimesFM 2.5 (231M, Apache-2.0, 16k context)
remains usable.

**Moirai is non-commercial across the whole family.**  Every checkpoint I
checked (1.0-R-large, 1.1-R-small, 1.1-R-large, 2.0-R-small,
moe-1.0-R-small) is CC-BY-NC-4.0 **[verified]**, even though the `uni2ts`
*code* is Apache-2.0.  This is the usual trap: permissive code, restricted
weights.  Moirai is out for anything pg_infer would ship, which is a real
loss because it is the only model here with a genuine mixture-distribution
head (Student-t + others, trained by negative log-likelihood).

**TimeGPT is API-only, confirmed from the vendor's own words.**  The
`nixtla` SDK is Apache-2.0, but its README states: "TimeGPT itself is
closed source.  The SDK is a client: every forecasting and anomaly
detection call is executed by the model rather than locally, so an API key
is required" **[verified]**.  It also claims TimeGPT "can be deployed on
your own infrastructure", which presumably means a licensed on-prem
appliance rather than downloadable weights **[unverified]**.  I found no
public weights.  For an open-source PostgreSQL extension this is a
non-starter regardless: it would make every imputation a billable outbound
network call from inside a backend.

**Chronos-2 is the strongest open option, and it is genuinely cheap.**
120M parameters **[verified]**, Apache-2.0, 8192 context, native
multivariate and covariate support, and the model card advertises CPU
inference.  I measured it at 43 ms per `predict_quantiles` call with a
1024-point context on CPU **[verified]**, and chronos-bolt-small at 15.5 ms
**[verified]**.  For comparison, pg_infer's own `describe()` costs 176 ms
and `similar_to()` 354 ms per DESIGN's performance table, so a Chronos call
is an order of magnitude *cheaper* than what pg_infer already does per
query.  Compute is not the obstacle here.  Nothing in section 4 changes
because of that.

**TabPFN-TS is the accuracy leader for imputation specifically but is
far too slow.**  Its licence is better than it first looks: HuggingFace
reports `license_name: priorlabs-1-1`, and the LICENSE text is Apache-2.0
with one added clause requiring you to display "Built with PriorLabs-TabPFN"
and to prefix derivative model names with "TabPFN" **[verified]**.  That is
commercially usable.  The problem is cost: the benchmark paper reports
"On an NVIDIA H100 GPU, a single forward pass over a chunk of 672 time
steps requires approximately one second" **[reproduced]**, and warns that
"systems operating on thousands of concurrent time series may find the
latency challenging".  One second per window on an H100 is roughly 10^4
times pg_infer's per-row budget for an index scan.

The classical baselines deserve more respect than the brief implies, and
section 2 shows why.

---

## Question 2: imputation is not forecasting, and almost nothing here does imputation

The distinction is real, it is load-bearing, and the honest answer is that
**none of the headline forecasting foundation models support interior
in-filling as an operation.**

What they do support is a weaker thing that is easy to mistake for it:
tolerating missing values *inside the context* while still only predicting
*forward of the context's end*.  Chronos does this by masking.  Reading
the library source **[verified]**: `chronos.py` builds
`attention_mask = ~torch.isnan(context)` and then sets
`token_ids[~attention_mask] = self.config.pad_token_id`; the paper says the
PAD token is used "to replace missing values".  `chronos_bolt.py` does the
patch-level equivalent, zeroing unobserved positions and concatenating the
mask as extra channels.  So a NaN in the middle of your history is handled
gracefully, but the output tensor still begins at `t = context_end + 1`.
I confirmed this behaviourally: feeding a context with `y[300:320] = NaN`
is accepted without error and returns a forecast of shape `(1, 12, 3)`
covering steps 512..523, not the gap at 300..320 **[verified]**.

I also enumerated the public API of `Chronos2Pipeline`: `['embed', 'fit',
'from_pretrained', 'inner_model', 'model', 'predict', 'predict_df',
'predict_fev', 'predict_quantiles', 'save_pretrained']` **[verified]**.
There is no `impute`, no `interpolate`, no `fill`.  The same holds for
Chronos-Bolt and TimesFM.

So using these for imputation means hacking it, and the two obvious hacks
are: forecast from the left and ignore the right context, or forecast from
both sides and blend.  I measured how much that costs.

### Experiment: does the hack work?

Setup: hourly synthetic series with daily and weekly seasonality plus
noise, a 12-step interior gap, chronos-bolt-small, 30 trials, median
forecast, MAE against truth **[verified]**.

| method | mean MAE | median MAE |
|---|---|---|
| forecast left-only | 0.357 | 0.354 |
| bidirectional, blended | 0.339 | 0.338 |
| linear interpolation | 2.312 | 2.328 |
| seasonal naive | 0.647 | 0.649 |

On clean seasonality the hack looks fine, and beats linear interpolation
by 6.5x.  That is the result a demo would show you.  But it is the easy
case: when a series is perfectly periodic, the right context carries almost
no information the left context lacks, so throwing it away costs nothing.

The case that actually separates the two tasks is one where the answer
depends on information *only the right context has*.  I used a level shift
of +12 beginning halfway through the gap: the left context cannot know it
happened, the right context shows it plainly **[verified]**.

| method | mean MAE |
|---|---|
| forecast left-only | 6.173 |
| forecast right-only | 6.119 |
| bidirectional, blended | 2.767 |
| linear interpolation | 3.998 |

Left-only is **2.23x worse** than the blend, and worse than plain linear
interpolation.  Both one-sided forecasts fail symmetrically, which is the
signature of the problem: each is ignoring half the evidence.  This is the
whole content of the imputation/forecasting distinction, reproduced in
twenty lines of code.

Note what the blend is *not*: it is not a model of the gap.  Averaging two
marginal forecasts with a linear ramp is a heuristic I invented for the
experiment.  It has no joint distribution, so it cannot answer "what is the
probability this value exceeded X at some point during the gap", and its
uncertainty is not the uncertainty of anything.  Section 3 measures what
happens when you try to use it as one anyway.

### The dedicated imputation literature

These models do the real task: they condition on past *and* future
observed values and model the interior.

- **BRITS** (NeurIPS 2018, arXiv:1805.10572): bidirectional RNN, treats
  imputed values as graph variables updated during training.  Point
  estimates.  Multivariate.
- **GP-VAE** (AISTATS 2020, arXiv:1907.04155): VAE with a Gaussian-process
  prior on a low-dimensional latent that evolves smoothly in time.
  Explicitly motivated by the fact that "naive applications of deep
  learning fall short in giving reliable confidence estimates"; provides
  "interpretable uncertainty estimates".  Probabilistic.
- **SAITS** (ESWA 2023, arXiv:2202.08516): two diagonally-masked
  self-attention blocks, weighted combination, joint-optimisation
  (ORT+MIT).  Point estimates natively.  Best-in-class among the
  supervised imputers in the benchmark below.
- **CSDI** (NeurIPS 2021, arXiv:2107.03502): conditional score-based
  diffusion, explicitly trained for imputation.  **This is the only one of
  the group that natively emits a full predictive distribution** (100
  samples per missing value in their protocol).
- **ImputeFormer** (KDD 2024, arXiv:2312.01728): low-rankness-induced
  transformer, aimed at spatiotemporal data where a structural prior helps.
- **PyPOTS** (arXiv:2305.18811, BSD-3-Clause, 2099 stars, last push
  2026-11-17 **[verified]**): the toolkit.  It is the right answer to
  "where do I get all of these", exposing ~50 algorithms behind one
  interface for imputation, forecasting, classification, clustering, and
  anomaly detection.  Importantly, PyPOTS' own README flags that many of
  the models it offers (marked with a wrench) "are not originally proposed
  as algorithms for POTS data ... and they cannot directly accept time
  series with missing values as input, let alone imputation"; PyPOTS makes
  them applicable by applying SAITS' embedding and training approach
  **[verified]**.  So even inside a toolkit devoted to imputation, most
  entries are retrofitted.

CSDI's reported numbers, since it is the probabilistic one
**[reproduced]** from the paper's Table 2 (CRPS, lower is better, mean with
standard error over five trials):

| method | healthcare 10% | 50% | 90% | air quality |
|---|---|---|---|---|
| Multitask GP | 0.489 (0.005) | 0.581 (0.003) | 0.942 (0.010) | 0.301 (0.003) |
| GP-VAE | 0.574 (0.003) | 0.774 (0.004) | 0.998 (0.001) | 0.397 (0.009) |
| V-RIN | 0.808 (0.008) | 0.831 (0.005) | 0.922 (0.003) | 0.526 (0.025) |
| unconditional diffusion | 0.360 (0.007) | 0.458 (0.008) | 0.671 (0.007) | 0.135 (0.001) |
| **CSDI** | **0.238 (0.001)** | **0.330 (0.002)** | **0.522 (0.002)** | **0.108 (0.001)** |

The paper's claim of "40-65% improvement over existing probabilistic
imputation methods" is consistent with these rows.  Note the datasets are
healthcare (PhysioNet) and air quality, both multivariate with
missing-at-random gaps, which is the brief's target regime.

### What is actually best for multivariate MAR gaps

There is a 2025 benchmark that settles this, and it is the most
decision-relevant paper I found: **"Are Time-Indexed Foundation Models the
Future of Time Series Imputation?"** (arXiv:2511.05980).  It is the first
large-scale study of zero-shot imputation: 33 out-of-domain datasets,
~1.3M imputation windows, four missingness regimes (pointwise 30%,
pointwise 70%, two-day blocks, four-day blocks).  Results **[reproduced]**,
z-normalised MAE averaged over regimes:

| model | MAE | avg rank | category |
|---|---|---|---|
| **TabPFN-TS** | **0.293** | **1.35** | foundation, zero-shot |
| MoTM | 0.371 | 3.62 | foundation, zero-shot |
| SAITS | 0.386 | 3.56 | supervised, trained per dataset |
| **Linear interpolation** | **0.506** | n/a | local heuristic |
| CSDI | 0.664 | n/a | supervised |
| TimesNet | 0.677 | n/a | supervised |

Three things in that table matter for this decision.

First, the winners are **not** the timeseries foundation models in
section 1.  TabPFN-TS is a *tabular* model (TabPFN-v2) fed handcrafted
time-index features (normalised time plus daily and weekly Fourier pairs),
and MoTM is a basis of implicit neural representations.  Both are
"time-indexed": they model value as a function of continuous time, which
is why they generalise to unobserved timestamps and can fill interior gaps
natively.  Chronos, TimesFM, and Moirai do not appear in the main table at
all.

Second, **linear interpolation beats CSDI, TimesNet, BRITS, and several
other supervised deep imputers** (0.506 vs 0.664, 0.677).  The paper is
blunt about why: supervised imputers "tend to overfit dataset-specific
temporal dynamics", and "sometimes perform worse than simple local
heuristics (e.g., Seasonal Naive, LOCF)".  For a general-purpose database
feature that must work on whatever column a user points at, this is the
single most important finding in the whole report.  The sophisticated
option is not reliably better than `interpolate()` out of domain.

Third, the paper explicitly tried the obvious foundation models for this
and excluded them for not working.  On MOMENT **[reproduced]**: "its
performance in our benchmark was found to be substantially lower than that
of the other methods ... the authors primarily advocate for fine-tuning
... their reported imputation experiments are limited to very short
missing segments ... Given its unsuitability for zero-shot time series
imputation, we excluded it."  On NuwaTS, likewise: "trails substantially
behind time-indexed foundation models."  MOMENT *does* have an imputation
path, its masked-reconstruction pretraining means `task_name:
"reconstruction"` fills masked patches, and the repo ships an imputation
tutorial, MIT-licensed **[verified]**.  But it needs fine-tuning per
dataset to be useful, which defeats the point of a drop-in feature.

---

## Question 3: calibrated uncertainty

This is where the brief's instinct is sharpest, and the literature
vindicates it.  **A model that emits q10/q50/q90 is not automatically
calibrated, and the metric the Chronos paper reports does not establish
calibration.**

### What the Chronos paper actually says

The Chronos paper (arXiv:2403.07815) reports WQL for probabilistic
performance.  Its own definition **[verified]** from the appendix: WQL is
the mean of the quantile (pinball) loss over nine levels
{0.1,...,0.9}, normalised by the sum of absolute values; the paper notes
WQL "approximates (a weighted average of) the continuous ranked
probability score (CRPS)" and that "many existing works use CRPS and WQL
synonymously".  On Benchmark II (27 unseen datasets) Chronos models "achieve
the 2nd to 4th spots" on aggregate relative WQL **[reproduced]**.

Two caveats the paper itself raises, which matter here.  The paper reports
WQL "deteriorates for larger vocabulary sizes" while MASE improves, and
attributes this to WQL being scale-dependent and so not closely correlated
with the (scale-invariant) training loss **[reproduced]**.  That is to say
the authors are already aware WQL measures something other than what they
optimise.  And on calibration proper, the paper puts it in *future work*:
"Chronos can be calibrated for a specific task with conformal methods ...
Chronos is especially attractive in the context of conformal prediction
since it requires no training set, so all available data can be used for
calibration" **[verified]**.  That sentence only makes sense if the raw
output is not already calibrated for the task.

### WQL does not measure calibration

The key citation is Chung et al. (2021), relayed by the paper below:
CRPS, WQL, and MSIS "measure a combination of both probabilistic
calibration and sharpness, rather than calibration alone", and this
"can skew towards prioritizing predictive sharpness".

**"Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated?"**
(Adler, Chang, Abdi, Smyth; arXiv:2510.16060) is the dedicated empirical
study, and it is the right source for this question.  It evaluates five
TSFMs (Chronos-Bolt, TimesFM, Moirai 2.0, TiRex, YingLong) plus ARIMA and
N-BEATS across six datasets, using calibration-specific metrics instead of
WQL:

- **PCE** (Probabilistic Calibration Error): mean absolute gap between
  each nominal quantile level and the empirical fraction of observations
  below the predicted quantile.  Range [0, 0.5].
- **CCE** (Centered Calibration Error): signed, so positive means
  overconfident and negative means underconfident.
- **SIW** (Scaled Interval Width): a sharpness surrogate, needed because
  a model that always predicts the marginal distribution is perfectly
  calibrated and useless.

Their findings **[reproduced]**:

- TSFMs are "generally better calibrated than the baseline models of
  N-BEATS and ARIMA", with PCE "close to or below 5%" on most datasets,
  and remaining there "even 64 time-steps out".  Their rule of thumb:
  PCE below 0.05 is well calibrated, above 0.15 is not.
- TSFMs "tend not to be either systematically over- or under-confident",
  unlike vision and language models which are reliably overconfident.
  Their explanation: TSFMs are "directly trained with a calibration-aware
  loss (i.e., trained to minimize WQL)".
- "No single TSFM significantly dominates in calibration performance."
- Prediction-head form barely matters, with one exception: Gaussian heads
  are "consistently under-confident" and worse than quantile, Student-t,
  or mixture heads.
- The explicit caution: "WQL alone is not necessarily a reliable indicator
  of calibration performance", with a figure showing WQL "incorrectly
  identifies ARIMA as the best calibrated model" on their Glucose dataset.
- A failure case: on their Patents dataset, *every* model (TSFM and
  baseline alike) has high calibration error and all are overconfident.

So: forecast-mode Chronos-Bolt and friends are, on the evidence, decently
calibrated *for forecasting*, and the brief's scepticism about q10/q90
being merely ordered turns out to be too harsh in that setting.  The
evidence is also narrower than it looks: one paper, six datasets,
marginal calibration at individual time steps only, no multivariate case.

### Quantiles are not even reliably ordered

Separately from calibration, I checked monotonicity, because the brief
asked.  Chronos-2's source contains no `sort`, no `cummax`, and no
crossing-correction **[verified]**.  TimesFM 2.5 exposes
`fix_quantile_crossing=True` as an **opt-in** compile flag **[verified]**,
which is itself an admission that crossings occur by default.

I measured it on chronos-bolt-small, 9 quantile levels, horizon 24, 40
series across four input classes **[verified]**:

| input class | crossing violations (adjacent quantile pairs) |
|---|---|
| iid Gaussian noise | 0 / 1920 (0.00%) |
| **sparse spikes** (5% of steps, N(50,5), else 0) | **281 / 1920 (14.64%)** |
| tiny variance on huge offset (1e6 + 1e-6·walk) | 0 / 1920 (0.00%) |
| weekly sine + noise | 4 / 1920 (0.21%) |

Overall 2.77%, worst single gap -0.053.  The violations concentrate
almost entirely in the sparse-spike case, and that is not a coincidence:
it is exactly the scaling-and-quantisation failure the Chronos paper
documents in its own limitations figure, where "data consists of unit
spikes every n observations ... when 1 > 15/n then the model cannot
possibly capture the spikes appropriately, since their value is not
represented accurately by tokens" **[reproduced]**.

A database column of event counts, error tallies, or sales of a
slow-moving SKU is exactly a sparse spiky series.  So the pathological
class for quantile ordering is a common class of real column, not an
adversarial construction.

### The decisive measurement: calibration under the imputation hack

Everything above is about calibration in *forecast* mode, which is the
mode these models were trained and evaluated in.  The feature under
consideration would use them in *imputation* mode.  I measured what that
does to coverage.

Setup: nominal 80% band (q10..q90), horizon 12, 120 trials,
chronos-bolt-small, forecasting from the left edge of a gap.  In a third
of trials I injected a level shift of N(0, 8) starting mid-gap, so that the
right context, which the model never sees, contains information that
changes the answer **[verified]**.

| regime | nominal | empirical coverage |
|---|---|---|
| no shift in gap (genuinely a forecasting task) | 80% | **71.4%** (n=80) |
| level shift inside gap (genuinely an imputation task) | 80% | **36.9%** (n=40) |

A nominal 80% interval that actually covers 36.9% of the time is not a
calibrated probability; it is a confidently wrong one.  And note that even
the clean case undershoots at 71.4%, consistent with the literature's
observation that these models are approximately but not exactly calibrated
on data they did not train on.

This is the crux of Q3 crossed with Q2.  The calibration that the Adler
paper credibly establishes is calibration *of the forecasting
distribution, for the forecasting task*.  Repurposing that distribution to
answer an imputation query silently voids the property that made it worth
shipping.  Returning 36.9%-coverage numbers through a SQL function that
advertises a probability is worse than returning NULL, because a caller
can handle NULL.

Conformal prediction is the standard repair and the literature is
encouraging about it for TSFMs specifically (arXiv:2507.08858 finds
calibration "more stable because more data are used for calibration",
especially when data is scarce).  But conformal calibration needs a
held-out calibration set per column per missingness pattern, which is a
substantial amount of machinery and state for a database extension to own.
And, per "Foundation Model Forecasts: Form and Function"
(arXiv:2510.19345), it buys coverage and not event probabilities:
"conformal methods guarantee coverage but not calibrated event
probabilities.  They answer 'where will 95% of realizations fall?' not
'what is Pr(event)?'" **[reproduced]**.  For a BayesStore-style feature
whose selling point is `P(value = x)`, that is the wrong guarantee.

That same paper adds a structural point worth recording.  It surveys
TSFMs and finds two-thirds emit only point or parametric forecasts, and
proves that marginals cannot determine path-dependent event probabilities
(infinitely many joint distributions share identical marginals).  It also
notes the direction of travel is *away* from the expressive form: "the
evolution from Chronos, a trajectory ensemble model, to ChronosBolt and
Chronos-2, which produce quantile forecasts" trades flexibility for speed
**[reproduced]**.  So the modern, fast, well-licensed models give you the
*least* expressive output, while original Chronos-T5 (the one that emits
sample paths you could marginalise however you like) is the slow one.

---

## Question 4: does this compose with pg_infer?

**No.  These are two unrelated features that would share a repository, and
bundling them is scope creep.**  I went looking for a genuine composition
and I do not think the one the brief proposes survives contact with what
pg_infer actually is.

### The domains do not overlap

I read the code and design docs to be sure.  pg_infer's object of study is
a *fixed artefact*: a vindex, which DESIGN describes as "an extracted,
query-optimized representation of a model's internal knowledge" holding
per-layer f16 gate vectors, token embeddings, feature metadata, and a
tokenizer, "built offline from model weights" and "read-only at query
time".  Every query function operates on that artefact.  `walk()` returns
`(layer, feature, activation, concept)`.  `describe()` returns
`(relation, target, confidence, layer)`.  `similar_to()` returns a scalar
derived from shared gate activations.  The index AM stores gate vectors in
8 KB pages **[verified]** from DESIGN and the function list in README.

Note what the dimensions are: **layer** and **feature**.  There is no time
axis, because there is nothing in a weight matrix that is indexed by time.
The user's data enters pg_infer only as text to be embedded and compared
against the model's features; the *values* in their columns are not the
object of study at all.  Even the column index is text-only: `CREATE INDEX
... USING infer (title)` supports `ORDER BY title <~> 'query'`, i.e. it
embeds a text column and ranks by semantic distance **[verified]** from
README.  I grepped the source for timestamp, interval, and time-bucket
handling and found nothing but a GUC for grid polling **[verified]**.

A timeseries imputation feature's object of study is the opposite: the
user's own column values, ordered by a timestamp, with the model as an
interchangeable black box.  The two features share the words "model" and
"PostgreSQL" and nothing else.  They would share no data structure, no
page format, no query primitive, no cost model, and no test fixture.  The
only thing they would genuinely share is `infer-client`'s HTTP transport
and its cancellation bridge to `CHECK_FOR_INTERRUPTS` **[verified]** from
`crates/infer-client/src/lib.rs`, which is reusable infrastructure, not a
conceptual link.  Reusing a transport is not a reason to put a feature in
a repo.

### The proposed composition, examined

The brief offers the strongest candidate: use the LM for structure
discovery (which columns are correlated or relevant) and the TS model for
the numbers.  It is the right thing to propose, and I think it fails on
three counts.

First, **pg_infer cannot currently answer the structure question.**  The
LM-for-structure step needs something like "which of these column names
are semantically related to the one with the gap".  pg_infer can produce a
number for that (`similar_to('temperature', 'humidity')`), but
`similar_to` is "the maximum shared gate activation ... the minimum
activation score of the best shared feature", falling back to
"cosine similarity * 10" when no features overlap **[verified]** from
DESIGN.  That is an unnormalised, uncalibrated, model-dependent score on
an arbitrary scale.  The `relation` labels that `describe()` returns are
worse: per a prior session's finding, they come from a hardcoded heuristic
over capitalisation, two magic thresholds (20.0, 30.0), and a 20-word
English geography list.  The project's own docs already concede the
failure mode, `docs/src/operations/bitnet-qualification.md` records that
`describe('France')` on BitNet returns `Zone`, `demographics`, `Barb`,
"structurally correct output ... but weak semantics ... the
feature-labelling heuristics were tuned on Gemma-class models".  So the
structure-discovery half would be built on the least trustworthy part of
pg_infer.

Second, **the structure question is better answered by the data.**  If you
want to know whether humidity helps predict temperature in *this* table,
computing the correlation over the observed rows is cheap, exact,
auditable, and does not care what a language model learned about the
English words "humidity" and "temperature".  An LM's view of two column
*names* is evidence about language, not about the table.  Worse, it fails
precisely where it would be most needed: on `col_a7`, `sensor_3142`, or
`v_meas_fixed`, which is what column names look like in real schemas.  I
can think of no case where the LM's answer should be preferred to the
data's when both are available, and the data is essentially always
available: you need observed rows to impute from anyway.

Third, **even granting the structure step, the numeric half still has to
work, and section 2 says it does not.**  The composition's value is capped
by its weaker half.  Chronos-class models have no imputation operation;
the hack costs 2.2x accuracy and drops an 80% interval to 36.9% coverage;
the models that actually win at imputation (TabPFN-TS, MoTM) are tabular
and time-indexed rather than timeseries foundation models, and the best of
them needs ~1 s per window on an H100.  A clever structure-discovery front
end feeding a numeric back end that cannot deliver a calibrated
distribution produces a feature that is sophisticated and wrong.

There is one composition I would call genuine, and it is small enough to
be a footnote rather than a feature: using pg_infer's existing text
similarity to *suggest* candidate covariate columns to a human who is
configuring an imputation job elsewhere.  That is a UX affordance on top
of `similar_to`, it needs no timeseries model in pg_infer, and it is
already expressible with the functions that exist.

---

## Question 5: prior art in databases

Nobody has shipped this, and the reason is informative: the adjacent
attempts have either stayed deliberately deterministic or died.

**TimescaleDB gap-filling, deterministic, correctly so.**
`time_bucket_gapfill()` has existed since 1.2.0 and buckets rows by time
interval while filling gaps; without an explicit algorithm, "gaps are left
as NULL in the returned data" **[verified]**.  The two algorithms are
`locf()` (last observation carried forward) and `interpolate()` (linear
between the bracketing observations).  Both return a single `DOUBLE
PRECISION` per bucket with no uncertainty column of any kind
**[verified]** from the docs' own worked example.  The brief is right that
this is not probabilistic.  I would add that it is not a gap in the
product: `interpolate()` is exact, explainable in one sentence, costs
nothing, and per the arXiv:2511.05980 benchmark **beats CSDI, TimesNet,
and BRITS out of domain** (0.506 vs 0.664, 0.677).  A deterministic
primitive that a user fully understands is a defensible design, not an
unfinished one.

**BigQuery ML ARIMA_PLUS, closest shipped thing, and it treats
imputation as cleaning, not inference.**  `ARIMA_PLUS` is a managed
in-database forecasting model type.  Its documented training pipeline
**[verified]** does: infer data frequency; handle irregular intervals;
handle duplicate timestamps by taking the mean; "interpolate missing data
using local linear interpolation"; detect and clean spike/dip outliers;
detect step changes; detect holiday effects; then STL decomposition and
ARIMA.  Note where imputation sits: it is **step four of preprocessing**,
it is **linear interpolation**, and it is deterministic.  Separately,
BigQuery ML offers `ML.IMPUTER` for feature columns, which the docs
describe as filling missing values with the column mean (or mode)
**[verified]**.  So Google, with every resource to do otherwise and a
fleet of timeseries foundation models in-house, ships mean-fill and linear
interpolation for the missing-value problem and reserves the model for
forecasting.  That is the most telling data point in this section.

Worth noting for section 1's licence discussion: BigQuery ML is also the
*only* sanctioned commercial path to TimesFM 3.0 **[verified]**, which
explains the carve-out in Google's licence.

**MADlib, retired.**  I checked the Apache Attic: MADlib "became a Top
Level Project in July 2017, retired in [date], and the move to the Attic is
In Progress" **[verified]**; both the TSA and ARIMA doc pages now open
with "This project has retired."  The GitHub repo shows 476 stars, not
archived, last push 2026-05-15 **[verified]**, but an Attic'd Apache
project is not a dependency to build on.  Its in-database ARIMA is prior
art for the *idea* of statistical timeseries modelling in SQL, and its
fate is prior art for the *sustainability* of it.

**PostgresML, does not do timeseries at all, and is stale.**  The brief
suspected this is the closest existing thing and might make a pg_infer
feature redundant.  I checked the source rather than the marketing.  Its
`Task` enum is `[regression, classification, decomposition, clustering,
question_answering, summarization, translation, text_classification,
text_generation, text2text, embedding, text_pair_classification,
conversation]` **[verified]**, no forecasting, no timeseries, no
imputation.  Its `Algorithm` enum lists 48 variants (linear, xgboost, svm,
lasso, ..., transformers, catboost, pca) with **no ARIMA, no ETS, and no
imputation algorithm** **[verified]**.  The README has zero matches for
"time series", "forecast", "arima", or "impute" **[verified]**.  So
PostgresML can run HF models in-database, as the brief says, but what it
runs are regression/classification/NLP tasks; it has no timeseries story
to be redundant with.  It is also at 6832 stars with **last push
2025-07-01** **[verified]**, roughly 17 months stale as of this writing.

**pgai, archived, and never did timeseries.**  Timescale's `pgai` is
embeddings, vectorizers, semantic search, RAG, and a semantic catalog for
text-to-SQL; its README has one incidental mention of "time-series" in a
marketing sentence about Timescale Cloud and no forecasting or imputation
functionality **[verified]**.  More to the point, the repository is
`archived: true` and the README now opens with "As of February 2026, this
project is no longer being maintained or supported" **[verified]**.

**pgvector-adjacent ML-in-PG generally.**  pgvector itself is a vector
type and index, orthogonal to this.  I did not find a maintained
PostgreSQL extension doing probabilistic timeseries imputation
**[verified]** to the extent a negative can be: no such thing surfaced in
the HF/PyPI/GitHub searches I ran.

The pattern across all five: the serious systems chose deterministic
interpolation on purpose (TimescaleDB, BigQuery ML), and the two
general-purpose "ML in Postgres" projects that might have grown into this
are respectively stale and archived.  Both of those died with far more
users than pg_infer has.  That is a strong prior about the maintenance
cost of in-database model serving, and it applies to pg_infer with extra
force given it has no production deployments yet (README: "No production
deployments exist yet").

---

## Verdict

**pg_infer should not take on timeseries-driven imputation.  There is no
minimum viable version worth building, so I am not proposing one.**

The reasons, in the order they would kill the feature:

**1. It is a different product.**  Section 4 is the decisive argument and
it does not depend on any model fact.  pg_infer's domain is a fixed
artefact indexed by *layer* and *feature*; imputation's domain is user
column values indexed by *time*.  They share a transport and nothing else.
The proposed LM-for-structure composition fails because the structure
question is better answered by correlating the observed data than by
asking a language model about column *names*.  It also fails because
pg_infer's structure-discovery surface (`similar_to`'s unnormalised
max-shared-gate score, `describe`'s hardcoded English heuristic) is its
least trustworthy part, already documented as degrading off Gemma-class
models.  Building
the feature would roughly double the project's conceptual surface while
pg_infer is still EXPERIMENTAL at v0.1.4-alpha with a SQL API that "may
change without notice" and no production deployments.

**2. The models do not do the task.**  No headline TSFM supports interior
in-filling.  Chronos tolerates NaN in the context and still only predicts
forward of its end **[verified]** in both source and behaviour;
`Chronos2Pipeline`'s public API has no impute method **[verified]**.  The
models that genuinely win at zero-shot imputation are TabPFN-TS and MoTM
(tabular and time-indexed, not timeseries foundation models), and
TabPFN-TS costs ~1 s per 672-step window on an H100 **[reproduced]**.
Shipping the hack instead costs 2.2x accuracy against a bidirectional
blend on the one case that distinguishes the tasks **[verified]**.

**3. The calibration claim cannot be honoured, which removes the whole
point.**  The BayesStore analogy lives or dies on returning a trustworthy
probability.  Forecast-mode TSFMs are decently calibrated (PCE at or
below 5% per arXiv:2510.16060 **[reproduced]**), but that property is
established for the forecasting task and does not transfer.  I measured a
nominal 80% band delivering **36.9% coverage** when the gap's answer
depends on right-context information **[verified]**.  Quantiles are not
even reliably ordered: 14.6% adjacent-pair crossings on sparse spiky
series **[verified]**, which is what a column of event counts or
slow-moving SKU sales looks like.  Conformal prediction would repair
coverage, at the cost of per-column held-out calibration state, and per
arXiv:2510.19345 it still yields coverage rather than event probabilities
**[reproduced]**, which is the wrong guarantee for `P(value = x)`.

**4. The sophisticated version is not reliably better than the thing that
already ships.**  Out of domain, linear interpolation beats CSDI,
TimesNet, and BRITS (MAE 0.506 vs 0.664, 0.677) **[reproduced]**.
TimescaleDB's `interpolate()` and BigQuery ML's local-linear preprocessing
step are not gaps in those products; they are the defensible answer.
Google ships mean-fill (`ML.IMPUTER`) and linear interpolation for missing
values despite owning TimesFM.

**5. The prior art is a graveyard.**  PostgresML: no timeseries task, no
ARIMA, 17 months since last push **[verified]**.  pgai: archived,
unmaintained since February 2026 **[verified]**.  MADlib: Apache Attic
**[verified]**.  Three projects with more users than pg_infer tried
in-database ML and stopped.

Note what is *not* on this list: compute cost.  Chronos-2 runs at 43 ms
and chronos-bolt-small at 15.5 ms per call on CPU **[verified]**, against
pg_infer's own 176 ms `describe()` and 354 ms `similar_to()`.  TTM is
0.81M parameters and 3.5 ms **[verified]**.  If the feature made sense,
the hardware would not stop it.  It does not make sense for reasons that
no amount of speed fixes.

### If the user wants this anyway

Then it should be a **separate extension** with its own name, repo,
version, and release cadence, sharing at most `infer-client`'s transport
crate.  It should also be honest about its form.  Concretely: expose
`locf`/`linear` deterministically (matching TimescaleDB's semantics so
users can migrate), and add a model-backed path only behind an explicit
opt-in that (a) runs a *bidirectional* context, never left-only, (b)
returns quantiles rather than a probability, (c) names the nominal level
and refuses to call it calibrated without a per-column conformal
calibration step, and (d) sorts quantiles before returning them, because
upstream will not.  Build it on TabPFN-TS or MoTM, not Chronos, because
those are the ones that do the task.  Budget for the latency that choice
implies.

That is a real project.  It is not pg_infer.

### Licence constraints to carry forward regardless

If any of this is revisited, three licence facts decide the shortlist and
are easy to get wrong because the surrounding code is permissive:

- **Moirai**: all checkpoints CC-BY-NC-4.0 (non-commercial) even though
  `uni2ts` is Apache-2.0 **[verified]**.  Unusable in shipped software.
- **TimesFM 3.0**: `timesfm-non-commercial-license-v1.0`, forbids
  revenue-generating and production use and forbids redistributing the
  weights; commercial use only via Google Cloud **[verified]**.  TimesFM
  2.5 and earlier remain Apache-2.0.
- **TabPFN-v2**: `priorlabs-1-1` is Apache-2.0 plus a clause requiring
  "Built with PriorLabs-TabPFN" displayed prominently and a "TabPFN"
  name prefix on derivative models **[verified]**.  Commercially usable
  with attribution.

Clean Apache-2.0/MIT with open weights: Chronos (all), Chronos-Bolt (all),
Chronos-2, TimesFM ≤2.5, Granite TTM/PatchTST/PatchTSMixer, Lag-Llama,
MOMENT (MIT), StatsForecast, PyPOTS (BSD-3-Clause).

---

## Sources

Primary, read directly:

- Chronos: https://arxiv.org/abs/2403.07815 (HTML v3; WQL definition in
  appendix, conformal calibration in §6.1 future work, quantisation
  limitation figure)
- Chronos-2: https://arxiv.org/abs/2510.15821 ·
  https://huggingface.co/amazon/chronos-2
- Chronos-Bolt: https://huggingface.co/amazon/chronos-bolt-base
- Chronos library source:
  https://github.com/amazon-science/chronos-forecasting
  (`src/chronos/chronos.py`, `chronos_bolt.py`, `chronos2/pipeline.py`)
- TimesFM: https://arxiv.org/abs/2310.10688 ·
  https://github.com/google-research/timesfm ·
  https://huggingface.co/google/timesfm-2.5-200m-pytorch ·
  licence: https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE
- TTM: https://arxiv.org/abs/2401.03955 ·
  https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2
- Moirai: https://arxiv.org/abs/2402.02592 ·
  https://huggingface.co/Salesforce/moirai-1.1-R-large
- MOMENT: https://arxiv.org/abs/2402.03885 ·
  https://github.com/moment-timeseries-foundation-model/moment
- Lag-Llama: https://arxiv.org/abs/2310.08278
- TimeGPT / Nixtla SDK: https://github.com/Nixtla/nixtla (§"License and
  access")
- **Imputation benchmark**: https://arxiv.org/abs/2511.05980, "Are
  Time-Indexed Foundation Models the Future of Time Series Imputation?"
- **Calibration study**: https://arxiv.org/abs/2510.16060, "Beyond
  Accuracy: Are Time Series Foundation Models Well-Calibrated?"
  (= Adler et al., https://openreview.net/forum?id=yeJlWzy8pV)
- **Forecast form taxonomy**: https://arxiv.org/abs/2510.19345 
  "Foundation Model Forecasts: Form and Function"
- CSDI: https://arxiv.org/abs/2107.03502 (Table 2 CRPS, Appendix E.3)
- SAITS: https://arxiv.org/abs/2202.08516 · BRITS:
  https://arxiv.org/abs/1805.10572 · GP-VAE:
  https://arxiv.org/abs/1907.04155 · ImputeFormer:
  https://arxiv.org/abs/2312.01728
- PyPOTS: https://arxiv.org/abs/2305.18811 ·
  https://github.com/WenjieDu/PyPOTS
- TabPFN: https://www.nature.com/articles/s41586-024-08328-6 ·
  https://huggingface.co/Prior-Labs/TabPFN-v2-reg
- TimescaleDB gapfill:
  https://docs.tigerdata.com/api/latest/hyperfunctions/gapfilling/time_bucket_gapfill/
- BigQuery ML ARIMA_PLUS:
  https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-time-series
- MADlib retirement: https://attic.apache.org/projects/madlib.html
- PostgresML source: https://github.com/postgresml/postgresml
  (`pgml-extension/src/orm/task.rs`, `orm/algorithm.rs`)
- pgai: https://github.com/timescale/pgai (archived)
- Conformal + TSFM: https://arxiv.org/abs/2507.08858
- QUAL (quantile alignment): https://arxiv.org/abs/2610.41100
- TSFM scepticism: https://arxiv.org/abs/2510.00742

Secondary / unverified:

- BayesStore (Wang et al., VLDB 2011): I could not retrieve the paper
  (dl.acm.org 403, vldb.org returned HTML not PDF).  The analogy is taken
  as the user framed it and none of my reasoning depends on the paper's
  details.  **[unverified]**
- TimeGPT "can be deployed on your own infrastructure" is a vendor claim;
  no public weights or terms found.  **[unverified]**
- MoTM (Le Naour et al. 2025): known only through the arXiv:2511.05980
  benchmark's description; I did not read the original.  **[unverified]**

Reproduction scripts and the venv for my own measurements are in
`/scratch/pg_infer/scratch-ts-research/` (throwaway).
