# Arsenal Match

Pitcher-similarity tools decide what "similar" means by hand, and none of them checks whether similarity
predicts anything. Arsenal Match learns a similarity metric from the outcomes of 1.4 million MLB pitches,
tests it on a held-out season against three published hand-built methods, and returns a pitcher's closest MLB
analogs together with how those analogs actually use their arsenals.

## What it Does

Pick an MLB pitcher-season in the app, or type in an arsenal of your own, and you get three lists of the ten
closest same-hand, same-role MLB pitchers side by side — my learned model, TJStats and HZB — all scored on one
scale, so the lists are comparable. Pitchers that TJStats or HZB rank in the top three while my model puts
them in the bottom half are flagged as "same measurements, different results". Pick any comp and you get his
pitches next to yours — velocity, movement, release, arm angle, and each pitch's whiff%, chase%, ground-ball%,
damage% and run value — plus the part a pitcher can actually borrow: that comp's gameplan, meaning his pitch
mix by count and batter hand with your own mix under it, shaded by the gap, and where he throws each pitch.

### What it is not

I started from a tool any pitcher at any level could use: enter whatever your league measures, get the MLB
pitchers you resemble, adjusted for level. It did not get there. What exists is a prototype at the MLB level,
trained and tested on MLB pitchers only. There is no tier or league adjustment anywhere in the project (D7),
so numbers from a lower level are compared to MLB numbers as if they were MLB numbers, and nothing here is
validated off MLB data. The MLB-level question it does answer is worth having on its own: does similarity
learned from outcomes pick better comps than similarity specified by hand? The gameplan view is useful at any
level, but the evidence behind it is MLB-only.

### Entering your own arsenal

There is no file upload. You choose a throwing hand, a role, a season to be compared against, and how much
your level measures. Then you fill in a table, one row per pitch type, with its usage % and its measurements.
Rows start at the MLB average for that pitch type and you edit them.

- **T1 (full Statcast)** is 12 numbers per pitch: velocity, ride, run, spin, extension, release height and
  side, arm angle, both approach angles, both release angles.
- **T3a (radar gun plus someone naming the pitch)** is three: velocity, ride, run. The model used is the one
  trained with everything else masked.
- Anything you don't enter is filled with the MLB average for that pitch type — location always, and every
  shape feature above your input level. A sparse entry is partly an average MLB pitcher.
- My model runs at both levels. TJStats needs all six of its measurements, so it runs only at T1. HZB needs
  pitch-by-pitch data, so it can't score a typed-in arsenal at all.

## Quick Start

```bash
git clone https://github.com/tomashigakithan/arsenal-match.git
cd arsenal-match
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app/app.py
```

The app runs from the committed weights and precomputed tables, so nothing has to be downloaded. To rebuild
everything from raw Statcast, see [SETUP.md](SETUP.md).

### Repo map

The pipeline runs in seven stages: **1** fetch → **2** features → **3** train → **4** evaluate → **5** comps
report → **6** app tables → **7** app. Each stage is a module under `src/`, run as
`python -m src.<package>.<module>`.

| Path | What it does | Stage |
|---|---|---|
| `src/data/` | `fetch_statcast.py` downloads MLB Statcast pitches from Baseball Savant, one month per file. | 1 |
| `src/features/` | `build_arsenal.py` cleans pitches, labels the 7 outcome classes, derives the physical features, assigns the season split and role, and builds the pitcher-season and pitch-type tables. | 2 |
| `src/models/` | The learned model: `arsenal_net.py` (network), `inputs.py` (what each data tier sees), `train_embedding.py` (sweep, ablation, publish). The published baselines: `physical_similarity.py` (TJStats, SEAM, raw Euclidean) and `hzb_similarity.py` (HZB). | 3; baselines used in 4–7 |
| `src/eval/` | `run_eval.py` drives the held-out test using `retrieval.py` (k-NN comps), `bootstrap.py` (paired CIs) and `calibration.py`. `comps_report.py` writes named comps and twin pairs. `slice_breakdown.py` regroups a saved run for new pitchers. `diagnostics.py` runs the 2025-only probes. `tjstats_fidelity.py` fits TJStats' σ to the site's own scores. | 4–5 |
| `src/app/` | `build_artifacts.py` precomputes the small tables the app loads; `gameplan_gap.py` holds the pitch pairing and the heat-map scale; `app.py` is the Streamlit app. | 6–7 |
| `data/` | Raw and processed Statcast (not committed; rebuilt by stages 1–2). `data/reference/` holds the hand-transcribed tjstats.ca scores. | 1–2, 4 |
| `models/` | Everything the app and the evaluation load instead of recomputing: see below. | written by 3 and 6; read by 4–7 |
| `docs/` | `design_decisions.md` (D1–D33, one entry per decision); `results/` and `figures/`, both indexed by `results/README.md`. | written by 3–5 |
| `notebooks/` | `results_overview.ipynb` reads the committed results back and walks through them, with its outputs saved. No training. | none |
| `videos/` | Demo and technical walkthrough. | none |

**What is in `models/`.** Four kinds of file, all committed, all small:

- `arsenal_net_t1.pt` and `arsenal_net_t3a.pt` — the trained weights, one per input level, written by
  `train_embedding.py --mode publish`. These are the model.
- `embeddings_{t1,t3a}.parquet` — each pitcher-season's embedding, so the evaluation and the app don't have to
  re-run the network.
- `published.json` — which sweep run shipped, so the weights trace back to the run that produced them.
- `models/app/` — the small precomputed tables the app reads (`build_artifacts.py`): the pitcher index, the
  arsenal and location tables, pitch mix by count, HZB distances, and two small JSON files. They are committed
  so the app runs from a clean clone with no Statcast download.

Training intermediates (`models/checkpoints/`, `models/sweep/`, `models/sample/`) are not committed. What
those runs produced is in the sweep and ablation tables under `docs/results/`.

## Video Links

- [Demo Video](https://drive.google.com/file/d/1uqTFKS0WqfT9D1a2K5gLYaOd3asynzA-/view?usp=sharing)
- [Technical Walkthrough](https://drive.google.com/file/d/1Yl91Ya44uwQ_IFLbmCfuwPBO5jJOf_zk/view?usp=sharing)

## Evaluation

2025 is the validation season and 2026 is the test, scored once with every choice already frozen. Every number
below is copied from a file in `docs/results/` that a script wrote; the index is
[`docs/results/README.md`](docs/results/README.md).

### How it was tested

The question is whether a pitcher's comps predict him.

- Queries are pitcher-seasons with at least 300 pitches. For each one, take the 10 most similar *other*
  pitchers from the same season, throwing hand and role (a starter is someone who started at least half his
  games): [`retrieval.py:67`](src/eval/retrieval.py#L67).
- Predict his run value per 100 pitches (RV/100, positive means runs saved) as the plain mean of theirs.
- Score with Pearson r and mean absolute error, against a no-similarity baseline: the mean of his whole
  candidate pool.
- Weights learn from 2023–24. Every choice — checkpoint, embedding size, auxiliary target, blend weight, which
  published baseline is the reference — is made on 2025. 2026 is scored once.
- Intervals are 95% paired-bootstrap intervals over pitchers, B = 2,000.

Every method faces that same protocol (D4, D5, D11, D20–D28).

**The target is noisy, and that caps every number here.** Split each 2025 season in half at random and a
pitcher's RV/100 in one half correlates with the other at 0.16. Stepped up to a full season (Spearman–Brown)
that is a reliability of 0.279, so even a perfect estimate of a pitcher's true RV/100 could only correlate
with his observed RV/100 at about 0.53 (`stat_reliability_val.csv`). Read every r below against 0.53, not
against 1. The rate stats are steadier: whiff% 0.781, GB% 0.733.

### What I compared against

Three published hand-built methods, re-implemented from their own descriptions and run on my Savant data under
my protocol. None was tuned on outcomes (D14) — only the learned model learns. Full implementation notes, the
gap-fills each one needed and the fidelity checks are in
[`docs/results/README.md`](docs/results/README.md#baseline-fidelity-are-my-rebuilds-the-real-methods).

- **TJStats** ([tjstats.ca](https://tjstats.ca/pitch-similarity/), T. Nestico) — z-scores six measurements
  within a pitch type, turns distance into similarity with a Gaussian kernel, matches pitch type to pitch
  type, and sums the per-pitch scores weighted by usage (D26). Reproduced to a mean error of 1.2–2.0 points
  against the site's own published scores.
- **HZB** (Healey, Zhao & Brooks 2017) — describes a pitcher as a set of pitch-type clusters and measures the
  distance between two pitchers as an Earth Mover's Distance between those sets. Pitch names never have to
  match (D27).
- **SEAM** (Wapner, Dalpiaz & Eck 2022) — a weighted distance over nine pitcher covariates, turned into a
  similarity by the paper's 1/d power and aggregated over pitch types (D19).
- **Raw Euclidean distance** on the ten unscaled measurements is a no-formula control.

The reference for the headline comparison is whichever published formula scores highest on 2025 RV/100. That
is HZB, and it stays the reference on 2026 (`alpha_star.json`; D28).

### Headline: do a pitcher's comps predict his run value?

**2025 (validation)**, from `method_comparison_val.csv`. n = 573 query pitchers; Δ is against HZB, paired.

| Method | r [95% CI] | MAE [95% CI] | Δr vs. HZB [95% CI] | ΔMAE vs. HZB [95% CI] |
|---|---|---|---|---|
| Raw Euclidean | 0.052 [−0.029, 0.130] | 0.800 [0.749, 0.851] | −0.093 [−0.188, 0.004] | −0.010 [−0.040, 0.021] |
| TJStats | 0.033 [−0.049, 0.117] | 0.820 [0.770, 0.873] | −0.112 [−0.201, −0.016] | +0.010 [−0.021, 0.041] |
| HZB | 0.145 [0.072, 0.215] | 0.810 [0.763, 0.859] | — | — |
| SEAM | 0.048 [−0.032, 0.130] | 0.812 [0.761, 0.863] | −0.096 [−0.179, −0.013] | +0.002 [−0.024, 0.028] |
| **Learned (T1, full Statcast)** | **0.240 [0.167, 0.315]** | **0.764 [0.715, 0.813]** | **+0.096 [0.011, 0.180]** | **−0.045 [−0.078, −0.015]** |
| Learned (T3a, radar gun + tagger) | 0.209 [0.132, 0.288] | 0.774 [0.725, 0.824] | +0.065 [−0.015, 0.152] | −0.036 [−0.065, −0.008] |
| *No similarity (pool mean)* | — | 0.786 | — | — |

**2026 (test)**, from `method_comparison_test.csv`. n = 557 query pitchers; the HZB reference and the blend
weight α\* = 1.0 are frozen from 2025 (`alpha_star.json`).

| Method | r [95% CI] | MAE [95% CI] | Δr vs. HZB [95% CI] | ΔMAE vs. HZB [95% CI] |
|---|---|---|---|---|
| Raw Euclidean | 0.038 [−0.048, 0.127] | 0.764 [0.708, 0.817] | −0.126 [−0.230, −0.015] | +0.020 [−0.011, 0.051] |
| TJStats | 0.081 [−0.002, 0.165] | 0.755 [0.704, 0.805] | −0.082 [−0.177, 0.010] | +0.011 [−0.016, 0.038] |
| HZB | 0.163 [0.073, 0.245] | 0.744 [0.692, 0.794] | — | — |
| SEAM | 0.052 [−0.034, 0.137] | 0.769 [0.717, 0.823] | −0.112 [−0.205, −0.011] | +0.025 [−0.002, 0.053] |
| **Learned (T1, full Statcast)** | **0.173 [0.083, 0.259]** | **0.733 [0.682, 0.783]** | **+0.010 [−0.090, 0.109]** | **−0.011 [−0.040, 0.019]** |
| Learned (T3a, radar gun + tagger) | 0.144 [0.059, 0.233] | 0.736 [0.681, 0.787] | −0.019 [−0.106, 0.068] | −0.008 [−0.033, 0.017] |
| *No similarity (pool mean)* | — | 0.722 | — | — |

On 2025 the learned model beat HZB by +0.096 in r, with the interval just clear of zero. On 2026 the two tie:
r 0.173 against 0.163, Δ +0.010 [−0.090, 0.109]. The 2025 edge did not replicate, and 2026 is the fair test,
because the model's checkpoint, embedding size and auxiliary target were all picked on 2025.

The learned model does beat the two weaker published methods on 2026: SEAM by +0.122 [0.003, 0.236] and
TJStats by +0.092 [−0.014, 0.193], though that second interval crosses zero (`baseline_deltas_test.csv`). Raw
Euclidean, TJStats and SEAM are not distinguishable from zero in either season. T3a, which sees only velocity
and movement, keeps most of T1's signal (0.144 against 0.173).

On absolute error, no method beats predicting the pool average on 2026 — 0.722 for the pool mean against 0.733
for the learned model. The comps rank pitchers better than chance, but averaging 10 comps' RV/100 is a noisier
point prediction than guessing the average.

### Which outcomes do comps capture?

One number per method hides what each kind of similarity is good at, so the same test runs on each of seven
stats (`stat_breakdown_{val,test}.csv`, `docs/figures/stat_breakdown_{val,test}.png`). Reliability is the
ceiling from `stat_reliability_{val,test}.csv`.

**2025 (validation)**

| Stat | Reliability | Raw r | TJStats r | HZB r | SEAM r | Learned r | Δr, learned − HZB [95% CI] |
|---|---|---|---|---|---|---|---|
| RV/100 | 0.279 | 0.052 | 0.033 | 0.145 | 0.048 | 0.240 | +0.096 [0.011, 0.180] |
| K% | 0.698 | 0.214 | 0.332 | 0.402 | 0.288 | 0.537 | +0.135 [0.066, 0.206] |
| BB% | 0.444 | 0.138 | 0.108 | 0.178 | 0.186 | 0.333 | +0.155 [0.066, 0.245] |
| Whiff% | 0.781 | 0.297 | 0.360 | 0.406 | 0.367 | 0.597 | +0.191 [0.130, 0.251] |
| Chase% | 0.515 | 0.064 | 0.042 | 0.127 | −0.013 | 0.207 | +0.080 [−0.021, 0.184] |
| GB% | 0.733 | 0.270 | 0.510 | 0.640 | 0.450 | 0.472 | −0.168 [−0.229, −0.108] |
| Damage% | 0.250 | 0.192 | 0.218 | 0.342 | 0.211 | 0.253 | −0.090 [−0.163, −0.021] |

**2026 (test)**

| Stat | Reliability | Raw r | TJStats r | HZB r | SEAM r | Learned r | Δr, learned − HZB [95% CI] |
|---|---|---|---|---|---|---|---|
| RV/100 | 0.265 | 0.038 | 0.081 | 0.163 | 0.052 | 0.173 | +0.010 [−0.090, 0.109] |
| K% | 0.683 | 0.221 | 0.421 | 0.472 | 0.365 | 0.509 | +0.037 [−0.023, 0.099] |
| BB% | 0.469 | 0.082 | 0.098 | 0.196 | 0.115 | 0.406 | +0.210 [0.131, 0.292] |
| Whiff% | 0.802 | 0.307 | 0.361 | 0.431 | 0.397 | 0.571 | +0.140 [0.076, 0.208] |
| Chase% | 0.638 | 0.123 | 0.096 | 0.203 | 0.130 | 0.335 | +0.132 [0.042, 0.224] |
| GB% | 0.733 | 0.217 | 0.452 | 0.611 | 0.438 | 0.434 | −0.177 [−0.241, −0.113] |
| Damage% | 0.458 | 0.134 | 0.205 | 0.262 | 0.153 | 0.223 | −0.040 [−0.129, 0.054] |

Three things hold in both seasons, and they are the findings I would defend. My comps predict walk rate better
than HZB's (+0.155, then +0.210). They predict swing-and-miss better (+0.191, then +0.140). And HZB's comps
predict ground-ball rate better than mine (−0.168, then −0.177). Every one of those six intervals excludes
zero.

The plate-discipline outcomes are where the learned space is stronger; contact quality is where the physical
formula is. HZB compares whole distributions of pitch shape, which is apparently what ground balls come from.
That split is one untested explanation for the RV/100 tie: run value mixes both kinds of outcome together.

### What else the evaluation shows

Each of these is a separate table, and none of them changed a choice:

- **It is the learning, not the feature list.** Given the learned model's exact 16 inputs, a hand-weighted
  z-scored distance loses on every stat (BB% Δr +0.164, whiff% +0.245), and adding the location shares to that
  distance changes nothing (`diagnostics/location_check_val.csv`).
- **The lists really are different.** The learned top 10 shares 1.5–1.9 names of 10 with each physical
  method's, against 2.4–4.1 between two physical methods and 0.7 by chance
  (`diagnostics/method_agreement_val.csv`).
- **The learned comps leave the arm slot.** Mean arm-angle gap to the top 10 is 11.7° for learned, 9.3° for
  HZB and 6.2° for TJStats, against 14.0° across the whole pool (`diagnostics/arm_angle_gap_val.csv`).
- **The auxiliary head is what makes the embedding work.** Asking it to also predict each pitcher's outcome
  mix lifts T1 from r 0.175 ± 0.012 to 0.224 ± 0.014 over three seeds; adding RV/100 to that target ties on r
  and makes retrieval decay after two or three epochs. Both tiers ship with the outcome mix only
  (`ablation_summary.csv`; D25).

| Tier | Auxiliary target | Retrieval r | MAE | Shipped |
|---|---|---|---|---|
| T1 (full Statcast) | none | 0.175 ± 0.012 | 0.786 | |
| T1 (full Statcast) | outcome mix | 0.224 ± 0.014 | 0.768 | ✓ |
| T1 (full Statcast) | outcome mix + RV/100 | 0.225 ± 0.020 | 0.778 | |
| T3a (radar gun + tagger) | none | 0.182 ± 0.019 | 0.781 | |
| T3a (radar gun + tagger) | outcome mix | 0.191 ± 0.019 | 0.774 | ✓ |
| T3a (radar gun + tagger) | outcome mix + RV/100 | 0.179 ± 0.023 | 0.786 | |

- **The outcome head is calibrated.** Expected calibration error per class is at most 0.012 on 2025 and 0.007
  on 2026, and log loss beats the class prior (1.109 against 1.689 on 2026). No temperature was fitted
  (`calibration_{val,test}.csv`, `docs/figures/reliability_{val,test}.png`; D12).
- **Whose twins actually pitch alike.** Pairs TJStats ranks top 3 while the learned space puts them in the
  bottom half differ more than the reverse case: mean |ΔRV/100| 1.001 against 0.980 on 2026, |Δwhiff%| 0.060
  against 0.041 (`twins_summary_test.csv`). The gaps are small and have no intervals. Named cases, including
  Marcus Stroman, and the reverse case where my model is the wrong one, are in `docs/results/comps_2025.md`.
- **Slices.** Bottom-quartile fastball velocity, and pitchers the model never trained on
  (`slices_{val,test}.csv`). The second one is the limitation below.

### The five limitations that matter most

There are 18, each tied to the file that shows it, in
[`docs/results/README.md`](docs/results/README.md#limitations-and-known-gaps). These five are the ones that
change how the results should be read.

1. **No method beats guessing the average on 2026.** The pool-mean MAE is 0.722 against the learned model's
   0.733 (`method_comparison_test.csv`). As a point prediction of RV/100, comps are worse than the pool
   average. On 2025 the learned model did beat it, 0.764 against 0.786.
2. **The RV/100 edge disappears on new pitchers.** On the 149 pitchers of 2026 never seen in training, learned
   scores 0.046 against HZB's 0.148 and TJStats' 0.131 (`slices_test.csv`). That is exactly the case the app
   is built for. The walk edge does survive there (BB% Δ +0.290 [0.143, 0.448]) and whiff% is borderline
   (+0.126 [0.007, 0.252]) (`stat_breakdown_new_test.csv`).
3. **The seed spread is bigger than the headline gap.** Three seeds of the shipped configuration score 0.240,
   0.214 and 0.217 on 2025 (`ablation.csv`) — a spread of 0.026 against a 2026 gap to HZB of +0.010. This
   setup cannot resolve a difference that small.
4. **The target is mostly noise.** RV/100's reliability is 0.279 on 2025 and 0.265 on 2026, capping any
   method's correlation at about 0.53 (`stat_reliability_{val,test}.csv`).
5. **MLB only, and two input levels.** Everything is trained and tested on MLB pitchers. T1 and T3a are the
   only levels tested, and no part of the project adjusts across levels (D7), so the pitcher at a data-poor
   level is the motivation, not a result.

## Future work

- **Feature-group permutation importance.** Shuffle one group of inputs (velocity, movement, spin, release,
  location) across pitchers and measure how far each stat's retrieval r drops, to show which inputs the model
  leans on for walks, whiffs and ground balls. Attempted late and left for later.
- **Learn TJStats' feature weights from outcomes** instead of comparing against them as published — the
  natural next experiment given D14.
- **Blending learned and physical similarity.** Mixing the two at a weight α was tried and left where it
  stopped: the best weight on 2025 was pure learned, so the blend had nothing to add
  (`alpha_sweep_{val,test}.csv`). Whether something smarter than a weighted average helps is open.
- **Sequence models.** Pitch-sequence architectures are recent prior art for this data and were ruled out here
  on time, not on merit.
- **Real lower-level data.** The motivating use case needs TrackMan or Hawk-Eye data from below MLB, plus a
  tested way to adjust across levels.

## References

Baselines re-implemented here:

- T. Nestico, **TJStats Pitch Similarity**, <https://tjstats.ca/pitch-similarity/> (methodology from the
  tool's "How It Works").
- G. Healey, S. Zhao and D. Brooks, **"Measuring Pitcher Similarity: Technical Details"**, viXra:1705.0098v1,
  2017, <https://vixra.org/abs/1705.0098>.
- J. Wapner, D. Dalpiaz and D. J. Eck, **"SEAM methodology for context-rich player matchup evaluations in
  baseball"**, arXiv:2005.07742v2, 2022 (v1: C. Young, D. Dalpiaz and D. J. Eck, 2020).

Methods and ideas this project builds on:

- M. Zaheer et al., **"Deep Sets"**, NeurIPS 2017 — the arsenal encoder's architecture.
- Y. Rubner, C. Tomasi and L. J. Guibas, **"The Earth Mover's Distance as a Metric for Image Retrieval"**,
  IJCV 40(2), 2000 — the distance HZB is built on.
- R. Flamary et al., **"POT: Python Optimal Transport"**, JMLR 22(78), 2021 — the exact EMD solver used here.
- M. A. Alcorn, **"(batter|pitcher)2vec"**, MIT Sloan Sports Analytics Conference, 2018 — precedent for
  learning player representations from outcomes.
- **"Using Clustering to Find Pitch Subtypes and Effective Pairings"**, SABR, 2020 — precedent for describing
  pitches by shape rather than by label.
- K. Lee and J. Ko, **"Unified Pitch Graphs for Diagnosing Pitching Strategy"**, arXiv:2609.03810, 2026.
- T. Tango, **Statcast Lab: pitcher similarity scores**, Baseball Savant, 2019 — the observation that
  motivated this project. Scherzer doesn't stand out on speed and movement, and Mikolas and Chatwood were each
  other's top comps at 99% while pitching very differently: a physical twin and a functional stranger,
  described by Savant's own analyst.

Design decisions and their tradeoffs are in [docs/design_decisions.md](docs/design_decisions.md). Every result
file is indexed in [docs/results/README.md](docs/results/README.md).

## Individual Contributions

Solo project by Toma Shigaki-Than. AI assistance is documented in [ATTRIBUTION.md](ATTRIBUTION.md) and in
file-level docstrings.
