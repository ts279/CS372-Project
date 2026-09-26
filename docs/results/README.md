# Results index

Every number in the top-level README comes from a file in this folder or in `docs/figures/`, and a script
writes every one of them. Nothing is typed by hand. This page says what each file holds, what it means, which
claim it backs, and which design decision it implements (`docs/design_decisions.md`, D1–D33). It also holds
the three things the README points at rather than prints: how faithful my rebuilds of the published baselines
are, the full list of limitations, and what the first training run changed.

`<split>` is `val` (2025, the validation season) or `test` (2026, scored once). To rebuild any of it, see
[SETUP.md](../../SETUP.md).

---

## The headline test — do a pitcher's comps predict his outcomes?

| File | Written by | What it holds | Backs | D |
|---|---|---|---|---|
| `method_comparison_<split>.csv` | `run_eval` | RV/100 r and MAE for every method, with paired-bootstrap 95% CIs and deltas against the reference baseline, plus the no-similarity pool-mean MAE | README *Headline* — the main result of the project | D4, D11, D20, D28 |
| `baseline_deltas_<split>.csv` | `run_eval` | Every method against **each** published baseline (TJStats, HZB, SEAM), paired CIs | the +0.122 over SEAM and +0.092 over TJStats on 2026 | D11, D28 |
| `alpha_star.json` | `run_eval --split val` | The blend weight α\* and the reference baseline, frozen on 2025 and read back on 2026 | that 2026 was scored with choices made in advance | D5, D28 |
| `alpha_sweep_<split>.csv` | `run_eval` | r and MAE of the learned ⊕ TJStats blend at α = 0.0 … 1.0 | README *Future work* — the blend was tried and added nothing | D4 |

**The blend, in one paragraph.** The idea was that learned and hand-built similarity might know different
things, so a weighted average of the two (α = 0 pure TJStats, α = 1 pure learned) could beat either. It
doesn't: on 2025 no interior α beats pure learned, so α\* = 1.0 was frozen and 2026 was scored with the
learned model alone.

## Model selection — what was chosen on 2025, and how

| File | Written by | What it holds | Backs | D |
|---|---|---|---|---|
| `sweep.csv`, `sweep_summary.csv` | `train_embedding` | The embedding-size sweep (8 / 16 / 32 numbers per pitcher × 3 seeds), per run and per configuration with mean ± SD | the hyperparameter search below | D21, D25 |
| `ablation.csv`, `ablation_summary.csv` | `train_embedding` | Input level (T1 / T3a) × auxiliary target (off / outcome mix / outcome mix + RV/100) × 3 seeds; a `selected` column marks what shipped | README *What else the evaluation shows* — the auxiliary head lifts T1 from 0.175 to 0.224 | D6, D10, D25 |
| `training_curves.csv` | `train_embedding` | Per-epoch training and validation log loss plus 2025 retrieval r, for every run | that retrieval r peaks early and then drifts down, which is why the kept epoch is chosen on r rather than on loss | D21 |

**How big the embedding should be** (`sweep_summary.csv`, 2025, mean ± SD over 3 seeds, auxiliary target held
fixed). The embedding is how many numbers stand for a whole arsenal; more is not better here.

| Numbers per pitcher | Retrieval r | MAE |
|---|---|---|
| 8 | 0.225 ± 0.020 | 0.778 |
| 16 | 0.220 ± 0.050 | 0.779 |
| 32 | 0.173 ± 0.044 | 0.785 |

8 and 16 are within one seed's worth of spread of each other, so the tie goes to the lower error and the
shipped models use 8 (D25).

## What the comps capture, and what caps it

| File | Written by | What it holds | Backs | D |
|---|---|---|---|---|
| `stat_breakdown_<split>.csv` | `run_eval` | The same test run separately on 7 stats × every method, with deltas against the reference baseline | README *Which outcomes do comps capture?* — the walk and whiff edge, and the ground-ball deficit | D22 |
| `stat_breakdown_new_test.csv` | `slice_breakdown` | The same breakdown restricted to 2026 pitchers never seen in training, regrouped from saved errors (nothing is re-scored) | limitation 2: the walk edge survives on new pitchers, whiff% is borderline, HZB still wins ground balls | D22, D28 |
| `stat_reliability_<split>.csv` | `run_eval` | How much of each stat is signal rather than luck (split each season in half at random, correlate the halves, step up to a full season), and the ceiling that puts on any method's r | README *The target is noisy* — RV/100 reliability 0.279, ceiling 0.528 | D10, D22 |
| `slices_<split>.csv` | `run_eval` | RV/100 by fastball-velocity band (bottom quartile vs. the rest) and by whether the model trained on that pitcher | limitation 2 — the learned RV/100 edge lives on pitchers the model trained on | D5 |

## Error analysis and calibration

| File | Written by | What it holds | Backs | D |
|---|---|---|---|---|
| `twins_<split>.csv` | `run_eval` | Every "physical twin, functional stranger" pair and its reverse, with both pitchers' stats | the named cases in `comps_2025.md` | D4 |
| `twins_summary_<split>.csv` | `run_eval` | Mean gap in each stat by pair type, and for each method's top 3 overall | README — whose twins actually pitch alike | D4 |
| `calibration_<split>.csv` | `run_eval` | Per-class calibration error, mean predicted vs. observed frequency, model log loss against the class prior | README — the outcome head's probabilities mean what they say, with no temperature fit | D12 |
| `reliability_bins_<split>.csv` | `run_eval` | The equal-count bins behind the reliability diagram, with the pitch count in each | the reliability figure; the counts also give how many pitches were scored (704,303 on 2025, 691,530 on 2026) | D12 |
| `comps_2025.md` | `comps_report` | Named top-10 comps from every method for about 20 hand-checkable 2025 pitchers, plus the biggest disagreements between methods | the qualitative cases: Marcus Stroman, Vesia/Cox, Scott/Fisher | D4, D23 |

**Three cases worth reading.** Marcus Stroman leads with an 89.7 mph sinker, and TJStats and the learned space
share none of his top 10: TJStats gives him four-seam pitchers like Antonio Senzatela (95.0 mph), the learned
space gives him sinker-and-cutter contact pitchers like Kyle Hendricks and Logan Webb. Alex Vesia and Austin
Cox are two left-handed relievers whose fastballs both average 92.7 mph; TJStats ranks Cox among Vesia's three
closest comps, the learned space puts him in the bottom half, and in 2025 Vesia saved 1.13 runs per 100
pitches while Cox cost 3.14. Tayler Scott and Braydon Fisher are the reverse case, where the learned space is
the wrong one. All three come from the largest-gap pairs, so they are extreme by construction; the mean gaps
in `twins_summary_<split>.csv` are the fair summary.

## Baseline fidelity: are my rebuilds the real methods?

All three baselines are re-implementations from published descriptions, run on my Savant data. None keeps its
own code, and none was tuned on outcomes (D14).

### TJStats — D26

**What the tool does.** It z-scores each measurement within a pitch type, converts distance to similarity with
a Gaussian kernel, matches each of the target's pitch types to the same type on the candidate, and sums the
per-pitch scores weighted by the target's usage.

**What I implemented.** The tool's default settings: six measurements (velocity, induced vertical break,
horizontal break, extension, release height, arm angle:
[`physical_similarity.py:118`](../../src/models/physical_similarity.py#L118)), raw Statcast pitch labels, a
pitch type counting once it is thrown at least 10 times, the kernel
([`:304`](../../src/models/physical_similarity.py#L304)) and the usage-weighted arsenal sum
([`:346`](../../src/models/physical_similarity.py#L346)).

**What I had to fill in.** Two things the site doesn't publish: the kernel width σ, and the population the
z-scores are taken over. I transcribed the site's own 2025 scores for Tarik Skubal (78 per-pitch cells), tried
8 variants, fitted σ per variant ([`tjstats_fidelity.py:219`](../../src/eval/tjstats_fidelity.py#L219)) and
kept the best by a rule fixed in advance ([`:312`](../../src/eval/tjstats_fidelity.py#L312)): six
measurements, z-scores over pitcher-season means with both hands pooled, **σ = 1.7444**.

**How close it is.**

- On Skubal, per-pitch mean error 2.33 points, Spearman 0.976.
- Frozen, then scored **once** on a held-out Zack Wheeler page: per-pitch mean error 2.02 over 81 cells,
  overall 1.23, and 18 of the 19 named top-20 comps shared. The bar, set in advance, was error ≤ 5 and overlap
  ≥ 15 of 20.
- A second check on a **2026** page the fit never saw, in the site's single-pitch mode — which drops the usage
  weighting and so tests the kernel and the z-scores alone — scored 19 named comps at a mean error of **1.30
  points** (Spearman 0.896, 16 of 19 in my own top 19, largest miss 5.8 points). Run it with
  `python -m src.eval.tjstats_fidelity --single-pitch`.
- The site's published methodology gives its kernel as e^(−d²/n), which implies σ = 1.7321 for six
  measurements. My fitted 1.7444 is 0.7% away from that.

**What differs by design.** The site has no role filter, a 10-pitch floor and a single-pitch mode. I run the
same formula on my own comparison pool — same hand, **same role**, at least 300 pitches, whole arsenal — so
that every method faces the same test. Run on the site's own pool it reproduces the site's ranking. The pool,
not the formula, is why the app's TJStats list for a given pitcher differs from the site's.

| File | Written by | What it holds |
|---|---|---|
| `tjstats_fit.json` | `tjstats_fidelity` | The chosen variant, σ = 1.7444, the selection rule, and the fit and held-out scores |
| `tjstats_fidelity_variants.csv` | `tjstats_fidelity` | All 8 candidate variants with their fitted σ, Spearman, error and top-20 overlap |
| `tjstats_fidelity_cells.csv` | `tjstats_fidelity` | Per-cell predicted vs. site score, which shows the remaining error sits on pitch-label differences |
| `tjstats_fidelity_heldout.csv` | `tjstats_fidelity` | The frozen variant scored once on a held-out tjstats.ca page |
| `tjstats_fidelity_2026.csv` | `tjstats_fidelity --single-pitch` | The same frozen configuration on a 2026 single-pitch page |

### HZB — D27

**What the paper does.** It describes a pitcher, against each batter hand, as a set of pitch-type clusters —
mean speed, horizontal movement and vertical movement, weighted by usage (eq. 1) — and measures the distance
between two pitchers as an Earth Mover's Distance between those sets, with a Mahalanobis ground distance
(eq. 2), then combines the two batter-hand sides by their league-average shares (eq. 4). Pitch names never
have to match, which is the point of using a transport distance.

**What I implemented.** Signatures at [`hzb_similarity.py:136`](../../src/models/hzb_similarity.py#L136), the
covariance at [`:153`](../../src/models/hzb_similarity.py#L153), the exact EMD at
[`:165`](../../src/models/hzb_similarity.py#L165) (POT's network simplex), the two sides combined at
[`:201`](../../src/models/hzb_similarity.py#L201).

**What I had to fill in.** Two gap-fills. Statcast labels and Hawk-Eye measurements stand in for Pitch Info
and PITCHf/x, which the paper says the EMD is not sensitive to. And when a pitcher never faced one batter
hand, I use the side both pitchers share; that touches 1,394 of 198,243 pitcher pairs in 2025 and **none** of
the 98,566 pairs where both pitchers have at least 300 pitches, so it never reaches the test.

**How close it is — and the limit.** The paper publishes no example comps and no distances, so there is
nothing to reproduce. All I can show is that my distance is a proper metric: D(A, A) = 0, symmetry, and no
violation of the triangle inequality in 10,000 random triples per hand, plus agreement to 4.4e-15 with an
independent SciPy Mahalanobis path ([`:272`](../../src/models/hzb_similarity.py#L272),
`python -m src.models.hzb_similarity --check`). That is a correctness check, not a fidelity check.

### SEAM — D19

**What the paper does.** A weighted distance over nine pitcher covariates, turned into a similarity by eq. 4's
1/d power, aggregated over pitch types.

**What I implemented.** A second configuration of the TJStats code
([`physical_similarity.py:153`](../../src/models/physical_similarity.py#L153)), with v2's power in the same
kernel function ([`:304`](../../src/models/physical_similarity.py#L304)). I cite v2 because v1 used a square
root and the formula in the code is v2's.

**What I had to fill in.** Three gap-fills. The weight matrix isn't published as numbers, so features are
z-scored within pitch type and the paper's default slider values are used — "stuff" 0.85, "release" 0.15
([`:152`](../../src/models/physical_similarity.py#L152)). Extension belongs to neither named group and is
counted as release. And pitch types are weighted by **usage** rather than v2's share of balls in play, because
that share depends on outcomes and would leak results into a baseline that is scored on predicting results.

**How close it is.** No check of the same kind is possible — the paper publishes no comps for a pitcher in my
seasons. This is a faithful implementation of a published formula with its unpublished parts documented, not a
verified replica.

## Diagnostics (2025 only — no claim or choice depends on them)

| File | Written by | What it holds | Backs | D |
|---|---|---|---|---|
| `diagnostics/location_check_val.csv` | `diagnostics` | A hand-built distance on the learned model's own 12 measurements, with and without the 4 attack-zone shares, against the learned model | README — location adds nothing to a hand-built distance; the learning is what helps | D17 |
| `diagnostics/t3a_vs_hzb_val.csv` | `diagnostics` | T1 and T3a against HZB, per stat, from the saved errors | what survives when location, spin and release are taken away: whiff% yes, walks no | D6 |
| `diagnostics/method_agreement_val.csv` | `diagnostics` | Mean top-10 overlap for every pair of methods, and the overlap two random lists would have | that the learned lists are not a re-skin of the physical ones | D4 |
| `diagnostics/arm_angle_gap_val.csv` | `diagnostics` | Mean arm-angle gap between a query and his top 10, per method, against the gap to his whole pool | that the learned space stops matching on arm slot | D4 |

## Data and app

| File | Written by | What it holds | Backs | D |
|---|---|---|---|---|
| `data_summary.csv` | `build_arsenal` | How many rows each of the 8 filter stages removed, pitches and pitcher-seasons per split with ratios, 300-pitch pitchers by role, and the 7 outcome-class shares | every dataset number in the README and SETUP (2,955,003 → 2,814,234 pitches; 50.4 / 25.0 / 24.6%) | D8, D13, D23 |
| `gameplan_gap_scale.csv` | `gameplan_gap` (run last by `build_artifacts`) | Percentiles of the usage gap over every comp cell the app can show | the heat map's full-red point, p90 = 33.8 points | D32 |

## Figures (`docs/figures/`)

| File | Written by | What it shows |
|---|---|---|
| `training_curves.png` | `train_embedding` | Training and validation log loss, and 2025 retrieval r, for the sweep's seed-372 runs |
| `stat_breakdown_<split>.png` | `run_eval` | Per-stat r by method with 95% CIs |
| `reliability_<split>.png` | `run_eval` | Reliability diagram per outcome class, with calibration error |

## Not committed

Two things `run_eval` writes are gitignored rather than committed:

- `errors_<split>.csv` (about 10 MB per split) holds one row per query × stat × method — the per-query table
  every summary above is grouped from. `slice_breakdown` and `diagnostics` read it when it is present.
- `docs/figures/alpha_sweep_<split>.png`, the α-blend curve. The numbers behind it are in
  `alpha_sweep_<split>.csv` and `alpha_star.json`, and no claim rests on the picture.

---

## Limitations and known gaps

This is a student project, and reviewing it turned up real weaknesses. Each one names the file that shows it.
The five that matter most for reading the results are in the top-level README.

**What the results themselves say**

1. **No method beats guessing the average on 2026.** The pool-mean MAE is 0.722; the learned model's is 0.733
   (`method_comparison_test.csv`). Comps rank pitchers better than chance, but as a point prediction of RV/100
   they are worse than the pool average. On 2025 the learned model did beat it, 0.764 against 0.786.
2. **The 2025 edge over HZB did not replicate.** +0.096 [0.011, 0.180] on 2025 became +0.010 [−0.090, 0.109]
   on 2026 — a tie (`method_comparison_{val,test}.csv`). Choices made on 2025 can look better on 2025 than
   they are, and this is what that looks like.
3. **The seed spread is bigger than the headline gap.** Three seeds of the shipped configuration score 0.240,
   0.214 and 0.217 on 2025 (`ablation.csv`), a spread of 0.026, against a 2026 gap to HZB of +0.010. A
   difference that small is not something this setup can resolve.
4. **The RV/100 edge disappears on new pitchers.** On the 149 pitchers never seen in training, learned scores
   0.046 against HZB's 0.148 and TJStats' 0.131 (`slices_test.csv`). That is the case the app is built for.
5. **HZB is better at contact, both years.** GB% 0.640 against 0.472 on 2025 and 0.611 against 0.434 on 2026
   (`stat_breakdown_{val,test}.csv`). Whatever whole-distribution shape matching knows about ground balls, the
   learned space does not.
6. **The target is mostly noise.** RV/100's reliability is 0.279 on 2025 and 0.265 on 2026
   (`stat_reliability_{val,test}.csv`), capping any method's correlation at about 0.53. Every r here should be
   read against that ceiling, not against 1.
7. **No multiple-comparison correction.** The per-stat table is 7 stats × 5 methods, and the slices add more.
   Some intervals will exclude zero by chance. Only the walk, whiff and ground-ball patterns repeat across
   both seasons, which is why those are the ones stated as findings.

**What the setup can't tell you**

8. **MLB only, and only two input levels.** Everything is trained and tested on MLB pitchers. T1 and T3a are
   the only levels tested, and there is no tier or league adjustment anywhere (D7), so the motivating use case
   — a pitcher at a data-poor level — is motivation, not a result.
9. **The baselines are rebuilds, not the original code.** TJStats is reproduced to a documented fidelity
   (above) but with a fitted σ. HZB and SEAM have two and three documented gap-fills and no published numbers
   to check against. A rebuild can be wrong in ways its own checks don't catch.
10. **k = 10 was fixed, never tuned**, and the prediction is an unweighted mean
    ([`retrieval.py:40`](../../src/eval/retrieval.py#L40); D20). No shrinkage, no distance weighting, no
    choice of k.
11. **One split, not cross-validation.** One training set, one validation season, one test season, with no
    pitcher-level folds — and most 2026 query pitchers appear in 2023–24 training, which is exactly what
    limitation 4 measures.
12. **Selection on the validation metric.** The kept epoch is the one with the best 2025 retrieval r (D21), so
    the 2025 numbers are optimistic by construction. Only 2026 is unbiased, and it was scored once.
13. **One architecture family.** Nothing here compares the Deep Sets encoder against, say, gradient boosting
    on the same inputs. The "learning, not the inputs" claim rests on beating a z-scored Euclidean distance on
    those same inputs (`diagnostics/location_check_val.csv`), which is a control, not a competitor.
14. **The GPU isn't deterministic.** Apple's MPS kernels on the auxiliary path give different numbers for the
    same seed (0.174 and 0.212 on two runs), which is why configurations are compared over three seeds and the
    shipped run is fixed in advance.

**What the model never sees**

15. **A pitcher is a season of averages.** Each pitch type is one row of season means, so nothing captures
    sequencing, tunnelling, how a pitch changed across the season, or a mid-season injury or role change.
16. **No context, and no command.** Run value is Statcast's `delta_run_exp`, which credits the pitcher for
    outcomes the batter, the defence, the park and the catcher also drive, with no adjustment for any of them.
    Location enters only as four attack-zone shares per pitch type (D17) — nothing about miss distance or
    intent — and the diagnostic says those shares add nothing to a hand-built distance.
17. **A few pitches go unlabelled.** 1,509 balls in play of 2.81M have no Savant contact code, so they carry
    no outcome label. They still count toward usage, shape and location (D13).
18. **A typed-in arsenal is partly an average.** In the app's own-arsenal mode, every feature above your input
    level, and location always, is filled with the MLB mean for that pitch type. The further your arsenal is
    from what you can measure, the more of the answer comes from the average rather than from you.

---

## Iteration history: what the first training run changed

`iteration1/` keeps the first full training run, superseded but not deleted, because the changes it forced are
the clearest evidence of evaluation-driven iteration in the project (D10, D21).

**Iteration 1 setup.** Three embedding sizes with the auxiliary head predicting the outcome profile **and**
RV/100; the ablation varied T1/T3a × auxiliary on/off. One seed each. Training both *ended* and *chose its
checkpoint* on 2025 pitch log loss.

**What it showed** (`iteration1/training_curves.csv`):

- The RV/100 output's 2025 error was worse than predicting the mean from the very first epoch (1.17–1.28
  against 1.07 for an untrained head). RV/100's reliability came out around 0.33 in that run — the 2025
  evaluation later put it at 0.279 — so the head was memorizing noise.
- 2025 retrieval r peaked early and then declined while pitch log loss kept improving: for `t1_emb8`, 0.269 at
  its best epoch against 0.177 at the epoch early stopping kept. Selecting on log loss was throwing away the
  better retrieval space.
- The same seed gave different results on two runs (r 0.174 and 0.212), because MPS kernels on the auxiliary
  path are not deterministic.

**What changed in iteration 2** (the shipped run):

- The kept checkpoint is the epoch with the best 3-epoch rolling mean of 2025 retrieval r, not the best log
  loss.
- Three seeds per configuration, compared on the mean; the shipped model is the seed-372 run, fixed in
  advance.
- The auxiliary target became an ablation axis: off / outcome mix / outcome mix + RV/100.
- Comps are matched on role as well as hand (D23).

**Result:** T1's 2025 retrieval r went from 0.177 at the kept epoch to 0.240 for the shipped model.

| File | What it is |
|---|---|
| `iteration1/sweep.csv`, `iteration1/ablation.csv` | The first run's per-run metrics |
| `iteration1/training_curves.csv`, `iteration1/training_curves.png` | Its per-epoch curves — the decline in retrieval r is visible here |
| `iteration1/train_embedding_log.txt` | The console log of the run that wrote them |
