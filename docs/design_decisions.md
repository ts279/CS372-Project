# Design decisions

One entry per decision. Each holds what was chosen, why, the alternatives that were considered and rejected,
and the evidence: a result file in `docs/results/`, a figure in `docs/figures/`, or the code that implements
it.

---

## D1. Learn the similarity metric and test it, instead of building another comps tool

- **What:** the contribution is a similarity metric learned from pitch outcomes, plus a held-out test of whether
  a similarity metric predicts anything at all. The comps tool is how the result is delivered, not the result.
- **Why:** hand-built comps tools already exist — [TJStats Pitch Similarity](https://tjstats.ca/pitch-similarity/)
  z-scores metrics within pitch type and applies an RBF kernel;
  [SeeMagnus](https://www.seemagnus.com/blog-posts-test/whats-my-major-league-pitcher-comp-a-tool-that-tells-you)
  computes a distance on five metrics from uploaded TrackMan data. Neither learns its metric or reports a
  held-out accuracy number. That gap is where this project sits. The framing: TJStats, HZB and SEAM are
  hand-built formulas run on my Savant data, kept as published apart from documented gap-fills; my model is
  trained from scratch and learns its own similarity from outcomes. The test is whether a held-out pitcher's
  comps' actual outcomes predict his. What the physical formulas miss shows up where learned comps beat
  physical comps, and in the physical-twin / functional-stranger pairs. A null result is still a finding.
- **Alternatives:** (1) another hand-distance comps tool — already exists twice; (2) a pitch-sequence model
  recommending counterfactual pitch calls — recent prior art, and it doesn't fit the time budget;
  (3) a feature-ablation study as the headline — no natural product, and its central claim (which features
  matter for lower-level pitchers) can't be validated on MLB-only data.
- **Evidence:** the prior-art review in [ATTRIBUTION.md](../ATTRIBUTION.md). Whether the learned metric helps is
  measured in D4.

## D2. Train the embedding through an outcome-prediction task

- **What:** a two-branch network. An arsenal encoder maps a pitcher's arsenal to a low-dimensional embedding
  (a set encoder, D18); the rest of the network combines that embedding with each pitch's own features and
  predicts the pitch's outcome over seven classes (D13). The embedding is the space comps are retrieved in.
- **Why:** a hand-specified distance fixes which features matter, and how much, before looking at any outcomes.
  Supervising the encoder on outcomes pushes the embedding to keep the arsenal information that predicts
  results and discard the rest. The cost: per-pitch outcomes are very noisy, so similarity is measured at the
  pitcher level (D5), where aggregation recovers signal.
- **Alternative:** a hand-specified distance only, as in the tools in D1.
- **Evidence:** `method_comparison_{val,test}.csv` — the learned embedding's comps predict RV/100 at r 0.240 on
  2025 and 0.173 on 2026, against 0.033 / 0.081 for TJStats and 0.145 / 0.163 for HZB. Training curves:
  `training_curves.csv`, `docs/figures/training_curves.png`.

## D3. Statcast's native run value, and a multi-class outcome instead of a binary one

- **What:** each pitch is one of seven classes (D13). Expected run value is Σₖ P(outcome k) × RVₖ, where RVₖ is
  the empirical mean of Statcast's `delta_run_exp` for that outcome in the same count and base-out state.
- **Why:** hand-building a run-expectancy table costs hours and adds error without changing the quantity
  measured. A binary target (contact vs. swing-and-miss) is simpler but rewards any pitch that avoids contact,
  including a 3-0 ball. Weighting outcomes by run value charges each outcome what it actually costs in that count.
- **Alternatives:** a hand-built run-expectancy table; a binary whiff/contact target.
- **Evidence:** the sign of `delta_run_exp` was checked before use — it is from the batting team's perspective
  (in June 2024, called strikes average −0.064 and swinging strikes −0.109), so the pitcher's RV/100 flips the
  sign. Implemented in `src/features/build_arsenal.py` (`aggregate_pitcher_seasons`).

## D4. Compare every similarity method under one protocol, and report the blend curve instead of a winner

- **What:** score five methods identically on held-out-pitcher retrieval (D20): raw Euclidean distance; TJStats
  rebuilt to the tool's defaults (D26); HZB (D27); SEAM's published score (D19); and the learned embedding.
  Then blend the learned similarity with TJStats at weight α, sweep α from 0 (pure TJStats) to 1 (pure learned),
  and plot retrieval quality against α. SEAM is its own comparison row and is never averaged into the blend.
- **Why:** a winner-take-all comparison makes the headline depend on the learned metric winning, which the
  design can't guarantee. The α curve is informative either way: an interior optimum means the two kinds of
  similarity carry complementary information, an endpoint means one of them is sufficient. Raw Euclidean on
  unscaled features is dominated by spin rate (rpm values dwarf mph and inches); it is kept on purpose as the
  naive control, and the z-scored methods are the serious baselines.
- **Why Savant's own similarity isn't a sixth method:** Savant's speed-and-movement similarity publishes no
  distance, weights or scaling and its page is JavaScript-only, so it can't be replicated faithfully; and
  Savant's "Player Similarity Scores" fingerprint pitchers on *outcomes*, which D17 keeps out of the fingerprint.
- **Alternative:** a head-to-head learned-vs-baseline comparison with a single winner.
- **Data provenance:** every number comes from Baseball Savant (`src/data/fetch_statcast.py`). The baselines are
  *formulas* re-implemented and run on that data; nothing was scraped from their tools.
- **Evidence:** `method_comparison_{val,test}.csv`, `baseline_deltas_{val,test}.csv`,
  `alpha_sweep_{val,test}.csv` (α* = 1.0 on 2025, `alpha_star.json`),
  and the top-10 overlap between methods in `diagnostics/method_agreement_val.csv`.

## D5. Split by season into train / validation / test, and hold out the pitcher, not the pitcher-season

- **What:** train on 2023–24, validate on 2025, test on 2026. Early stopping, the sweep, the checkpoint, the aux
  target, α, and the headline baseline are all chosen on 2025. 2026 is scored once, at the end. Within an
  evaluation season, when a pitcher is the query every season of his is removed from the index.
- **Why:** with a two-way split, tuning would happen on the test season and the reported number would be
  optimistic. Validating on held-out 2023–24 pitchers keeps tuning off the test season but shares seasons with
  training, so it can't detect year-to-year drift — and drift is exactly what a future-season test faces. A
  random pitch-level split would put the same pitcher on both sides and measure memorization. Holding out only
  the query's pitcher-season would leave his own earlier arsenal in the index as his nearest neighbour, and
  every method would look good.
- **Alternatives:** the original two-way split; a held-out group of 2023–24 pitchers; a random pitch-level
  split; holding out the pitcher-season only.
- **Cost:** an extra download; a partial test season (2026 runs to the day before the pull); and fewer usable
  queries, because a pitcher needs enough pitches in the evaluation season.
- **Evidence:** `SPLIT_BY_YEAR` in `src/features/build_arsenal.py`; `data_summary.csv` — train 1,417,523 pitches
  / 1,606 pitcher-seasons, 2025 704,713 / 803, 2026 691,998 / 809 (50.4 / 25.0 / 24.6%); 573 and 557 retrieval
  queries in `method_comparison_{val,test}.csv`.

## D6. Simulate lower-data tiers by masking features in one pipeline

- **What:** two input tiers exist anywhere in the project. **T1** is full Statcast. **T3a** is a radar gun plus a
  pitch tagger: pitch family, velocity, movement rounded to the nearest 3 inches, and count — no location, spin,
  release, or angles, and contact quality collapsed to "in play". Both are evaluated. Two further tiers sketched
  early on (T2, ball flight without batted-ball detail; T3b, pitch type and count only) were **cut entirely**,
  app inputs included.
- **Why:** real lower-level data isn't available here (no college TrackMan, and the minor-league pull is parked,
  D8). Masking keeps everything else fixed, so any drop in retrieval quality is attributable to the missing
  features. T3a is the tier furthest from full Statcast that still describes the pitch itself, which makes it
  the most informative single comparison. Masked MLB pitchers are still MLB pitchers, so the tier results
  measure information loss and nothing more — no claim is made about college or high-school pitchers.
- **Alternative:** validating on real lower-level data.
- **Evidence:** `ablation_summary.csv` (2025, mean ± SD over 3 seeds) — with the outcome-profile aux target,
  T1 r 0.224 ± 0.014 vs. T3a 0.191 ± 0.019; with no aux head, 0.175 vs. 0.182. On 2026 the shipped T3a scores
  0.144 against T1's 0.173 (`method_comparison_test.csv`). Per-stat: `diagnostics/t3a_vs_hzb_val.csv`.

## D7. No level adjustment ships

- **What:** the project has no way to adjust a pitcher's metrics across levels. A velocity offset was planned
  first as a swept parameter, then as an app slider labelled an untested assumption, and finally cut altogether.
- **Why:** the true gap between levels can't be estimated from MLB-only data, so any constant would be an
  untested assumption dressed as a feature. A sweep would have made the assumption visible, but it can't make
  it true, and shipping a control that nothing validates invites the reader to trust it.
- **Alternatives:** a single fixed offset built into the model; Bauer Units, which SeeMagnus recommends for
  sub-90 mph fastballs; the sweep; the slider.
- **Evidence:** none, and that is the point — no result about level adjustment is claimed anywhere. The
  consequence is stated in the README's Limitations.

## D8. Scope the data to MLB Statcast 2023–2026

- **What:** MLB regular-season pitches pulled from Baseball Savant. 2023–2025 is 2,147,196 pitches; 2026 runs
  from March to the day before the pull. `build_arsenal` removes rows that aren't usable pitches and prints the
  count at each step: non-regular-season games, pitch-clock violations (no pitch thrown), bunt attempts (D13),
  untracked pitches, five pitch types (D15), and position players pitching.
- **Why:** position players pitching in blowouts throw 40–60 mph lobs; they are not the population a comp tool
  serves, and statsapi's primary position is a rule rather than a judgment call. Minor-league Statcast was
  probed and parked: the query returned the right schema and zero rows, and TJStats already covers MiLB, so
  adding it wouldn't set this project apart. College TrackMan: no data access. The consequence is that every
  result here is about MLB pitchers.
- **Alternatives:** adding minor-league Statcast; adding college TrackMan.
- **Evidence:** `data_summary.csv` — 2,955,003 raw pitches in, 2,814,234 kept, with the rows removed by each of
  the eight filter stages; `src/data/fetch_statcast.py`.

## D9. Commit the final model and app artifacts, not the data

- **What:** commit the final weights and the small precomputed tables the app loads, each well under 50 MB.
  Keep raw data (325 MB), processed data, training checkpoints, sweep runs, and per-query error tables out of git.
- **Why:** a slightly larger repository buys a Quick Start that runs from a clean clone, with no 325 MB download
  and no retraining run.
- **Alternative:** gitignore all model files.
- **Evidence:** [`.gitignore`](../.gitignore); `models/app/` and `models/arsenal_net_{t1,t3a}.pt`.

## D10. An auxiliary head, so the arsenal embedding can't be bypassed

- **What:** a second output head whose **only input is the arsenal embedding**. It predicts the pitcher-season's
  **outcome profile** (the share of his pitches in each of the seven classes) and, in one ablation arm, his
  **RV/100**, z-scored on the training seasons. Training loss = pitch cross-entropy + λ × auxiliary loss.
- **Why:** the main head also sees each pitch's own velocity, movement, location and count, so it can predict
  outcomes well while largely ignoring the embedding. If that happens the embedding gets almost no training
  signal and the retrieval space is close to arbitrary, while the outcome head still looks fine. The auxiliary
  head can only lower its loss if the embedding itself carries outcome information. It doesn't force the *main*
  head to use the embedding — but retrieval runs on the embedding, so what the embedding contains is what
  matters. RV/100 is standardized because raw RV/100 spans several runs while the profile cross-entropy moves in
  fractions of a nat; unscaled, the squared error would swamp the outcome-mix loss.
- **Alternatives:** no auxiliary head; removing the per-pitch features so the embedding is the only input (the
  model could then no longer account for location and count); predicting RV/100 *instead of* the profile (a
  noisier target, and two arsenals can reach the same run value in different ways — whiffs vs. weak contact).
- **What the first full run showed:** the RV/100 output overfits from the first epoch — its 2025 squared error
  (1.17–1.28 standardized) is worse than an untrained head's (1.07), which is roughly what predicting the mean
  scores. The cause is noise: in that run RV/100's full-season reliability came out at 0.33, and the 2025
  evaluation later put it at 0.279 (`stat_reliability_val.csv`) — either way most of the target is luck, and
  the head memorizes it. The profile output does generalize, slightly (2025 cross-entropy 1.6913 vs.
  1.6931 for the league-average profile, against a floor of 1.6835). The auxiliary target therefore became an
  ablation axis (off / profile / profile + RV/100), and the shipped models use profile only.
- **Evidence:** `docs/results/iteration1/` (the first run); `ablation_summary.csv` — T1 r 0.175 ± 0.012 with no
  aux head, 0.224 ± 0.014 with the profile, 0.225 ± 0.020 with profile + RV/100, and the profile arm has the
  lower MAE (0.768 vs. 0.778), so it ships (D25). The 0.33 figure is the first run's, kept in
  `docs/results/README.md` (*Iteration history*); 0.279 is the current one.

## D11. Paired bootstrap confidence intervals, resampling pitchers

- **What:** 95% intervals from a paired bootstrap over query pitchers — resample the held-out pitchers with
  replacement 2,000 times, apply the same resamples to every method and every α, and report each method's
  interval *and* the interval of each pairwise difference.
- **Why:** the held-out pitcher is the unit of the test, and one pitcher's pitches are correlated, so resampling
  pitches would make the intervals look narrower than they are. Pairing cancels the shared difficulty of a given
  pitcher, so a difference interval that excludes zero is real evidence; two separately computed intervals can
  overlap even when one method is consistently better. A t-test would assume roughly normal errors, and
  per-pitcher run-value errors are skewed.
- **Alternatives:** point estimates only; separate unpaired intervals; a t-test.
- **Limit:** the intervals cover uncertainty about *which pitchers were queried*, not about the composition of
  the retrieval index or the training run.
- **Evidence:** `src/eval/bootstrap.py` (B = 2000); every interval in `method_comparison_*.csv`,
  `baseline_deltas_*.csv`, `alpha_sweep_*.csv`, `slices_*.csv`, `stat_breakdown_*.csv`.

## D12. Check the calibration of the outcome head

- **What:** evaluate the 7-class head's probabilities, not just its top choice, on both seasons: a reliability
  diagram per class (equal-count bins), expected calibration error, and log loss against a class-prior baseline.
  The planned temperature fit was cut; miscalibration would be reported as a finding, not corrected.
- **Why:** expected run value multiplies the class probabilities directly, so a head that is systematically too
  confident or too timid biases it even when the top choice is right. Accuracy is nearly meaningless here —
  single pitches are noisy enough that the most likely class is usually "ball". Equal-count bins rather than
  uniform ones, because a rare class like damage (~2.5%) would otherwise land almost entirely in one bin.
- **Alternatives:** accuracy and log loss only; fitting a temperature on 2025.
- **Evidence:** `calibration_{val,test}.csv`, `reliability_bins_{val,test}.csv`,
  `docs/figures/reliability_{val,test}.png`. Per-class ECE is at most 0.012 on 2025 and 0.007 on 2026 (both the
  ball class); log loss 1.113 vs. 1.696 for the class prior on 2025, and 1.109 vs. 1.689 on 2026. The head is
  calibrated without a temperature fit.

## D13. Outcome classes: what counts as poor contact, a flare, or damage

- **What:** seven classes — called strike, ball, whiff, foul, **poor**, **flare**, **damage**. Contact comes from
  Savant's `launch_speed_angle` code: poor = 1 Weak / 2 Topped / 3 Under, flare = 4 Flare-Burner, damage =
  5 Solid / 6 Barrel. A foul tip is a whiff; a hit-by-pitch is a ball. Balls in play with no contact code get no
  label, but their pitches still count toward usage, shape and location. Removed entirely: pitchouts,
  pitch-clock violations, and **bunt attempts** (foul bunts, missed bunts, bunt foul tips, and bunts put in play,
  found by "bunt" in the play description) — no label, and they count toward nothing.
- **Why:** flares and burners are kept separate from both poor contact and damage because they go for hits far
  more often than the other weak-contact codes. A bunt's outcome reflects the batter's intent to bunt, not how
  the pitch played, and the bunt pitch leaves the fingerprint for the same reason: a pitcher who happens to be
  bunted on more often would otherwise have those pitches counted as part of how he attacks hitters, when the
  batter chose not to swing at all. A foul tip is a swing the catcher holds and is scored as a strike like a
  swinging miss, so it is grouped with whiffs.
- **Alternatives:** Savant's six contact codes as six classes; a single "in play" class (still used at T3a,
  which has no batted-ball quality); dropping only the bunt *label* and keeping the pitch in the fingerprint.
- **Evidence:** `data_summary.csv` — shares of labelled pitches: ball 36.1%, foul 18.1%, called strike 16.4%,
  whiff 12.2%, poor 10.5%, flare 4.2%, damage 2.5%; the bunt filter removes 10,964 pitches.

## D14. Who sets the weights in "similar"

- **What:** no hand-chosen weighting anywhere in this project. The published baselines use **their published
  weights exactly**; only the learned model learns its own.
- **Why:** the question is whether a learned notion of similarity beats hand-specified ones. Tuning the
  baselines' weights would add a third, unpublished hand-specified method — and would tune it on the same data
  the comparison uses. Keeping every published baseline as published, and letting only the learned model learn,
  is what makes the comparison controlled.
- **Alternative:** an expert-weighted distance reflecting my own pitching judgment. Reasonable as a
  product feature; it would blur the experiment.
- **Evidence:** the `TJSTATS`, `SEAM` and `RAW_EUCLIDEAN` configs in `src/models/physical_similarity.py`; where a
  published method leaves a weight unspecified, the gap-fill is stated in D19 and D27.

## D15. Pitch identity: family plus shape, not the Statcast name; an arsenal threshold of 3%

- **What:** in the learned model each pitch type is described by its **family** (Savant's groups: fastball =
  FF/SI/FC, breaking = SL/ST/SV/CU/KC/CS, offspeed = CH/FS/FO/SC) plus its **shape** (velocity, movement, spin,
  release, approach and release angles). The Statcast pitch name is never an input. KN, EP, FA, PO and UN are
  removed. A pitch type joins a pitcher-season's arsenal at **≥ 3% usage**, and arsenal usage is renormalized
  over the types that qualify; sub-3% pitches stay in the pitch-level training data as real pitches with real
  outcomes, they just aren't part of the fingerprint.
- **Why:** Statcast names are a classifier's labels, and the line between a slider and a sweeper, or a cutter
  and a hard slider, is partly arbitrary — shape says what the pitch actually does, and the family keeps the
  broad role that shape alone can blur. The removed types are rare and either unrepeatable (KN, EP) or not pitch
  descriptions at all (FA, PO, UN). A pitch thrown 1% of the time is noise in a fingerprint.
- **Alternatives:** the Statcast name as a categorical input; no usage threshold.
- **Evidence:** `data_summary.csv` (the KN/EP/FA/PO/UN filter removes 8,451 pitches); `aggregate_pitch_types` in
  `src/features/build_arsenal.py`, which prints how many pitcher-season × type rows clear 3%. Note that the
  TJStats and SEAM baselines deliberately keep matching on the raw Statcast name, as published (D26, D19).

## D16. Same-hand comps only

- **What:** a right-hander is compared only with right-handers, a left-hander only with left-handers, for every
  method.
- **Why:** a lefty and a righty with mirror-image arsenals face platoon matchups the opposite way around, so one
  isn't a usable gameplan comp for the other.
- **Implementation:** horizontal features are converted to arm-side-positive values, so both hands can share one
  standardization population; retrieval is then restricted to the same hand.
- **Alternative:** cross-hand comps on mirrored features.
- **Evidence:** `add_features` (arm-side columns) in `src/features/build_arsenal.py`; the `hand` argument
  throughout `src/models/physical_similarity.py` and `src/eval/retrieval.py`.

## D17. Location in the fingerprint: where he throws, never how it turned out

- **What:** each pitch type's fingerprint includes its **Savant attack-zone shares** — the fraction thrown to
  heart, shadow, chase and waste, on Savant's units where 100% is the zone edge (horizontally half the plate
  plus a ball radius, vertically the batter's own `sz_top`/`sz_bot`; rings at heart < 67% ≤ shadow < 133% ≤
  chase < 200% ≤ waste). **Results never enter the fingerprint**: no strike%, whiff%, chase rate or run value.
  Location is masked at T3a, where a radar gun and a scorebook can't record it.
- **Why:** two pitchers with the same stuff can play very differently depending on where they throw it. And
  letting results into the fingerprint would be target leakage: the model
  would "predict" run value by reading it.
- **Alternatives:** shape only; raw mean plate location (which loses the spread — a pitch that lives on the
  edges versus one that lives in the middle); outcome rates as features (rejected as leakage).
- **Evidence:** `attack_zone()` in `src/features/build_arsenal.py`; league-wide shares from the build are heart
  24.7%, shadow 40.9%, chase 23.8%, waste 10.6%, in line with Savant. What location is *worth* is measured in
  `diagnostics/location_check_val.csv`: adding the four zone shares to a hand-built distance on the same inputs
  changes nothing (every interval crosses zero), while the learned model beats that same-input distance on every
  stat.

## D18. A set encoder for the arsenal

- **What:** the arsenal encoder is a **Deep Sets** encoder (Zaheer et al. 2017): one shared small network runs on
  every pitch type, the results are combined by a **usage-weighted average**, and a final layer maps that to the
  embedding.
- **Why:** arsenals differ in size (2 to 9 types) and have no natural order. A set encoder handles any size and
  gives the same answer whatever order the pitches are listed in. Usage weighting means a pitch he throws 40% of
  the time shapes the fingerprint more than one he throws 5% of the time.
- **Alternatives:** fixed slots ("four-seam columns, slider columns…") zero-filled when absent — which ties the
  model to the Statcast names D15 removes and wastes capacity on empty slots; an attention-based set encoder —
  more flexible, but more parameters and harder to explain, and nothing shows the simpler pooling falls short.
- **Evidence:** `SetEncoder` in `src/models/arsenal_net.py`. No alternative encoder was trained, so this is a
  design choice, not a tested result.

## D19. SEAM's similarity as a published baseline, sharing the TJStats code

- **What:** SEAM's pitcher similarity score, implemented as a second configuration of the TJStats code. Formula
  source is **v2** (Wapner, Dalpiaz & Eck, arXiv:2005.07742v2, 2022, eq. 4); v1 writes the same score with a
  square root, and v2 replaced it with the 1/d_p power. Both methods z-score features within pitch type, take a
  weighted squared distance, and turn it into a similarity with an exponential. They differ in:

  | | TJStats | SEAM |
  |---|---|---|
  | Features | 6 (D26) | 9: velocity, spin, horizontal and vertical break, horizontal and vertical **release angle**, release side and height, extension |
  | Per-pitch similarity | exp(−d²/2σ²) | exp(−(Δ′VΔ)^(1/n)), v2 eq. 4 |
  | Pitch labels | Statcast | KC → CU, FO → FS; SC removed (§4) |

- **Why a third published baseline:** SEAM is an academic similarity score built on the same Statcast data, so
  "learned vs. hand-specified" doesn't rest on one website's choices.
- **Gap-fills (three, all documented):** (1) **V isn't published numerically** — the paper says it trades "stuff"
  (velocity, spin, movement) against "release" via a slider whose default is 0.85, so here features are z-scored
  within pitch type and stuff gets 0.85, release 0.15; (2) **extension is in neither named group** and is counted
  as release; (3) **arsenal weighting** — v2 weights pitch types by each type's share of the pitcher's *balls in
  play*, which depends on outcomes and would leak results into a baseline scored on predicting results, so types
  are weighted by usage over all pitches, as TJStats does and as v1 describes.
- **Alternative:** implementing v1's square-root form; using v2's balls-in-play weights as written.
- **Evidence:** the `SEAM` config in `src/models/physical_similarity.py`; its rows in
  `method_comparison_{val,test}.csv` and `baseline_deltas_{val,test}.csv`. Release angles are computed as in the
  paper, running Statcast's velocity at 50 ft back to the release point under constant acceleration
  (`release_angles` in `src/features/build_arsenal.py`). The 1/n power compresses SEAM's scores into a narrow
  band (about 0.28–0.34 for top comps), so its numbers are not comparable to TJStats' on their face — only the
  ranking is, which is why the retrieval test uses an unweighted mean of the top 10 (D20).

## D20. Retrieval protocol: same-season index, 10 comps, plain mean, r and MAE

- **What:** for each query pitcher-season, take the 10 most similar *other* pitchers from the **same season**,
  hand (D16) and role (D23), each with ≥ 300 pitches (D24), and predict his stat as the **plain mean** of theirs.
  Report Pearson r and MAE next to a no-similarity reference: the mean of his whole candidate pool. Every method
  is scored by this one function.
- **Why:** a same-season index removes the query's other seasons by construction and compares contemporaneous
  run values. The mean is unweighted because similarity scales differ by method (TJStats' kernel values, SEAM's
  compressed scores, embedding distances), so weighting would make the comparison depend on the scale rather
  than on *who* the comps are.
- **Alternatives:** an index spanning several seasons; similarity-weighted means; a different k (k = 10 was fixed
  in advance and never tuned).
- **Cost:** RV/100 is noisy even at 300 pitches, so absolute r values are modest. Methods are compared against
  each other with paired intervals (D11), not against a perfect score.
- **Evidence:** `src/eval/retrieval.py`; every table in `docs/results/`.

## D21. Training choices for the embedding network

- **What:** early stopping and the LR schedule run on **2025 pitch log loss** only (patience 3, ReduceLROnPlateau
  halving after one flat epoch). Plain cross-entropy, no class weights. The sweep varies the **embedding
  dimension** {8, 16, 32} with λ = 1; the aux target is a separate experiment (D6, D10) so the sweep and the
  ablation don't claim the same work. Regularization: dropout 0.1, weight decay 1e-4 (AdamW), early stopping,
  gradient clipping at norm 1, LayerNorm. Every training arsenal is embedded at every step (~1,600 sets, cheap),
  so each qualifying pitcher-season gets equal weight in the auxiliary loss. 2025 retrieval r is logged every
  epoch, and **the checkpoint kept is the epoch with the best 3-epoch rolling mean of that r** — selection on the
  project's objective, on validation data only. **3 seeds per configuration (372–374).**
- **Why:** an earlier version stopped on pitch loss + λ × aux loss, and the RV/100 term swung from noise and
  stopped runs at epoch 1 while pitch loss was still improving. Pitch log loss averages ~700k validation pitches;
  the RV/100 error averages ~500 pitcher-seasons. Retrieval r peaks early and then declines while pitch log loss
  keeps improving (0.269 at its best epoch vs. 0.177 at the one early stopping kept), which is why the *kept*
  epoch is chosen on r even though training *ends* on log loss. A 3-epoch rolling mean damps epoch-to-epoch noise
  in r. Three seeds because MPS kernels on the aux path aren't deterministic — the same seed gave 2025 r of 0.174
  and 0.212 on two runs — so configs are compared on the mean over seeds and the shipped model is the seed-372
  run, fixed in advance so no seed is cherry-picked.
- **Consequence to state plainly:** validation r is optimistically biased by that selection. 2026 is the
  unbiased number.
- **Alternatives:** stopping on the combined loss; keeping the final epoch; one seed.
- **Evidence:** `sweep_summary.csv` (2025, 3 seeds, profile + RV/100 target) — d_emb 8 r 0.225 ± 0.020, 16
  0.220 ± 0.050, 32 0.173 ± 0.044, and 8 is selected (D25); per-run rows in `sweep.csv`; per-epoch curves in
  `training_curves.csv` and `docs/figures/training_curves.png`.

## D22. Per-stat breakdown of the retrieval test

- **What:** run the D20 test separately on seven pitcher-season stats, for every method, with paired intervals.
  **RV/100 stays the headline.**

  | Stat | Definition (cleaned pitch table, D13) |
  |---|---|
  | K% | strikeouts / plate appearances |
  | BB% | walks / plate appearances (intentional walks have no tracked pitch, so none survive the filter) |
  | Whiff% | whiffs, foul tips included / swings |
  | Chase% | swings at pitches outside the zone (Statcast `zone` 11–14) / pitches outside the zone |
  | GB% | ground balls (Statcast `bb_type`) / balls in play |
  | Damage% | solid + barrel contact / balls in play with a contact code |
  | RV/100 | −100 × mean run value per pitch, from the pitcher's side |

- **Why:** a single RV/100 number can't say *what* a similarity method captures. Comps might share strikeout
  ability, a stable and physically driven skill, but not contact quality, which is noisy. Each of the six rate
  stats is also more reliable than RV/100, whose split-half reliability is 0.28 on 2025.
- **Alternative:** RV/100 only.
- **Cost, stated plainly:** seven stats × five methods is 35 comparisons with no multiple-comparison correction,
  so some intervals will exclude zero by chance. The table reports every cell, not only the ones that clear zero.
- **Evidence:** `stat_breakdown_{val,test}.csv`, `stat_reliability_{val,test}.csv`,
  `docs/figures/stat_breakdown_{val,test}.png`, and new pitchers only in `stat_breakdown_new_test.csv`. On 2026
  the learned model beats HZB on BB% (+0.210 [0.131, 0.292]), Whiff% (+0.140 [0.076, 0.208]) and Chase%
  (+0.132 [0.042, 0.224]); HZB wins GB% (−0.177 [−0.241, −0.113]). The same pattern holds on 2025.

## D23. Similarity is role-blind; every comparison is same-role

- **What:** similarity uses raw pitch metrics only — role is never an input. But every *performance* comparison
  uses same-role comps, like same-hand: the retrieval test, the stat breakdown, the comps report, and the app.
  **Starter** = started at least half his appearances that season, where a start means he threw his team's first
  pitch of the game (openers count).
- **Why:** a starter's and a reliever's outcomes aren't comparable. Relievers throw harder in short stints and
  face batters once. Judging a starter by relievers' run values would blame the method for the role gap. Keeping
  role out of the similarity itself leaves the arsenal comparison physical.
- **Alternatives:** role as a similarity input; no role matching at all — the June-sample comps showed the
  problem, with Luis Castillo's TJStats list coming back mostly relievers.
- **Evidence:** `pitcher_roles` in `src/features/build_arsenal.py`; role counts per split in `data_summary.csv`
  (2025: 229 SP, 344 RP with ≥ 300 pitches).

## D24. Qualifiers: 300 pitches per pitcher, Savant's arsenal qualifier per pitch type

- **What:** pitcher-level comparisons (auxiliary targets, retrieval queries and comps, the stat breakdown) need
  **300 pitches** in the season. Per-pitch-type grades in the app use Savant's arsenal qualifier instead: 2.5
  pitches of that type per team game, scaled to the team games played (D30 scales it again for relievers).
- **Why not one qualifier everywhere:** applied per pitch type, Savant's rule would strip most relievers'
  secondary pitches; MLB's pitcher qualifier (1 inning per team game) drops nearly every reliever.
- **Alternative:** one threshold for both purposes.
- **Evidence:** `MIN_PITCHES` in `src/models/inputs.py`; `type_grades` in `src/app/build_artifacts.py`;
  the qualifiers recorded in `models/app/meta.json`.

## D25. Config selection: ties on retrieval r are broken by MAE

- **What:** configs are ranked by mean 2025 retrieval r over 3 seeds. Any config within one seed-SD of the top
  mean counts as **tied**, and among the tied ones the lowest mean 2025 MAE wins. The rule applies to the sweep
  (embedding size) and to each tier's aux target. `--mode publish` re-applies it to saved runs without retraining.
- **Why:** on mean r alone, T1 picked profile + RV/100 over profile by 0.0008 — about 1/25 of the seed SD. With
  RV/100 in the target, 2025 r peaked at epoch 2–3 and then fell (0.27 → 0.085 by epoch 29 for seed 372) while
  pitch log loss kept improving: negative transfer from a noisy target. The shipped model would have been an
  epoch-3 snapshot at a sharp peak picked on 2025, which is exactly the number most likely to shrink on 2026.
  Profile alone climbs steadily and holds (0.24 at epoch 23 for seed 372) and has the lower MAE.
- **Effect:** only T1's aux target changes (profile + RV/100 → profile). The embedding size stays 8 (16 was tied
  on r; MAE 0.7785 vs. 0.7782) and T3a's pick stays profile, so T1 and T3a now differ only in inputs.
- **Caveat, stated plainly:** the rule was written after seeing the 2025 results. Every input to it is 2025 data,
  which D5 reserves for choices, and 2026 was not touched.
- **Alternatives:** mean r alone (a coin flip at this gap); the best single seed (noisier); retraining with a
  smaller RV weight λ (costs a retrain, and it would still be a 2025 choice).
- **Evidence:** `select_config` in `src/models/train_embedding.py`; the `selected` column of
  `ablation_summary.csv` and `sweep_summary.csv`; `training_curves.csv`.

## D26. TJStats rebuilt to the tool's defaults, with its kernel width σ fitted to the site

- **What:** TJStats is reproduced at the tool's default settings — Full Arsenal, Same Hand, Min Pitches 10, and
  the six default metrics (velocity, iVB, HB, extension, release height, arm angle). Pitch types are matched on
  the raw Statcast `pitch_type`, as published, and a type counts if the pitcher threw it ≥ 10 times. Per-pitch
  similarity is s = exp(−d²/2σ²) on the z-scored metrics; the overall score is Σ over the target's types of
  (target usage × s), with an unmatched type counting 0. Z-scores are taken within each pitch type over that
  season's pitcher-season × type rows, both hands pooled with horizontal features mirrored to arm side, and
  **σ = 1.7444**.
- **Why:** TJStats is the tool this project is motivated against and the physical side of the α blend. An earlier
  version used all ten metrics, the ≥ 3% arsenal rows, and a fixed exp(−d²/10), which is not what the site
  computes; old and new TJStats share only about 6 of their 10 comps per 2025 query. A baseline has to be the
  published method, verified against its own outputs, before beating it means anything.
- **How σ and the z population were chosen** (the two things the tool doesn't publish): the site's scores for
  Tarik Skubal were transcribed by hand (2025, 78 per-pitch cells, `data/reference/tjstats_site_2025.csv`).
  Eight variants were tried — 6 or 10 metrics × z-scores over all pitches or over pitcher-season means × same
  hand or both hands — each with its own σ fitted by minimizing per-pitch MAE. The lowest-MAE variant won, and it
  was also best on Spearman, which doesn't depend on σ (0.976, MAE 2.33 points); the 10-metric variants were far
  worse (MAE 8.6–9.2). The winner and its σ were then **frozen** and scored once on a held-out Zack Wheeler page:
  per-pitch MAE 2.02 points over 81 cells, overall MAE 1.23, and 18 of the 19 named top-20 comps in common —
  clearing the bar set in advance (MAE ≤ 5, overlap ≥ 15 of 20).
- **Independent corroboration of σ:** the site's published methodology gives its kernel as S = e^(−d²/n),
  normalized by the number of selected metrics, which implies σ = √(n/2) = 1.7321 for six metrics. The fitted
  1.7444 is 0.7% away, and the two produce scores that differ by at most 0.4 points. σ is left at the fitted
  value because the 2026 test was scored with it.
- **What still differs:** every remaining mismatch traces to pitch labels — the site shows FS for a few pitchers
  whose pitches my Savant pull labels CH. Those cells account for both top-20 misses.
- **What differs by design:** the app and the evaluation run this formula on *my* comparison pool — same hand,
  same role, ≥ 300 pitches, full arsenal (D20, D23, D24) — while the site has no role filter, a 10-pitch floor,
  and a single-pitch mode. Running the site's own pool reproduces the site's ranking; the pool is the reason the
  app's list differs, not the formula.
- **A bug this rebuild surfaced:** reviewing the shared merge step showed that a metric missing for a whole
  pitch type became a literal 0 instead of the pitch-type average. The fix only touches seasons with missing
  values — none in 2025, 7 rows in 2026 — and it landed before the one-shot 2026 test.
- **Alternatives:** keeping the old unverified formula; choosing σ by retrieval r on 2025 (which would turn a
  published baseline into a tuned one); transcribing more pages.
- **A second, harder check:** the frozen config was also scored on a **2026** single-pitch page, a season
  sigma was never fitted to and a mode that applies no usage weighting, so it tests the kernel and the
  z-scores on their own: 19 named comps at a mean error of 1.30 points, Spearman 0.896, 16 of 19 in my own
  top 19 (`tjstats_fidelity_2026.csv`, `python -m src.eval.tjstats_fidelity --single-pitch`).
- **Evidence:** `src/eval/tjstats_fidelity.py`; `tjstats_fit.json`, `tjstats_fidelity_variants.csv`,
  `tjstats_fidelity_cells.csv`, `tjstats_fidelity_heldout.csv`, `tjstats_fidelity_2026.csv`; the `TJSTATS`
  config in `src/models/physical_similarity.py`.

## D27. HZB (Healey, Zhao & Brooks 2017) as a published baseline

- **What:** HZB, from "Measuring Pitcher Similarity: Technical Details" (viXra:1705.0098v1), implemented as
  published. A pitcher's signature against each batter hand is his set of pitch-type clusters — mean (speed,
  horizontal movement, vertical movement) in mph and inches, weighted by that type's share of his pitches to that
  hand (eq. 1). The ground distance between clusters is Mahalanobis, with Σ the covariance of every cluster mean
  in that pitcher-hand × batter-hand configuration that season (eq. 2). D_R and D_L are Earth Mover's Distances
  between signatures, and D = f_R·D_R + f_L·D_L, where f is the league-average share of pitches that pitchers of
  that hand throw to each batter hand (eqs. 3–4).
- **Why:** HZB is a different *family* of hand-built similarity. It compares whole pitch distributions without
  matching pitch labels, and it whitens the features jointly instead of z-scoring each one separately. Adding it
  means the headline can't come from beating a single weak formula (D28) — and in fact it is the strongest
  published baseline on 2025.
- **Gap-fills (two):** (1) Statcast pitch-type labels and Hawk-Eye measurements in place of Pitch Info's reviewed
  labels and PITCHf/x — the paper notes the EMD is not sensitive to pitch-classification vagaries; (2) if a
  pitcher never faced one batter hand, D uses the side both pitchers share, with f renormalized. In 2025 that
  touches 1,394 of 198,243 pitcher pairs (0.7%) and **none** of the 98,566 pairs where both pitchers have ≥ 300
  pitches, so it never reaches the retrieval test.
- **How faithful it is, honestly:** only as far as maths tests go. The paper publishes no example comps or
  distances to check against, so there is no fidelity number like TJStats'. What can be checked is that the
  distance is a metric: D(A, A) = 0, symmetry, and the triangle inequality on 10,000 random triples per hand
  (`python -m src.models.hzb_similarity --check`), plus D(b, a) recomputed by an independent path (SciPy's
  Mahalanobis, no Cholesky whitening), which differs by ≤ 4.4e-15. All pass on full 2025 and on the sample.
- **Alternatives:** leaving it out; an entropy-regularized (Sinkhorn) EMD, which is faster but approximate and
  isn't what the paper uses.
- **Evidence:** `src/models/hzb_similarity.py`; the `hzb` rows of `method_comparison_{val,test}.csv` and
  `baseline_deltas_{val,test}.csv`.

## D28. Headline Δ = learned vs. the strongest published baseline

- **What:** the headline comparison is learned (T1) against whichever published baseline (TJStats, HZB or SEAM)
  has the highest **2025** RV/100 r. That choice is written to `alpha_star.json` and reused unchanged on 2026.
  Every Δ in the method comparison, the per-stat breakdown and the slices is taken against it. The α blend stays
  learned ⊕ TJStats.
- **Why:** the claim is that learned similarity beats hand-built physical similarity. If the comparison is
  against the *best* published formula, and that formula is chosen on validation data, the result can't come
  from picking a weak opponent after seeing the test set. The strongest on 2025 is HZB.
- **Alternatives:** keeping TJStats as the reference (the tool the README motivates, but not the strongest);
  reporting the best of the three on 2026, which would be choosing on test data.
- **Evidence:** `main` in `src/eval/run_eval.py`; `alpha_star.json`; `baseline_deltas_{val,test}.csv`, which
  reports every method against all three published baselines regardless.

## D29. The app's gameplan keeps small samples

- **What:** the gameplan tables (pitch mix by count and batter hand, per-pitch grades, locations) show every
  pitch the comp threw, with no usage or count floor. Each row carries its pitch count; shares are whole percents
  that add to 100, with "<1%" for a nonzero share that rounds to 0. Grades below Savant's qualifier (D24) are
  flagged "small sample" rather than hidden.
- **Why (my words):** "Keep small samples. They show where he doesn't or shouldn't throw a pitch."
- **What changed:** the gameplan previously used only the ≥ 3% arsenal and hid unqualified grades, so a
  reliever's grades section was always empty. D15's 3% threshold still defines the arsenal the model reads.
- **Alternative:** keeping the 3% filter and hiding unqualified grades; a minimum count per table cell.
- **Evidence:** the gameplan section of `src/app/app.py`; `arsenal.parquet`, built by
  `src/app/build_artifacts.py`, keeps every pitch type.

## D30. Relievers get a lower per-pitch grade qualifier

- **What:** a reliever's per-pitch-type qualifier is the starter bar × (median reliever season pitches ÷ median
  starter season pitches) over the pitcher-seasons the app shows (≥ 300 pitches, 2023–26 pooled) — 0.441 in
  practice, from medians of 179 and 405 pitches. The starter bar stays Savant's 2.5 pitches per team game,
  scaled to the team games in my data for that season.
- **Why (my words):** "Relievers pitch significantly fewer innings, so they should definitely be on a
  lower threshold."
- **Alternatives:** one bar for both roles, which flags nearly every reliever's pitches; medians over every
  pitcher-season including call-ups under 300 pitches (a lower ratio, ~0.28); a fixed share such as 1/3.
- **Scope:** app only. Grades below the bar are still shown and flagged (D29); nothing in the evaluation reads
  the qualifier.
- **Evidence:** `reliever_scale` and `type_grades` in `src/app/build_artifacts.py`; `reliever_scale` in
  `models/app/meta.json`.

## D31. The app's twin flags check HZB as well as TJStats

- **What:** the app flags a comp as "same measurements, different results" if he is in TJStats' **or** HZB's top
  3 and in the learned model's bottom half of the pool, with a "Flagged by" column naming the method and its rank.
- **Why (my words):** "should list both."
- **Scope:** app only. The error analysis (`twins_*.csv`, `twins_summary_*.csv`) keeps the TJStats-only rule it
  was run with, which was set before HZB was added, and the README says so.
- **Alternatives:** TJStats only (the eval rule); HZB only (the strongest published baseline on 2025).
- **Evidence:** `twin_flags` in `src/app/app.py`; `twins_summary_{val,test}.csv` for the eval-side rule.

## D32. The gameplan heat map: his row under the comp's, red set by the pool

- **What:** in the app's pitch-mix table the selected pitcher's own row sits under each of the comp's (batter
  hand × count state) rows, and each comp cell is shaded green → red by the usage gap to his paired pitch (D33).
  A cell is fully red at the 90th-percentile |gap| over every comp cell the app can show — each pitcher-season
  with ≥ 300 pitches paired with every comp in the top 10 of any of the three lists, 2023–26 — which is
  **33.8 points** (2,279 pitchers, 55,684 pairs, 2.23M cells).
- **Why:** full red should mean a bigger gap than 9 in 10 comp-vs-pitcher cells across MLB, so the colour means
  the same thing on every page. The fixed 20-point scale it replaces sat near the 75th percentile, so too much
  of the table read as red.
- **Scope:** app only; nothing in the evaluation reads it. The percentile is over cells (one pitch × one row),
  not over whole pairs.
- **Alternatives:** the fixed 20 points; side-by-side columns instead of his row under the comp's; percentiles
  over the eval's learned top 10 only, or over every candidate pair in the pool.
- **Evidence:** `cell_gaps` and `write_gap_scale` in `src/app/gameplan_gap.py`; `models/app/gap_scale.json`;
  `gameplan_gap_scale.csv` (p50 10.6, p75 21.4, p90 33.8, p95 42.1).

## D33. The app pairs pitches by type; slider and sweeper are the only cross-type pair

- **What:** wherever the app compares a comp's pitches with the selected pitcher's — the pitch-by-pitch table,
  the pitch-mix heat map, the zone table — a pitch is paired only with the same Statcast type. The one exception
  is a slider with a sweeper, and only when one pitcher throws one of the two and the other throws the other; if
  either throws both, each is paired with its own type. Pitches with no partner stand alone.
- **Why (my words):** "don't pair Change/Split in pitch by pitch comparisons, they are different
  pitches. sweeper/slider is fine for now, but if one of the pitcher's uses both (or both pitcher's) we need to
  separate them clearly." This replaces an earlier rule that paired a pitch with the closest shape in its family,
  which matched changeups to splitters.
- **Scope:** app display only. The model's pitch identity (D15: family + shape) is unchanged and nothing in the
  evaluation pairs pitches. The heat-map scale (D32) was recomputed on the new pairs.
- **Alternatives:** the old family + shape pairing; more cross-type pairs (curveball / knuckle curve, slider /
  slurve).
- **Evidence:** `pair_types` and `CROSS_PAIRS` in `src/app/gameplan_gap.py`.
