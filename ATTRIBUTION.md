# ATTRIBUTION

Claude Code wrote most of the code in this repository and drafted most of the documentation. I specified the
work, ran it, read the output, decided what was wrong, and edited what shipped. I did not write the bulk of
`src/` myself, and this file says so plainly. Below, one section per AI tool, then my own part, then the data,
libraries and prior work I used.

## 1. Claude Code (Anthropic)

This is where the project was built. It ran on my Mac with access to the repo, the data and the GPU, so it
edited files and ran git and Python directly.

**What it generated.** First drafts of essentially every module:

| File | What Claude Code drafted |
|---|---|
| `src/data/fetch_statcast.py` | The resumable month-by-month Statcast pull, the season windows, `--sample` and `--refresh` |
| `src/features/build_arsenal.py` | The load → filter → features → label → aggregate pipeline, including the approach-angle and release-angle derivations, Savant attack zones, arm-side normalization, and `data_summary.csv` |
| `src/models/inputs.py` | The input-level definitions, the train-only scaler, and the padded arsenal-set construction |
| `src/models/arsenal_net.py` | The Deep Sets encoder, the pitch-outcome head, and the auxiliary head |
| `src/models/train_embedding.py` | The training loop, the sweep, the ablation, checkpoint selection, and `--mode publish` |
| `src/models/physical_similarity.py` | TJStats and SEAM as two configurations of one implementation, plus raw Euclidean |
| `src/models/hzb_similarity.py` | HZB from the paper: signatures, Mahalanobis whitening, the exact EMD, and the metric checks |
| `src/eval/retrieval.py`, `bootstrap.py`, `calibration.py` | The held-out retrieval test, the paired bootstrap, and the calibration report |
| `src/eval/run_eval.py`, `slice_breakdown.py`, `diagnostics.py`, `comps_report.py`, `tjstats_fidelity.py` | The evaluation driver and every table it writes; the new-pitcher regrouping; the 2025 diagnostics; the named-comps report; the σ fit and the fidelity checks |
| `src/app/app.py`, `build_artifacts.py`, `gameplan_gap.py` | The Streamlit app, its precomputed tables, and the heat map's pitch pairing and colour scale |
| `README.md`, `SETUP.md`, `docs/design_decisions.md`, `docs/results/README.md`, `notebooks/results_overview.ipynb` | Drafted from my decisions and the committed results |

It also transcribed one of the two tjstats.ca reference pages from a screenshot I supplied
(`data/reference/tjstats_site_2026.csv`; I typed in the other one), ran the short checks — `--sample` smoke
tests and headless app tests — and wrote every commit message.

**How that changed over follow-up prompts.** The working loop was: I described a piece of work in words, Claude
Code drafted it, I ran it and read the output, and the next round fixed what the output showed. Some things
changed more than once:

- The evaluation started as one comparison against one hand-built baseline and ended as five methods scored by
  one protocol with paired confidence intervals, because a single winner-take-all number made the headline
  depend on my model winning.
- The first training setup chose its checkpoint on pitch log loss. The curves showed retrieval quality peaking
  early and then decaying while the loss kept improving, so checkpoint selection was rewritten to track
  retrieval instead, and the auxiliary target became an ablation axis.
- The app's gameplan heat map was built, then reverted when I decided it cluttered the page, then restored when
  I decided the comparison was the point of the page. Its colour scale changed from a fixed 20-point gap to an
  MLB percentile after the fixed version painted most of the table red.
- The input levels started at four (T1, T2, T3a, T3b) and were cut to two, and the app's level-adjustment
  slider was deleted entirely, because nothing in the project validates an adjustment across levels.
- The README was rewritten twice: once when the multi-level framing stopped being true, and once more to cut it
  to a reference and move the long material into `docs/results/README.md`.

**What had to be debugged, fixed or reworked.** These are the ones that changed a result or a decision:

- **A pitch-type merge bug.** When a measurement was missing for a whole pitch type, the TJStats rebuild turned
  it into a literal 0 instead of the pitch-type average, which silently moved similarity scores. Found while
  rebuilding TJStats, fixed before the one-shot 2026 test.
- **SEAM cited to the wrong version.** The code implemented v2's 1/d power while the comment cited v1, which
  uses a square root. Caught in review, re-checked against both arXiv versions, re-cited to v2.
- **Two identical runs, two different numbers.** Apple's MPS kernels on the auxiliary path are not
  deterministic, so the same seed scored 0.174 and 0.212. The fix was three seeds per configuration, a shipped
  seed fixed in advance, and comparisons on seed means.
- **A cleanup that deleted cited outputs.** A tidy-up removed `method_agreement_val.csv` and `twins_test.csv`
  as uncited, when both backed numbers already written down. They were restored, and the rule became: a file
  stays if any number cites it.
- **The season windows missed the openers.** The 2024 and 2025 pulls started too late to include the Seoul and
  Tokyo Series, which are regular-season games. The windows moved to March 15 and the affected months were
  re-pulled.
- **Bunts left in the fingerprint.** The first version dropped the bunt *label* and kept the pitch, so bunt
  attempts still counted toward usage, shape, location and run value. They became a filter stage instead.
- **Comps that looked wrong.** A starter's TJStats list came back mostly relievers in an early sample. That is
  what produced same-role matching.

## 2. Claude (claude.ai chat)

Used for second opinions and for having machine-learning concepts explained to me while I was learning them. It
never saw the code or the data, and it never built anything. Its output is not in this repository except where
it shaped my own framing: the statement of the test in my words — that the question is whether a held-out
pitcher's comps' actual outcomes predict his, and that a null result is still a finding — was written in that
chat before it went into the design log. What I brought back was wording and understanding, not code. Where it
was wrong, I found out by running the code, not from the chat.

## 3. Cowork (Anthropic)

The earliest planning document for this project — the scope, the deliverables and the first read of the prior
art — was drafted in a Cowork session, before the work moved into Claude Code. None of that text is in this
repository. What survived from it is the project's shape, including the correction that cut the claim from
beating a published baseline down to running a controlled comparison.

No other AI tools were used on this project. The repository's history begins with an earlier, unrelated project
of mine in the same repo; the Arsenal Match work starts at the commit that replaced it.

## 4. My own part

- **The problem and the motivation.** A pitcher at a level with no Hawk-Eye has no way to find out which
  big-leaguer he resembles, or how that pitcher actually attacks hitters. That is why I started this, and it is
  why the input levels exist at all.
- **Pushing back on redundancy.** I found TJStats and SeeMagnus and asked whether this project was redundant,
  given that comp tools already exist. It nearly was. The answer became the contribution: those tools define
  similarity by hand and never test it, so what is missing is a learned metric and a held-out test, not another
  list.
- **Cutting the claim to what I could deliver.** An early plan leaned on beating a published baseline, which I
  could not control and did not have the time for. I cut it down to a controlled comparison, which is earned by
  running the experiment however the result comes out.
- **The test framing.** That the baselines stay as published, run on the same Savant data, while my model learns
  its own similarity; that the test is whether a held-out pitcher's comps' outcomes predict his; and that what
  the physical formulas miss should show up both where learned comps win and in the physical-twin,
  functional-stranger pairs.
- **"Physical twin, functional stranger"** as the name for where the two kinds of similarity disagree, and as
  the app's headline flag.
- **The product.** That the useful output is not the comp list but the comp's gameplan: pitch mix by count and
  batter hand, locations, and per-pitch grades.
- **The per-stat breakdown** (D22) — my idea, and what turned a single tied number into the project's clearest
  finding.

**The baseball calls that became design-log entries.** Each of these was a question put to me with options, and
the answer is why the code does what it does:

- The seven outcome classes, including keeping flares separate, foul tips as whiffs, and hit-by-pitch as a ball
  (D13).
- Bunt attempts removed entirely, not just left unlabelled (D13).
- Pitch identity is family plus shape, never the Statcast name, and a pitch is in the arsenal at 3% usage
  (D15).
- Same-hand comps only (D16).
- Location means where he throws it, and results never enter the fingerprint (D17).
- SEAM's gap-fills: extension counts as release, pitch types weighted by usage rather than by balls in play,
  and SEAM stays its own row rather than being averaged into the blend (D19).
- 300 pitches for pitcher-level comparisons, and Savant's per-pitch-type qualifier in the app (D20, D24).
- Movement to the nearest 3 inches at T3a, so that level stays in the ablation (D6).
- Similarity stays role-blind, but every performance comparison is same-role, with a starter defined as someone
  who started at least half his games (D23).
- Keep small samples in the app's gameplan, because they show where a pitcher doesn't or shouldn't throw a pitch
  (D29).
- Scale the reliever grade bar to reliever workload, because relievers throw far fewer innings (D30).
- Flag twins from HZB as well as from TJStats (D31).
- Pair pitches by type in the comparison, with slider and sweeper the only cross pair, because a changeup and a
  splitter are different pitches (D33).
- The three-season split — train 2023–24, validate 2025, test 2026 once (D5) — and the March 15 window start
  that catches the openers (D8).
- The fidelity protocol for TJStats: transcribe the site's own scores, fit its unpublished kernel width to them,
  then freeze it and score once on a held-out page, with the pass bar set in advance (D26).

**The runs.** Every full run whose numbers are committed was run from my own terminal on my own machine, with
the console output tee'd to `logs/` (gitignored): the Statcast pull including the openers re-pull, the feature
build, the full training sweep and ablation, the 2025 evaluation, the one-shot 2026 evaluation, the comps
report, the app-table build, the diagnostics, and the 2025 re-runs that checked the final code cleanup changed
no number. Claude Code ran the `--sample` smoke tests and the headless app tests.

**What I caught in the shipped app.** I compared the app's TJStats list for a 2026 pitcher against the site's
own list and asked whether the measurements matched at all. They do — the six are the same, and re-running my
rebuild at the site's own settings reproduces its ranking. The difference was the candidate pool: mine is
same-role and at least 300 pitches, and the site's is neither. That question is what produced the second
fidelity check, on a 2026 page the kernel width was never fitted to, which came back at a mean error of 1.30
points with 16 of 19 named comps shared (`docs/results/tjstats_fidelity_2026.csv`). The app now says why its
list differs from the site's.

## 5. Data

- **Statcast pitch-level data, 2023–2026.** Baseball Savant (baseballsavant.mlb.com), © MLB Advanced Media.
  Retrieved with `src/data/fetch_statcast.py` and used for non-commercial academic purposes.
- **MLB Stats API** (statsapi.mlb.com), for players' primary positions, to filter out position players who
  pitch. © MLB Advanced Media.
- **`data/reference/tjstats_site_2025.csv` and `tjstats_site_2026.csv`** — similarity scores read off
  tjstats.ca screenshots and typed in by hand, with the tool's settings recorded in `data/reference/README.md`.
  I transcribed the 2025 pages; Claude Code transcribed the 2026 page from a screenshot I supplied. These are
  the site's own outputs, used only to fit and check my rebuild of its method.

## 6. Libraries

Versions are pinned loosely in `requirements.txt` and exactly in `requirements-lock.txt`.

| Library | Use |
|---|---|
| pandas, NumPy, PyArrow | Data handling and parquet I/O |
| PyTorch | The embedding model and its training loop |
| SciPy | Pairwise distances (`scipy.spatial.distance.cdist`) |
| POT (Python Optimal Transport) | The exact Earth Mover's Distance for the HZB baseline (`ot.emd2`) |
| Streamlit | The app |
| Matplotlib | Figures |
| pybaseball | Statcast download from Baseball Savant (`src/data/fetch_statcast.py`) |
| requests | MLB Stats API position lookup (`src/features/build_arsenal.py`) |

## 7. Methods and prior work

Baselines re-implemented here:

- **[TJStats Pitch Similarity](https://tjstats.ca/pitch-similarity/)** (T. Nestico). Its z-score plus
  Gaussian-kernel similarity is rebuilt at the tool's default settings; D26 and `docs/results/README.md` have
  the two unpublished details and the fidelity checks.
- **HZB.** G. Healey, S. Zhao and D. Brooks, "Measuring Pitcher Similarity: Technical Details",
  viXra:1705.0098v1, 2017. Implemented from the paper with two documented gap-fills (D27).
- **SEAM.** J. Wapner, D. Dalpiaz and D. J. Eck, "SEAM methodology for context-rich player matchup evaluations
  in baseball", arXiv:2005.07742v2, 2022; v1 by C. Young, D. Dalpiaz and D. J. Eck, 2020. The **v2** formula
  (eq. 4, the 1/d power) is the one implemented, with three documented gap-fills (D19).

Methods and ideas built on:

- **Deep Sets.** M. Zaheer et al., NeurIPS 2017 — the arsenal encoder.
- **The Earth Mover's Distance as a metric for image retrieval.** Y. Rubner, C. Tomasi and L. J. Guibas,
  IJCV 40(2), 99–121, 2000 — the distance HZB is built on.
- **POT: Python Optimal Transport.** R. Flamary et al., JMLR 22(78), 1–8, 2021 — the exact EMD solver.
- **(batter|pitcher)2vec.** M. A. Alcorn, MIT Sloan Sports Analytics Conference, 2018 — precedent for learning
  player representations from outcomes.
- **"Using Clustering to Find Pitch Subtypes and Effective Pairings."** SABR, 2020 — precedent for describing
  pitches by shape rather than by label.
- **Unified Pitch Graphs.** K. Lee and J. Ko, arXiv:2609.03810, 2026.
- **Statcast Lab: pitcher similarity scores.** T. Tango, Baseball Savant, 2019 — the observation that motivated
  the project.
- **Baseball Savant attack zones** (heart / shadow / chase / waste) — the location definition used in
  `src/features/build_arsenal.py`.
- **[SeeMagnus MLB comp tool](https://www.seemagnus.com/blog-posts-test/whats-my-major-league-pitcher-comp-a-tool-that-tells-you).**
  Prior art for matching pitchers to MLB comps from TrackMan/YakkerTech data. Its use of Bauer Units for slower
  fastballs is the kind of level adjustment this project considered and cut (D7): nothing here adjusts across
  levels.
- **Baseball Savant "Similar Pitchers."** Prior art for pitch-level similarity.
- **FanGraphs Stuff+/Location+, Baseball Prospectus StuffPro/PitchPro, Driveline Stuff+.** Established pitch
  feature sets that informed which measurements this project uses.

---

Every module in `src/` that contains code carries this note in its docstring, as the course requires, and so
does the notebook. The six `__init__.py` files are empty package markers, with nothing to attribute.

```python
"""
<what the module does>

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
```
