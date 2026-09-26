# Self-Assessment: Arsenal Match

Toma Shigaki-Than, solo project. Line numbers refer to the files in this repository.

## Machine Learning (15 selections)

1. **Completed project individually without a partner (10 pts).** `README.md` → *Individual Contributions*.

2. **Deployed model as functional web application with user interface (10 pts).** A Streamlit app,
   [`src/app/app.py:674–787`](src/app/app.py#L674-L787). It runs the trained network on a typed-in arsenal at
   query time ([`:210–225`](src/app/app.py#L210-L225)) and ranks comps from the learned embeddings
   ([`:235–238`](src/app/app.py#L235-L238)). Run steps: `README.md` → *Quick Start*.

3. **Compared multiple model architectures or approaches quantitatively with controlled experimental setup
   (7 pts).** Six similarity methods are scored under one protocol:
   [`src/eval/run_eval.py:121–139`](src/eval/run_eval.py#L121-L139),
   [`src/eval/retrieval.py:67–110`](src/eval/retrieval.py#L67-L110), and bootstrap CIs in
   [`src/eval/bootstrap.py:46–76`](src/eval/bootstrap.py#L46-L76). Results are in
   `docs/results/method_comparison_{val,test}.csv` and the *Headline* table of `README.md`.

4. **Conducted ablation study systematically varying at least two independent design choices (7 pts).** The
   input set (T1 vs. T3a) is crossed with the auxiliary loss (off / outcome mix / outcome mix + RV/100), with 3
   seeds each: [`src/models/train_embedding.py:536–556`](src/models/train_embedding.py#L536-L556). The summary
   table is in `docs/results/ablation_summary.csv` and `README.md` → *What else the evaluation shows*.

5. **Defined and trained a custom neural network architecture (5 pts).** A Deep Sets arsenal encoder with an
   outcome head and an auxiliary head, [`src/models/arsenal_net.py:29–92`](src/models/arsenal_net.py#L29-L92).
   It is trained from scratch in
   [`src/models/train_embedding.py:239–328`](src/models/train_embedding.py#L239-L328).

6. **Conducted systematic hyperparameter tuning using validation data (5 pts).** Embedding sizes 8, 16 and 32 ×
   3 seeds, compared on the 2025 validation season:
   [`src/models/train_embedding.py:512–533`](src/models/train_embedding.py#L512-L533). Results are in
   `docs/results/sweep_summary.csv` and `docs/results/README.md` → *Model selection*.

7. **Applied regularization techniques to prevent overfitting (5 pts).** Dropout at
   [`src/models/arsenal_net.py:40`](src/models/arsenal_net.py#L40) and
   [`:62`](src/models/arsenal_net.py#L62). Early stopping at
   [`src/models/train_embedding.py:306–310`](src/models/train_embedding.py#L306-L310).

8. **Applied feature engineering (5 pts).** Approach angles, attack zones, arm-side normalization and
   usage/location shares:
   [`src/features/build_arsenal.py:252–293`](src/features/build_arsenal.py#L252-L293),
   [`:305–308`](src/features/build_arsenal.py#L305-L308) and
   [`:373–398`](src/features/build_arsenal.py#L373-L398).

9. **Analyzed model behavior on edge cases or out-of-distribution examples (5 pts).** Two slices: slow-fastball
   pitchers, and pitchers never seen in training
   ([`src/eval/run_eval.py:176–197`](src/eval/run_eval.py#L176-L197)). Results are in
   `docs/results/slices_{val,test}.csv`, discussed in `README.md` → limitation 2.

10. **Documented a design decision where you chose between ML approaches based on technical tradeoffs (3 pts).**
    D21 in `docs/design_decisions.md` keeps the checkpoint with the best validation retrieval r rather than the
    best log loss. The evidence is `docs/results/iteration1/training_curves.csv`.

11. **In ATTRIBUTION.md, provided a substantive account of how AI development tools were used (3 pts).**
    `ATTRIBUTION.md` §1, including the list of what had to be debugged or reworked.

12. **Implemented proper train/validation/test split with documented split ratios (3 pts).** The split is by
    season: 2023–24 train, 2025 validation, 2026 test
    ([`src/features/build_arsenal.py:50`](src/features/build_arsenal.py#L50)). The ratios are in
    `docs/results/data_summary.csv` and D5.

13. **Used learning rate scheduling (3 pts).** `ReduceLROnPlateau` at
    [`src/models/train_embedding.py:247`](src/models/train_embedding.py#L247).

14. **Implemented gradient clipping (3 pts).**
    [`src/models/train_embedding.py:285`](src/models/train_embedding.py#L285).

15. **Used appropriate data loading with batching and shuffling (3 pts).** A seeded `DataLoader` with
    `BatchSampler(RandomSampler)` at
    [`src/models/train_embedding.py:254`](src/models/train_embedding.py#L254).

## Following Directions

- Self-assessment submitted, following the 15-selection limit, with evidence (3 pts): this file.
- SETUP.md with step-by-step installation instructions (1 pt).
- ATTRIBUTION.md with attributions of all sources, including AI generation (1 pt).
- requirements.txt included and accurate (1 pt).
- README *What it Does* (1 pt), *Quick Start* (1 pt), *Video Links* (1 pt) and *Evaluation* (1 pt).
- Demo video (2 pts) and technical walkthrough (2 pts), linked in `README.md` → *Video Links*.

## Project Cohesion and Motivation

- README states a single, unified research question (3 pts): the opening paragraph of `README.md`.
- Demo video communicates why the project matters to a non-technical audience (3 pts).
- Connected to concrete research papers (3 pts): HZB and SEAM are rebuilt as baselines (`README.md` → *What I
  compared against*), with the full list under *References*.
- Technical walkthrough shows the components working together (2 pts).
- Clear progression from problem → approach → solution → evaluation (2 pts): the README's section order.
- Design choices explicitly justified (2 pts): `docs/design_decisions.md` (D1–D33).
- Evaluation metrics directly measure the stated objective (2 pts): comps are scored on how well they predict a
  held-out pitcher's outcomes ([`src/eval/retrieval.py:88–110`](src/eval/retrieval.py#L88-L110)).
- No superfluous ML components (2 pts): every item above is a step from pitches to comps.
