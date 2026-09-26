# SETUP

## What you need

- Python 3.9. I developed on 3.9.6; newer 3.9.x is fine.
- macOS or Linux. Training uses the Apple Silicon GPU (MPS) when it is there and falls back to CPU otherwise.
- About 900 MB of free disk space to run the app: the checkout is around 20 MB and the virtual environment is
  around 840 MB, nearly all of it PyTorch. Rebuilding from raw Statcast needs about 500 MB more.

## Install and run the app

You do not need any data to run the app. It loads the committed weights and the precomputed tables in
`models/`.

1. **Clone the repo and make a virtual environment.**

   ```bash
   git clone https://github.com/tomashigakithan/arsenal-match.git
   cd arsenal-match
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   You should see your shell prompt gain a `(.venv)` prefix.

2. **Install the dependencies.**

   ```bash
   pip install -r requirements-lock.txt
   ```

   `requirements-lock.txt` pins the exact versions the results were produced with; `requirements.txt` has the
   loose pins and works too. Either way pip ends with a `Successfully installed …` line naming torch, pandas,
   streamlit and POT. On macOS you will also see a pip-version notice and a LibreSSL warning. Both are
   harmless.

3. **Start the app.**

   ```bash
   streamlit run src/app/app.py
   ```

   Streamlit prints a `Local URL: http://localhost:8501` line and opens a browser tab. A pitcher is already
   selected in the sidebar, and the page should show three comp lists side by side. If those lists are there,
   everything you need is committed and working.

## Rebuild everything from raw Statcast

Run every command from the repository root, with the virtual environment active. Each stage prints its row
counts and shapes as it goes and writes its own numbers under `docs/results/`, so you can check each step
against what is committed. Every script also takes `--sample`, which runs the same code on a small slice of
data in seconds and writes under `models/sample/` instead of `models/`. Run the sample first if you want to
know a stage works before spending the time.

1. **Download the Statcast pitches, 2023–2026.**

   ```bash
   python -m src.data.fetch_statcast                    # 2023-2025
   python -m src.data.fetch_statcast --seasons 2026     # the test season, through yesterday
   ```

   This writes one parquet file per month into `data/raw/` and prints a `[save]` line with each file's row
   count. 2023–2025 alone is about 325 MB and 2.1M pitches. The download is resumable: re-running it skips
   months already on disk. It needs network access to baseballsavant.mlb.com.

2. **Build the features.**

   ```bash
   python -m src.features.build_arsenal
   ```

   It prints how many rows each of the eight filter stages removed, then the shape of each table it saves, and
   ends with a `[done]` line and a time. You should end up with `data/processed/pitches.parquet` plus the
   arsenal and pitcher-season tables, and 2,814,234 pitches kept out of 2,955,003. It needs statsapi.mlb.com
   once, to look up player positions; the answer is cached in `data/processed/player_positions.parquet`.

3. **Train.**

   ```bash
   python -m src.models.train_embedding --mode all
   ```

   It logs which device it is using (`mps` or `cpu`), then runs the embedding-size sweep (3 sizes × 3 seeds)
   and the ablation (2 input levels × 3 auxiliary targets × 3 seeds), both selected on 2025. It ends with two
   `[publish]` lines naming the runs that shipped, and a `[done]` line. About 20 minutes on an Apple-silicon
   GPU. It writes `models/arsenal_net_{t1,t3a}.pt`, the embeddings, and the sweep, ablation and training-curve
   tables under `docs/results/`.

   Expect small differences from the committed numbers. Apple's MPS kernels on the auxiliary path are not
   deterministic, so the same seed can score differently between runs. That is why every configuration is
   compared over three seeds.

4. **Evaluate on 2025, the validation season.**

   ```bash
   python -m src.eval.run_eval --split val
   python -m src.eval.comps_report
   ```

   `run_eval` prints how many pitcher-seasons are eligible, then the method comparison (raw Euclidean,
   TJStats, HZB, SEAM, learned T1 and T3a), the per-stat breakdown, each stat's reliability, the slices, the
   twin pairs and the calibration summary, with a `[save]` line per file. This is also the run that picks the
   blend weight α\* and the reference baseline and freezes them in `docs/results/alpha_star.json`.
   `comps_report` then writes the named comps to `docs/results/comps_2025.md` and prints how many targets it
   covered.

5. **Score 2026 once.**

   ```bash
   python -m src.eval.run_eval --split test
   python -m src.eval.slice_breakdown --split test
   ```

   The test run reads α\* and the reference baseline back from step 4 instead of choosing them again, then
   prints the same set of tables for 2026. `slice_breakdown` regroups the saved per-query errors for the
   pitchers that never appeared in training and writes `stat_breakdown_new_test.csv`. It scores nothing new.

6. **Build the app's tables.**

   ```bash
   python -m src.app.build_artifacts
   ```

   It prints a `[save]` line and a size for each file it writes into `models/app/`, ending with the HZB
   distance matrices and the heat map's colour scale. Those files are committed, so this step only matters if
   you changed the data or the model. `python -m src.app.gameplan_gap` reruns the colour-scale step alone, in
   about a minute.

7. **Run the 2025 diagnostics (optional).**

   ```bash
   python -m src.eval.diagnostics
   ```

   Four probes — a hand-built distance on the model's own inputs, T3a against HZB, how much the methods' lists
   overlap, and the arm-angle gap — written to `docs/results/diagnostics/`. It reads the HZB tables from step 6
   and never touches 2026.

8. **Check the baselines (optional).**

   ```bash
   python -m src.eval.tjstats_fidelity                  # refit TJStats' kernel width to the tjstats.ca scores
   python -m src.eval.tjstats_fidelity --single-pitch   # re-score the 2026 page only, fitting nothing
   python -m src.models.hzb_similarity --season 2025 --check
   ```

   The fidelity run rewrites `docs/results/tjstats_fit.json` and prints the chosen variant and its σ. The HZB
   check ends with `[hzb] checks overall: PASS` if the distance is a proper metric.

9. **Read the results back (optional).**

   `notebooks/results_overview.ipynb` loads the committed tables in `docs/results/` and walks through them in
   the order of the project. It is committed **with its outputs saved**, so you can read it without running
   anything. To re-run it yourself you need Jupyter, which is not in `requirements.txt` because nothing else in
   the project uses it:

   ```bash
   pip install jupyter
   jupyter lab notebooks/results_overview.ipynb
   ```

Raw and processed data are not committed, because of their size. `src/data/fetch_statcast.py` is the data
access script.

## Repository layout

```
src/
  data/       data access (the Statcast download)
  features/   arsenal feature construction
  models/     the embedding model, its training, and the three published baselines
  eval/       the held-out retrieval experiments
  app/        the Streamlit app and its precomputed tables
data/         raw/ and processed/ (not committed; see data/README.md)
models/       the trained weights and the tables the app loads (the README's repo map has the detail)
notebooks/    results_overview.ipynb (reads the committed results; no training)
docs/         design_decisions.md, results/ (indexed by results/README.md) and figures/
videos/       the demo and the technical walkthrough
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — you are not in the repository root. Every command runs
  from the folder that contains `src/`, as `python -m src.<package>.<module>`.
- **A pip-version notice, or `NotOpenSSLWarning … LibreSSL`** — harmless on macOS. Ignore both.
- **A NumPy warning about matmul** — also harmless. NumPy 2 on Apple's Accelerate framework warns spuriously;
  the distances that matter go through `scipy.spatial.distance.cdist`.
- **`FileNotFoundError` under `data/processed/`** — you are running a rebuild stage before the one that feeds
  it. The steps above are in order, and everything from step 4 on needs step 2.
- **The Statcast download stops partway** — run it again. It skips the months already in `data/raw/`.
- **Training numbers don't match the committed ones** — expected on Apple silicon; see step 3.
- **A comp list looks short in the app** — a pitcher-season needs at least 300 pitches to be eligible as a
  comp, so a pool can be small. The count is shown above the lists.

## Notes

- Splits are by season, never random: train on 2023–24, validate on 2025, test on 2026. A random split would
  put the same pitcher on both sides and measure memorization (`docs/design_decisions.md`, D5).
- Console output from the long runs goes to `logs/`, which is gitignored.
