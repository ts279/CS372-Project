"""
Per-stat breakdown on the "new pitchers" slice, recomputed from a saved eval run (no re-scoring).

The slices table (slices_<split>.csv) reports RV/100 only. This splits the per-stat breakdown the
same way: for each of the 7 stats (D22), every base method's r and MAE on the query pitchers who
never appeared in 2023-24 training, with paired-bootstrap deltas vs. the headline baseline (D28).
It reuses run_eval.compare, so the pairing, bootstrap (B = 2000) and seed match the main tables.

Pipeline: a follow-up to stage 4 (evaluate) that regroups its saved errors. Stages: fetch -> features ->
train -> evaluate -> comps report -> app tables -> app.

Inputs:  docs/results/errors_<split>.csv (saved per-query predictions), alpha_star.json (the
         headline baseline, frozen on 2025), data/processed/pitcher_seasons.parquet (who was in training)
Output:  docs/results/stat_breakdown_new_<split>.csv

Usage (from the repo root):
    python -m src.eval.slice_breakdown --split test

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.eval.retrieval import HEADLINE, STATS
from src.eval.run_eval import BASE_METHODS, SEASON_OF, compare
from src.models.train_embedding import Paths


def new_pitcher_breakdown(errors: pd.DataFrame, train_pitchers: set, reference: str) -> pd.DataFrame:
    """7 stats x base methods on the queries not in 2023-24 training, deltas vs. `reference`."""
    # why (D17): "new" is the app's real use case (a pitcher the model never saw), so the per-stat edge
    # has to be checked there, not only on the full pool where most queries were trained on.
    new = lambda d: ~d["pitcher"].isin(train_pitchers).to_numpy()  # noqa: E731, same mask as run_eval.slices
    frames = [compare(errors, s, BASE_METHODS, reference, new) for s in STATS]
    return pd.concat(frames, ignore_index=True).drop(columns=["baseline_mae"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-stat breakdown on new pitchers, from a saved eval run.")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()
    paths = Paths.make(False)

    errors = pd.read_csv(paths.results / f"errors_{args.split}.csv")
    ref = json.loads((paths.results / "alpha_star.json").read_text())["headline_baseline"]
    seasons = pd.read_parquet(paths.data / "pitcher_seasons.parquet", columns=["pitcher", "split"])
    train_pitchers = set(seasons.loc[seasons["split"] == "train", "pitcher"])
    n_new = errors.loc[(errors["method"] == ref) & (errors["stat"] == HEADLINE), "pitcher"].isin(train_pitchers).eq(False).sum()
    print(f"[slice] {args.split} ({SEASON_OF[args.split]}): {len(errors):,} saved error rows; "
          f"{len(train_pitchers):,} training pitchers; {n_new} new query pitchers; reference {ref}")

    out = new_pitcher_breakdown(errors, train_pitchers, ref)
    path = paths.results / f"stat_breakdown_new_{args.split}.csv"
    out.to_csv(path, index=False)
    print(f"[slice] wrote {path} ({len(out)} rows)")
    show = out[out["method"] == "learned"][["stat", "r", "delta_r", "delta_r_lo", "delta_r_hi"]]
    print(f"\n[slice] learned vs. {ref}, new pitchers:\n{show.round(3).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
