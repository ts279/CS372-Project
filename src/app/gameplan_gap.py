"""
The gameplan heat map's pitch pairing and usage gaps, and the scale its colors are read on (D32).

The app (src/app/app.py) shows each comp's pitch mix by count and batter hand with the selected pitcher's
own mix under it, and shades each comp cell by the usage gap between them. This module holds the pieces
that are pure functions of the app tables, so the app and the scale calibration compute the same gap:
    - pair_types / paired_pitch: which of his pitches each comp pitch is compared with (D33);
    - usage_mix / row_shares: pitch counts and shares by (batter hand, count state);
    - cell_gaps: the gap, in percentage points, for every comp cell of the table.

Pipeline stage 6 of 7 (app tables), run at the end of build_artifacts or on its own. The scale is the
90th-percentile |gap| over every comp cell the app can show: every pitcher-season with >= 300 pitches,
paired with each comp in the top 10 of any of the app's three lists (learned T1, TJStats, HZB).

Inputs:  models/app/{index,arsenal,usage_by_count}.parquet, hzb_distance.npz, meta.json (build_artifacts.py)
Outputs: models/app/gap_scale.json (read by the app); docs/results/gameplan_gap_scale.csv
         (--sample: models/sample/app/gap_scale.json and models/sample/results/gameplan_gap_scale.csv)

Usage (from the repo root):
    python -m src.app.gameplan_gap [--sample]

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.models.inputs import MIN_PITCHES
from src.models.physical_similarity import TJSTATS, arsenal_similarity
from src.models.train_embedding import ROOT, Paths

K = 10  # comps per list, as in the app and the eval (D20)
PERCENTILE = 90  # full red = a bigger gap than 9 of 10 comp cells (D32)
PITCHER_KEYS = ["pitcher", "game_year", "p_throws"]
# Different Statcast codes the app may pair as one pitch (D33). A changeup and a splitter are two pitches.
CROSS_PAIRS = [frozenset({"SL", "ST"})]


# ---------------------------------------------------------------------------
# Pairing and gaps (used by the app)
# ---------------------------------------------------------------------------
def arsenal_rows(arsenal: pd.DataFrame, r: pd.Series) -> pd.DataFrame:
    """One pitcher-season's >= 3% arsenal, indexed by pitch type."""
    a = arsenal[arsenal["in_arsenal"] & (arsenal["pitcher"] == r["pitcher"]) & (arsenal["game_year"] == r["game_year"])
                & (arsenal["p_throws"] == r["p_throws"])]
    return a.set_index("pitch_type")


def pair_types(his: pd.DataFrame, theirs: pd.DataFrame) -> List[Tuple[Optional[str], Optional[str]]]:
    """Pair his pitch types with the comp's (pass every pitch each threw): the same code when both throw it.
    The only pair of different codes is a slider with a sweeper, when he throws one of the two and the comp
    throws the other (D33). Leftovers stand alone."""
    pairs = {t: t for t in his.index if t in theirs.index}
    for group in CROSS_PAIRS:
        a, b = [t for t in his.index if t in group], [u for u in theirs.index if u in group]
        # why: a slider and a sweeper are often the same pitch tagged two ways, but a pitcher who throws both
        # has two different pitches, so then only same-code pairs count (D33).
        if len(a) == 1 and len(b) == 1 and a[0] != b[0]:
            pairs[a[0]] = b[0]
    out = [(t, pairs.get(t)) for t in his.sort_values("usage", ascending=False).index]
    return out + [(None, u) for u in theirs.sort_values("usage", ascending=False).index if u not in pairs.values()]


def usage_mix(usage: pd.DataFrame, r: pd.Series, states: List[str]) -> pd.DataFrame:
    """Pitch counts by (batter hand, count state) x pitch type for one pitcher-season: every row, and every
    pitch he threw, most-thrown first."""
    u = usage[(usage["pitcher"] == r["pitcher"]) & (usage["game_year"] == r["game_year"])
              & (usage["p_throws"] == r["p_throws"])].astype({"pitch_type": str})
    order = u.groupby("pitch_type")["n"].sum().sort_values(ascending=False, kind="stable").index.tolist()
    mix = u.pivot_table(index=["stand", "count_state"], columns="pitch_type", values="n", aggfunc="sum",
                        fill_value=0, observed=True)
    return mix.reindex(index=pd.MultiIndex.from_product([["L", "R"], states]), columns=order, fill_value=0)


def paired_pitch(comp_types: List[str], his_types: List[str],
                 pairs: List[Tuple[Optional[str], Optional[str]]]) -> Dict[str, Optional[str]]:
    """For each of the comp's pitches, the pitch of his it's compared with: the pitch-by-pitch pairing
    (pair_types: same type, or slider with sweeper, D33); otherwise the same Statcast type if he threw it
    and it isn't paired already; otherwise None (he has no such pitch)."""
    paired = {u: t for t, u in pairs if t is not None and u is not None}
    used = set(paired.values())
    return {u: paired.get(u, u if u in his_types and u not in used else None) for u in comp_types}


def row_shares(counts: np.ndarray) -> np.ndarray:
    """Pitch counts -> shares of the row (all zeros if the row has no pitches)."""
    total = counts.sum()
    return counts / total if total else np.zeros(len(counts))


def cell_gaps(mix: pd.DataFrame, his_mix: pd.DataFrame, his_for: Dict[str, Optional[str]]) -> np.ndarray:
    """Comp share minus his paired share, in percentage points, for every cell of the comp's mix table
    (rows = batter hand x count state, columns = the comp's pitches). A pitch he doesn't throw counts as 0%."""
    his_col = {t: j for j, t in enumerate(his_mix.columns)}
    out = np.zeros(mix.shape)
    for i, key in enumerate(mix.index):
        share, his_share = row_shares(mix.loc[key].to_numpy()), row_shares(his_mix.loc[key].to_numpy())
        for j, u in enumerate(mix.columns):
            t = his_for[u]
            out[i, j] = 100 * (share[j] - (his_share[his_col[t]] if t in his_col else 0.0))
    return out


# ---------------------------------------------------------------------------
# The scale: every comp cell the app can show
# ---------------------------------------------------------------------------
def top_k(scores: np.ndarray, k: int = K) -> np.ndarray:
    """Positions of the k highest scores; ties keep pool order, as the app's rank(method="first")."""
    return np.argsort(-scores, kind="stable")[:k]


def comp_pairs(index: pd.DataFrame, arsenal: pd.DataFrame, hzb: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]]
               ) -> pd.DataFrame:
    """(query, comp) pitcher-season pairs: every query with >= 300 pitches x each list's top 10 in his pool
    (same season, hand, role, >= 300 pitches, not himself; D16, D20, D23, D24). One row per distinct pair."""
    q = index[index["n_pitches"] >= MIN_PITCHES].reset_index(drop=True)
    emb_cols = [c for c in q.columns if c.startswith("t1_e")]
    rows = []
    for (season, hand), g in q.groupby(["game_year", "p_throws"]):
        pitchers, S = arsenal_similarity(arsenal, TJSTATS, int(season), hand)
        tj_row = {int(p): i for i, p in enumerate(pitchers["pitcher"])}
        ids, D = hzb.get((int(season), hand), (np.array([], "int64"), None))
        hzb_col = {int(c): j for j, c in enumerate(ids)}
        for role, p in g.groupby("role"):
            pid = p["pitcher"].to_numpy("int64")
            emb = p[emb_cols].to_numpy("float64")
            learned = -cdist(emb, emb)
            tj = np.array([[S[tj_row[a], tj_row[b]] if a in tj_row and b in tj_row else 0.0 for b in pid] for a in pid])
            hz = np.array([[-float(D[hzb_col[a], hzb_col[b]]) if a in hzb_col and b in hzb_col else -np.inf
                            for b in pid] for a in pid]) if D is not None else None
            for i, a in enumerate(pid):
                keep = np.arange(len(pid)) != i
                picks = set()
                for m in [learned, tj] + ([hz] if hz is not None and a in hzb_col else []):
                    picks.update(np.flatnonzero(keep)[top_k(m[i, keep])].tolist())
                rows += [(int(a), int(pid[j]), int(season), hand) for j in picks]
    out = pd.DataFrame(rows, columns=["pitcher", "comp", "game_year", "p_throws"])
    print(f"[gap] {len(q):,} query pitcher-seasons; {len(out):,} distinct (query, comp) pairs")
    return out


def all_gaps(pairs: pd.DataFrame, arsenal: pd.DataFrame, usage: pd.DataFrame, states: List[str]) -> np.ndarray:
    """|gap| for every comp cell of every pair's heat map, exactly as the app computes it."""
    mixes: Dict[Tuple, pd.DataFrame] = {}
    rows: Dict[Tuple, pd.DataFrame] = {k: g.set_index("pitch_type") for k, g in arsenal.groupby(PITCHER_KEYS)}
    by_pitcher = {k: g for k, g in usage.groupby(PITCHER_KEYS)}

    def mix_of(key: Tuple) -> pd.DataFrame:
        if key not in mixes:
            mixes[key] = usage_mix(by_pitcher[key], pd.Series(dict(zip(PITCHER_KEYS, key))), states)
        return mixes[key]

    out = []
    for a, b, season, hand in pairs.itertuples(index=False, name=None):
        ka, kb = (a, season, hand), (b, season, hand)
        mix, his_mix = mix_of(kb), mix_of(ka)
        pairs_ab = pair_types(rows[ka], rows[kb])
        his_for = paired_pitch(mix.columns.tolist(), his_mix.columns.tolist(), pairs_ab)
        out.append(np.abs(cell_gaps(mix, his_mix, his_for)).ravel())
    return np.concatenate(out)


def load_hzb(path: Path) -> Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]]:
    """{(season, hand): (pitcher ids, HZB distance matrix)} from build_artifacts' hzb_distance.npz."""
    if not path.exists():
        return {}
    z = np.load(path)
    keys = [k[: -len("_pitcher")] for k in z.files if k.endswith("_pitcher")]
    return {(int(k.split("_")[0]), k.split("_")[1]): (z[f"{k}_pitcher"], z[f"{k}_distance"]) for k in keys}


def write_gap_scale(app_dir: Path, results_dir: Path) -> float:
    """Compute the heat map's full-red gap from the app tables and write it for the app and the README."""
    t0 = time.time()
    index = pd.read_parquet(app_dir / "index.parquet")
    arsenal = pd.read_parquet(app_dir / "arsenal.parquet")
    usage = pd.read_parquet(app_dir / "usage_by_count.parquet")
    states = json.loads((app_dir / "meta.json").read_text())["count_states"]
    pairs = comp_pairs(index, arsenal, load_hzb(app_dir / "hzb_distance.npz"))
    gaps = all_gaps(pairs, arsenal, usage, states)
    qs = [50, 75, PERCENTILE, 95]
    vals = np.percentile(gaps, qs)
    full_red = round(float(vals[qs.index(PERCENTILE)]), 1)
    print(f"[gap] {len(gaps):,} comp cells; |gap| percentiles (points): "
          + ", ".join(f"p{q} {v:.1f}" for q, v in zip(qs, vals)) + f"  ({time.time() - t0:.0f} s)")
    (app_dir / "gap_scale.json").write_text(json.dumps({
        "gap_full_red": full_red, "percentile": PERCENTILE, "n_cells": int(len(gaps)), "n_pairs": int(len(pairs)),
        "n_queries": int(pairs[["pitcher", "game_year", "p_throws"]].drop_duplicates().shape[0])}, indent=2) + "\n")
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"percentile": qs, "abs_gap_points": np.round(vals, 2), "n_cells": len(gaps), "n_pairs": len(pairs)}) \
        .to_csv(results_dir / "gameplan_gap_scale.csv", index=False)
    print(f"[save] {(app_dir / 'gap_scale.json').relative_to(ROOT)}; {(results_dir / 'gameplan_gap_scale.csv').relative_to(ROOT)}")
    return full_red


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate the gameplan heat map's color scale (D32).")
    ap.add_argument("--sample", action="store_true", help="use models/sample/app/")
    args = ap.parse_args()
    paths = Paths.make(args.sample)
    write_gap_scale(paths.models / "app", paths.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
