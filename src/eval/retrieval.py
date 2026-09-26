"""
Held-out-pitcher retrieval: do a pitcher's nearest comps predict his outcomes?

For each query pitcher-season, retrieve the k most similar *other* pitchers from the same season,
throwing hand, and role. Then predict each of his stats as the plain mean of theirs. Any
similarity method plugs in as a matrix: learned embedding, TJStats, SEAM, raw Euclidean, or an
alpha blend.

Protocol (D4, D5, D20, D23):
    - index = the same season as the query (one season in the index, so holding out the query's
      pitcher also removes every season of his; otherwise his own other season would be his
      nearest comp and every method would look good, D5);
    - candidates: same hand (D16), same role (D23), >= MIN_PITCHES pitches (D24), a different pitcher;
    - queries: pitcher-seasons with >= MIN_PITCHES pitches;
    - prediction: unweighted mean of the top-k. Unweighted, because similarity scales differ by
      method and only the ranking should matter;
    - stats: RV/100 (the headline) plus the per-stat breakdown (D22);
    - per-query errors are kept, so the paired bootstrap (D11) and the slices (velocity band,
      seen vs. new pitchers) are group-bys on one table.

Pipeline: the test behind stage 4 (evaluate); training also logs its 2025 r every epoch (D21).
Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

Inputs:  a pitcher-season table (pitcher_seasons.parquet rows) and a similarity matrix aligned with it
Outputs: the top-k comps per query; one row per query x stat (actual, predicted, errors); summaries

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.models.inputs import MIN_PITCHES

K_NEIGHBORS = 10
HEADLINE = "pitcher_rv100"
# Stat column -> display name (D22). RV/100 is the headline; the rest are the per-stat breakdown.
STATS: Dict[str, str] = {
    "pitcher_rv100": "RV/100", "k_pct": "K%", "bb_pct": "BB%", "whiff_pct": "whiff%",
    "chase_pct": "chase%", "gb_pct": "GB%", "damage_pct": "Damage%",
}
QUERY_COLS = ["pitcher", "game_year", "p_throws", "role", "player_name", "n_pitches"]


def embedding_similarity(emb: np.ndarray) -> np.ndarray:
    """Similarity in the learned space: negative Euclidean distance (higher = more similar)."""
    return -cdist(emb, emb)


def eligible(ps: pd.DataFrame, min_pitches: int = MIN_PITCHES) -> np.ndarray:
    """Pitcher-seasons that can be a query or a comp."""
    return ps["n_pitches"].to_numpy() >= min_pitches


def candidate_pool(ps: pd.DataFrame, i: int, ok: np.ndarray) -> np.ndarray:
    """Rows that can be comps for query row i: eligible, same hand, same role, a different pitcher."""
    return np.flatnonzero(ok & (ps["p_throws"].to_numpy() == ps["p_throws"].iat[i])
                          & (ps["role"].to_numpy() == ps["role"].iat[i])
                          & (ps["pitcher"].to_numpy() != ps["pitcher"].iat[i]))


def top_k_comps(ps: pd.DataFrame, sim: np.ndarray, k: int = K_NEIGHBORS,
                min_pitches: int = MIN_PITCHES) -> Tuple[np.ndarray, np.ndarray, list]:
    """(query rows, top-k comp rows [n_queries, k], candidate pools) within one season.

    ps: pitcher-seasons from ONE season, rows aligned with sim.
    sim[i, j]: similarity of candidate j to query i (rows = queries; may be asymmetric).
    """
    if ps["game_year"].nunique() != 1:
        raise ValueError("retrieval expects one season; the index is same-season by design (D20)")
    ok = eligible(ps, min_pitches)
    queries, tops, pools = [], [], []
    for i in np.flatnonzero(ok):
        cand = candidate_pool(ps, i, ok)
        if len(cand) < k:
            continue
        queries.append(i)
        tops.append(cand[np.argsort(-sim[i, cand], kind="stable")[:k]])
        pools.append(cand)
    return np.array(queries, dtype=int), np.array(tops, dtype=int).reshape(-1, k), pools


def knn_errors(ps: pd.DataFrame, sim: np.ndarray, stats: Sequence[str] = (HEADLINE,),
               k: int = K_NEIGHBORS, min_pitches: int = MIN_PITCHES) -> pd.DataFrame:
    """One row per query x stat: his actual value, his comps' mean, and the no-similarity baseline.

    The baseline predicts the mean of his whole candidate pool (same season, hand, role): what
    you'd guess knowing nothing about his arsenal.
    """
    q, top, pools = top_k_comps(ps, sim, k, min_pitches)
    base = ps.iloc[q][QUERY_COLS].reset_index(drop=True)
    frames = []
    for stat in stats:
        v = ps[stat].to_numpy(dtype="float64")
        f = base.copy()
        f["stat"] = stat
        f["actual"] = v[q]
        f["predicted"] = np.nanmean(v[top], axis=1)
        f["baseline"] = np.array([np.nanmean(v[pool]) for pool in pools])
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out = out[np.isfinite(out["actual"])].reset_index(drop=True)
    out["abs_error"] = (out["predicted"] - out["actual"]).abs()
    out["baseline_abs_error"] = (out["baseline"] - out["actual"]).abs()
    return out


def summarize(errors: pd.DataFrame) -> pd.DataFrame:
    """Per stat: Pearson r (comps' mean vs. actual), MAE, and the no-similarity baseline MAE."""
    rows = []
    for stat, g in errors.groupby("stat", sort=False):
        rows.append({"stat": stat, "n_queries": len(g), "r": float(np.corrcoef(g["predicted"], g["actual"])[0, 1]),
                     "mae": float(g["abs_error"].mean()), "baseline_mae": float(g["baseline_abs_error"].mean())})
    return pd.DataFrame(rows)


def headline(errors: pd.DataFrame) -> Dict[str, float]:
    """Summary of the headline stat (RV/100) as a dict."""
    return summarize(errors[errors["stat"] == HEADLINE]).iloc[0].to_dict()
