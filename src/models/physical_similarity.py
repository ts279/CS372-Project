"""
Hand-specified pitcher similarity: TJStats' Pitch Similarity tool (rebuilt to its defaults) and
SEAM's published formula, as configurations of one implementation, plus the naive raw-Euclidean
row. These are the physical baselines the learned embedding is compared against
(docs/design_decisions.md, D4 and D19). Each keeps its published weights; only the learned model
learns its own (D14).

Pipeline: the hand-built baselines scored in stage 4 (evaluate) and shown in stage 7 (app).
Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

Inputs:  arsenal_pitch_type.parquet from src/features/build_arsenal.py (one row per
         pitcher-season x pitch type, with shape/release means, pitch count n, and usage).
         Optional: per-type pitch-level moments (load_pitch_moments), needed only by a config
         that z-scores over pitches (a variant tested in src/eval/tjstats_fidelity.py, not the default)
Outputs: an arsenal-level similarity matrix for one season and one throwing hand, and
         top-k comps (CLI prints them for a sanity check)

Both methods, per pitch type t (matched on the Statcast pitch-type label, as published):
    1. z-score each feature within pitch type t;
    2. q = sum_k v_k * (z_ak - z_bk)^2      (v = diagonal feature weights);
    3. per-pitch similarity s_t = kernel(q):
         TJStats: Gaussian (RBF) kernel      s = exp(-q / (2 sigma^2))
         SEAM:    s = exp(-q ** (1 / n))     (= exp(-||x_a - x_b||_V) with the 1/d_p power)
Arsenal similarity of candidate b to target a (TJStats' "Full Arsenal" mode):
    S(a, b) = sum_t w_a,t * s_t(a, b),  w_a,t = target's usage of t (share of ALL his pitches);
    0 if b doesn't throw t.

TJStats, rebuilt to the tool's defaults (Same Hand on, Min Pitches 10, default metrics):
    metrics velocity, iVB, HB, extension, release height, arm angle; a pitch type counts if the
    pitcher threw it >= 10 times; types matched on the raw Statcast pitch_type; z-scores within
    each pitch type over the season's pitcher-season x type rows (both hands, mirrored); RBF
    kernel width TJSTATS_SIGMA. Same-hand comps are applied by the caller (the eval, D16).
    sigma is the one number the tool doesn't publish: it was fitted to the site's own 2025
    Skubal scores and checked on a held-out Wheeler screenshot (src/eval/tjstats_fidelity.py,
    docs/results/tjstats_fit.json).

SEAM is never averaged with TJStats: it is its own comparison row, and the alpha blend uses
TJStats as its physical side (D4, D19).

Sources:
    TJStats Pitch Similarity (T. Nestico), https://tjstats.ca/pitch-similarity/ ("How It Works":
        z-scores within each pitch type; a Gaussian/RBF kernel turns distance into similarity;
        the full-arsenal score is usage-weighted over pitch types)
    SEAM formula: Wapner, Dalpiaz & Eck (2022), "SEAM methodology for context-rich player matchup
        evaluations in baseball", arXiv:2005.07742v2, eq. (4) (the 1/d_p power) and Sections 4-5.
        v1 (Young, Dalpiaz & Eck 2020, arXiv:2005.07742v1, Section 4.1) used a square root instead;
        v1 Section 3 is the source for z-scoring the features first.

Usage (from the repo root):
    python -m src.models.physical_similarity --season 2025
    python -m src.models.physical_similarity --sample --season 2025

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["pitcher", "game_year", "p_throws"]
MOMENT_HANDS = ("L", "R", "both")  # pitch-level moments are kept per hand and pooled


@dataclass(frozen=True)
class PhysicalSimilarityConfig:
    """One hand-specified similarity method.

    features:         per-pitch-type columns compared (means over the pitcher-season)
    weights:          diagonal of V, applied to squared z-score differences (same order as features)
    distance:         "rbf" -> s = exp(-q / (2 sigma^2)); "seam" -> s = exp(-q ** (1 / n));
                      "euclidean" -> the raw row (no z-scores, no kernel; raw_euclidean_similarity)
    type_map:         relabel Statcast pitch types before matching (merged rows are usage-weighted)
    drop_types:       pitch types the method excludes
    rows:             "arsenal" -> types with >= 3% usage (in_arsenal); "min_pitches" -> types
                      thrown >= min_type_pitches times
    min_type_pitches: the pitch-count floor when rows == "min_pitches"
    z_population:     "pitcher" -> z-score over the season's pitcher-season x type rows (unweighted);
                      "pitch" -> over every pitch of that type in the season (needs `moments`)
    z_hands:          "both" -> both hands pooled (horizontal features are arm-side mirrored);
                      "same" -> only pitchers of the target's hand
    sigma:            RBF kernel width (distance "rbf" only)
    """
    name: str
    features: Tuple[str, ...]
    weights: Tuple[float, ...]
    distance: str
    type_map: Tuple[Tuple[str, str], ...] = ()
    drop_types: Tuple[str, ...] = ()
    rows: str = "arsenal"
    min_type_pitches: int = 0
    z_population: str = "pitcher"
    z_hands: str = "both"
    sigma: float = 0.0


@dataclass
class TypeBlock:
    """One pitch type's pairwise distances among the pitchers who throw it (one season, one hand)."""
    match_type: str
    rows: np.ndarray   # row indices into the `pitchers` table
    usage: np.ndarray  # each row's usage of this type (its weight when it is the target)
    q: np.ndarray      # [len(rows), len(rows)] weighted squared z-score distance


# The ten per-type shape/release metrics built by src/features/build_arsenal.py (TJSTATS_FEATURES).
TEN_METRICS = ("release_speed", "ivb_in", "hb_arm_in", "release_spin_rate", "release_extension",
               "release_height", "release_side_arm", "vaa", "haa_arm", "arm_angle")
# The tool's default metrics: Velocity, iVB, HB, Extension, Release Height, Arm Angle.
TJSTATS_DEFAULT_METRICS = ("release_speed", "ivb_in", "hb_arm_in", "release_extension",
                           "release_height", "arm_angle")

# why (D26): sigma is the only number the tool doesn't publish, and "z-score within each pitch type" doesn't
# say over which population. Both were fitted to the site's own 2025 Skubal per-pitch scores (78 cells):
# 8 variants (6 or 10 metrics x pitch- or pitcher-level z-scores x same or both hands), sigma fitted
# per variant by minimizing the MAE; the lowest-MAE variant won (6 metrics, pitcher-season means,
# both hands: Spearman 0.976, MAE 2.3 pts). Frozen, then checked once on a held-out Wheeler page
# (per-pitch MAE 2.0, overall MAE 1.2, top-20 overlap 18/19). Recompute with
# `python -m src.eval.tjstats_fidelity`; values in docs/results/tjstats_fit.json.
TJSTATS_SIGMA = 1.7444
TJSTATS_Z_POPULATION = "pitcher"  # pitcher-season x type rows with n >= 10, unweighted
TJSTATS_Z_HANDS = "both"          # both hands pooled; horizontal features are arm-side mirrored
TJSTATS_MIN_TYPE_PITCHES = 10  # the tool's "Min Pitches" default

# TJStats: the tool's defaults. Equal weights; the RBF kernel carries the scale.
TJSTATS = PhysicalSimilarityConfig(
    name="tjstats",
    features=TJSTATS_DEFAULT_METRICS,
    weights=(1.0,) * len(TJSTATS_DEFAULT_METRICS),
    distance="rbf",
    rows="min_pitches",
    min_type_pitches=TJSTATS_MIN_TYPE_PITCHES,
    z_population=TJSTATS_Z_POPULATION,
    z_hands=TJSTATS_Z_HANDS,
    sigma=TJSTATS_SIGMA,
)

# SEAM (v2): nine pitcher covariates (Section 4). V is not published numerically. The paper
# says V scales the characteristics and weights "stuff" (velocity, spin, movement) against
# "release" (release angles and release point) via the app's slider, whose default is 0.85
# (Fig. 2). Here: features are z-scored (as v1 does), then stuff gets 0.85 and release 0.15.
# Extension is in neither named group; it is counted as release (author's call, D19). Preprocessing
# follows Section 4: knuckle-curves become curveballs, forkballs become splitters, screwballs are removed.
SEAM_STUFF_WEIGHT = 0.85
SEAM = PhysicalSimilarityConfig(
    name="seam",
    features=("release_speed", "release_spin_rate", "hb_arm_in", "ivb_in",                        # stuff
              "rel_angle_h_arm", "rel_angle_v", "release_side_arm", "release_height", "release_extension"),  # release
    weights=(SEAM_STUFF_WEIGHT,) * 4 + (1.0 - SEAM_STUFF_WEIGHT,) * 5,
    distance="seam",
    type_map=(("KC", "CU"), ("FO", "FS")),
    drop_types=("SC",),
)

# Raw Euclidean (D4): the ten metrics, unscaled, on the arsenal rows (>= 3% usage). Kept separate
# from TJStats so rebuilding TJStats doesn't move this row.
RAW_EUCLIDEAN = PhysicalSimilarityConfig(
    name="raw",
    features=TEN_METRICS,
    weights=(1.0,) * len(TEN_METRICS),
    distance="euclidean",
)


# ---------------------------------------------------------------------------
# Rows and z-scores
# ---------------------------------------------------------------------------
def prepare_arsenal(by_type: pd.DataFrame, config: PhysicalSimilarityConfig) -> pd.DataFrame:
    """The pitch-type rows this method compares: drop its excluded types, merge relabeled ones.

    Rows are the >= 3% arsenal (rows="arsenal") or every type thrown >= min_type_pitches times
    (rows="min_pitches"). `usage` stays the share of the pitcher-season's *total* pitches, which
    is TJStats' weight.
    """
    if config.rows == "arsenal":
        keep = by_type["in_arsenal"]
    elif config.rows == "min_pitches":
        if "n" not in by_type.columns:
            raise ValueError(f"{config.name} keeps pitch types by count: by_type needs the pitch-count column 'n'")
        # Rows with no pitch count (the app's user-entered query arsenal) are kept: every type he
        # lists is one he throws.
        keep = by_type["n"].isna() | (by_type["n"] >= config.min_type_pitches)
    else:
        raise ValueError(f"unknown rows rule {config.rows!r}")
    df = by_type[keep & ~by_type["pitch_type"].isin(config.drop_types)].copy()
    df["match_type"] = df["pitch_type"].replace(dict(config.type_map))
    feats = list(config.features)
    # Usage-weighted mean when two Statcast types collapse into one (e.g. SEAM's KC -> CU).
    # why: each feature is weighted only by rows where it was measured, so a missing value stays
    # missing (and gets the documented fill downstream) instead of summing to a literal 0.
    keys = KEYS + ["match_type"]
    x = df[feats].astype("float64")
    num = pd.concat([df[keys], x.mul(df["usage"], axis=0)], axis=1).groupby(keys)[feats].sum(min_count=1)
    den = pd.concat([df[keys], x.notna().mul(df["usage"], axis=0)], axis=1).groupby(keys)[feats].sum()
    out = num / den.replace(0.0, np.nan)
    out["usage"] = df.groupby(keys)["usage"].sum()
    return out.reset_index()


def standardize_within_type(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """z-score each feature within pitch type, across the rows given (one season).

    Missing values become 0 after standardizing, i.e. the pitch-type average.
    """
    z = df.copy()
    x = df[features].astype("float64")  # Statcast ships some columns as nullable Float64
    g = x.groupby(df["match_type"])
    mean, std = g.transform("mean"), g.transform("std")
    z[features] = ((x - mean) / std.replace(0.0, np.nan)).fillna(0.0)
    return z


def pitch_type_moments(pitches: pd.DataFrame, features: Sequence[str] = TEN_METRICS) -> pd.DataFrame:
    """Per pitch type: mean and SD of each feature over every pitch of that type (one season).

    pitches: pitch rows of ONE season (pitch_type, p_throws, and the feature columns).
    Returns one row per (hands, match_type), hands in {"L", "R", "both"}, with columns
    <feature>_mean and <feature>_std (ddof = 1; NaN pitches skipped).
    why: TJStats z-scores "within each pitch type" without saying over what; the pitch-level
    population is one of the two readings tested in src/eval/tjstats_fidelity.py.
    """
    if pitches["game_year"].nunique() > 1:
        raise ValueError("pitch_type_moments expects one season's pitches")
    feats = list(features)
    x = pitches[feats].astype("float64")
    frames = []
    for hands in MOMENT_HANDS:
        mask = np.ones(len(pitches), bool) if hands == "both" else (pitches["p_throws"] == hands).to_numpy()
        g = x[mask].groupby(pitches.loc[mask, "pitch_type"].to_numpy())
        m = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
        m.index.name = "match_type"
        frames.append(m.reset_index().assign(hands=hands))
    out = pd.concat(frames, ignore_index=True)
    print(f"[moments] {len(pitches):,} pitches -> {len(out)} (hands x pitch type) rows, {len(feats)} features")
    return out


def load_pitch_moments(season: int, path: Optional[Path] = None,
                       features: Sequence[str] = TEN_METRICS) -> pd.DataFrame:
    """pitch_type_moments for one season, reading only the needed columns of pitches.parquet."""
    path = path or ROOT / "data" / "processed" / "pitches.parquet"
    cols = ["game_year", "p_throws", "pitch_type"] + list(features)
    pitches = pd.read_parquet(path, columns=cols, filters=[("game_year", "==", season)])
    print(f"[moments] read {len(pitches):,} {season} pitches from {path.name}")
    return pitch_type_moments(pitches, features)


def standardize_with_moments(df: pd.DataFrame, features: List[str], moments: pd.DataFrame) -> pd.DataFrame:
    """z-score each feature with externally supplied per-type moments (one hands group).

    Missing values (and types absent from `moments`) become 0, i.e. the pitch-type average.
    """
    missing = [f"{f}_{s}" for f in features for s in ("mean", "std") if f"{f}_{s}" not in moments.columns]
    if missing:
        raise ValueError(f"moments lack columns {missing}")
    m = df[["match_type"]].merge(moments, on="match_type", how="left")
    z = df.copy()
    x = df[features].astype("float64").to_numpy()
    mean = m[[f"{f}_mean" for f in features]].to_numpy("float64")
    std = m[[f"{f}_std" for f in features]].to_numpy("float64")
    std[std == 0.0] = np.nan
    z[features] = np.nan_to_num((x - mean) / std, nan=0.0)
    return z


def standardized_rows(by_type: pd.DataFrame, config: PhysicalSimilarityConfig, season: int, hand: str,
                      moments: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """This method's rows for one season and hand, z-scored over its z population."""
    rows = prepare_arsenal(by_type[by_type["game_year"] == season], config)
    feats = list(config.features)
    if config.z_population == "pitch":
        if moments is None:
            raise ValueError(f"{config.name} z-scores over pitch-level moments: pass "
                             f"moments=load_pitch_moments({season}) (or pitch_type_moments(<that season's pitches>))")
        group = hand if config.z_hands == "same" else "both"
        return standardize_with_moments(rows[rows["p_throws"] == hand], feats, moments[moments["hands"] == group])
    if config.z_population != "pitcher":
        raise ValueError(f"unknown z population {config.z_population!r}")
    # why (SEAM, and the pitcher-population reading of TJStats): z-scores use every pitcher-season of
    # that pitch type in the season, both hands unless z_hands == "same" (the horizontal features are
    # arm-side normalized); comps are then same-hand only (D16).
    pop = rows if config.z_hands == "both" else rows[rows["p_throws"] == hand]
    z = standardize_within_type(pop, feats)
    return z[z["p_throws"] == hand]


# ---------------------------------------------------------------------------
# Distances, kernel, arsenal similarity
# ---------------------------------------------------------------------------
def weighted_sq_distance(za: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pairwise q = sum_k v_k (z_ak - z_bk)^2 among rows of one pitch type."""
    diff = za[:, None, :] - za[None, :, :]
    return (diff ** 2 * weights).sum(axis=-1)


def kernel(q: np.ndarray, config: PhysicalSimilarityConfig) -> np.ndarray:
    """Per-pitch similarity from the weighted squared distance q."""
    n = len(config.features)
    if config.distance == "rbf":
        if not config.sigma > 0:
            raise ValueError(f"{config.name}: the RBF kernel needs sigma > 0 (got {config.sigma})")
        # why: the tool's published kernel: Gaussian in the z-scored distance, s in (0, 1].
        return np.exp(-q / (2.0 * config.sigma ** 2))
    if config.distance == "seam":
        return np.exp(-(q ** (1.0 / n)))
    raise ValueError(f"unknown distance {config.distance!r}")


def type_distances(by_type: pd.DataFrame, config: PhysicalSimilarityConfig, season: int, hand: str,
                   moments: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[TypeBlock]]:
    """(pitchers, one TypeBlock per matched pitch type) for one season and hand.

    Everything before the kernel, so a kernel width can be swept without recomputing distances.
    """
    z = standardized_rows(by_type, config, season, hand, moments)
    feats = list(config.features)
    pitchers = z[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    index = {tuple(r): i for i, r in enumerate(pitchers.itertuples(index=False))}
    weights = np.asarray(config.weights)
    blocks = []
    for t, grp in z.groupby("match_type"):
        rows = np.array([index[tuple(r)] for r in grp[KEYS].itertuples(index=False)])
        q = weighted_sq_distance(grp[feats].to_numpy(dtype="float64"), weights)
        blocks.append(TypeBlock(str(t), rows, grp["usage"].to_numpy(), q))
    return pitchers, blocks


def similarity_from_blocks(n_pitchers: int, blocks: List[TypeBlock], config: PhysicalSimilarityConfig) -> np.ndarray:
    """S[a, b] = sum_t usage_a,t * kernel(q_t(a, b)); candidates without type t add 0."""
    S = np.zeros((n_pitchers, n_pitchers))
    for b in blocks:
        s = kernel(b.q, config)
        w = b.usage[:, None]                 # target's usage of this type
        S[np.ix_(b.rows, b.rows)] += w * s   # candidates without this type add 0
    return S


def arsenal_similarity(by_type: pd.DataFrame, config: PhysicalSimilarityConfig,
                       season: int, hand: str, moments: Optional[pd.DataFrame] = None
                       ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Arsenal similarity matrix for one season and one throwing hand.

    Returns (pitchers, S), where pitchers lists the pitcher-seasons in row order and
    S[a, b] = sum_t usage_a,t * s_t(a, b). Rows are targets, columns candidates, and S is not
    symmetric: it is weighted by the target's usage.
    moments: pitch-level per-type moments (load_pitch_moments(season)); required only when
    config.z_population == "pitch" (a fidelity-check variant; shipped TJStats and SEAM don't use it).
    """
    pitchers, blocks = type_distances(by_type, config, season, hand, moments)
    return pitchers, similarity_from_blocks(len(pitchers), blocks, config)


def raw_euclidean_similarity(by_type: pd.DataFrame, season: int, hand: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """The naive baseline (D4): unscaled Euclidean distance on the ten metrics, per matched pitch type.

    S[a, b] = -sum_t usage_a,t * d_t(a, b). If candidate b doesn't throw t, d_t is the largest
    distance between any two pitchers who throw t (the worst possible match).
    why (D4): no z-scores and no kernel, on purpose: this is the naive baseline, and the z-scored
    methods are the serious ones. Unscaled, spin rate (thousands of rpm) dominates velocity (tens of
    mph) and movement (inches).
    """
    rows = prepare_arsenal(by_type[by_type["game_year"] == season], RAW_EUCLIDEAN)
    feats = list(RAW_EUCLIDEAN.features)
    x = rows[feats].astype("float64")
    rows[feats] = x.fillna(x.groupby(rows["match_type"]).transform("mean"))  # missing -> the type's mean
    rows = rows[rows["p_throws"] == hand]

    pitchers = rows[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    index = {tuple(r): i for i, r in enumerate(pitchers.itertuples(index=False))}
    D = np.zeros((len(pitchers), len(pitchers)))
    for _, grp in rows.groupby("match_type"):
        idx = np.array([index[tuple(r)] for r in grp[KEYS].itertuples(index=False)])
        d = cdist(grp[feats].to_numpy("float64"), grp[feats].to_numpy("float64"))
        contrib = np.full((len(idx), len(pitchers)), d.max() if d.size else 0.0)
        contrib[:, idx] = d
        D[idx] += grp["usage"].to_numpy()[:, None] * contrib
    return pitchers, -D


def top_k(pitchers: pd.DataFrame, S: np.ndarray, target_row: int, k: int = 5) -> pd.DataFrame:
    """The k most similar other pitchers to one target (the target itself excluded)."""
    scores = S[target_row].copy()
    same = pitchers["pitcher"].to_numpy() == pitchers.loc[target_row, "pitcher"]
    scores[same] = -np.inf
    order = np.argsort(-scores)[:k]
    out = pitchers.iloc[order].copy()
    out["similarity"] = scores[order]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Print TJStats and SEAM comps for a few pitchers.")
    ap.add_argument("--sample", action="store_true", help="use data/processed/sample/")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--n-targets", type=int, default=3, help="pitchers with the most pitches, per hand")
    args = ap.parse_args()

    base = ROOT / "data" / "processed" / ("sample" if args.sample else "")
    by_type = pd.read_parquet(base / "arsenal_pitch_type.parquet")
    names = pd.read_parquet(base / "pitcher_seasons.parquet")[KEYS + ["player_name", "n_pitches"]]
    print(f"[similarity] {len(by_type):,} pitcher-season x type rows; season {args.season}")
    # Both shipped configs z-score over pitcher-season means, so no pitch-level moments are needed here.
    # The pitch-level variant is only one of the 8 candidates in src/eval/tjstats_fidelity.py (D26).
    moments = None

    for hand in ["R", "L"]:
        results = {}
        for cfg in (TJSTATS, SEAM):
            pitchers, S = arsenal_similarity(by_type, cfg, args.season, hand, moments)
            results[cfg.name] = (pitchers.merge(names, on=KEYS, how="left"), S)
            print(f"[similarity] {cfg.name}: {len(pitchers)} {hand}HP pitcher-seasons; "
                  f"self-similarity median {np.median(np.diag(S)):.3f} (= usage covered by his rows)")
        pitchers = results["tjstats"][0]
        targets = pitchers.sort_values("n_pitches", ascending=False).head(args.n_targets).index
        for t in targets:
            print(f"\n  {hand}HP target: {pitchers.loc[t, 'player_name']} ({args.season})")
            for name, (p, S) in results.items():
                row = int(p.index[p["pitcher"] == pitchers.loc[t, "pitcher"]][0])
                comps = top_k(p, S, row)
                print(f"    {name:<8} " + "; ".join(f"{r.player_name} {r.similarity:.2f}" for r in comps.itertuples()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
