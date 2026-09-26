"""
HZB pitcher similarity: Healey, Zhao & Brooks (2017), implemented as published. It is a third
published physical baseline next to TJStats and SEAM (src/models/physical_similarity.py; D27).

Pipeline: a baseline scored in stage 4 (evaluate) and precomputed for the app in stage 6 (app tables).
Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

Inputs:  pitches.parquet from src/features/build_arsenal.py (one row per pitch; uses pitcher,
         game_year, p_throws, stand, pitch_type, release_speed, pfx_x, pfx_z)
Outputs: a symmetric pitcher x pitcher similarity matrix S = -D for one season and one throwing
         hand, with the same API as arsenal_similarity, so run_eval's align() takes it unchanged.
         The CLI prints Sigma and f per configuration, top-5 comps, and (with --check) metric tests.

The method (the paper's sections and equations):
    Sec. 1   pitch = (s, x, z): s = initial speed (mph); (x, z) = movement in inches relative to
             a spinless pitch at the same speed; +x = catcher's right, +z = up.
             Statcast: s = release_speed, x = pfx_x * 12, z = pfx_z * 12. Not mirrored, because
             every comparison is within one pitcher hand.
    Eq. (1)  signature vs. RHB: P_R = {(mu_1, w_1), ..., (mu_m, w_m)}; mu_i = his mean (s, x, z)
             for pitch type i vs. RHB, w_i = his share of pitches of type i vs. RHB. Same for P_L.
             No minimum count or usage share is stated, so every type he threw to that hand counts.
    Eq. (2)  ground distance G(i, j) = [(mu_i - mu_j) Sigma^-1 (mu_i - mu_j)^T]^(1/2) (Mahalanobis).
             Sigma = the covariance of the population of mean vectors mu_i for one platoon
             configuration (e.g. RHP vs. RHB, one season): every cluster mean of every pitcher in
             it, unweighted, no pitch floor. It's the same as Euclidean distance after whitening.
    Sec. 3.1 D_R, D_L = Earth Mover's Distance between signatures (Rubner et al. 2000) with ground
             distance G. Both sides' weights sum to 1, so the exact EMD is ot.emd2(a, b, M).
    Eq. (3)/(4)  D(A, B) = f_hR * D_R(A, B) + f_hL * D_L(A, B); f_hR, f_hL = league-average shares
             of pitches that pitchers of hand h throw to RHB / LHB (per season). Small D = similar.

Data scope (shared by every method, not specific to HZB): pitches.parquet is build_arsenal's
filtered table (regular season; no bunt attempts, KN/EP/FA/PO/UN pitch types, position players,
or pitches missing core tracking), so "every type he threw" means every type in that table.

Gap-fills (the only two):
    (1) Statcast pitch_type labels in place of Pitch Info's manually reviewed ones (Sec. 3.1 notes
        the EMD isn't sensitive to how pitches are classified), and Statcast/Hawk-Eye measurements
        in place of PITCHf/x.
    (2) If a pitcher threw no pitches to one batter hand, that pair's D uses only the side both
        pitchers have, with f renormalized to 1. If they share no side, D = +inf (never a comp).

Sources:
    Healey, G., Zhao, S., & Brooks, D. (2017). Measuring Pitcher Similarity: Technical Details.
        viXra:1705.0098v1, https://vixra.org/pdf/1705.0098v1.pdf (eqs. 1-4, Sections 1-4).
    Rubner, Y., Tomasi, C., & Guibas, L. J. (2000). The Earth Mover's Distance as a metric for
        image retrieval. International Journal of Computer Vision, 40(2), 99-121.
    Flamary, R., et al. (2021). POT: Python Optimal Transport. JMLR 22(78), 1-8 (ot.emd2).

Usage (from the repo root):
    python -m src.models.hzb_similarity --sample --season 2025 --check

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import ot
import pandas as pd
from scipy.spatial.distance import cdist

from src.models.inputs import MIN_PITCHES
from src.models.physical_similarity import top_k

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["pitcher", "game_year", "p_throws"]
PITCH_COLUMNS = KEYS + ["stand", "pitch_type", "release_speed", "pfx_x", "pfx_z"]
FEATURES = ["speed_mph", "x_in", "z_in"]  # the paper's (s, x, z)
BATTER_HANDS = ("R", "L")
TOL = 1e-9


@dataclass
class Platoon:
    """One platoon configuration (this pitcher hand vs. one batter hand) in one season.

    sigma:   3x3 covariance of every cluster mean in the configuration (eq. 2)
    whiten:  L with Sigma^-1 = L L^T, so ||(mu_i - mu_j) L|| is the Mahalanobis distance
    means:   cluster means (s, x, z), one row per pitcher x pitch type, grouped by pitcher row
    weights: each cluster's share of that pitcher's pitches to this batter hand
    start, end: per pitcher row, its clusters are means[start:end] (start == end: no pitches)
    D:       pairwise EMD on this side; NaN where either pitcher has no pitches to this hand
    """
    stand: str
    sigma: np.ndarray
    whiten: np.ndarray
    means: np.ndarray
    weights: np.ndarray
    start: np.ndarray
    end: np.ndarray
    D: np.ndarray

    @property
    def has_side(self) -> np.ndarray:
        """True for pitchers who threw at least one pitch to this batter hand."""
        return self.end > self.start


def load_pitches(base: Path, season: int) -> pd.DataFrame:
    """The columns HZB needs, one season only (the full pitch file is 2.8M rows)."""
    df = pd.read_parquet(base / "pitches.parquet", columns=PITCH_COLUMNS, filters=[("game_year", "==", season)])
    print(f"[hzb] loaded {len(df):,} pitches x {df.shape[1]} columns for {season}")
    return df


def movement_frame(pitches: pd.DataFrame, season: int, hand: str) -> pd.DataFrame:
    """Pitches from one season and throwing hand, with the paper's (s, x, z) in mph and inches."""
    keep = ((pitches["game_year"] == season) & (pitches["p_throws"] == hand)).fillna(False).to_numpy(bool)
    df = pitches.loc[keep, KEYS + ["stand", "pitch_type"]].copy()
    df["speed_mph"] = pitches.loc[keep, "release_speed"].astype("float64")
    # why: Statcast pfx_x/pfx_z are the paper's (x, z), movement vs. a spinless pitch, but in feet;
    # x12 gives the paper's inches. Catcher's view, not arm-side mirrored: HZB never compares across hands.
    df["x_in"] = pitches.loc[keep, "pfx_x"].astype("float64") * 12.0
    df["z_in"] = pitches.loc[keep, "pfx_z"].astype("float64") * 12.0
    out = df.dropna(subset=FEATURES + ["stand", "pitch_type"])
    print(f"[hzb] {hand}HP {season}: {len(out):,} pitches ({len(df) - len(out):,} dropped for missing values)")
    return out


def batter_hand_shares(moves: pd.DataFrame) -> Dict[str, float]:
    """f_hR, f_hL (eq. 4): league-wide share of this pitcher hand's pitches thrown to RHB / LHB.

    moves: movement_frame() output, i.e. every pitch of one pitcher hand in one season.
    """
    share = moves["stand"].value_counts(normalize=True)
    return {b: float(share.get(b, 0.0)) for b in BATTER_HANDS}


def signatures(moves: pd.DataFrame) -> pd.DataFrame:
    """Eq. (1): one row per pitcher x batter hand x pitch type, with mean (s, x, z), count n and weight w.

    moves: movement_frame() output (one pitcher hand, one season).
    w = n / (his pitches to that batter hand). Every pitch type counts, however rare (no threshold
    is published).
    """
    g = moves.groupby(KEYS + ["stand", "pitch_type"], observed=True, sort=True)
    sig = g[FEATURES].mean()
    sig["n"] = g.size()
    sig = sig.reset_index()
    sig["weight"] = sig["n"] / sig.groupby(KEYS + ["stand"])["n"].transform("sum")
    print(f"[hzb] signatures: {len(sig):,} pitcher x batter-hand x type clusters, "
          f"{sig['pitcher'].nunique()} pitchers")
    return sig.sort_values(KEYS + ["stand", "pitch_type"]).reset_index(drop=True)


def platoon_covariance(means: np.ndarray) -> np.ndarray:
    """Eq. (2)'s Sigma: sample covariance of all cluster means in one configuration (unweighted)."""
    # why: np.cov's default ddof = 1. The paper doesn't say; with ~1,000-2,600 means per configuration,
    # ddof only rescales Sigma by n / (n - 1) < 1.001.
    return np.cov(means, rowvar=False)


def whitener(sigma: np.ndarray) -> np.ndarray:
    """L with Sigma^-1 = L L^T. Row vectors mu @ L then have Euclidean distance = Mahalanobis distance."""
    return np.linalg.cholesky(np.linalg.inv(sigma))


def side_distances(means: np.ndarray, weights: np.ndarray, start: np.ndarray, end: np.ndarray,
                   L: np.ndarray) -> np.ndarray:
    """D_side[a, b] = EMD between the two pitchers' signatures on one side (eq. 2 ground distance).

    Whitens once, builds the cluster-to-cluster ground distances with one cdist, then solves the
    upper triangle only (the EMD is symmetric). The diagonal is solved too, so --check can test D(A, A) = 0.
    """
    Y = np.einsum("nk,kj->nj", means, L)  # einsum, not @: avoids numpy 2.0 + Accelerate matmul warnings
    G = cdist(Y, Y)
    n = len(start)
    D = np.full((n, n), np.nan)
    have = np.flatnonzero(end > start)
    for i, a in enumerate(have):
        sa, ea = start[a], end[a]
        wa = weights[sa:ea]
        for b in have[i:]:
            sb, eb = start[b], end[b]
            # why: exact EMD (network simplex), because both signatures carry total mass 1 (Sec. 3.1).
            D[a, b] = D[b, a] = ot.emd2(wa, weights[sb:eb], G[sa:ea, sb:eb])
    return D


def build_platoon(sig: pd.DataFrame, pitchers: pd.DataFrame, stand: str) -> Platoon:
    """Sigma, whitener, and pairwise EMD for one batter hand. sig and pitchers are one pitcher hand."""
    side = sig[sig["stand"] == stand]
    row_of = {k: i for i, k in enumerate(pitchers[KEYS].itertuples(index=False, name=None))}
    rows = np.array([row_of[k] for k in side[KEYS].itertuples(index=False, name=None)], dtype=int)
    start = np.searchsorted(rows, np.arange(len(pitchers)), side="left")
    end = np.searchsorted(rows, np.arange(len(pitchers)), side="right")
    means = side[FEATURES].to_numpy("float64")
    sigma = platoon_covariance(means)
    L = whitener(sigma)
    D = side_distances(means, side["weight"].to_numpy("float64"), start, end, L)
    return Platoon(stand, sigma, L, means, side["weight"].to_numpy("float64"), start, end, D)


def combine_sides(platoons: Dict[str, Platoon], f: Dict[str, float]) -> np.ndarray:
    """Eq. (4): D = f_R D_R + f_L D_L, with gap-fill (2) for pitchers missing a batter hand."""
    DR, DL = platoons["R"].D, platoons["L"].D
    okR, okL = np.isfinite(DR), np.isfinite(DL)
    D = np.full(DR.shape, np.inf)  # no shared side: never a comp
    both = okR & okL
    D[both] = f["R"] * DR[both] + f["L"] * DL[both]
    # why (gap-fill 2): one side only -> f renormalized to 1 on that side, i.e. that side's D alone.
    D[okR & ~okL] = DR[okR & ~okL]
    D[okL & ~okR] = DL[okL & ~okR]
    return D


def hzb_distance(pitches: pd.DataFrame, season: int,
                 hand: str) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Platoon], Dict[str, float]]:
    """(pitchers, D, platoons, f) for one season and throwing hand; D is symmetric with a 0 diagonal."""
    moves = movement_frame(pitches, season, hand)
    sig = signatures(moves)
    pitchers = sig[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    f = batter_hand_shares(moves)
    platoons = {b: build_platoon(sig, pitchers, b) for b in BATTER_HANDS}
    D = combine_sides(platoons, f)
    print(f"[hzb] {hand}HP {season}: D is {D.shape[0]} x {D.shape[1]}")
    return pitchers, D, platoons, f


def hzb_similarity(pitches: pd.DataFrame, season: int, hand: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Same API as arsenal_similarity: (pitchers sorted by KEYS, S), with S = -D (higher = more similar)."""
    pitchers, D, _, _ = hzb_distance(pitches, season, hand)
    return pitchers, 0.0 - D  # 0.0 - D keeps the diagonal at +0.0; unmatched pairs become -inf


def gap_fill_counts(platoons: Dict[str, Platoon], mask: np.ndarray) -> Dict[str, int]:
    """Distinct pitcher pairs (both in mask) by how many batter hands they share: 2, 1 (gap-fill 2), or 0."""
    hasR, hasL = platoons["R"].has_side, platoons["L"].has_side
    iu = np.triu_indices(len(hasR), k=1)
    keep = mask[iu[0]] & mask[iu[1]]
    a, b = iu[0][keep], iu[1][keep]
    shared = (hasR[a] & hasR[b]).astype(int) + (hasL[a] & hasL[b]).astype(int)
    return {"pairs": int(len(a)), "both_sides": int((shared == 2).sum()),
            "one_side_gap_fill": int((shared == 1).sum()), "no_shared_side": int((shared == 0).sum())}


# ---------------------------------------------------------------------------
# Checks (--check)
# ---------------------------------------------------------------------------

def pair_distance(platoons: Dict[str, Platoon], f: Dict[str, float], a: int, b: int) -> float:
    """D(a, b) recomputed from scratch for one pair: Mahalanobis via scipy (no whitening), then EMD."""
    parts = {}
    for stand, p in platoons.items():
        if p.has_side[a] and p.has_side[b]:
            M = cdist(p.means[p.start[a]:p.end[a]], p.means[p.start[b]:p.end[b]],
                      "mahalanobis", VI=np.linalg.inv(p.sigma))
            parts[stand] = float(ot.emd2(p.weights[p.start[a]:p.end[a]], p.weights[p.start[b]:p.end[b]], M))
    if not parts:
        return float("inf")
    total = sum(f[s] for s in parts)
    return sum(f[s] * d for s, d in parts.items()) / total


def random_triples(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n random triples (a, b, c) of distinct indices in [0, m)."""
    out = np.empty((0, 3), dtype=int)
    while len(out) < n:
        t = rng.integers(0, m, size=(2 * n, 3))
        t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
        out = np.vstack([out, t])
    return out[:n]


def metric_checks(D: np.ndarray, platoons: Dict[str, Platoon], f: Dict[str, float], eligible: np.ndarray,
                  n_samples: int = 10_000, seed: int = 372) -> Dict[str, float]:
    """D(A, A) = 0, symmetry (matrix and recomputed D(b, a)), and the triangle inequality among eligible pitchers."""
    rng = np.random.default_rng(seed)
    n = len(D)
    out: Dict[str, float] = {"max_self": float(np.max(np.abs(np.diag(D))))}
    fin = np.isfinite(D) & np.isfinite(D).T
    out["max_asym_matrix"] = float(np.max(np.abs(D[fin] - D.T[fin]))) if fin.any() else 0.0
    out["inf_mismatch"] = int((np.isinf(D) != np.isinf(D.T)).sum())
    # Recompute D(b, a) with the reverse argument order and an independent ground distance
    # (scipy's Mahalanobis, no Cholesky), then compare to the fast path's D(a, b).
    pairs = rng.integers(0, n, size=(n_samples, 2))
    rev = np.array([pair_distance(platoons, f, b, a) for a, b in pairs])
    fwd = D[pairs[:, 0], pairs[:, 1]]
    both = np.isfinite(rev) & np.isfinite(fwd)
    out["max_asym_recomputed"] = float(np.max(np.abs(rev[both] - fwd[both]))) if both.any() else 0.0
    out["recompute_inf_mismatch"] = int((np.isinf(rev) != np.isinf(fwd)).sum())
    # Whitening matches scipy's Mahalanobis on every cluster pair, per side.
    out["max_whiten_err"] = max(
        float(np.max(np.abs(cdist(np.einsum("nk,kj->nj", p.means, p.whiten), np.einsum("nk,kj->nj", p.means, p.whiten))
                            - cdist(p.means, p.means, "mahalanobis", VI=np.linalg.inv(p.sigma)))))
        for p in platoons.values())
    idx = np.flatnonzero(eligible)
    t = idx[random_triples(len(idx), n_samples, rng)]
    viol = D[t[:, 0], t[:, 2]] - D[t[:, 0], t[:, 1]] - D[t[:, 1], t[:, 2]]
    viol = viol[np.isfinite(viol)]
    out["triples"] = int(len(t))
    out["triangle_violations"] = int((viol > TOL).sum())
    out["max_triangle_violation"] = float(max(viol.max(), 0.0)) if len(viol) else 0.0
    return out


def checks_pass(c: Dict[str, float]) -> bool:
    """All metric checks within tolerance."""
    return (c["max_self"] <= TOL and c["max_asym_matrix"] <= TOL and c["inf_mismatch"] == 0
            and c["max_asym_recomputed"] <= 1e-7 and c["recompute_inf_mismatch"] == 0
            and c["max_whiten_err"] <= 1e-7 and c["triangle_violations"] == 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def usage_string(pitches: pd.DataFrame, season: int, pitcher: int, hand: str, top: int = 4) -> str:
    """His most-used pitch types this season (both batter hands), e.g. 'SI 41% SL 30% ...'."""
    p = pitches[((pitches["game_year"] == season) & (pitches["p_throws"] == hand)
                 & (pitches["pitcher"] == pitcher)).fillna(False).to_numpy(bool)]
    share = p["pitch_type"].value_counts(normalize=True).head(top)
    return " ".join(f"{t} {v:.0%}" for t, v in share.items())


def print_config(platoons: Dict[str, Platoon], f: Dict[str, float], hand: str) -> None:
    """Sigma per platoon configuration and the batter-hand shares f."""
    print(f"[hzb] {hand}HP f: f_{hand}R = {f['R']:.4f}, f_{hand}L = {f['L']:.4f}")
    for stand, p in platoons.items():
        print(f"[hzb] Sigma {hand}HP vs {stand}HB ({len(p.means):,} cluster means; rows/cols = s mph, x in, z in):")
        for row in p.sigma:
            print("        " + "  ".join(f"{v:9.3f}" for v in row))


def main() -> int:
    ap = argparse.ArgumentParser(description="HZB (Healey, Zhao & Brooks 2017) comps, plus metric checks.")
    ap.add_argument("--sample", action="store_true", help="use data/processed/sample/")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--n-targets", type=int, default=3, help="pitchers with the most pitches, per hand")
    ap.add_argument("--check", action="store_true", help="test D(A,A)=0, symmetry, triangle inequality")
    args = ap.parse_args()

    base = ROOT / "data" / "processed" / ("sample" if args.sample else "")
    pitches = load_pitches(base, args.season)
    names = pd.read_parquet(base / "pitcher_seasons.parquet")[KEYS + ["player_name", "n_pitches"]]
    ok_all, total_s = True, 0.0

    for hand in ["R", "L"]:
        t0 = time.perf_counter()
        pitchers, D, platoons, f = hzb_distance(pitches, args.season, hand)
        secs = time.perf_counter() - t0
        total_s += secs
        pitchers = pitchers.merge(names, on=KEYS, how="left")
        print(f"[hzb] {hand}HP: {len(pitchers)} pitchers, {secs:.1f} s")
        print_config(platoons, f, hand)

        eligible = (pitchers["n_pitches"].fillna(0).to_numpy() >= MIN_PITCHES)
        for label, mask in [("all pitchers", np.ones(len(pitchers), bool)),
                            (f">= {MIN_PITCHES} pitches", eligible)]:
            print(f"[hzb] {hand}HP gap-fill (2), {label}: {gap_fill_counts(platoons, mask)}")
        print(f"[hzb] {hand}HP pitchers with no pitches vs RHB: {int((~platoons['R'].has_side).sum())}, "
              f"vs LHB: {int((~platoons['L'].has_side).sum())}")

        S = 0.0 - D
        targets = pitchers.sort_values("n_pitches", ascending=False).head(args.n_targets).index
        for t in targets:
            pid = pitchers.loc[t, "pitcher"]
            print(f"\n  {hand}HP target: {pitchers.loc[t, 'player_name']} ({args.season}) "
                  f"[{usage_string(pitches, args.season, pid, hand)}]")
            for r in top_k(pitchers, S, int(t)).itertuples():
                print(f"    D {-r.similarity:6.3f}  {r.player_name:<26} [{usage_string(pitches, args.season, r.pitcher, hand)}]")
        print()

        if args.check:
            c = metric_checks(D, platoons, f, eligible)
            ok = checks_pass(c)
            ok_all &= ok
            print(f"[hzb] {hand}HP checks {'PASS' if ok else 'FAIL'}: " + ", ".join(f"{k}={v:.3g}" for k, v in c.items()))

    print(f"[hzb] total distance time, both hands: {total_s:.1f} s")
    if args.check:
        print(f"[hzb] checks overall: {'PASS' if ok_all else 'FAIL'}")
        return 0 if ok_all else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
