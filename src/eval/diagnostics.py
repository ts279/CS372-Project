"""
Diagnostics on 2025 (val) only: where the learned model's edge comes from, and how much the methods
agree. These are diagnostics, not results. They change no claim, test, or choice, and 2026 is untouched.

Pipeline: optional, after stage 6 (app tables), whose HZB distances it reads. Stages: fetch -> features ->
train -> evaluate -> comps report -> app tables -> app.

1. Location check. The learned model is the only method that sees location (each pitch type's
   attack-zone shares, D17). Does its walk/whiff edge come from that input or from learning?
   Two hand-built distances get the learned T1 model's exact arsenal inputs, z-scored within pitch type:
       phys12       the 12 shape and release features
       phys12_loc   the same 12 + the 4 attack-zone shares (heart, shadow, chase, waste)
   phys12_loc vs. phys12 = what location adds to a hand-built distance; learned vs. phys12_loc =
   what learning adds on the same inputs. Plain raw Euclidean is unscaled on purpose (spin in rpm
   swamps everything), so 0-1 zone shares added to it would test nothing.
2. T3a vs. HZB per stat, from the saved 2025 errors. The T3a embedding sees velocity, coarse
   movement, and pitch family only: no location, spin, or release.
3. Method agreement: mean top-10 overlap for each pair of learned / TJStats / HZB / SEAM, with the
   overlap two random top-10 lists would have.
4. Arm-angle gap: mean |arm-angle difference| (degrees, primary fastball) between each query and his
   top-10 comps, for learned / TJStats / HZB, next to the gap to his whole candidate pool (no skill).
   TJStats uses arm angle directly; HZB never sees it; the learned model reads it as one of 12 inputs.

Inputs:  data/processed/{pitcher_seasons,arsenal_pitch_type}.parquet, models/embeddings_t1.parquet,
         models/app/hzb_distance.npz (HZB distances from build_artifacts, the eval's hzb_distance code),
         docs/results/errors_val.csv (the saved comps' predictions for raw, HZB, learned, T3a)
Outputs: docs/results/diagnostics/location_check_val.csv, t3a_vs_hzb_val.csv, method_agreement_val.csv,
         arm_angle_gap_val.csv

Usage (from the repo root):
    python -m src.eval.diagnostics

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import dataclasses
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.eval.retrieval import STATS, knn_errors, top_k_comps
from src.eval.run_eval import align, compare, learned_matrix, physical_matrix, season_table
from src.features.build_arsenal import LOCATION_FEATURES, SHAPE_FEATURES
from src.models.physical_similarity import KEYS, RAW_EUCLIDEAN, prepare_arsenal, standardize_within_type
from src.models.train_embedding import Paths

YEAR = 2025  # val only: diagnostics never touch 2026 (D5)
REPORT = {"bb_pct": "BB%", "whiff_pct": "whiff%", "chase_pct": "chase%", "gb_pct": "GB%", "pitcher_rv100": "RV/100"}
LOC_VARIANTS = {"phys12": list(SHAPE_FEATURES), "phys12_loc": list(SHAPE_FEATURES) + list(LOCATION_FEATURES)}
SAVED = ["learned", "tjstats", "hzb", "seam"]  # recomputed here, checked against errors_val.csv, compared for agreement
ARM = ["learned", "tjstats", "hzb"]


def zscored_euclidean(by_type: pd.DataFrame, season: int, hand: str, features: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Raw Euclidean's structure on z-scored features: per matched pitch type, weighted by the target's
    usage; a candidate without the type gets the block's worst distance (as raw_euclidean_similarity)."""
    cfg = dataclasses.replace(RAW_EUCLIDEAN, name="diag", features=tuple(features), weights=(1.0,) * len(features))
    rows = prepare_arsenal(by_type[by_type["game_year"] == season], cfg)
    rows = rows[rows["p_throws"] == hand].reset_index(drop=True)
    # why: z-scores put mph, inches, rpm, and 0-1 zone shares on one scale, so location can count at all.
    rows = standardize_within_type(rows, features)

    pitchers = rows[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    index = {tuple(r): i for i, r in enumerate(pitchers.itertuples(index=False))}
    D = np.zeros((len(pitchers), len(pitchers)))
    for _, grp in rows.groupby("match_type"):
        idx = np.array([index[tuple(r)] for r in grp[KEYS].itertuples(index=False)])
        d = cdist(grp[features].to_numpy("float64"), grp[features].to_numpy("float64"))
        contrib = np.full((len(idx), len(pitchers)), d.max() if d.size else 0.0)
        contrib[:, idx] = d
        D[idx] += grp["usage"].to_numpy()[:, None] * contrib
    return pitchers, -D


def season_matrix(ps: pd.DataFrame, per_hand) -> np.ndarray:
    """Stack two per-hand (pitchers, S) results into one season matrix aligned with ps."""
    out = np.full((len(ps), len(ps)), -np.inf)
    for hand in ["R", "L"]:
        pitchers, S = per_hand(hand)
        align(ps, pitchers, S, out)
    return out


def hzb_from_artifacts(path: Path, season: int, hand: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """HZB similarity (minus the EMD) among pitcher-seasons with >= 300 pitches, as the app loads it."""
    z = np.load(path)
    ids = z[f"{season}_{hand}_pitcher"]
    pitchers = pd.DataFrame({"pitcher": ids.astype("int64"), "game_year": season, "p_throws": hand})
    return pitchers, -z[f"{season}_{hand}_distance"].astype("float64")


def errors_for(ps: pd.DataFrame, sims: Dict[str, np.ndarray]) -> pd.DataFrame:
    frames = []
    for name, sim in sims.items():
        e = knn_errors(ps, sim, stats=list(STATS))
        e.insert(0, "method", name)
        frames.append(e)
    return pd.concat(frames, ignore_index=True)


def check_against_saved(recomputed: pd.DataFrame, saved: pd.DataFrame) -> None:
    """The recomputed comps must reproduce the eval's saved predictions (same queries, same comps)."""
    key = ["method", "pitcher", "p_throws", "stat"]
    m = recomputed.merge(saved[key + ["predicted"]], on=key, suffixes=("", "_saved"))
    for name, g in m.groupby("method"):
        gap = (g["predicted"] - g["predicted_saved"]).abs()
        print(f"[diag] {name}: {len(g):,} rows matched to errors_val.csv; {int((gap > 1e-9).sum())} differ (max {gap.max():.2e})")


def deltas(errors: pd.DataFrame, methods: List[str], references: List[str]) -> pd.DataFrame:
    out = pd.concat([compare(errors, s, methods, ref) for ref in references for s in STATS], ignore_index=True)
    return out[out["method"] != out["reference"]].drop(columns=["baseline_mae"]).reset_index(drop=True)


def agreement(ps: pd.DataFrame, sims: Dict[str, np.ndarray], methods: List[str]) -> pd.DataFrame:
    """Mean top-10 overlap for each pair of methods over the same queries, and the overlap by chance."""
    tops = {m: top_k_comps(ps, sims[m]) for m in methods}
    q0, t0, pools = tops[methods[0]]
    k = t0.shape[1]
    chance = float(np.mean([k * k / len(p) for p in pools]))  # expected overlap of two random top-k lists
    rows = []
    for a, b in combinations(methods, 2):
        (qa, ta, _), (qb, tb, _) = tops[a], tops[b]
        assert np.array_equal(qa, qb), "every method must be scored on the same queries"
        ov = np.array([len(set(x) & set(y)) for x, y in zip(ta, tb)])
        rows.append({"method_a": a, "method_b": b, "n_queries": len(ov), "mean_overlap": ov.mean(),
                     "median_overlap": float(np.median(ov)), "share_no_overlap": float((ov == 0).mean()),
                     "chance_overlap": chance})
    return pd.DataFrame(rows)


def arm_angle_gap(ps: pd.DataFrame, by_type: pd.DataFrame, sims: Dict[str, np.ndarray], methods: List[str]) -> pd.DataFrame:
    """Mean |arm-angle gap| (degrees, primary fastball) from each query to his top-10 comps, per method,
    plus the gap to his whole candidate pool (what comps picked at random would give)."""
    fb = by_type[KEYS + ["pitch_type", "arm_angle"]].rename(columns={"pitch_type": "fb_type", "arm_angle": "fb_arm"})
    angle = ps[KEYS + ["fb_type"]].merge(fb, on=KEYS + ["fb_type"], how="left")["fb_arm"].to_numpy("float64")
    rows = []
    for m in methods:
        q, top, pools = top_k_comps(ps, sims[m])
        per_query = np.nanmean(np.abs(angle[top] - angle[q][:, None]), axis=1)
        rows.append({"method": m, "n_queries": len(q), "mean_abs_gap_deg": np.nanmean(per_query),
                     "median_abs_gap_deg": np.nanmedian(per_query)})
    pool_gap = np.array([np.nanmean(np.abs(angle[p] - angle[i])) for i, p in zip(q, pools)])
    rows.append({"method": "pool", "n_queries": len(q), "mean_abs_gap_deg": np.nanmean(pool_gap),
                 "median_abs_gap_deg": np.nanmedian(pool_gap)})
    return pd.DataFrame(rows)


def show(d: pd.DataFrame, label: str) -> None:
    d = d[d["stat"].isin(REPORT)].copy()
    d["delta_r (95% CI)"] = d.apply(lambda r: f"{r.delta_r:+.3f} [{r.delta_r_lo:+.3f}, {r.delta_r_hi:+.3f}]", axis=1)
    d["stat"] = d["stat"].map(REPORT)
    print(f"\n[diag] {label}\n{d[['stat', 'method', 'reference', 'r', 'delta_r (95% CI)']].round(3).to_string(index=False)}")


def main() -> int:
    paths = Paths.make(False)
    out_dir = paths.results / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons = pd.read_parquet(paths.data / "pitcher_seasons.parquet")
    by_type = pd.read_parquet(paths.data / "arsenal_pitch_type.parquet")
    ps = season_table(seasons, YEAR)
    saved = pd.read_csv(paths.results / "errors_val.csv")
    print(f"[diag] {YEAR}: {len(ps):,} pitcher-seasons; {len(by_type):,} pitch-type rows; {len(saved):,} saved error rows")

    sims = {name: season_matrix(ps, lambda h, f=feats: zscored_euclidean(by_type, YEAR, h, f))
            for name, feats in LOC_VARIANTS.items()}
    sims["learned"] = learned_matrix(ps, pd.read_parquet(paths.models / "embeddings_t1.parquet"))
    sims["tjstats"] = physical_matrix(ps, by_type, None, YEAR, "tjstats")
    sims["seam"] = physical_matrix(ps, by_type, None, YEAR, "seam")
    sims["hzb"] = season_matrix(ps, lambda h: hzb_from_artifacts(paths.models / "app" / "hzb_distance.npz", YEAR, h))

    recomputed = errors_for(ps, sims)
    check_against_saved(recomputed[recomputed["method"].isin(SAVED)], saved)

    # 1. location check
    errors = pd.concat([saved[saved["method"].isin(["raw", "learned"])],
                        recomputed[recomputed["method"].isin(list(LOC_VARIANTS))]], ignore_index=True)
    loc = deltas(errors, ["raw", "phys12", "phys12_loc", "learned"], ["learned", "raw", "phys12"])
    loc.to_csv(out_dir / "location_check_val.csv", index=False)
    show(loc[(loc["reference"] == "phys12") & (loc["method"] == "phys12_loc")], "location added to a hand-built distance")
    show(loc[(loc["reference"] == "learned") & loc["method"].isin(list(LOC_VARIANTS))], "hand-built on the learned inputs vs. learned")
    show(loc[(loc["reference"] == "raw") & (loc["method"] == "phys12_loc")], "phys12_loc vs. plain raw")

    # 2. T3a (no location) vs. HZB, saved errors only
    t3a = deltas(saved, ["hzb", "learned_t3a", "learned"], ["hzb"])
    t3a.to_csv(out_dir / "t3a_vs_hzb_val.csv", index=False)
    show(t3a, "learned T1 and T3a vs. HZB")

    # 3. method agreement
    agree = agreement(ps, sims, SAVED)
    agree.to_csv(out_dir / "method_agreement_val.csv", index=False)
    print(f"\n[diag] top-10 overlap, {YEAR}:\n{agree.round(2).to_string(index=False)}")

    # 4. arm-angle gap
    arm = arm_angle_gap(ps, by_type, sims, ARM)
    arm.to_csv(out_dir / "arm_angle_gap_val.csv", index=False)
    print(f"\n[diag] |arm-angle gap| to the top 10, primary fastball, {YEAR}:\n{arm.round(2).to_string(index=False)}")
    print(f"\n[diag] wrote {out_dir}/: location_check_val.csv ({len(loc)} rows), t3a_vs_hzb_val.csv ({len(t3a)}), "
          f"method_agreement_val.csv ({len(agree)}), arm_angle_gap_val.csv ({len(arm)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
