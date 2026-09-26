"""
TJStats fidelity check: how closely does our TJStats rebuild reproduce tjstats.ca's own scores,
and what kernel width sigma (the tool's one unpublished number) does the site use? (D26)

Pipeline: an optional check next to stage 4 (evaluate); its sigma is frozen into the TJStats baseline.
Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

The tool publishes its method ("How It Works"): z-score each metric within each pitch type; a
Gaussian/RBF kernel turns the distance into a similarity; the full-arsenal score is usage-weighted
over the target's pitch types. It doesn't publish sigma, nor which population the z-scores are
taken over. So:
    1. Variants (8): metrics {the tool's 6 defaults, all 10 of ours} x z population {every pitch of
       the type, pitcher-season means of the type} x hands {same hand, both hands mirrored}.
    2. On the FIT set (the site's 2025 Tarik Skubal page), per variant: Spearman(-d^2, site %) over
       the per-pitch cells (sigma-free); sigma fitted by minimizing the per-pitch MAE of 100*s vs.
       site %; overall MAE; top-20 overlap with the site's list.
    3. Selection rule, fixed before the held-out page is scored (a train/test split on the site's own
       pages, so the held-out check can't be tuned to): lowest per-pitch MAE; within
       0.1 pt -> higher Spearman. The chosen variant and sigma (rounded to 4 decimals) are frozen.
    4. The HELD-OUT page (2025 Zack Wheeler) is scored once with the frozen variant and sigma.
       Bar: per-pitch and overall MAE <= 5 points; top-20 overlap >= 15/20.
Every cell goes through src/models/physical_similarity.py (type_distances -> kernel ->
similarity_from_blocks), the same functions the eval's TJStats row uses.

Pool (as on the site with its defaults): every 2025 regular-season pitcher of the target's hand,
no role filter, no 300-pitch floor; a pitch type counts if thrown >= 10 times.

Inputs:  data/reference/tjstats_site_2025.csv (hand-transcribed from two tjstats.ca screenshots;
         see data/reference/README.md), data/processed/{arsenal_pitch_type,pitcher_seasons,pitches}.parquet,
         data/raw/statcast_2025_*.parquet (team only, to resolve the one hidden name)
Outputs: docs/results/tjstats_fidelity_variants.csv  one row per variant (fit metrics; held-out
                                                     columns are labeled diagnostics, not used to choose)
         docs/results/tjstats_fidelity_cells.csv     per-cell site vs. ours (all variants on the fit
                                                     set; the chosen variant on the held-out set)
         docs/results/tjstats_fidelity_heldout.csv   the chosen variant on both pages vs. the bar
         docs/results/tjstats_fit.json               the frozen variant + sigma (copied into physical_similarity.py's TJSTATS constants)

Sources:
    TJStats Pitch Similarity (T. Nestico), https://tjstats.ca/pitch-similarity/ ("How It Works")

Usage (from the repo root):
    python -m src.eval.tjstats_fidelity

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import unicodedata
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from src.models.physical_similarity import (
    KEYS, ROOT, TEN_METRICS, TJSTATS, TJSTATS_DEFAULT_METRICS, PhysicalSimilarityConfig, TypeBlock,
    arsenal_similarity, kernel, load_pitch_moments, similarity_from_blocks, type_distances,
)

SEASON = 2025
SITE_CSV = ROOT / "data" / "reference" / "tjstats_site_2025.csv"
SITE_CSV_2026 = ROOT / "data" / "reference" / "tjstats_site_2026.csv"
SINGLE_PITCH_SEASON = 2026  # the single-pitch page is from a season sigma was never fitted to
RESULTS = ROOT / "docs" / "results"
METRIC_SETS: Dict[str, Tuple[str, ...]] = {"6": TJSTATS_DEFAULT_METRICS, "10": TEN_METRICS}
Z_POPULATIONS = ("pitch", "pitcher")
HANDS = ("same", "both")
SIGMA_BOUNDS = (0.05, 50.0)
SIGMA_DECIMALS = 4
TIE_PT = 0.1          # per-pitch MAE tie band for the selection rule
BAR_MAE = 5.0         # held-out bar, points
BAR_OVERLAP = 15      # held-out bar, of 20
TOP_N = 20


# ---------------------------------------------------------------------------
# Site data and name resolution
# ---------------------------------------------------------------------------
def fold(name: str) -> str:
    """Lowercase, accents removed, for name matching (as in src/eval/comps_report.py)."""
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().strip()


def last_first_to_first_last(name: str) -> str:
    """'Gómez, Yoendrys' -> 'Yoendrys Gómez'."""
    last, _, first = str(name).partition(", ")
    return f"{first} {last}".strip() if first else last


def load_site(path=SITE_CSV) -> pd.DataFrame:
    """The transcribed site table (one row per target x row x pitch type)."""
    site = pd.read_csv(path, dtype={"comp": "string", "team": "string"})
    print(f"[site] {len(site)} rows; sets {site['set'].value_counts().to_dict()}")
    return site


def name_lookup(ps: pd.DataFrame, hand: str, season: int = SEASON) -> Dict[str, List[int]]:
    """Folded 'first last' -> pitcher ids, over one season's pitchers of one hand."""
    p = ps[(ps["game_year"] == season) & (ps["p_throws"] == hand)]
    out: Dict[str, List[int]] = {}
    for pid, name in zip(p["pitcher"].astype(int), p["player_name"]):
        out.setdefault(fold(last_first_to_first_last(name)), []).append(pid)
    return out


def resolve_names(site: pd.DataFrame, ps: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add a pitcher id to every named site row; return the names that don't match exactly once."""
    site = site.copy()
    site["pitcher"] = np.nan
    site["target_id"] = np.nan
    problems = []
    for hand in site["target_hand"].unique():
        look = name_lookup(ps, hand)
        rows = site["target_hand"] == hand
        for name in pd.unique(pd.concat([site.loc[rows, "target"], site.loc[rows, "comp"].dropna()])):
            ids = look.get(fold(name), [])
            if len(ids) != 1:
                problems.append(f"{name} ({hand}HP): {len(ids)} matches")
                continue
            site.loc[rows & (site["comp"] == name), "pitcher"] = ids[0]
            site.loc[rows & (site["target"] == name), "target_id"] = ids[0]
    print(f"[names] resolved {site.loc[site['row_kind'] == 'comp', 'pitcher'].notna().sum()} comp cells; "
          f"problems: {problems or 'none'}")
    return site, problems


def pitcher_teams(season: int = SEASON) -> pd.Series:
    """pitcher -> set of teams he pitched for (regular season), from raw Statcast."""
    cols = ["pitcher", "home_team", "away_team", "inning_topbot", "game_type"]
    raw = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((ROOT / "data" / "raw").glob(f"statcast_{season}_*.parquet"))])
    raw = raw[raw["game_type"] == "R"]
    team = raw["home_team"].where(raw["inning_topbot"] == "Top", raw["away_team"])
    print(f"[teams] {len(raw):,} {season} regular-season raw rows")
    return team.groupby(raw["pitcher"].astype(int)).agg(lambda s: set(s))


def hidden_row_candidates(site: pd.DataFrame, by_type: pd.DataFrame, teams: pd.Series) -> pd.DataFrame:
    """Candidates for a comp row with no name: same hand, team shown, and exactly the matched types shown.

    Uses only pitch-type sets (>= 10 pitches) and team; no similarity value (no circularity).
    """
    out = []
    for (tset, rank), g in site[(site["row_kind"] == "comp") & site["comp"].isna()].groupby(["set", "rank"]):
        hand, team = g["target_hand"].iat[0], g["team"].iat[0]
        typed = g[g["pitch_type"] != "OVERALL"]
        target_types = set(typed["pitch_type"])
        shown = set(typed.loc[typed["site_pct"].notna(), "pitch_type"])
        rows = by_type[(by_type["game_year"] == SEASON) & (by_type["p_throws"] == hand) & (by_type["n"] >= TJSTATS.min_type_pitches)]
        types = rows.groupby(rows["pitcher"].astype(int))["pitch_type"].agg(set)
        for pid, ts in types.items():
            if (ts & target_types) == shown and team in teams.get(pid, set()):
                out.append({"set": tset, "rank": rank, "pitcher": pid, "team": team})
    cand = pd.DataFrame(out, columns=["set", "rank", "pitcher", "team"])
    print(f"[hidden] {len(cand)} candidate(s) for the unnamed row(s)")
    return cand


def usage_check(site: pd.DataFrame, by_type: pd.DataFrame) -> pd.DataFrame:
    """Our usage (% of all his pitches) vs. the site's displayed usage, per target and pitch type."""
    u = site[site["row_kind"] == "usage"][["set", "target", "target_id", "pitch_type", "site_pct"]].copy()
    ours = by_type[by_type["game_year"] == SEASON][["pitcher", "pitch_type", "usage", "n"]].copy()
    ours["pitcher"] = ours["pitcher"].astype(int)
    u = u.merge(ours, left_on=["target_id", "pitch_type"], right_on=["pitcher", "pitch_type"], how="left")
    u["ours_pct"] = 100.0 * u["usage"]
    u["diff_pts"] = u["ours_pct"] - u["site_pct"]
    return u[["set", "target", "pitch_type", "site_pct", "ours_pct", "diff_pts", "n"]]


# ---------------------------------------------------------------------------
# Variants and cells
# ---------------------------------------------------------------------------
def variant_config(metrics: str, z_population: str, hands: str, sigma: float = 1.0) -> PhysicalSimilarityConfig:
    """TJStats with one reading of the unpublished details (rows, kernel, matching stay the tool's)."""
    feats = METRIC_SETS[metrics]
    return dataclasses.replace(TJSTATS, name=f"tjstats_{metrics}_{z_population}_{hands}", features=feats,
                               weights=(1.0,) * len(feats), z_population=z_population, z_hands=hands,
                               sigma=sigma)


def block_lookup(pitchers: pd.DataFrame, blocks: List[TypeBlock]) -> Tuple[Dict[int, int], Dict[str, Tuple[TypeBlock, Dict[int, int]]]]:
    """pitcher id -> row in `pitchers`; pitch type -> (block, row in pitchers -> row in block)."""
    row_of = {int(p): i for i, p in enumerate(pitchers["pitcher"])}
    by_t = {b.match_type: (b, {int(r): j for j, r in enumerate(b.rows)}) for b in blocks}
    return row_of, by_t


def target_cells(site_set: pd.DataFrame, pitchers: pd.DataFrame, blocks: List[TypeBlock]) -> pd.DataFrame:
    """One row per site comp x target pitch type: site %, our d^2 (NaN where either lacks the type)."""
    row_of, by_t = block_lookup(pitchers, blocks)
    target = int(site_set["target_id"].dropna().iat[0])
    cells = site_set[(site_set["row_kind"] == "comp") & (site_set["pitch_type"] != "OVERALL")].copy()
    d2 = []
    for pid, t in zip(cells["pitcher"], cells["pitch_type"]):
        entry = by_t.get(t)
        if entry is None or np.isnan(pid) or int(pid) not in row_of:
            d2.append(np.nan)
            continue
        b, j = entry
        i_t, i_c = j.get(row_of.get(target, -1)), j.get(row_of[int(pid)])
        d2.append(np.nan if i_t is None or i_c is None else float(b.q[i_t, i_c]))
    cells["d2"] = d2
    cells["used"] = cells["site_pct"].notna() & cells["d2"].notna()
    cells["flag"] = np.select([cells["site_pct"].notna() & cells["d2"].isna(),
                               cells["site_pct"].isna() & cells["d2"].notna()],
                              ["site matched, ours not", "ours matched, site not"], "")
    return cells


def per_pitch_mae(d2: np.ndarray, site_pct: np.ndarray, sigma: float) -> float:
    """MAE (points) of 100 * exp(-d^2 / 2 sigma^2) against the site's integer %."""
    return float(np.mean(np.abs(100.0 * np.exp(-d2 / (2.0 * sigma ** 2)) - site_pct)))


def fit_sigma(d2: np.ndarray, site_pct: np.ndarray) -> float:
    """sigma minimizing the per-pitch MAE: log-spaced grid, then a bounded 1-D refinement in log sigma."""
    grid = np.exp(np.linspace(np.log(SIGMA_BOUNDS[0]), np.log(SIGMA_BOUNDS[1]), 4001))
    maes = np.array([per_pitch_mae(d2, site_pct, s) for s in grid])
    i = int(np.argmin(maes))
    lo, hi = np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, len(grid) - 1)])
    res = minimize_scalar(lambda ls: per_pitch_mae(d2, site_pct, float(np.exp(ls))), bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-8})
    best = float(np.exp(res.x)) if res.fun <= maes[i] else float(grid[i])
    return round(best, SIGMA_DECIMALS)


def overall_and_top(site_set: pd.DataFrame, pitchers: pd.DataFrame, blocks: List[TypeBlock],
                    config: PhysicalSimilarityConfig, exclude_ranks: Tuple[int, ...] = ()) -> Dict:
    """Overall MAE vs. the site's OVERALL column and top-20 overlap, from the full arsenal matrix."""
    S = similarity_from_blocks(len(pitchers), blocks, config)
    row_of, _ = block_lookup(pitchers, blocks)
    target = int(site_set["target_id"].dropna().iat[0])
    scores = S[row_of[target]].copy()
    scores[row_of[target]] = -np.inf
    top = pitchers["pitcher"].to_numpy()[np.argsort(-scores, kind="stable")[:TOP_N]].astype(int)

    ov = site_set[(site_set["row_kind"] == "comp") & (site_set["pitch_type"] == "OVERALL")
                  & ~site_set["rank"].isin(exclude_ranks) & site_set["pitcher"].notna()].copy()
    ov["ours_pct"] = [100.0 * scores[row_of[int(p)]] if int(p) in row_of else np.nan for p in ov["pitcher"]]
    named = set(ov["pitcher"].astype(int))
    return {"overall": ov, "overall_mae": float((ov["ours_pct"] - ov["site_pct"]).abs().mean()),
            "top20_overlap": len(named & set(top)), "overlap_of": len(named), "top20": top,
            "target_row_scores": scores}


def score_set(site_set: pd.DataFrame, pitchers: pd.DataFrame, blocks: List[TypeBlock],
              config: PhysicalSimilarityConfig, exclude_ranks: Tuple[int, ...] = ()) -> Dict:
    """Per-pitch cells + MAE, overall MAE, top-20 overlap for one page at a fixed sigma."""
    cells = target_cells(site_set, pitchers, blocks)
    cells = cells[~cells["rank"].isin(exclude_ranks)].copy()
    cells["ours_pct"] = 100.0 * kernel(cells["d2"].to_numpy(), config)
    cells["abs_err"] = (cells["ours_pct"] - cells["site_pct"]).abs()
    used = cells[cells["used"]]
    out = overall_and_top(site_set, pitchers, blocks, config, exclude_ranks)
    out.update(cells=cells, n_cells=len(used), per_pitch_mae=float(used["abs_err"].mean()),
               spearman=float(spearmanr(-used["d2"], used["site_pct"]).correlation))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_type = pd.read_parquet(ROOT / "data" / "processed" / "arsenal_pitch_type.parquet")
    ps = pd.read_parquet(ROOT / "data" / "processed" / "pitcher_seasons.parquet",
                         columns=KEYS + ["player_name", "n_pitches"])
    print(f"[load] arsenal_pitch_type {by_type.shape}; {SEASON} rows {int((by_type['game_year'] == SEASON).sum())}, "
          f"of which n >= {TJSTATS.min_type_pitches}: {int(((by_type['game_year'] == SEASON) & (by_type['n'] >= TJSTATS.min_type_pitches)).sum())}")
    moments = load_pitch_moments(SEASON, features=TEN_METRICS)
    return by_type, ps, moments


def run_variants(site: pd.DataFrame, by_type: pd.DataFrame, moments: pd.DataFrame,
                 hold_exclude: Tuple[int, ...]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Fit sigma per variant on the fit page; held-out columns are computed only as labeled diagnostics."""
    fit, hold = site[site["set"] == "fit"], site[site["set"] == "holdout"]
    fit_hand, hold_hand = fit["target_hand"].iat[0], hold["target_hand"].iat[0]
    rows, cells_all, cache = [], [], {}
    for metrics in METRIC_SETS:
        for zpop in Z_POPULATIONS:
            for hands in HANDS:
                cfg = variant_config(metrics, zpop, hands)
                pf, bf = type_distances(by_type, cfg, SEASON, fit_hand, moments)
                cells = target_cells(fit, pf, bf)
                used = cells[cells["used"]]
                rho = float(spearmanr(-used["d2"], used["site_pct"]).correlation)
                sigma = fit_sigma(used["d2"].to_numpy(), used["site_pct"].to_numpy())
                cfg = dataclasses.replace(cfg, sigma=sigma)
                f = score_set(fit, pf, bf, cfg)
                ph, bh = type_distances(by_type, cfg, SEASON, hold_hand, moments)
                h = score_set(hold, ph, bh, cfg, hold_exclude)  # diagnostic only; never used to choose
                cache[cfg.name] = {"cfg": cfg, "fit": f, "hold": h, "fit_blocks": (pf, bf)}
                cells_all.append(f["cells"].assign(variant=cfg.name))
                rows.append({"variant": cfg.name, "metrics": metrics, "z_population": zpop, "hands": hands,
                             "n_cells": f["n_cells"], "spearman": rho, "sigma": sigma,
                             "fit_per_pitch_mae": f["per_pitch_mae"], "fit_overall_mae": f["overall_mae"],
                             "fit_top20_overlap": f["top20_overlap"],
                             "fit_cells_site_only": int((f["cells"]["flag"] == "site matched, ours not").sum()),
                             "fit_cells_ours_only": int((f["cells"]["flag"] == "ours matched, site not").sum()),
                             "diag_holdout_per_pitch_mae": h["per_pitch_mae"],
                             "diag_holdout_overall_mae": h["overall_mae"],
                             "diag_holdout_top20_overlap": h["top20_overlap"]})
                print(f"[variant] {cfg.name:<26} cells {f['n_cells']}  rho {rho:.3f}  sigma {sigma:.4f}  "
                      f"per-pitch MAE {f['per_pitch_mae']:.2f}  overall MAE {f['overall_mae']:.2f}  top20 {f['top20_overlap']}/20")
    return pd.DataFrame(rows), pd.concat(cells_all, ignore_index=True), cache


def choose(variants: pd.DataFrame) -> str:
    """Lowest fit per-pitch MAE; variants within TIE_PT of it are tied -> the higher Spearman wins."""
    best = variants["fit_per_pitch_mae"].min()
    tied = variants[variants["fit_per_pitch_mae"] <= best + TIE_PT]
    return str(tied.sort_values(["spearman", "fit_per_pitch_mae"], ascending=[False, True])["variant"].iat[0])


def check_one_implementation(by_type: pd.DataFrame, moments: pd.DataFrame, chosen: Dict, fit: pd.DataFrame) -> Optional[bool]:
    """arsenal_similarity(TJSTATS) must reproduce the chosen variant's Skubal overall scores exactly.

    Returns None (with a warning) if physical_similarity.py's constants are not the frozen variant yet.
    """
    cfg = chosen["cfg"]
    same = (TJSTATS.features == cfg.features and TJSTATS.z_population == cfg.z_population
            and TJSTATS.z_hands == cfg.z_hands and TJSTATS.sigma == cfg.sigma
            and TJSTATS.min_type_pitches == cfg.min_type_pitches)
    if not same:
        print(f"[check] WARNING: TJSTATS in physical_similarity.py ({TJSTATS.features}, {TJSTATS.z_population}, "
              f"{TJSTATS.z_hands}, sigma {TJSTATS.sigma}) is not the frozen variant; update its constants.")
        return None
    hand = fit["target_hand"].iat[0]
    pitchers, S = arsenal_similarity(by_type, TJSTATS, SEASON, hand, moments)
    target = int(fit["target_id"].dropna().iat[0])
    row = int(np.flatnonzero(pitchers["pitcher"].to_numpy() == target)[0])
    theirs = chosen["fit"]["target_row_scores"]
    ours = S[row].copy()
    ours[row] = -np.inf
    assert np.array_equal(pitchers["pitcher"].to_numpy(), chosen["fit_blocks"][0]["pitcher"].to_numpy())
    assert np.allclose(ours, theirs, rtol=0.0, atol=1e-12), "arsenal_similarity(TJSTATS) != fidelity Skubal scores"
    print(f"[check] arsenal_similarity(TJSTATS) reproduces the fidelity script's {len(ours) - 1} Skubal overall scores")
    return True


def single_pitch_check(out_path: Path = RESULTS / "tjstats_fidelity_2026.csv") -> pd.DataFrame:
    """Score a tjstats.ca *single-pitch* page with the frozen config, on a season it was not fitted to.

    The site can rank candidates on one pitch type instead of the whole arsenal. That mode is this module's
    per-pitch similarity with the usage weighting left out, so it checks the kernel and the z-scores on their
    own. Settings are the tool's defaults, as transcribed in data/reference/README.md: same hand, a type
    counting once thrown >= 10 times, no role filter and no season-pitch floor (D26).

    Nothing here is fitted: sigma and the variant stay frozen at the values chosen on 2025.
    """
    site = load_site(SITE_CSV_2026)
    season = int(site["season"].iat[0]) if "season" in site.columns else SINGLE_PITCH_SEASON
    hand, ptype = site["target_hand"].iat[0], site["pitch_type"].iat[0]
    by_type = pd.read_parquet(ROOT / "data" / "processed" / "arsenal_pitch_type.parquet")
    ps = pd.read_parquet(ROOT / "data" / "processed" / "pitcher_seasons.parquet",
                         columns=KEYS + ["player_name", "n_pitches"])

    pitchers, blocks = type_distances(by_type, TJSTATS, season, hand)
    block = next(b for b in blocks if b.match_type == ptype)
    s = 100.0 * kernel(block.q, TJSTATS)                      # per-pitch similarity, in the site's percent
    look = name_lookup(ps, hand, season)
    ids = pitchers["pitcher"].astype(int).to_numpy()
    slot = {int(ids[r]): i for i, r in enumerate(block.rows)}  # pitcher id -> row within this pitch type

    def resolve(name: str) -> Optional[int]:
        hits = look.get(fold(name), [])
        return hits[0] if len(hits) == 1 else None

    target_id = resolve(site.loc[site["row_kind"] == "target", "target"].iat[0])
    if target_id is None or target_id not in slot:
        raise SystemExit(f"[single-pitch] target not found among {season} {hand}HP {ptype} throwers")
    ours = pd.Series(s[slot[target_id]], index=[int(ids[r]) for r in block.rows])

    rows = []
    for r in site[site["row_kind"] == "comp"].itertuples():
        pid = resolve(r.comp)
        rows.append({"rank": r.rank, "comp": r.comp, "pitcher": pid, "site_pct": r.site_pct,
                     "ours_pct": round(float(ours[pid]), 2) if pid in ours.index else np.nan})
    out = pd.DataFrame(rows)
    out["abs_error"] = (out["ours_pct"] - out["site_pct"]).abs()
    matched = out.dropna(subset=["ours_pct"])
    # Overlap of the two top-N lists: the site's named comps against ours over the same candidate pool.
    ours_top = set(ours.drop(index=target_id).sort_values(ascending=False).head(len(out)).index)
    out["in_our_top20"] = out["pitcher"].isin(ours_top)
    summary = {"season": season, "target": site["target"].iat[0], "pitch_type": ptype,
               "n_named": len(out), "n_scored": len(matched),
               "per_pitch_mae": round(float(matched["abs_error"].mean()), 3),
               "spearman": round(float(matched[["site_pct", "ours_pct"]].corr(method="spearman").iloc[0, 1]), 4),
               "overlap": int(out["in_our_top20"].sum())}
    for k, v in summary.items():
        out[k] = v
    out.to_csv(out_path, index=False)
    print(f"[single-pitch] {summary['target']} {season} {ptype}: {summary['n_scored']}/{summary['n_named']} named "
          f"comps scored, per-pitch MAE {summary['per_pitch_mae']} pts, Spearman {summary['spearman']}, "
          f"{summary['overlap']}/{summary['n_named']} in our top {summary['n_named']}")
    print(out[["rank", "comp", "site_pct", "ours_pct", "abs_error", "in_our_top20"]].to_string(index=False))
    print(f"[save] {out_path.relative_to(ROOT)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit TJStats' kernel width to tjstats.ca and score a held-out page.")
    ap.add_argument("--single-pitch", action="store_true",
                    help="only re-score the single-pitch page in data/reference/tjstats_site_2026.csv")
    args = ap.parse_args()
    if args.single_pitch:
        single_pitch_check()
        return 0
    site = load_site()
    by_type, ps, moments = load_inputs()
    site, name_problems = resolve_names(site, ps)
    teams = pitcher_teams()
    hidden = hidden_row_candidates(site, by_type, teams)
    names = ps[ps["game_year"] == SEASON].set_index(ps.loc[ps["game_year"] == SEASON, "pitcher"].astype(int))["player_name"]
    hold_exclude: Tuple[int, ...] = ()
    hidden_note = "no unnamed rows"
    for (tset, rank), g in hidden.groupby(["set", "rank"]):
        who = ", ".join(f"{names.get(p, p)} ({p})" for p in g["pitcher"])
        if len(g) == 1:
            m = (site["set"] == tset) & (site["rank"] == rank) & (site["row_kind"] == "comp")
            site.loc[m, "pitcher"] = int(g["pitcher"].iat[0])
            hidden_note = f"{tset} rank {rank}: unique candidate {who}; used"
        else:
            hold_exclude = hold_exclude + (int(rank),)
            hidden_note = (f"{tset} rank {rank}: {len(g)} candidates ({who}); row excluded from the MAEs, "
                           f"overlap counted over the {TOP_N - 1} named rows")
    print(f"[hidden] {hidden_note}")
    usage = usage_check(site, by_type)
    print("[usage] ours vs. site (%):\n" + usage.round(2).to_string(index=False))

    variants, fit_cells, cache = run_variants(site, by_type, moments, hold_exclude)
    chosen_name = choose(variants)
    variants["chosen"] = variants["variant"] == chosen_name
    chosen = cache[chosen_name]
    cfg = chosen["cfg"]
    print(f"\n[choose] {chosen_name}  sigma {cfg.sigma:.4f} (frozen; rule: lowest fit per-pitch MAE, "
          f"ties within {TIE_PT} pt -> higher Spearman)")

    fit, hold = chosen["fit"], chosen["hold"]
    passes = {"per_pitch": hold["per_pitch_mae"] <= BAR_MAE, "overall": hold["overall_mae"] <= BAR_MAE,
              "overlap": hold["top20_overlap"] >= BAR_OVERLAP}
    top_hidden = sorted(set(hidden["pitcher"]) & set(hold["top20"])) if len(hidden) else []
    print(f"[held-out] Wheeler: per-pitch MAE {hold['per_pitch_mae']:.2f} over {hold['n_cells']} cells; "
          f"overall MAE {hold['overall_mae']:.2f} over {len(hold['overall'])} rows; "
          f"top-20 overlap {hold['top20_overlap']}/{hold['overlap_of']} named "
          f"(hidden-row candidates in our top 20: {[names.get(p, p) for p in top_hidden] or 'none'})")
    print(f"[held-out] bar (MAE <= {BAR_MAE}, overlap >= {BAR_OVERLAP}): {passes}")

    # Diagnostic: the site shows Skubal's FS column (3 pitches in our data, below the 10 floor).
    diag_cfg = dataclasses.replace(cfg, min_type_pitches=1)
    fit_set = site[site["set"] == "fit"]
    pd_, bd = type_distances(by_type, diag_cfg, SEASON, fit_set["target_hand"].iat[0], moments)
    diag_cells = target_cells(fit_set, pd_, bd)
    fs = diag_cells[(diag_cells["pitch_type"] == "FS") & diag_cells["site_pct"].notna()]
    fs_note = "; ".join(f"{c}: site {s:.0f}, ours {100 * np.exp(-d / (2 * cfg.sigma ** 2)):.1f} (target FS kept at n >= 1)"
                        for c, s, d in zip(fs["comp"], fs["site_pct"], fs["d2"]))
    print(f"[diag] target-FS cells: {fs_note or 'none'}")

    check = check_one_implementation(by_type, moments, chosen, site[site["set"] == "fit"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    variants.to_csv(RESULTS / "tjstats_fidelity_variants.csv", index=False)
    hold_cells = hold["cells"].assign(variant=chosen_name)
    cells_cols = ["variant", "set", "target", "rank", "comp", "pitcher", "pitch_type", "site_pct", "d2", "used", "flag"]
    cells = pd.concat([fit_cells[cells_cols], hold_cells[cells_cols]], ignore_index=True)
    cells = cells.merge(pd.DataFrame({"variant": list(cache), "sigma": [cache[v]["cfg"].sigma for v in cache]}),
                        on="variant")
    cells["ours_pct"] = 100.0 * np.exp(-cells["d2"] / (2.0 * cells["sigma"] ** 2))
    cells["abs_err"] = (cells["ours_pct"] - cells["site_pct"]).abs()
    overall_rows = []
    for tag, res in (("fit", fit), ("holdout", hold)):
        o = res["overall"][["set", "target", "rank", "comp", "pitcher", "pitch_type", "site_pct", "ours_pct"]].copy()
        o["variant"], o["sigma"], o["used"], o["flag"] = chosen_name, cfg.sigma, True, ""
        o["abs_err"] = (o["ours_pct"] - o["site_pct"]).abs()
        overall_rows.append(o)
    cells = pd.concat([cells] + overall_rows, ignore_index=True)
    cells["chosen"] = cells["variant"] == chosen_name
    cells.to_csv(RESULTS / "tjstats_fidelity_cells.csv", index=False)

    summary = pd.DataFrame([
        {"set": tag, "target": site.loc[site["set"] == tag, "target"].iat[0], "variant": chosen_name, "sigma": cfg.sigma,
         "n_cells": r["n_cells"], "per_pitch_mae": r["per_pitch_mae"], "n_overall": len(r["overall"]),
         "overall_mae": r["overall_mae"], "top20_overlap": r["top20_overlap"], "overlap_of": r["overlap_of"],
         "bar_mae": BAR_MAE, "bar_overlap": BAR_OVERLAP,
         "passes_per_pitch": r["per_pitch_mae"] <= BAR_MAE, "passes_overall": r["overall_mae"] <= BAR_MAE,
         "passes_overlap": r["top20_overlap"] >= BAR_OVERLAP}
        for tag, r in (("fit", fit), ("holdout", hold))])
    summary.to_csv(RESULTS / "tjstats_fidelity_heldout.csv", index=False)

    fit_json = {
        "generated_by": "python -m src.eval.tjstats_fidelity", "date": str(date.today()), "season": SEASON,
        "source": "https://tjstats.ca/pitch-similarity/ (How It Works); site values in data/reference/tjstats_site_2025.csv",
        "chosen_variant": chosen_name, "sigma": cfg.sigma, "metrics": list(cfg.features),
        "z_population": cfg.z_population, "z_hands": cfg.z_hands, "min_type_pitches": cfg.min_type_pitches,
        "kernel": "s = exp(-d^2 / (2 sigma^2)), d^2 = sum of squared z-score differences",
        "selection_rule": f"lowest fit (Skubal) per-pitch MAE after the sigma fit; within {TIE_PT} pt -> higher Spearman",
        "fit": {k: summary.iloc[0][k] for k in ["n_cells", "per_pitch_mae", "overall_mae", "top20_overlap"]},
        "holdout": {k: summary.iloc[1][k] for k in ["n_cells", "per_pitch_mae", "overall_mae", "top20_overlap", "overlap_of"]},
        "holdout_passes": {k: bool(v) for k, v in passes.items()},
        "hidden_row": hidden_note, "name_problems": name_problems, "target_fs_diagnostic": fs_note,
        "usage_check_pct": usage[["set", "pitch_type", "site_pct", "ours_pct", "diff_pts", "n"]].round(2).to_dict("records"),
        "one_implementation_check": check,
    }
    fit_json = json.loads(json.dumps(fit_json, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    (RESULTS / "tjstats_fit.json").write_text(json.dumps(fit_json, indent=2) + "\n")

    print("\n" + variants.drop(columns=["variant"]).round(3).to_string(index=False))
    print(f"\n[write] {RESULTS / 'tjstats_fidelity_variants.csv'} ({len(variants)} rows), "
          f"tjstats_fidelity_cells.csv ({len(cells)} rows), tjstats_fidelity_heldout.csv, tjstats_fit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
