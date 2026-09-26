"""
Precompute the small tables the app loads, so the app never touches the raw or processed data.

Pipeline stage 6 of 7 (app tables). Stages: fetch -> features -> train -> evaluate -> comps report ->
app tables -> app. The tables are committed (D9) so the app runs from a clean clone.

Inputs:  data/processed/{pitches,arsenal_pitch_type,pitcher_seasons}.parquet (build_arsenal.py)
         models/embeddings_{t1,t3a}.parquet, models/published.json (train_embedding.py)
Outputs (models/app/, committed; --sample writes models/sample/app/):
    index.parquet           one row per pitcher-season: name, hand, role, pitch counts, primary FB velo,
                            arm angle, top 3 pitches, RV/100 and the per-stat rates, T1 + T3a embeddings
    arsenal.parquet         one row per pitcher-season x pitch type he threw: in_arsenal (>= 3%) flag,
                            pitch count, usage, shape means, attack-zone shares, per-type outcome
                            grades, and Savant's arsenal qualifier
    usage_by_count.parquet  pitch mix by count state x batter hand (the gameplan to transfer)
    locations.parquet       attack-zone shares by pitch type x batter hand
    hzb_distance.npz        HZB distances among pitcher-seasons with >= 300 pitches, per season x hand
    meta.json               team games per season, qualifiers, published models
    gap_scale.json          the gameplan heat map's full-red usage gap (gameplan_gap.py, D32; also
                            docs/results/gameplan_gap_scale.csv)

Qualifiers (D24, D29, D30): comps need 300 pitches; a per-pitch-type grade is flagged "small sample" unless
that type was thrown 2.5 times per team game, scaled to the team games played so far (Savant's rule).
A reliever's bar is the starter bar x (median reliever season pitches / median starter season pitches),
over the pitcher-seasons the app shows (>= 300 pitches).

Usage (from the repo root):
    python -m src.app.build_artifacts [--sample]

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

import numpy as np
import pandas as pd

from src.app.gameplan_gap import write_gap_scale
from src.features.build_arsenal import (CONTACT_CLASSES, LOCATION_FEATURES, OUT_OF_ZONE, SHAPE_FEATURES,
                                        SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS)
from src.models.hzb_similarity import KEYS as HZB_KEYS, hzb_distance, load_pitches
from src.models.inputs import ATTACK_ZONES, KEYS, MIN_PITCHES
from src.models.train_embedding import Paths

SAVANT_PITCHES_PER_TEAM_GAME = 2.5
TEAMS = 30
STAT_COLS = ["pitcher_rv100", "k_pct", "bb_pct", "whiff_pct", "chase_pct", "gb_pct", "damage_pct"]
# Count states from the pitcher's side.
# why: four count states instead of all 12 counts keep each gameplan row's sample readable. The alternative,
# a full 12-count grid, splits one pitcher-season thin enough that most cells stop meaning anything.
COUNT_STATES = ["first pitch", "ahead", "even", "behind"]


def count_state(balls: pd.Series, strikes: pd.Series) -> pd.Series:
    """first pitch (0-0), ahead (more strikes than balls), even (equal, after 0-0), behind (more balls)."""
    b, s = balls.astype(int), strikes.astype(int)
    return pd.Series(np.select([(b == 0) & (s == 0), s > b, s == b], COUNT_STATES[:3], "behind"), index=balls.index)


def team_games(pitches: pd.DataFrame) -> dict:
    """Average team games per season = 2 x distinct games / 30 teams (162 in a full season)."""
    g = pitches.groupby("game_year")["game_pk"].nunique()
    return {int(y): round(2 * n / TEAMS, 1) for y, n in g.items()}


def reliever_scale(seasons: pd.DataFrame) -> float:
    """Median season pitches of relievers / of starters, over pitcher-seasons with >= 300 pitches (D30)."""
    # why: relievers pitch far fewer innings (D30), so one bar for both flags nearly every reliever pitch.
    # The pool is the pitcher-seasons the app shows (D24), so call-ups don't drag the medians down.
    q = seasons[seasons["n_pitches"] >= MIN_PITCHES]
    if q["role"].nunique() < 2:  # --sample (one month) can lack 300-pitch relievers
        q = seasons
    med = q.groupby("role")["n_pitches"].median()
    scale = round(float(med["RP"] / med["SP"]), 3)
    print(f"[app] median season pitches: SP {med['SP']:.0f}, RP {med['RP']:.0f} ({len(q):,} pitcher-seasons); "
          f"reliever bar = {scale} x starter bar")
    return scale


def type_grades(pitches: pd.DataFrame, games: dict, roles: pd.DataFrame, rp_scale: float) -> pd.DataFrame:
    """Per pitcher-season x pitch type: whiff%, chase%, GB%, Damage%, RV/100, and the Savant qualifier.
    Counts are season totals per pitch type (every count, both batter hands)."""
    desc = pitches["description"]
    swing = desc.isin(SWING_DESCRIPTIONS)
    out_zone = pitches["zone"].between(*OUT_OF_ZONE)
    bip = desc == "hit_into_play"
    f = pd.DataFrame({
        "n": 1, "swings": swing, "whiffs": desc.isin(WHIFF_DESCRIPTIONS), "out_zone": out_zone,
        "chases": out_zone & swing, "bip": bip, "gb": bip & (pitches["bb_type"] == "ground_ball"),
        "bip_coded": pitches["label7"].isin(CONTACT_CLASSES), "damage": pitches["label7"] == "damage",
    }).fillna(False).astype(int)
    f["rv"] = -pitches["delta_run_exp"].astype("float64")
    f[KEYS + ["pitch_type"]] = pitches[KEYS + ["pitch_type"]]
    g = f.groupby(KEYS + ["pitch_type"], observed=True).sum().reset_index()
    g["type_whiff_pct"] = g["whiffs"] / g["swings"].replace(0, np.nan)
    g["type_chase_pct"] = g["chases"] / g["out_zone"].replace(0, np.nan)
    g["type_gb_pct"] = g["gb"] / g["bip"].replace(0, np.nan)
    g["type_damage_pct"] = g["damage"] / g["bip_coded"].replace(0, np.nan)
    g["type_rv100"] = 100 * g["rv"] / g["n"]
    g = g.merge(roles[KEYS + ["role"]], on=KEYS, how="left")
    starter_bar = g["game_year"].map(lambda y: SAVANT_PITCHES_PER_TEAM_GAME * games[int(y)])
    g["savant_min"] = starter_bar * np.where(g["role"] == "RP", rp_scale, 1.0)
    g["qualified"] = g["n"] >= g["savant_min"]
    return g[KEYS + ["pitch_type", "type_whiff_pct", "type_chase_pct", "type_gb_pct", "type_damage_pct",
                     "type_rv100", "savant_min", "qualified"]]


def shares(pitches: pd.DataFrame, by: list, col: str, name: str) -> pd.DataFrame:
    """Share of `col` values within each `by` group, long format."""
    c = pitches.groupby(by + [col], observed=True).size().rename("n").reset_index()
    c[name] = c["n"] / c.groupby(by, observed=True)["n"].transform("sum")
    return c


def hzb_tables(paths: Paths, index: pd.DataFrame) -> Dict[str, np.ndarray]:
    """HZB distances among pitcher-seasons with >= 300 pitches, keyed "<season>_<hand>_{pitcher,distance}"."""
    # why: HZB compares pitch-level distributions and the app never loads pitches, so it's precomputed.
    # The metric is fitted on every pitcher of that season and hand, as in run_eval; only the rows the
    # app can show are kept. Unmatched pairs stay at +inf (they rank last).
    out = {}
    for season in sorted(index["game_year"].unique().tolist()):
        pitches = load_pitches(paths.data, int(season))
        for hand in ["R", "L"]:
            pitchers, D, _, _ = hzb_distance(pitches, int(season), hand)
            n = pitchers.merge(index[HZB_KEYS + ["n_pitches"]], on=HZB_KEYS, how="left")["n_pitches"]
            ok = (n >= MIN_PITCHES).to_numpy()
            out[f"{season}_{hand}_pitcher"] = pitchers["pitcher"].to_numpy("int64")[ok]
            out[f"{season}_{hand}_distance"] = D[np.ix_(ok, ok)].astype("float32")
            print(f"[app] HZB {season} {hand}HP: {int(ok.sum())} x {int(ok.sum())}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the app's precomputed tables.")
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    paths = Paths.make(args.sample)
    out = paths.models / "app"
    out.mkdir(parents=True, exist_ok=True)

    cols = KEYS + ["game_pk", "pitch_type", "description", "zone", "bb_type", "label7", "delta_run_exp",
                   "balls", "strikes", "stand", "attack_zone"]
    pitches = pd.read_parquet(paths.data / "pitches.parquet", columns=cols)
    by_type = pd.read_parquet(paths.data / "arsenal_pitch_type.parquet")
    seasons = pd.read_parquet(paths.data / "pitcher_seasons.parquet")
    games = team_games(pitches)
    rp_scale = reliever_scale(seasons)
    print(f"[app] {len(pitches):,} pitches; team games per season {games}")

    # index: pitcher-seasons + embeddings
    t = by_type.sort_values("usage", ascending=False).groupby(KEYS).head(3)
    label = (t["pitch_type"] + " " + (100 * t["usage"]).round().astype(int).astype(str))
    top3 = label.groupby([t[k] for k in KEYS]).agg(" · ".join).rename("top_pitches").reset_index()
    index = seasons[KEYS + ["player_name", "role", "n_pitches", "n_games", "n_starts", "fb_type", "fb_velo",
                            "arm_angle"] + STAT_COLS].merge(top3, on=KEYS, how="left")
    for tier in ["t1", "t3a"]:
        path = paths.models / f"embeddings_{tier}.parquet"
        if path.exists():
            e = pd.read_parquet(path)
            ecols = [c for c in e.columns if c.startswith("e") and c[1:].isdigit()]
            index = index.merge(e[KEYS + ecols].rename(columns={c: f"{tier}_{c}" for c in ecols}), on=KEYS, how="left")
    index.to_parquet(out / "index.parquet", index=False)

    # why: every type he threw. The learned model reads the >= 3% arsenal (in_arsenal), TJStats keeps
    # types thrown >= 10 times itself (D26), and the gameplan shows every pitch with its count (D29).
    ars = by_type[KEYS + ["pitch_type", "family", "n", "usage", "usage_arsenal", "in_arsenal"]
                        + SHAPE_FEATURES + LOCATION_FEATURES]
    ars = ars.merge(type_grades(pitches, games, seasons, rp_scale), on=KEYS + ["pitch_type"], how="left")
    print(f"[app] arsenal: {len(ars):,} pitcher-season x pitch-type rows; qualified {int(ars['qualified'].sum()):,}")
    ars.to_parquet(out / "arsenal.parquet", index=False)

    pitches["count_state"] = count_state(pitches["balls"], pitches["strikes"])
    shares(pitches, KEYS + ["count_state", "stand"], "pitch_type", "usage").to_parquet(out / "usage_by_count.parquet", index=False)
    shares(pitches.dropna(subset=["attack_zone"]), KEYS + ["pitch_type", "stand"], "attack_zone", "share") \
        .to_parquet(out / "locations.parquet", index=False)
    np.savez_compressed(out / "hzb_distance.npz", **hzb_tables(paths, index))

    published = json.loads((paths.models / "published.json").read_text()) if (paths.models / "published.json").exists() else {}
    (out / "meta.json").write_text(json.dumps({
        "team_games": games, "min_pitches": MIN_PITCHES, "savant_pitches_per_team_game": SAVANT_PITCHES_PER_TEAM_GAME,
        "reliever_scale": rp_scale,
        "count_states": COUNT_STATES, "attack_zones": ATTACK_ZONES, "published": published}, indent=2) + "\n")
    write_gap_scale(out, paths.results)
    for f in sorted(out.iterdir()):
        print(f"[save] {f.relative_to(paths.models.parent)}  {f.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
