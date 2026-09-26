"""
Build the cleaned pitch table and the pitcher-season arsenal tables from raw Statcast.

Pipeline stage 2 of 7 (features). Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

Inputs:
    data/raw/statcast_YYYY_MM.parquet   monthly Statcast pulls, 2023-2026 (src/data/fetch_statcast.py)
    statsapi.mlb.com                    each pitcher's primary position (cached after the first run)

Outputs (data/processed/, or data/processed/sample/ with --sample):
    pitches.parquet             one row per pitch: split, context, physical features, pitch family,
                                attack zone, outcome label, and whether the pitch type is in the arsenal
    arsenal_pitch_type.parquet  one row per pitcher-season x pitch type: usage, arsenal flag (>= 3%),
                                mean shape/release features, and attack-zone location shares
    pitcher_seasons.parquet     one row per pitcher-season: pitch count, run value per 100 pitches,
                                usage by pitch type, and the 7-class outcome profile
    player_positions.parquet    cache of statsapi primary positions (always in data/processed/)
    docs/results/data_summary.csv   filter counts, split sizes/ratios, class shares (full build only;
                                    the sample writes data/processed/sample/data_summary.csv)

Stages, with row counts printed at each: load -> filter -> features -> label -> aggregate.
Splits are by season (docs/design_decisions.md, D5): 2023-24 train, 2025 validation, 2026 test.
The outcome classes and pitch-identity rules are the author's calls (D13, D15). Every outcome column
built here (run value, the per-stat rates, the outcome profile) is a target, never a model input (D17).

Usage (from the repo root):
    python -m src.features.build_arsenal --sample   # June of each season -> data/processed/sample/
    python -m src.features.build_arsenal            # everything on disk

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
POSITIONS_CACHE = PROCESSED / "player_positions.parquet"

SPLIT_BY_YEAR = {2023: "train", 2024: "train", 2025: "val", 2026: "test"}
SAMPLE_MONTH = "06"

# ---------------------------------------------------------------------------
# Outcome classes (author's calls, design_decisions.md D13)
# ---------------------------------------------------------------------------
CLASSES = ["called_strike", "ball", "whiff", "foul", "poor", "flare", "damage"]
CONTACT_CLASSES = ["poor", "flare", "damage"]

DESCRIPTION_TO_CLASS: Dict[str, str] = {
    "called_strike": "called_strike",
    "ball": "ball",
    "blocked_ball": "ball",
    "hit_by_pitch": "ball",
    "swinging_strike": "whiff",
    "swinging_strike_blocked": "whiff",
    "foul_tip": "whiff",
    "foul": "foul",
}

# Savant launch_speed_angle code -> contact class.
# Codes: 1 Weak, 2 Topped, 3 Under, 4 Flare/Burner, 5 Solid Contact, 6 Barrel.
CONTACT_BUCKETS: Dict[int, str] = {1: "poor", 2: "poor", 3: "poor", 4: "flare", 5: "damage", 6: "damage"}

# Bunt attempts are removed entirely (author's call, D13): no label, and they don't count toward
# usage, shape, location, or run value.
BUNT_DESCRIPTIONS = {"foul_bunt", "missed_bunt", "bunt_foul_tip"}

# Per-stat breakdown (D22): event and description groups behind K%, BB%, whiff%, chase%, GB%, Damage%.
K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk"}                       # intentional walks have no tracked pitch, so none survive the filter
NOT_A_PA = {"truncated_pa"}                # PA cut short (e.g., inning-ending caught stealing)
SWING_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "foul", "hit_into_play"}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}  # foul tip = whiff (D13)
OUT_OF_ZONE = (11, 14)                     # Statcast zone codes outside the strike zone

# Removed entirely: pitch-clock violations (no pitch was thrown, nothing tracked).
NOT_A_PITCH = {"automatic_ball", "automatic_strike"}

# ---------------------------------------------------------------------------
# Pitch identity (author's calls, D15)
# ---------------------------------------------------------------------------
# Removed entirely: knuckleball, eephus, generic "fastball", pitchout, and Statcast's "unknown".
DROP_PITCH_TYPES = {"KN", "EP", "FA", "PO", "UN"}

# Savant's pitch groups. The learned model sees the family and the pitch's shape, never the
# Statcast name; the TJStats and SEAM baselines match on the Statcast name, as published.
PITCH_FAMILY: Dict[str, str] = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball",
    "SL": "breaking", "ST": "breaking", "SV": "breaking", "CU": "breaking", "KC": "breaking", "CS": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "SC": "offspeed",
}
FAMILIES = ["fastball", "breaking", "offspeed"]

MIN_ARSENAL_USAGE = 0.03  # a pitch type joins the arsenal at >= 3% of the pitcher-season's pitches

# ---------------------------------------------------------------------------
# Location: Savant attack zones (D17)
# ---------------------------------------------------------------------------
# why: location is described by *where he throws*, never by results. Zone units follow Savant's
# attack-zone definition: 100% = the edge of the zone. Horizontally that edge is half the 17-in
# plate plus a ball radius (~10 in); vertically it is the batter's own sz_top / sz_bot.
ZONE_HALF_WIDTH_FT = 10.0 / 12.0
ATTACK_ZONES = ["heart", "shadow", "chase", "waste"]
ATTACK_ZONE_EDGES = (0.67, 1.33, 2.00)  # heart < 67% <= shadow < 133% <= chase < 200% <= waste

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
LOAD_COLS = [
    "game_pk", "game_date", "game_year", "game_type", "at_bat_number", "pitch_number",
    "pitcher", "player_name", "batter", "p_throws", "stand", "age_pit",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b", "inning",
    "pitch_type", "description", "events", "des", "type", "launch_speed_angle", "bb_type", "zone", "inning_topbot",
    "launch_speed", "launch_angle", "delta_run_exp",
    "release_speed", "release_spin_rate", "spin_axis", "release_extension",
    "release_pos_x", "release_pos_y", "release_pos_z", "arm_angle",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
]

# A pitch without these can't be described physically, so it is dropped.
REQUIRED_TRACKING = [
    "pitch_type", "release_speed", "release_pos_x", "release_pos_y", "release_pos_z",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
]

# why: every shape metric a method reads is built here, so all methods see the same numbers (D4).
# The ten below are TJStats' full metric list (the tool's defaults use six of them, D26; raw Euclidean
# uses all ten); SEAM adds its two release angles (D19). The learned T1 model reads all 12.
TJSTATS_FEATURES = [
    "release_speed", "ivb_in", "hb_arm_in", "release_spin_rate", "release_extension",
    "release_height", "release_side_arm", "vaa", "haa_arm", "arm_angle",
]
SEAM_EXTRA_FEATURES = ["rel_angle_h_arm", "rel_angle_v"]
SHAPE_FEATURES = TJSTATS_FEATURES + SEAM_EXTRA_FEATURES
LOCATION_FEATURES = [f"loc_{z}" for z in ATTACK_ZONES]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def raw_files(sample: bool) -> List[Path]:
    """Monthly raw files to load: all of them, or June of each season for --sample."""
    files = sorted(RAW.glob("statcast_*.parquet"))
    if sample:
        files = [f for f in files if f.stem.endswith(f"_{SAMPLE_MONTH}")]
    if not files:
        raise SystemExit(f"No raw files found in {RAW}. Run src.data.fetch_statcast first.")
    return files


def load_raw(files: List[Path]) -> pd.DataFrame:
    """Concatenate the needed columns from each monthly file."""
    frames = [pd.read_parquet(f, columns=LOAD_COLS) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] {len(files)} files -> {len(df):,} pitches, {df.shape[1]} columns")
    print(f"[load] pitches by season: {df['game_year'].value_counts().sort_index().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
FILTER_LOG: List[Dict[str, object]] = []  # rows removed per stage, for docs/results/data_summary.csv


def _report(stage: str, before: int, after: int) -> None:
    print(f"[filter] {stage:<44} {before:>10,} -> {after:>10,}  (-{before - after:,})")
    FILTER_LOG.append({"section": "filter_removed", "item": stage, "value": before - after})


def fetch_positions(pitcher_ids: List[int]) -> pd.DataFrame:
    """Primary position for each pitcher id from statsapi.mlb.com, cached to disk.

    Only ids missing from the cache are requested, 150 per call.
    """
    cache = pd.read_parquet(POSITIONS_CACHE) if POSITIONS_CACHE.exists() else pd.DataFrame(
        columns=["pitcher", "full_name", "primary_position"])
    missing = sorted(set(pitcher_ids) - set(cache["pitcher"].astype(int)))
    rows = []
    for i in range(0, len(missing), 150):
        ids = ",".join(str(x) for x in missing[i:i + 150])
        resp = requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={ids}", timeout=30)
        resp.raise_for_status()
        for p in resp.json().get("people", []):
            rows.append({"pitcher": int(p["id"]), "full_name": p.get("fullName"),
                         "primary_position": p.get("primaryPosition", {}).get("abbreviation")})
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(POSITIONS_CACHE, index=False)
    print(f"[filter] positions: {len(cache):,} cached, {len(rows):,} newly fetched from statsapi")
    return cache


def is_bunt(df: pd.DataFrame) -> pd.Series:
    """Bunt attempts: bunt descriptions, plus balls in play whose play text says bunt."""
    in_play_bunt = (df["description"] == "hit_into_play") & df["des"].str.contains("bunt", case=False, na=False)
    return df["description"].isin(BUNT_DESCRIPTIONS) | in_play_bunt


def filter_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are not usable, competitive, tracked pitches from real pitchers."""
    n = len(df)
    df = df[df["game_type"] == "R"]
    _report("regular season only", n, len(df)); n = len(df)

    df = df[df["balls"].between(0, 3) & df["strikes"].between(0, 2)]
    _report("valid count (0-3 balls, 0-2 strikes)", n, len(df)); n = len(df)

    df = df[~df["description"].isin(NOT_A_PITCH)]
    _report("pitch-clock violations (no pitch thrown)", n, len(df)); n = len(df)

    df = df[~is_bunt(df)].drop(columns=["des"])  # des is free text, only needed to find bunts
    _report("bunt attempts (author's call, D13)", n, len(df)); n = len(df)

    df = df.dropna(subset=REQUIRED_TRACKING)
    _report("missing pitch type or core tracking", n, len(df)); n = len(df)

    df = df[~df["pitch_type"].isin(DROP_PITCH_TYPES)]
    _report("KN / EP / FA / PO / UN pitch types", n, len(df)); n = len(df)

    unknown = set(df["pitch_type"].unique()) - set(PITCH_FAMILY)
    if unknown:
        raise SystemExit(f"Pitch types with no family: {sorted(unknown)}. Add them to PITCH_FAMILY or DROP_PITCH_TYPES.")

    # why (D8): position players pitching in blowouts throw 40-60 mph lobs that are not the
    # population a comp tool serves. statsapi's primary position is a rule, not a judgment call.
    pos = fetch_positions(df["pitcher"].astype(int).unique().tolist())
    keep = set(pos.loc[pos["primary_position"].isin(["P", "TWP"]), "pitcher"].astype(int))
    df = df[df["pitcher"].astype(int).isin(keep)]
    _report("position players pitching (statsapi)", n, len(df))

    for col in ["release_spin_rate", "release_extension", "arm_angle", "sz_top"]:
        print(f"[filter] still-null {col}: {df[col].isna().mean():.2%} (kept; aggregation skips NaN)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def approach_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Vertical and horizontal approach angle (degrees) at the front of home plate.

    Statcast gives the pitch's velocity (vx0, vy0, vz0) and acceleration (ax, ay, az) at
    y = 50 ft. Constant acceleration lets us solve for the velocity at the plate
    (y = 17/12 ft) and take the angle of that velocity vector.
    """
    y_plate = 17.0 / 12.0
    vy_f = -np.sqrt(df["vy0"] ** 2 - 2.0 * df["ay"] * (50.0 - y_plate))
    t = (vy_f - df["vy0"]) / df["ay"]
    vz_f = df["vz0"] + df["az"] * t
    vx_f = df["vx0"] + df["ax"] * t
    df["vaa"] = -np.degrees(np.arctan(vz_f / vy_f))
    df["haa"] = -np.degrees(np.arctan(vx_f / vy_f))
    return df


def release_angles(df: pd.DataFrame) -> pd.DataFrame:
    """SEAM's horizontal and vertical release angles (degrees), from the velocity at release.

    The same constant-acceleration model, run backwards from y = 50 ft to the release point
    (y = release_pos_y): launch_h = arctan(vx / vy), launch_v = arctan(vz / sqrt(vx^2 + vy^2)).
    """
    vy_r = -np.sqrt(df["vy0"] ** 2 + 2.0 * df["ay"] * (df["release_pos_y"] - 50.0))
    t = (vy_r - df["vy0"]) / df["ay"]  # negative: release happens before y = 50 ft
    vx_r = df["vx0"] + df["ax"] * t
    vz_r = df["vz0"] + df["az"] * t
    df["rel_angle_h"] = np.degrees(np.arctan(vx_r / vy_r))
    df["rel_angle_v"] = np.degrees(np.arctan(vz_r / np.sqrt(vx_r ** 2 + vy_r ** 2)))
    return df


def attack_zone(df: pd.DataFrame) -> pd.Series:
    """Savant attack zone (heart / shadow / chase / waste) for each pitch; NaN without a zone."""
    x = df["plate_x"] / ZONE_HALF_WIDTH_FT
    mid = (df["sz_top"] + df["sz_bot"]) / 2.0
    half = (df["sz_top"] - df["sz_bot"]) / 2.0
    z = (df["plate_z"] - mid) / half
    r = np.maximum(np.abs(x), np.abs(z))  # rectangular rings: the farther of the two directions
    heart, shadow, chase = ATTACK_ZONE_EDGES
    zone = np.select([r < heart, r < shadow, r < chase, r >= chase], ATTACK_ZONES, default="")
    return pd.Series(pd.Categorical(np.where(r.isna(), None, zone), categories=ATTACK_ZONES), index=df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive physical features (inches, degrees, arm-side-positive), family, zone, and split."""
    df = approach_angles(df)
    df = release_angles(df)
    df["ivb_in"] = df["pfx_z"] * 12.0
    df["release_height"] = df["release_pos_z"]
    # why (D16): Statcast x-coordinates are from the catcher's view, so a lefty's arm side has the
    # opposite sign from a righty's. Flipping righties makes "arm side" positive for everyone,
    # so one population can be standardized together. Comps are still same-hand only.
    arm_sign = np.where(df["p_throws"] == "L", 1.0, -1.0)
    df["hb_arm_in"] = df["pfx_x"] * 12.0 * arm_sign
    df["release_side_arm"] = df["release_pos_x"] * arm_sign
    df["haa_arm"] = df["haa"] * arm_sign
    df["rel_angle_h_arm"] = df["rel_angle_h"] * arm_sign
    df["family"] = pd.Categorical(df["pitch_type"].map(PITCH_FAMILY), categories=FAMILIES)
    df["attack_zone"] = attack_zone(df)
    df["split"] = df["game_year"].map(SPLIT_BY_YEAR)
    for col in ["on_1b", "on_2b", "on_3b"]:
        df[col] = df[col].notna()
    print(f"[features] added approach + release angles, arm-side features, family, attack_zone, split -> {df.shape}")
    print(f"[features] pitches by split: {df['split'].value_counts().to_dict()}")
    return df


def sanity_check_features(df: pd.DataFrame) -> None:
    """Print checks a pitching coach would recognize."""
    common = df["pitch_type"].value_counts().index[:7]
    print("[check] means by pitch type (VAA: fastballs flattest; release angle v: breaking balls launched upward):")
    print(df[df["pitch_type"].isin(common)].groupby("pitch_type")[
        ["release_speed", "ivb_in", "vaa", "rel_angle_v"]].mean().round(1).to_string())
    ff = df[df["pitch_type"] == "FF"]
    print("[check] FF by hand (arm-side columns should have the same sign for L and R):")
    print(ff.groupby("p_throws")[["hb_arm_in", "release_side_arm", "haa_arm", "rel_angle_h_arm"]]
          .mean().round(2).to_string())
    print(f"[check] family shares: {df['family'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"[check] attack-zone shares, all pitches: {df['attack_zone'].value_counts(normalize=True).round(3).to_dict()}")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def label_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Assign the 7-class outcome label, plus the 5-class version used by lower data tiers.

    Unlabeled pitches keep label = NaN, and `label_status` says why.
    """
    label = df["description"].map(DESCRIPTION_TO_CLASS)
    status = pd.Series(np.where(label.notna(), "ok", "unmapped description"), index=df.index)

    in_play = df["description"] == "hit_into_play"
    code = df["launch_speed_angle"].where(in_play)
    bucket = code.map(CONTACT_BUCKETS)
    label[in_play] = bucket[in_play]
    status[in_play & bucket.notna()] = "ok"
    status[in_play & code.isna()] = "in play, no launch_speed_angle"

    df["label7"] = pd.Categorical(label, categories=CLASSES)
    # why (D6, D13): the T3a tier has no batted-ball quality, so poor/flare/damage collapse to in_play.
    label5 = label.where(~label.isin(CONTACT_CLASSES), "in_play")
    df["label5"] = pd.Categorical(label5, categories=CLASSES[:4] + ["in_play"])
    df["label_status"] = status

    print(f"[label] status: {status.value_counts().to_dict()}")
    unmapped = df.loc[status == "unmapped description", "description"].value_counts().to_dict()
    if unmapped:
        print(f"[label] WARNING unmapped descriptions: {unmapped}")
    print(f"[label] 7-class shares of labeled pitches: "
          f"{df['label7'].value_counts(normalize=True, sort=False).round(3).to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
KEYS = ["pitcher", "game_year", "split", "p_throws"]


def aggregate_pitch_types(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher-season x pitch type: usage, arsenal flag, shape means, location shares."""
    g = df.groupby(KEYS + ["pitch_type"], observed=True)
    out = g[SHAPE_FEATURES].mean()
    out["n"] = g.size()
    out = out.reset_index()
    out["family"] = out["pitch_type"].map(PITCH_FAMILY)

    # Location: share of this pitch type's pitches in each attack zone (pitches without a zone skipped).
    loc = pd.crosstab([df[k] for k in KEYS + ["pitch_type"]], df["attack_zone"], normalize="index")
    loc = loc.reindex(columns=ATTACK_ZONES, fill_value=0.0)
    loc.columns = LOCATION_FEATURES
    out = out.merge(loc.reset_index(), on=KEYS + ["pitch_type"], how="left")

    totals = out.groupby(KEYS)["n"].transform("sum")
    out["usage"] = out["n"] / totals
    # why (D15): a pitch he throws 1% of the time is noise in his fingerprint. >= 3% is the author's
    # threshold. Arsenal usage is renormalized to sum to 1 across the pitch types that qualify; the
    # sub-3% pitches stay in the pitch table as real pitches with real outcomes.
    out["in_arsenal"] = out["usage"] >= MIN_ARSENAL_USAGE
    arsenal_total = out["usage"].where(out["in_arsenal"]).groupby([out[k] for k in KEYS]).transform("sum")
    out["usage_arsenal"] = (out["usage"] / arsenal_total).where(out["in_arsenal"])

    print(f"[aggregate] pitcher-season x pitch type: {len(out):,} rows; in arsenal (>= 3%): "
          f"{int(out['in_arsenal'].sum()):,}")
    return out


def pitcher_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Games, starts, and role per pitcher-season (D23). Role is only for same-role comparisons,
    never a similarity input.

    A start = he threw his team's first pitch of the game (the earliest pitch of each half of the
    1st inning). Starter ("SP") = started at least half his appearances; otherwise "RP".
    """
    first = (df[df["inning"] == 1]
             .sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
             .drop_duplicates(["game_pk", "inning_topbot"]))
    starts = first.groupby(KEYS, observed=True).size().rename("n_starts")
    games = df.groupby(KEYS, observed=True)["game_pk"].nunique().rename("n_games")
    out = pd.concat([games, starts], axis=1).fillna({"n_starts": 0}).reset_index()
    out["n_starts"] = out["n_starts"].astype(int)
    out["role"] = np.where(out["n_starts"] >= 0.5 * out["n_games"], "SP", "RP")
    return out


def outcome_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and rates behind the per-stat breakdown (D22), per pitcher-season. Targets only, never inputs."""
    ev, desc = df["events"], df["description"]
    pa = ev.notna() & ~ev.isin(NOT_A_PA)
    swing = desc.isin(SWING_DESCRIPTIONS)
    out_zone = df["zone"].between(*OUT_OF_ZONE)
    bip = desc == "hit_into_play"
    flags = pd.DataFrame({
        "n_pa": pa, "n_k": pa & ev.isin(K_EVENTS), "n_bb": pa & ev.isin(BB_EVENTS),
        "n_swings": swing, "n_whiffs": desc.isin(WHIFF_DESCRIPTIONS),
        "n_out_zone": out_zone, "n_chases": out_zone & swing,
        "n_bip": bip, "n_gb": bip & (df["bb_type"] == "ground_ball"),
        "n_bip_coded": df["label7"].isin(CONTACT_CLASSES), "n_damage": df["label7"] == "damage",
    }).fillna(False).astype(int)
    flags[KEYS] = df[KEYS]
    c = flags.groupby(KEYS, observed=True).sum()
    rates = {"k_pct": ("n_k", "n_pa"), "bb_pct": ("n_bb", "n_pa"), "whiff_pct": ("n_whiffs", "n_swings"),
             "chase_pct": ("n_chases", "n_out_zone"), "gb_pct": ("n_gb", "n_bip"), "damage_pct": ("n_damage", "n_bip_coded")}
    for name, (num, den) in rates.items():
        c[name] = c[num] / c[den].replace(0, np.nan)
    return c.reset_index()


def primary_fastball(by_type: pd.DataFrame) -> pd.DataFrame:
    """Most-thrown fastball-family pitch type and its velocity, per pitcher-season (for reports and slices)."""
    fb = by_type[by_type["family"] == "fastball"].sort_values("n", ascending=False).drop_duplicates(KEYS)
    return fb[KEYS + ["pitch_type", "release_speed"]].rename(columns={"pitch_type": "fb_type", "release_speed": "fb_velo"})


def aggregate_pitcher_seasons(df: pd.DataFrame, by_type: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher-season: pitch count, role, run value, per-stat rates, usage by type, outcome profile."""
    g = df.groupby(KEYS, observed=True)
    out = g.agg(
        n_pitches=("pitch_type", "size"),
        n_labeled=("label7", "count"),
        player_name=("player_name", "first"),
        age=("age_pit", "median"),
        # why (D3): delta_run_exp is from the batting team's side. Flipping the sign and scaling to 100
        # pitches matches Savant's pitcher run value convention: positive = runs prevented.
        pitcher_rv100=("delta_run_exp", lambda s: -100.0 * s.mean()),
        arm_angle=("arm_angle", "mean"),
    ).reset_index()
    out = out.merge(pitcher_roles(df), on=KEYS, how="left")
    out = out.merge(outcome_stats(df), on=KEYS, how="left")
    out = out.merge(primary_fastball(by_type), on=KEYS, how="left")

    ars = by_type[by_type["in_arsenal"]].groupby(KEYS).agg(n_arsenal_types=("pitch_type", "size")).reset_index()
    out = out.merge(ars, on=KEYS, how="left")

    usage = by_type.pivot_table(index=KEYS, columns="pitch_type", values="usage", fill_value=0.0)
    usage.columns = [f"usage_{c}" for c in usage.columns]
    out = out.merge(usage.reset_index(), on=KEYS, how="left")

    # Outcome profile = share of each outcome class among labeled pitches. This is a *target*
    # (the auxiliary head's, D10), never an input to the fingerprint.
    prof = pd.crosstab([df[k] for k in KEYS], df["label7"], normalize="index")
    prof = prof.reindex(columns=CLASSES, fill_value=0.0)
    prof.columns = [f"p_{c}" for c in CLASSES]
    out = out.merge(prof.reset_index(), on=KEYS, how="left")

    print(f"[aggregate] pitcher-seasons: {len(out):,} rows; by split: {out['split'].value_counts().to_dict()}")
    q = out["n_pitches"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0).astype(int).to_dict()
    print(f"[aggregate] pitches per pitcher-season, quantiles: {q}")
    print(f"[aggregate] arsenal size (types >= 3%): {out['n_arsenal_types'].value_counts().sort_index().to_dict()}")
    q300 = out[out["n_pitches"] >= 300]
    print(f"[aggregate] role (SP = started >= half his games), >= 300 pitches: "
          f"{q300.groupby(['split', 'role']).size().to_dict()}")
    print(f"[aggregate] per-stat medians, >= 300 pitches: "
          f"{q300[['k_pct', 'bb_pct', 'whiff_pct', 'chase_pct', 'gb_pct', 'damage_pct']].median().round(3).to_dict()}")
    return out


def flag_arsenal_pitches(df: pd.DataFrame, by_type: pd.DataFrame) -> pd.DataFrame:
    """Mark each pitch with whether its pitch type made its pitcher-season's arsenal."""
    flags = by_type[KEYS + ["pitch_type", "in_arsenal"]]
    df = df.merge(flags, on=KEYS + ["pitch_type"], how="left")
    print(f"[aggregate] pitches whose type is in the arsenal: {df['in_arsenal'].mean():.2%}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_parquet(path, index=False)
    print(f"[save] {path.relative_to(ROOT)}  {df.shape}")


def write_data_summary(n_loaded: int, df: pd.DataFrame, seasons: pd.DataFrame, path: Path) -> None:
    """Write the numbers the docs quote (filter counts, split sizes and ratios, class shares) to CSV."""
    rows: List[Dict[str, object]] = [{"section": "load", "item": "raw pitches", "value": n_loaded}]
    rows += FILTER_LOG
    rows.append({"section": "load", "item": "pitches kept", "value": len(df)})
    for split in ["train", "val", "test"]:
        n = int((df["split"] == split).sum())
        rows.append({"section": "split_pitches", "item": split, "value": n})
        rows.append({"section": "split_pitch_share", "item": split, "value": round(n / len(df), 4)})
        rows.append({"section": "split_pitcher_seasons", "item": split, "value": int((seasons["split"] == split).sum())})
    q300 = seasons[seasons["n_pitches"] >= 300]
    for (split, role), n in q300.groupby(["split", "role"]).size().items():
        rows.append({"section": "role_300_pitches", "item": f"{split} {role}", "value": int(n)})
    for cls, share in df["label7"].value_counts(normalize=True, sort=False).items():
        rows.append({"section": "label7_share", "item": cls, "value": round(float(share), 4)})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[save] {path.relative_to(ROOT)}  ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--sample", action="store_true", help="June of each season only -> data/processed/sample/")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = PROCESSED / "sample" if args.sample else PROCESSED

    df = load_raw(raw_files(args.sample))
    n_loaded = len(df)
    df = filter_pitches(df)
    df = add_features(df)
    sanity_check_features(df)
    df = label_outcomes(df)

    by_type = aggregate_pitch_types(df)
    seasons = aggregate_pitcher_seasons(df, by_type)
    df = flag_arsenal_pitches(df, by_type)

    save(df, out_dir, "pitches.parquet")
    save(by_type, out_dir, "arsenal_pitch_type.parquet")
    save(seasons, out_dir, "pitcher_seasons.parquet")
    summary = out_dir / "data_summary.csv" if args.sample else ROOT / "docs" / "results" / "data_summary.csv"
    write_data_summary(n_loaded, df, seasons, summary)
    print(f"[done] {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
