"""
Model inputs: which columns each data tier sees, and how the processed tables become arrays.

Shared by training (train_embedding.py), evaluation, and the app, so a query is always encoded
exactly the way the training data was.

Pipeline: a helper for stage 3 (train), stage 4 (evaluate), and stage 7 (app). Stages: fetch ->
features -> train -> evaluate -> comps report -> app tables -> app.

Inputs:  the processed tables from src/features/build_arsenal.py
    pitches.parquet             per pitch: shape, location, count, outcome label
    arsenal_pitch_type.parquet  per pitcher-season x pitch type: mean shape, location shares, usage
    pitcher_seasons.parquet     per pitcher-season: pitch counts, RV/100, outcome profile
Outputs: float32 arrays (pitch features, padded arsenal sets, auxiliary targets), plus the
    standardization statistics. Those are fit on the training seasons only and saved with the model.

Tiers (docs/design_decisions.md, D6; only T1 and T3a are evaluated):
    t1   full Statcast: 12 shape features, location, count, platoon, family; 7 outcome classes
    t3a  radar gun + tagger: family, velocity, coarse movement, count, platoon; no location and
         no spin, release, or angles; 5 outcome classes (contact quality collapsed to in_play)

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

KEYS = ["pitcher", "game_year", "split", "p_throws"]
FAMILIES = ["fastball", "breaking", "offspeed"]
ATTACK_ZONES = ["heart", "shadow", "chase", "waste"]
LOCATION_SHARES = [f"loc_{z}" for z in ATTACK_ZONES]
CLASSES7 = ["called_strike", "ball", "whiff", "foul", "poor", "flare", "damage"]
CLASSES5 = CLASSES7[:4] + ["in_play"]
N_COUNTS = 12  # balls 0-3 x strikes 0-2, encoded as 3 * balls + strikes

SHAPE_T1 = [
    "release_speed", "ivb_in", "hb_arm_in", "release_spin_rate", "release_extension",
    "release_height", "release_side_arm", "vaa", "haa_arm", "arm_angle", "rel_angle_h_arm", "rel_angle_v",
]
MOVEMENT = ["ivb_in", "hb_arm_in"]

# Author's call (D6): at T3a, movement is known only to the nearest 3 in.
COARSE_MOVEMENT_IN = 3.0

# Author's call (D24): a pitcher-season needs this many pitches for its RV/100 and outcome profile
# to be used as a target, a retrieval query, or a retrieval comp.
MIN_PITCHES = 300


@dataclass(frozen=True)
class Tier:
    """What one data tier can measure."""
    name: str
    shape: Tuple[str, ...]   # physical features, per pitch and as per-type means
    location: bool           # pitch location per pitch, attack-zone shares per type
    coarse_movement: bool    # round movement to COARSE_MOVEMENT_IN
    label: str               # "label7" or "label5"

    @property
    def classes(self) -> List[str]:
        return CLASSES7 if self.label == "label7" else CLASSES5


TIERS: Dict[str, Tier] = {
    "t1": Tier("t1", tuple(SHAPE_T1), location=True, coarse_movement=False, label="label7"),
    "t3a": Tier("t3a", ("release_speed", "ivb_in", "hb_arm_in"), location=False, coarse_movement=True, label="label5"),
}


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------
@dataclass
class Scaler:
    """Per-column z-scoring. Fit on training rows only; missing values become 0 (the mean)."""
    # why (D5): the mean and SD come from 2023-24 only, so no 2025 or 2026 statistic leaks into the inputs.
    columns: List[str]
    mean: List[float]
    std: List[float]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Scaler":
        x = df.astype("float64")
        std = x.std().replace(0.0, 1.0).fillna(1.0)
        return cls(list(df.columns), x.mean().tolist(), std.tolist())

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.columns].astype("float64").to_numpy()
        z = (x - np.asarray(self.mean)) / np.asarray(self.std)
        return np.nan_to_num(z, nan=0.0).astype("float32")

    def to_dict(self) -> Dict[str, list]:
        return {"columns": self.columns, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "Scaler":
        return cls(list(d["columns"]), list(d["mean"]), list(d["std"]))


def coarsen(x: pd.DataFrame, step: float = COARSE_MOVEMENT_IN) -> pd.DataFrame:
    """Round movement to the nearest `step` inches, as a lower tier would know it."""
    return (x / step).round() * step


def one_hot(values: pd.Series, categories: List) -> np.ndarray:
    """One column per category; unknown or missing values are all zeros."""
    codes = pd.Categorical(values, categories=categories).codes
    out = np.zeros((len(values), len(categories)), dtype="float32")
    ok = codes >= 0
    out[np.flatnonzero(ok), codes[ok]] = 1.0
    return out


# ---------------------------------------------------------------------------
# Pitch-level inputs (the outcome head)
# ---------------------------------------------------------------------------
def pitch_continuous(pitches: pd.DataFrame, tier: Tier) -> pd.DataFrame:
    """Unscaled continuous per-pitch inputs for one tier."""
    out = pitches[list(tier.shape)].astype("float64").copy()
    if tier.coarse_movement:
        out[MOVEMENT] = coarsen(out[MOVEMENT])
    if tier.location:
        # why: location relative to the batter. plate_x is from the catcher's view, so for a
        # right-handed batter "inside" is negative x; flipping it makes positive = inside for everyone.
        sign = np.where(pitches["stand"].to_numpy() == "R", -1.0, 1.0)
        out["plate_x_in"] = pitches["plate_x"].astype("float64").to_numpy() * sign
        # why: height as a fraction of this batter's own zone (0 = bottom edge, 1 = top edge).
        height = (pitches["sz_top"] - pitches["sz_bot"]).astype("float64")
        out["plate_z_zone"] = (pitches["plate_z"] - pitches["sz_bot"]).astype("float64") / height
    return out


def pitch_matrix(pitches: pd.DataFrame, tier: Tier, scaler: Scaler) -> np.ndarray:
    """[n_pitches, d_pitch]: scaled continuous inputs + count, platoon, family (+ attack zone at T1)."""
    count = 3 * pitches["balls"].astype(int) + pitches["strikes"].astype(int)
    parts = [
        scaler.transform(pitch_continuous(pitches, tier)),
        one_hot(count, list(range(N_COUNTS))),
        (pitches["stand"] == pitches["p_throws"]).to_numpy(dtype="float32")[:, None],  # platoon: same-hand matchup
        one_hot(pitches["family"].astype(str), FAMILIES),
    ]
    if tier.location:
        parts.append(one_hot(pitches["attack_zone"].astype(str), ATTACK_ZONES))
    return np.concatenate(parts, axis=1)


def pitch_labels(pitches: pd.DataFrame, tier: Tier) -> np.ndarray:
    """Class index per pitch, -1 where unlabeled."""
    return pd.Categorical(pitches[tier.label].astype(object), categories=tier.classes).codes.astype("int64")


# ---------------------------------------------------------------------------
# Arsenal sets (the set encoder)
# ---------------------------------------------------------------------------
def arsenal_continuous(by_type: pd.DataFrame, tier: Tier) -> pd.DataFrame:
    """Unscaled per-pitch-type shape means for one tier."""
    out = by_type[list(tier.shape)].astype("float64").copy()
    if tier.coarse_movement:
        out[MOVEMENT] = coarsen(out[MOVEMENT])
    return out


def arsenal_sets(by_type: pd.DataFrame, seasons: pd.DataFrame, tier: Tier,
                 scaler: Scaler) -> Tuple[np.ndarray, np.ndarray]:
    """Padded arsenal sets aligned with the rows of `seasons`.

    Returns X [n_seasons, max_types, d_type] and W [n_seasons, max_types]. W holds each pitch
    type's usage within the arsenal (sums to 1) and is 0 on padding. Only types with >= 3% usage
    are included (D15). The Statcast pitch name is never an input: only family + shape (+ location).
    """
    # why (D18): a Deep Sets encoder needs one tensor per batch, so arsenals of 2-8 types are padded to
    # the largest; the padding slots get weight 0, so the usage-weighted pooling ignores them.
    ars = by_type[by_type["in_arsenal"]].reset_index(drop=True)
    parts = [scaler.transform(arsenal_continuous(ars, tier)), one_hot(ars["family"].astype(str), FAMILIES)]
    if tier.location:
        parts.append(ars[LOCATION_SHARES].astype("float64").fillna(0.0).to_numpy(dtype="float32"))
    feats = np.concatenate(parts, axis=1)

    row_of = {k: i for i, k in enumerate(seasons[KEYS].itertuples(index=False, name=None))}
    rows = np.array([row_of.get(k, -1) for k in ars[KEYS].itertuples(index=False, name=None)])
    keep = rows >= 0
    rows, feats, usage = rows[keep], feats[keep], ars.loc[keep, "usage_arsenal"].to_numpy(dtype="float32")

    # Pack the arsenal rows into [season, slot] without a Python loop: sort by season row, then give each
    # pitch type its position within its own season (slot 0, 1, 2, ...). `starts` marks where each season's
    # block begins, so subtracting the block start from the running index is that row's slot.
    order = np.argsort(rows, kind="stable")
    rows, feats, usage = rows[order], feats[order], usage[order]
    slot = np.zeros(len(rows), dtype=int)
    if len(rows):
        starts = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
        slot = np.arange(len(rows)) - np.repeat(starts, np.diff(np.r_[starts, len(rows)]))
    max_types = int(slot.max()) + 1 if len(rows) else 1

    X = np.zeros((len(seasons), max_types, feats.shape[1]), dtype="float32")
    W = np.zeros((len(seasons), max_types), dtype="float32")
    X[rows, slot] = feats
    W[rows, slot] = usage
    return X, W


# ---------------------------------------------------------------------------
# Auxiliary targets (D10)
# ---------------------------------------------------------------------------
def outcome_profile(seasons: pd.DataFrame, tier: Tier) -> np.ndarray:
    """[n_seasons, n_classes] share of labeled pitches in each class (5-class: contact summed)."""
    p = seasons[[f"p_{c}" for c in CLASSES7]].astype("float64").to_numpy()
    if tier.label == "label5":
        p = np.concatenate([p[:, :4], p[:, 4:].sum(axis=1, keepdims=True)], axis=1)
    return p.astype("float32")


def rv_stats(seasons: pd.DataFrame) -> Tuple[float, float]:
    """Mean and SD of RV/100 over *training* pitcher-seasons with enough pitches."""
    tr = seasons[(seasons["split"] == "train") & (seasons["n_pitches"] >= MIN_PITCHES)]["pitcher_rv100"]
    return float(tr.mean()), float(tr.std())


def aux_targets(seasons: pd.DataFrame, tier: Tier, rv_mean: float,
                rv_std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outcome profile, z-scored RV/100, and a mask of pitcher-seasons with enough pitches."""
    profile = outcome_profile(seasons, tier)
    # why: RV/100 is in runs (SD ~1-2), the profile loss is in fractions of a nat. The z-score puts
    # the two on one scale so the squared error doesn't swamp the outcome-mix loss (D10).
    rv_z = ((seasons["pitcher_rv100"].astype("float64") - rv_mean) / rv_std).to_numpy(dtype="float32")
    ok = (seasons["n_pitches"].to_numpy() >= MIN_PITCHES) & np.isfinite(profile).all(axis=1) & np.isfinite(rv_z)
    return np.nan_to_num(profile), np.nan_to_num(rv_z), ok
