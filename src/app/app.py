"""
Arsenal Match: who pitches like him? Find MLB pitchers whose arsenals get similar results, then see
how they attack hitters.

Pick an MLB pitcher-season, or enter your own arsenal at one of the two tested input levels: full
Statcast (T1) or radar gun + pitch tagging (T3a). The page shows, top to bottom:
    - his profile: pitch mix, fastball shape and release, and his season's results with percentiles;
    - three lists of the 10 closest same-hand, same-role MLB pitchers: our learned model, TJStats,
      and HZB (the two published hand-built methods the app can run), each scored on one similarity scale;
    - "same measurements, different results" flags: TJStats' or HZB's top 3, our model's bottom half
      (the error analysis's rule, run_eval.twins, with HZB added in the app, D31);
    - for the clicked comp: how he compares to the pitcher (every pitch's shape and grades, and season
      results) and how he attacks hitters (pitch mix by count and batter hand, locations), each next to
      the pitcher's own.

Loads only committed artifacts: models/arsenal_net_{t1,t3a}.pt and models/app/* (built by
src/app/build_artifacts.py). Never the raw or processed data.

Pipeline stage 7 of 7 (app). Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

Run (from the repo root):
    streamlit run src/app/app.py
    ARSENAL_MATCH_SAMPLE=1 streamlit run src/app/app.py    # against models/sample/ (development)
Deep link to a pitcher: http://localhost:8501/?pitcher=<MLBAM id>&season=<year>

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import html
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

from src.app.gameplan_gap import (arsenal_rows, cell_gaps, pair_types, paired_pitch, row_shares,  # noqa: E402
                                  usage_mix)
from src.features.build_arsenal import LOCATION_FEATURES, PITCH_FAMILY, SHAPE_FEATURES  # noqa: E402
from src.models import inputs as I  # noqa: E402
from src.models.physical_similarity import TJSTATS, arsenal_similarity  # noqa: E402
from src.models.train_embedding import load_model  # noqa: E402

MODELS = ROOT / "models" / ("sample" if os.environ.get("ARSENAL_MATCH_SAMPLE") else "")
APP = MODELS / "app"
K = 10
TWIN_TOP = 3  # same measurements, different results: a published method's top 3, our model's bottom half
TWIN_METHODS = ["tjstats", "hzb"]  # the published methods whose top 3 can flag a pair (D31)
LIST_HEIGHT = 35 * (K + 1) + 3  # px: the 10 rows plus the header, no inner scroll

# Input levels: only the two tested ones (T1 and T3a; T2 and T3b were cut, D6).
TIERS_UI: Dict[str, Tuple[str, str]] = {
    "Full Statcast (T1)": ("t1", "Tested on held-out MLB pitchers."),
    "Radar gun + pitch tagging (T3a)": ("t3a", "Tested on held-out MLB pitchers, with movement rounded to the nearest 3 in."),
}
EDIT_COLS = {"t1": ["release_speed", "ivb_in", "hb_arm_in", "release_spin_rate", "release_extension", "release_height",
                    "release_side_arm", "arm_angle", "vaa", "haa_arm", "rel_angle_h_arm", "rel_angle_v"],
             "t3a": ["release_speed", "ivb_in", "hb_arm_in"]}
EDIT_LABELS = {"pitch_type": "Pitch type (Statcast code)", "usage_pct": "Usage %", "release_speed": "Velo (mph)",
               "ivb_in": "Ride (in)", "hb_arm_in": "Run (in, arm side)", "release_spin_rate": "Spin (rpm)",
               "release_extension": "Extension (ft)", "release_height": "Release height (ft)",
               "release_side_arm": "Release side (ft, arm side)", "arm_angle": "Arm angle (°)",
               "vaa": "Vertical approach angle (°)", "haa_arm": "Horizontal approach angle (°)",
               "rel_angle_h_arm": "Release angle, horizontal (°)", "rel_angle_v": "Release angle, vertical (°)"}

METHODS = {"learned": "Our model", "tjstats": "TJStats", "hzb": "HZB"}
CAPTIONS = {
    "learned": "Learned from 2023–24 MLB pitch results: arsenals that should get similar results. Picked from "
               "arsenal data alone; results are never an input.",
    "tjstats": "Published formula (tjstats.ca): matches each pitch to the same pitch type on 6 measurements: velo, "
               "ride, run, extension, release height, arm angle.",
    "hzb": "Published method (Healey, Zhao & Brooks, 2017): compares one pitcher's whole spread of velo and "
           "movement to another's, without needing pitch names to match.",
}
TJSTATS_URL = "https://tjstats.ca/pitch-similarity/"
HZB_URL = "https://vixra.org/abs/1705.0098"  # Healey, Zhao & Brooks (2017), viXra:1705.0098
INTRO = (
    "Find the MLB pitchers whose arsenals should play like a given pitcher's, then see how they attack hitters. "
    f"<a href='{TJSTATS_URL}' target='_blank'>TJStats</a> and <a href='{HZB_URL}' target='_blank'>HZB</a> match "
    "pitchers whose pitches measure alike. Our model learned from 2023–24 pitch results which parts of an arsenal "
    "drive outcomes, and matches on those."
)
SIM_HELP = ("Standard deviations closer than the average same-hand, same-role pitcher, by this list's method. "
            "Every method scores in its own units; this puts all three lists on one scale.")
HOW_TO_READ = """
**Three lists, three definitions of "similar."**
- **TJStats** is a published, hand-built formula from [tjstats.ca](https://tjstats.ca/pitch-similarity/). It matches each pitch to the same pitch type on the
  other pitcher and compares six measurements: velocity, ride, run, extension, release height, and arm angle. We
  rebuilt it at the tool's default settings, tuned its one free setting to similarity scores published on
  tjstats.ca, and checked it on a pitcher we didn't tune on. All three lists here use the same candidates —
  same hand, same role, at least 300 pitches that season, whole arsenal — so the methods are comparable. The
  site itself has no role filter, counts any pitch type thrown 10 times, and can rank one pitch at a time, so
  its own list for the same pitcher will look different.
- **HZB** ([Healey, Zhao & Brooks, 2017](https://vixra.org/abs/1705.0098)) is a published method that compares one pitcher's whole spread of pitches
  (velocity and movement) to another's, so pitch names don't have to match. We implemented it from the paper,
  filling in two details the paper leaves open.
- **Our model** is a neural network trained on 2023–24 MLB pitches to predict what each pitch did: called strike,
  ball, whiff, foul, weak contact, flare, or damage. To do that, it condenses a pitcher's arsenal (every pitch's
  velocity, movement, release, usage, and location) into a short profile that keeps what matters for results. Its
  list is the pitchers with the closest profiles.

**What's different about ours.** TJStats and HZB decide by hand which measurements matter and how much. Our model
learned that from outcomes. So it can call two arsenals close even when they measure differently, and far apart
when they measure alike (the "same measurements, different results" pairs below the lists). Results are never an
input: the comps come from arsenal data alone, and their actual results are what we tested against. On pitchers
held out of training (2025 and 2026), our comps predicted walk rate and swing-and-miss better than HZB's, the
stronger published method, in both years. HZB's comps predicted ground-ball rate better, and on overall run value
the two tied in 2026.

**Your own arsenal.** Because the model needs only arsenal data, it can place any pitcher among MLB pitchers, not
just one with a big-league track record. Choose "Your own arsenal" in the sidebar and enter your pitches, from full
tracking data or just a radar gun and pitch types. You get the MLB pitchers whose arsenals should play like yours,
and how they attack hitters. TJStats needs all six of its measurements, and HZB needs pitch-by-pitch data, so with
a radar gun only our list runs.
"""
# Wide enough that "Christopher Sánchez (LHP, starter)" fits in the pitcher picker.
SIDEBAR_CSS = "<style>section[data-testid='stSidebar'] {width: 360px !important; min-width: 360px !important;}</style>"
SUMMARY_STATS = {"k_pct": "K%", "bb_pct": "BB%", "whiff_pct": "Whiff%", "chase_pct": "Chase%", "gb_pct": "GB%",
                 "damage_pct": "Damage%", "pitcher_rv100": "Run value/100"}
# why: these are the seven stats the held-out test scored (D22), so the page shows exactly what was measured.
# Run value leads because it is the headline metric, with the noise warning it needs (D10).
STATS_WHY = ("Why these stats: run value/100 adds up every pitch's effect on runs, so it's the test's headline, "
             "but it's noisy from year to year; the other six each isolate one part of pitching. All seven are "
             "what the held-out test scored.")
OUTCOME_STATS = {"k_pct": "K%", "bb_pct": "BB%", "whiff_pct": "Whiff%", "chase_pct": "Chase%", "gb_pct": "GB%",
                 "damage_pct": "Damage%", "pitcher_rv100": "Run value/100"}
DIFF_FEATURES = {"release_speed": "Velo (mph)", "ivb_in": "Ride (in)", "hb_arm_in": "Run (in)",
                 "release_extension": "Extension (ft)",
                 "release_height": "Release height (ft)", "arm_angle": "Arm angle (°)"}
GRADES = {"type_whiff_pct": "Whiff%", "type_chase_pct": "Chase%", "type_gb_pct": "GB%",
          "type_damage_pct": "Damage%", "type_rv100": "Run value/100"}
# Short names for paired pitches of different types ("Change / Split").
PITCH_SHORT = {"FF": "4-seam", "CU": "Curve", "KC": "K-curve", "CS": "Slow curve", "CH": "Change", "FS": "Split",
               "FO": "Fork", "SC": "Screw"}
PITCH_NAMES = {"FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper", "SV": "Slurve",
               "CU": "Curveball", "KC": "Knuckle curve", "CS": "Slow curve", "CH": "Changeup", "FS": "Splitter",
               "FO": "Forkball", "SC": "Screwball"}
ZONES = {"heart": "Heart", "shadow": "Edges", "chase": "Chase", "waste": "Waste"}
HAND = {"R": "RHP", "L": "LHP"}
# Percentiles follow Baseball Savant: higher = better for the pitcher, so these two are flipped.
LOWER_IS_BETTER = {"bb_pct", "damage_pct"}
PCT_BAR = ("<div style='margin-top:-0.5rem'><div style='height:6px;border-radius:3px;background:rgba(128,128,128,0.2)'>"
           "<div style='height:6px;border-radius:3px;width:{p}%;background:rgba(28,131,225,0.85)'></div></div>"
           "<div style='font-size:0.8rem;opacity:0.8;margin-top:3px'>{label}</div></div>")
GLOSSARY = """
- **K% / BB%:** strikeouts / walks per batter faced.
- **Whiff%:** swings and misses per swing.
- **Chase%:** swings at pitches outside the strike zone, per pitch outside the zone.
- **GB%:** ground balls per ball in play.
- **Damage%:** percentage of batted balls in play that Baseball Savant rates as solid contact or barrels.
- **Run value/100:** how much each pitch changed the batting team's expected runs (Statcast's run value),
  per 100 pitches, from the pitcher's side: positive = better for the pitcher.
- **Counts:** *ahead* = more strikes than balls; *behind* = more balls than strikes; *even* = equal, after 0-0.
- **Zones:** Heart = the middle of the strike zone; Edges = the borders of the zone; Chase = just off the plate;
  Waste = well off it (Statcast's attack zones).
"""


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models() -> Dict[str, Tuple]:
    return {t: load_model(MODELS / f"arsenal_net_{t}.pt") for t in ["t1", "t3a"] if (MODELS / f"arsenal_net_{t}.pt").exists()}


@st.cache_data
def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    index = pd.read_parquet(APP / "index.parquet")
    arsenal = pd.read_parquet(APP / "arsenal.parquet")  # every pitch type thrown; in_arsenal = the >= 3% arsenal
    usage = pd.read_parquet(APP / "usage_by_count.parquet")
    locs = pd.read_parquet(APP / "locations.parquet")
    meta = json.loads((APP / "meta.json").read_text())
    return index, arsenal, usage, locs, meta


@st.cache_data
def load_hzb() -> Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]]:
    """{(season, hand): (pitcher ids, HZB distance matrix)}; empty if the tables predate HZB."""
    path = APP / "hzb_distance.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    keys = [k[: -len("_pitcher")] for k in z.files if k.endswith("_pitcher")]
    return {(int(k.split("_")[0]), k.split("_")[1]): (z[f"{k}_pitcher"], z[f"{k}_distance"]) for k in keys}


@st.cache_data
def tjstats_matrix(season: int, hand: str) -> Tuple[pd.DataFrame, np.ndarray]:
    _, arsenal, _, _, _ = load_tables()
    return arsenal_similarity(arsenal, TJSTATS, season, hand)


# ---------------------------------------------------------------------------
# Query encoding and retrieval
# ---------------------------------------------------------------------------
def query_arsenal(rows: pd.DataFrame, hand: str, season: int) -> pd.DataFrame:
    """User-entered pitch rows -> arsenal rows in the same format as the training data."""
    q = rows.copy()
    q["pitcher"], q["game_year"], q["split"], q["p_throws"] = -1, season, "query", hand
    q["family"] = q["pitch_type"].map(PITCH_FAMILY)
    q["usage"] = q["usage_pct"] / q["usage_pct"].sum()
    q["in_arsenal"] = q["usage"] >= 0.03
    q["usage_arsenal"] = (q["usage"] / q.loc[q["in_arsenal"], "usage"].sum()).where(q["in_arsenal"])
    return q


@torch.no_grad()
def embed_query(q: pd.DataFrame, tier: str, models: Dict) -> np.ndarray:
    model, ckpt = models[tier]
    x, w = I.arsenal_sets(q, q[I.KEYS].drop_duplicates(), I.TIERS[tier], I.Scaler.from_dict(ckpt["arsenal_scaler"]))
    return model.embed(torch.from_numpy(x), torch.from_numpy(w)).numpy()[0]


def pool(index: pd.DataFrame, season: int, hand: str, role: str, exclude: int = -1) -> pd.DataFrame:
    """Candidate comps: same season, hand, role, >= 300 pitches, not the query himself (D16, D20, D23, D24)."""
    p = index[(index["game_year"] == season) & (index["p_throws"] == hand) & (index["role"] == role)
              & (index["n_pitches"] >= I.MIN_PITCHES) & (index["pitcher"] != exclude)]
    return p.reset_index(drop=True)


def learned_scores(p: pd.DataFrame, emb: np.ndarray, tier: str) -> np.ndarray:
    """Learned similarity of every pool member: minus the embedding distance (higher = closer)."""
    cols = [c for c in p.columns if c.startswith(f"{tier}_e")]
    return -cdist(emb[None, :], p[cols].to_numpy("float64"))[0]


def tjstats_scores(p: pd.DataFrame, q_key: Tuple, season: int, hand: str, extra: pd.DataFrame = None) -> pd.Series:
    """TJStats similarity of every pool member to the query (higher = closer)."""
    if extra is None:
        pitchers, S = tjstats_matrix(season, hand)
    else:
        _, arsenal, _, _, _ = load_tables()
        pitchers, S = arsenal_similarity(pd.concat([arsenal, extra], ignore_index=True), TJSTATS, season, hand)
    row = {k: i for i, k in enumerate(pitchers[["pitcher", "game_year", "p_throws"]].itertuples(index=False, name=None))}
    s = pd.Series(S[row[q_key]], index=list(row.keys()))
    return p.apply(lambda r: s.get((r["pitcher"], r["game_year"], r["p_throws"]), 0.0), axis=1)


def hzb_scores(p: pd.DataFrame, pitcher: int, season: int, hand: str) -> pd.Series:
    """HZB similarity (minus the EMD) of every pool member to an MLB pitcher; NaN if not precomputed."""
    ids, D = load_hzb().get((season, hand), (np.array([], "int64"), None))
    col = {int(c): j for j, c in enumerate(ids)}
    if pitcher not in col:
        return pd.Series(np.nan, index=p.index)
    row = D[col[pitcher]]
    return p["pitcher"].map(lambda c: -float(row[col[c]]) if c in col else -np.inf)


def rank_methods(p: pd.DataFrame) -> pd.DataFrame:
    """Add <method>_z (similarity as a pool z-score) and <method>_rank (1 = closest) for each method."""
    out = p.copy()
    for m in METHODS:
        s = out[m].astype("float64")
        finite = s[np.isfinite(s)]
        # why: each method scores similarity in its own units (a distance, a kernel %, an EMD). Z-scoring
        # over the candidate pool, as the α blend does (run_eval.pool_zscore), puts the three on one scale.
        out[f"{m}_z"] = (s - finite.mean()) / (finite.std(ddof=0) or 1.0)
        out[f"{m}_rank"] = s.rank(ascending=False, method="first")
    return out


# ---------------------------------------------------------------------------
# Display helpers (every table is rounded here)
# ---------------------------------------------------------------------------
def fmt(stat: str, v: float, signed: bool = False) -> str:
    """Rates as percentages with one decimal; run value/100 always signed, two decimals."""
    if pd.isna(v):
        return "—"
    x, digits = (v, 2) if stat in ("pitcher_rv100", "type_rv100") else (100 * v, 1)
    return f"{x:+.{digits}f}" if signed or stat in ("pitcher_rv100", "type_rv100") else f"{x:.{digits}f}"


def num(v: float, digits: int = 1, signed: bool = False) -> str:
    return "—" if pd.isna(v) else (f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}")


def first_last(name: str) -> str:
    """Statcast's "Burns, Chase" -> "Chase Burns"."""
    last, _, first = name.partition(", ")
    return f"{first} {last}" if first else name


def last_name(name: str) -> str:
    return name.partition(", ")[0]


def pitch_name(code: Optional[str]) -> str:
    return PITCH_NAMES.get(code, code) if code else "—"


def mix_text(top_pitches: str) -> str:
    """ "FF 58 · SL 34 · CH 6" -> "Four-seam 58%, Slider 34%, Changeup 6%"."""
    parts = [p.split() for p in top_pitches.split(" · ") if p.strip()]
    return ", ".join(f"{pitch_name(p[0])} {p[1]}%" for p in parts if len(p) == 2)


def top_comps(ranked: pd.DataFrame, method: str) -> pd.DataFrame:
    return ranked.sort_values(f"{method}_rank").head(K)


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def percentile(index: pd.DataFrame, target: pd.Series, stat: str) -> int:
    """His percentile among same-season, same-role pitchers with >= 300 pitches (both hands); higher = better
    for the pitcher, as on Baseball Savant."""
    peers = index[(index["game_year"] == target["game_year"]) & (index["role"] == target["role"])
                  & (index["n_pitches"] >= I.MIN_PITCHES)][stat].dropna()
    v = target[stat]
    p = 100 * ((peers < v).mean() + 0.5 * (peers == v).mean())
    return int(np.clip(round(100 - p if stat in LOWER_IS_BETTER else p), 1, 100))


def profile(target: pd.Series, index: pd.DataFrame, arsenal: pd.DataFrame) -> None:
    """The selected MLB pitcher at a glance: hand, role, season, and pitch mix; his primary fastball's shape
    and his release; then his season's results with percentiles."""
    st.subheader(f"{first_last(target['player_name'])} ({HAND[target['p_throws']]}, {target['role']})")
    st.markdown(f"<p style='font-size:1.15rem;margin-top:-0.5rem'>{int(target['game_year'])} · "
                f"{mix_text(target['top_pitches'])}</p>", unsafe_allow_html=True)
    a = arsenal_rows(arsenal, target)
    fb = a.loc[target["fb_type"]] if target["fb_type"] in a.index else None
    w = a["usage_arsenal"] / a["usage_arsenal"].sum()
    name = pitch_name(target["fb_type"]) if fb is not None else "Fastball"
    shape = [(f"{name} velo (mph)", num(target["fb_velo"])),
             (f"{name} ride (in)", num(fb["ivb_in"]) if fb is not None else "—"),
             (f"{name} run (in)", num(fb["hb_arm_in"]) if fb is not None else "—"),
             ("Extension (ft)", num((w * a["release_extension"]).sum())),
             ("Release height (ft)", num((w * a["release_height"]).sum())),
             ("Arm angle", f"{target['arm_angle']:.0f}°")]
    for col, (label, value) in zip(st.columns(len(shape)), shape):
        col.metric(label, value)
    cols = st.columns(len(SUMMARY_STATS))
    for col, (c, label) in zip(cols, SUMMARY_STATS.items()):
        pct = percentile(index, target, c)
        col.metric(label, fmt(c, target[c]))
        col.markdown(PCT_BAR.format(p=pct, label=f"{ordinal(pct)} percentile"), unsafe_allow_html=True)
    st.caption(f"Percentiles are among "
               f"{int(target['game_year'])} {target['role']}s with 300+ pitches; higher = better for the pitcher, as on "
               "Baseball Savant (so a low BB% or Damage% ranks high).")


def comp_list(top: pd.DataFrame, method: str) -> pd.DataFrame:
    """Rank, name, and similarity on the shared pool-z-score scale."""
    return pd.DataFrame({"#": range(1, len(top) + 1), "Pitcher": [first_last(n) for n in top["player_name"]],
                         "Similarity": top[f"{method}_z"].round(1).to_numpy()})


def list_key(method: str, season: int, target_id: int) -> str:
    """Widget key for one comp list. The counter changes when another list is clicked (remember_click)."""
    return f"list_{method}_{season}_{target_id}_{st.session_state.get('list_gen', {}).get(method, 0)}"


def remember_click(key: str, ids: List[int], method: str) -> None:
    """A click on a comp (his row's checkbox or any of his cells) makes him the comp to compare with and borrow
    from, and clears the other lists."""
    picked = st.session_state[key]["selection"]
    rows = picked["rows"] or [cell[0] for cell in picked["cells"]]
    if rows and rows[0] < len(ids):
        st.session_state["picked"] = ids[rows[0]]
        # why: one comp selected at a time. Streamlit can't unselect a table from code, so the other lists
        # get a new key, which redraws them with nothing selected.
        gen = st.session_state.setdefault("list_gen", {})
        for other in METHODS:
            if other != method:
                gen[other] = gen.get(other, 0) + 1


def twin_flags(ranked: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    """In TJStats' or HZB's top 3 but in our model's bottom half of the pool: same measurements, different
    results. "Flagged by" names the method(s) whose top 3 he is in."""
    # why: the error analysis's rule (run_eval.twins: TJStats top 3, our bottom half), with HZB's top 3 added
    # in the app so both published methods are checked (D31). The README's pairs are the TJStats ones.
    learned_pct = (ranked["learned_rank"] - 1) / max(len(ranked) - 1, 1)
    in_top = pd.DataFrame({m: ranked[f"{m}_rank"] <= TWIN_TOP for m in methods})
    f = ranked[in_top.any(axis=1) & (learned_pct > 0.5)]
    f = f.assign(best=f[[f"{m}_rank" for m in methods]].min(axis=1)).sort_values(["best", "learned_rank"])
    out = pd.DataFrame({"Pitcher": [first_last(n) for n in f["player_name"]],
                        "Flagged by": [" + ".join(METHODS[m] for m in methods if in_top.at[i, m]) for i in f.index]})
    for m in methods:
        out[f"{METHODS[m]} rank"] = f[f"{m}_rank"].astype(int).to_numpy()
    out["Our model rank"] = [f"{int(r)} of {len(ranked)}" for r in f["learned_rank"]]
    for c, label in SUMMARY_STATS.items():
        out[label] = [fmt(c, v) for v in f[c]]
    return out


def short_names(target: pd.Series, comp: pd.Series) -> Tuple[str, str]:
    """Last names for column headers; full names if the two share one."""
    a, b = last_name(target["player_name"]), last_name(comp["player_name"])
    return (a, b) if a != b else (first_last(target["player_name"]), first_last(comp["player_name"]))


def pitcher_rows(arsenal: pd.DataFrame, r: pd.Series) -> pd.DataFrame:
    """Every pitch type one pitcher-season threw (not only the >= 3% arsenal), indexed by pitch type (D29)."""
    a = arsenal[(arsenal["pitcher"] == r["pitcher"]) & (arsenal["game_year"] == r["game_year"])
                & (arsenal["p_throws"] == r["p_throws"])]
    return a.set_index("pitch_type")


def pair_all(his: pd.DataFrame, theirs: pd.DataFrame) -> List[Tuple[Optional[str], Optional[str]]]:
    """Every pitch either pitcher threw, paired by type (a slider with a sweeper only when neither throws
    both; pair_types, D33)."""
    partner = paired_pitch(theirs.index.tolist(), his.index.tolist(), pair_types(his, theirs))
    his_to = {t: u for u, t in partner.items() if t is not None}
    rows = [(t, his_to.get(t)) for t in his.sort_values("usage", ascending=False).index]
    return rows + [(None, u) for u in theirs.sort_values("usage", ascending=False).index if partner[u] is None]


def pair_label(t: Optional[str], u: Optional[str], him: str, other: str) -> str:
    """His pitch t next to the comp's pitch u: one name if they're the same type, "Change / Split" if not."""
    if t is None:
        return f"{pitch_name(u)} (only {other})"
    if u is None:
        return f"{pitch_name(t)} (only {him})"
    return pitch_name(t) if t == u else f"{PITCH_SHORT.get(t, pitch_name(t))} / {PITCH_SHORT.get(u, pitch_name(u))}"


def vs_cell(theirs_v: float, his_v: float, show: Callable[[float], str], diff: Callable[[float], str]) -> str:
    """The comp's value, with the difference from his in parentheses: "86.0 (-2.6)"."""
    if pd.isna(theirs_v):
        return "—"
    return show(theirs_v) if pd.isna(his_v) else f"{show(theirs_v)} ({diff(theirs_v - his_v)})"


def usage_pct(v: float) -> str:
    return "—" if pd.isna(v) else ("<1" if 0 < v < 0.005 else f"{100 * v:.0f}")


def small_sample_note(rows: pd.DataFrame, role: str, meta: dict) -> str:
    """The per-pitch qualifier in words (Savant's 2.5 per team game; scaled for relievers, D30)."""
    # why: relievers get a bar scaled to their workload (D30), so the bar comes from his rows, not one constant.
    bar = f"fewer than {rows['savant_min'].iloc[0]:.0f} thrown" if len(rows) else "below the qualifier"
    rule = ("Savant's qualifier, 2.5 per team game" if role == "SP" else
            f"Savant's 2.5 per team game, scaled to a reliever's workload: the median reliever throws "
            f"{100 * meta.get('reliever_scale', 1.0):.0f}% as many pitches as the median starter")
    return f"{bar} ({rule})"


def compare(target: pd.Series, comp: pd.Series, arsenal: pd.DataFrame, meta: dict) -> None:
    """Every pitch's shape and grades (the comp's, with the difference from his paired pitch), then season results."""
    his, theirs = pitcher_rows(arsenal, target), pitcher_rows(arsenal, comp)
    pairs = pair_all(his, theirs)
    him, other = short_names(target, comp)
    get = lambda d, t, c: d.at[t, c] if t is not None else np.nan  # noqa: E731
    # why: every pitch either threw, small samples included and marked (D29), as the old per-pitch grades table did.
    small = [(u is not None and not bool(theirs.at[u, "qualified"])) or (t is not None and not bool(his.at[t, "qualified"]))
             for t, u in pairs]
    feat = pd.DataFrame({"Pitch": [pair_label(t, u, him, other) + (" *" if sm else "") for (t, u), sm in zip(pairs, small)],
                         f"{him} %": [usage_pct(get(his, t, "usage")) for t, _ in pairs],
                         f"{other} %": [usage_pct(get(theirs, u, "usage")) for _, u in pairs]})
    for c, name in DIFF_FEATURES.items():
        d = 0 if c == "arm_angle" else 1
        feat[name] = [vs_cell(get(theirs, u, c), get(his, t, c), partial(num, digits=d), partial(num, digits=d, signed=True))
                      for t, u in pairs]
    for c, name in GRADES.items():
        feat[name] = [vs_cell(get(theirs, u, c), get(his, t, c), partial(fmt, c), partial(fmt, c, signed=True))
                      for t, u in pairs]
    res = pd.DataFrame({"Stat": list(OUTCOME_STATS.values()),
                        him: [fmt(c, target[c]) for c in OUTCOME_STATS],
                        other: [fmt(c, comp[c]) for c in OUTCOME_STATS],
                        "Difference": [fmt(c, comp[c] - target[c], signed=True) for c in OUTCOME_STATS]})

    st.markdown(f"#### How {first_last(comp['player_name'])} compares to {first_last(target['player_name'])}")
    st.markdown("**Pitch by pitch**")
    st.markdown(table_html(feat, left=["Pitch"]), unsafe_allow_html=True)
    mixed = [(t, u) for t, u in pairs if t is not None and u is not None and t != u]
    example = (f". \"{pair_label(*mixed[0], him, other)}\" is {him}'s {pitch_name(mixed[0][0]).lower()} next to "
               f"{other}'s {pitch_name(mixed[0][1]).lower()}") if mixed else ""
    st.caption(f"{other}'s numbers, with the difference from {him}'s same pitch underneath: (+1.0) on velo means "
               f"{other} throws it 1 mph harder{example}. * Small sample for either pitcher: "
               f"{small_sample_note(theirs, comp['role'], meta)}, so read those grades loosely.")
    left, _ = st.columns([2, 3])
    with left:
        st.markdown("**Results**")
        st.dataframe(res, hide_index=True, width="stretch")


def percents(counts: np.ndarray) -> List[str]:
    """Pitch counts -> whole percents that add to exactly 100 (largest remainder); "<1%" for a
    nonzero share that rounds to 0; "—" if there are no pitches at all."""
    n = np.asarray(counts, "float64")
    if n.sum() == 0:
        return ["—"] * len(n)
    x = 100 * n / n.sum()
    whole = np.floor(x).astype(int)
    whole[np.argsort(-(x - whole), kind="stable")[: 100 - whole.sum()]] += 1
    return ["<1%" if w == 0 and c > 0 else f"{w}%" for w, c in zip(whole, n)]


@st.cache_data
def gap_full_red() -> float:
    """The usage gap, in percentage points, at which a heat-map cell is fully red (D32)."""
    # why: the scale is the gap bigger than 9 of 10 comp cells across the pool (gameplan_gap.py), so red
    # means unusual for a comp, not a hand-picked number. 20 points was the fixed scale before D32.
    path = APP / "gap_scale.json"
    return float(json.loads(path.read_text())["gap_full_red"]) if path.exists() else 20.0


def gap_color(gap_points: float, full_red: float) -> str:
    """Cell background for a usage gap: green (0 points) through yellow to red (>= full_red)."""
    # why: a translucent fill, so the cell text stays readable in both the light and the dark theme.
    t = min(abs(gap_points) / full_red, 1.0)
    return f"hsla({120 * (1 - t):.0f}, 70%, 45%, 0.35)"


MIX_CSS = ("<style>table.mix{border-collapse:collapse;width:100%;font-size:0.875rem;margin-bottom:0.5rem}"
           "table.mix th,table.mix td{padding:4px 10px;text-align:right;white-space:nowrap;"
           "border-bottom:1px solid rgba(128,128,128,0.2)}"
           "table.mix th{font-weight:600;background:rgba(128,128,128,0.08)}"
           "table.mix th.l,table.mix td.l{text-align:left}"
           "table.mix tr.his td{opacity:0.7;border-bottom:1px solid rgba(128,128,128,0.45)}"
           "table.mix td .d{display:block;font-size:0.75rem;opacity:0.65}"
           "table.mix.wrap th{white-space:normal;vertical-align:bottom}</style>")


def table_html(df: pd.DataFrame, left: List[str]) -> str:
    """A table in the pitch-mix style, with each "value (difference)" cell split over two lines."""
    # why: 14 columns of "93.0 (+0.2)" overflow st.dataframe; the difference on its own small line under the
    # value keeps every column on screen.
    esc = lambda x: html.escape(str(x), quote=True)  # noqa: E731

    def cell(col: str, v: str) -> str:
        value, sep, diff = str(v).partition(" (")
        body = esc(v) if col in left or not sep else f"{esc(value)}<span class='d'>({esc(diff)}</span>"
        return f"<td class='l'>{body}</td>" if col in left else f"<td>{body}</td>"

    out = [MIX_CSS, "<table class='mix wrap'><thead><tr>"]
    out += [f"<th class='l'>{esc(c)}</th>" if c in left else f"<th>{esc(c)}</th>" for c in df.columns]
    out += ["</tr></thead><tbody>"]
    for row in df.itertuples(index=False):
        out.append("<tr>" + "".join(cell(c, v) for c, v in zip(df.columns, row)) + "</tr>")
    return "".join(out + ["</tbody></table>"])


def mix_html(comp: pd.Series, mix: pd.DataFrame, target: Optional[pd.Series] = None,
             his_mix: Optional[pd.DataFrame] = None, his_for: Optional[Dict[str, Optional[str]]] = None) -> str:
    """The pitch-mix table as HTML. For an MLB pitcher, each of the comp's rows is followed by his row, and
    each comp cell is shaded by the gap between their shares."""
    # why: st.dataframe can't shade single cells by a value from another row, so the table is HTML. Both
    # shares are printed, so the color isn't the only cue.
    esc = lambda x: html.escape(str(x), quote=True)  # noqa: E731
    him, other = short_names(target, comp) if target is not None else (None, last_name(comp["player_name"]))
    head = ["Batter", "Count"] + (["Pitcher"] if target is not None else []) + ["Pitches"]
    out = [MIX_CSS, "<table class='mix'><thead><tr>"]
    out += [f"<th class='l'>{h}</th>" if h != "Pitches" else f"<th>{h}</th>" for h in head]
    # why: a column where his slider sits under the comp's sweeper (D33) is labeled with both names.
    heads = [pair_label(his_for[u], u, him, other) if target is not None and his_for.get(u) not in (None, u)
             else pitch_name(u) for u in mix.columns]
    out += [f"<th>{esc(h)}</th>" for h in heads] + ["</tr></thead><tbody>"]
    gaps, full_red = (cell_gaps(mix, his_mix, his_for), gap_full_red()) if target is not None else (None, None)
    for row, ((stand, state), counts) in enumerate(zip(mix.index, mix.to_numpy())):
        pct = percents(counts)
        lead = [f"vs {stand}HB", state] + ([other] if target is not None else [])
        out.append("<tr class='comp'>" + "".join(f"<td class='l'>{esc(c)}</td>" for c in lead)
                   + f"<td>{int(counts.sum())}</td>")
        if target is None:
            out += [f"<td>{p}</td>" for p in pct] + ["</tr>"]
            continue
        his_counts = his_mix.loc[(stand, state)].to_numpy()
        his_pct = dict(zip(his_mix.columns, percents(his_counts)))
        for p, gap in zip(pct, gaps[row]):
            out.append(f"<td style='background:{gap_color(gap, full_red)}'>{p}</td>")
        out.append("</tr><tr class='his'><td></td><td></td>" + f"<td class='l'>{esc(him)}</td>"
                   + f"<td>{int(his_counts.sum())}</td>")
        for u in mix.columns:
            t = his_for[u]
            out.append(f"<td>{his_pct.get(t, '—') if t else '—'}</td>")
        out.append("</tr>")
    return "".join(out + ["</tbody></table>"])


def gameplan(comp: pd.Series, arsenal: pd.DataFrame, usage: pd.DataFrame, locs: pd.DataFrame, meta: dict,
             target: Optional[pd.Series] = None) -> None:
    """Pitch mix by count and batter hand and locations, each next to the selected pitcher's own if he's an MLB
    pitcher; for your own arsenal, the comp's per-pitch grades as well. Every pitch the comp threw (D29)."""
    key = (comp["pitcher"], comp["game_year"], comp["p_throws"])
    sel = lambda d: d[(d["pitcher"] == key[0]) & (d["game_year"] == key[1]) & (d["p_throws"] == key[2])]  # noqa: E731
    # why: every pitch he threw, small samples included (D29): a pitch he barely throws to lefties, or
    # only when ahead, is part of the plan. The pitch counts next to each row keep small samples reading as small.
    mix = usage_mix(usage, comp, meta["count_states"])
    order = mix.columns.tolist()
    st.markdown(f"#### How {last_name(comp['player_name'])} attacks hitters")

    st.markdown("**Pitch mix by count and batter handedness**")
    his_for: Dict[str, Optional[str]] = {}
    if target is None:
        st.markdown(mix_html(comp, mix), unsafe_allow_html=True)
    else:
        # why: the gameplan is only useful next to what he does now, so his own mix sits under each of the
        # comp's rows, compared pitch for pitch by the same type-and-shape pairing as the table above (D15).
        his_mix = usage_mix(usage, target, meta["count_states"])
        pairs = pair_types(pitcher_rows(arsenal, target), pitcher_rows(arsenal, comp))
        his_for = paired_pitch(order, his_mix.columns.tolist(), pairs)
        st.markdown(mix_html(comp, mix, target, his_mix, his_for), unsafe_allow_html=True)
        him, other = short_names(target, comp)
        # why: full red is the 90th-percentile gap over every comp cell the app can show (D32), so the colour
        # means the same thing on every pitcher's page instead of rescaling to whoever is on screen.
        st.caption(f"Each {other} cell is shaded by the gap between his share and {him}'s in the row below: green = "
                   f"the same share, turning yellow and then red as the gap grows. Full red = {gap_full_red():.0f}+ points "
                   f"apart, a bigger gap than 9 in 10 comp-vs-pitcher cells across MLB. Pitches only {him} throws "
                   "aren't shown here.")

    a = sel(arsenal).sort_values("usage", ascending=False)
    if target is None:
        # For an MLB pitcher these grades are in the pitch-by-pitch table above, next to his own.
        grades = pd.DataFrame({"Pitch": [pitch_name(t) for t in a["pitch_type"]], "Pitches": a["n"].astype(int).to_numpy(),
                               "Sample": ["" if ok else "small sample" for ok in a["qualified"].fillna(False)],
                               "Usage": percents(a["n"].to_numpy()),
                               "Velo": [num(v) for v in a["release_speed"]], "Ride": [num(v) for v in a["ivb_in"]],
                               "Run": [num(v) for v in a["hb_arm_in"]]})
        for c, name in GRADES.items():
            grades[name] = [fmt(c, v) for v in a[c]]
        st.markdown("**Per-pitch grades**")
        st.dataframe(grades, hide_index=True, width="stretch")
        st.caption(f"Small sample: {small_sample_note(a, comp['role'], meta)}, so read those grades loosely.")

    def zone_counts(r: pd.Series) -> pd.DataFrame:
        d = locs[(locs["pitcher"] == r["pitcher"]) & (locs["game_year"] == r["game_year"])
                 & (locs["p_throws"] == r["p_throws"])].astype({"pitch_type": str, "stand": str})
        z = d.pivot_table(index=["pitch_type", "stand"], columns="attack_zone", values="n", aggfunc="sum",
                          fill_value=0, observed=True)
        return z.reindex(columns=meta["attack_zones"], fill_value=0)

    loc = zone_counts(comp)
    keep = [(t, s) for t in order for s in ["L", "R"] if (t, s) in loc.index]
    loc = loc.reindex(pd.MultiIndex.from_tuples(keep)) if keep else loc.iloc[0:0]
    cells = [percents(r) for r in loc.to_numpy()]
    other = last_name(comp["player_name"])
    if target is not None:
        # why: each zone share next to his, for the same pitch pairing and batter hand as the mix table.
        him, other = short_names(target, comp)
        his_loc = zone_counts(target)
        for i, ((u, s), counts) in enumerate(zip(loc.index, loc.to_numpy())):
            t = his_for.get(u)
            if t is None or (t, s) not in his_loc.index or his_loc.loc[(t, s)].sum() == 0:
                continue
            gap = 100 * (row_shares(counts.astype(float)) - row_shares(his_loc.loc[(t, s)].to_numpy(float)))
            cells[i] = [f"{p} ({g:+.0f})" if p != "—" else p for p, g in zip(cells[i], gap)]
    where = pd.DataFrame(cells, columns=[ZONES.get(z, z) for z in loc.columns])
    where.insert(0, "Pitches", loc.sum(axis=1).astype(int).to_numpy())
    where.insert(0, "Batter", [f"vs {s}HB" for _, s in loc.index])
    where.insert(0, "Pitch", [pitch_name(u) if target is None else pair_label(his_for.get(u), u, him, other)
                              for u, _ in loc.index])
    st.markdown(f"**Where {other} throws it** (each pitch's zones by batter hand)")
    st.dataframe(where, hide_index=True, width="stretch")
    if target is not None:
        st.caption(f"{other}'s share of each pitch in each zone, with the difference from {him}'s paired pitch "
                   "(same batter hand) in parentheses, in points.")


def query_param_int(name: str) -> Optional[int]:
    v = st.query_params.get(name)
    return int(v) if v is not None and str(v).isdigit() else None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Arsenal Match", layout="wide")
    st.title("Arsenal Match")
    st.markdown(f"<p style='font-size:1.1rem;line-height:1.55'>{INTRO}</p>", unsafe_allow_html=True)
    st.info("Tested on MLB pitchers only (2025–26). College and high school are future work.")
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    models = load_models()
    index, arsenal, usage, locs, meta = load_tables()
    if "in_arsenal" not in arsenal.columns:
        st.error("models/app/arsenal.parquet is out of date. Rebuild it: python -m src.app.build_artifacts")
        st.stop()
    seasons = sorted(index["game_year"].unique().tolist())
    want_season, want_pitcher = query_param_int("season"), query_param_int("pitcher")
    default_season = want_season if want_season in seasons else (2025 if 2025 in seasons else seasons[-1])

    with st.sidebar:
        mode = st.radio("Look up", ["An MLB pitcher", "Your own arsenal"])
        season = st.selectbox("Season", seasons, index=seasons.index(default_season))
        if mode == "An MLB pitcher":
            choices = index[(index["game_year"] == season) & (index["n_pitches"] >= I.MIN_PITCHES)] \
                .sort_values("player_name").reset_index(drop=True)
            labels = [f"{first_last(n)} ({HAND[h]}, {r})"
                      for n, h, r in zip(choices["player_name"], choices["p_throws"], choices["role"])]
            hit = np.flatnonzero(choices["pitcher"].to_numpy() == want_pitcher)
            pick = st.selectbox("Pitcher", labels, index=int(hit[0]) if len(hit) else 0)
            target = choices.iloc[labels.index(pick)]
            hand, role, tier = target["p_throws"], target["role"], "t1"
        else:
            hand = st.radio("Throws", ["R", "L"], horizontal=True)
            role = st.radio("Role", ["SP", "RP"], horizontal=True,
                            help="Comps are always the same role: starters with starters, relievers with relievers.")
            tier_label = st.selectbox("What can you measure?", list(TIERS_UI))
            tier, note = TIERS_UI[tier_label]
            st.caption(note)

    t0 = time.perf_counter()
    if mode == "An MLB pitcher":
        target_id = int(target["pitcher"])
        p = pool(index, season, hand, role, exclude=target_id)
        p["learned"] = learned_scores(p, target[[c for c in index.columns if c.startswith("t1_e")]].to_numpy("float64"), "t1")
        p["tjstats"] = tjstats_scores(p, (target_id, season, hand), season, hand)
        p["hzb"] = hzb_scores(p, target_id, season, hand)
        actual = target
        profile(target, index, arsenal)
    else:
        base = arsenal[arsenal["in_arsenal"] & (arsenal["game_year"] == season) & (arsenal["p_throws"] == hand)] \
            .groupby("pitch_type")[SHAPE_FEATURES + LOCATION_FEATURES].mean()
        default = pd.DataFrame({"pitch_type": ["FF", "SL", "CH"], "usage_pct": [55.0, 30.0, 15.0]})
        default = default.join(base.round(1), on="pitch_type")
        st.subheader("Your arsenal")
        st.caption("One row per pitch. Starting values are the MLB averages for that pitch; edit them to yours.")
        edited = st.data_editor(default[["pitch_type", "usage_pct"] + EDIT_COLS[tier]], num_rows="dynamic",
                                width="stretch", hide_index=True, column_config=EDIT_LABELS)
        rows = edited.merge(default.drop(columns=["usage_pct"] + EDIT_COLS[tier]), on="pitch_type", how="left")
        for c in SHAPE_FEATURES + LOCATION_FEATURES:
            if c not in rows:
                rows[c] = np.nan
        if tier != "t1":
            rows[[c for c in SHAPE_FEATURES if c not in EDIT_COLS[tier]]] = np.nan  # not measured at this level
        q = query_arsenal(rows.dropna(subset=["pitch_type", "usage_pct"]), hand, season)
        target_id, actual = -1, None
        p = pool(index, season, hand, role)
        p["learned"] = learned_scores(p, embed_query(q, tier, models), tier)
        p["tjstats"] = tjstats_scores(p, (-1, season, hand), season, hand, extra=q) if tier == "t1" else np.nan
        p["hzb"] = np.nan  # HZB needs pitch-by-pitch data
    ranked = rank_methods(p)
    ms = (time.perf_counter() - t0) * 1000
    shown_methods = [m for m in METHODS if ranked[m].notna().any()]

    with st.expander("How to read this page"):
        st.markdown(HOW_TO_READ + "\n\n" + STATS_WHY + "\n" + GLOSSARY)

    st.caption("Click a pitcher in any list to see " + ("how he compares and " if actual is not None else "")
               + "how he attacks hitters.")
    shown: List[int] = []
    for col, (m, title) in zip(st.columns(3), METHODS.items()):
        with col:
            # why: the description sits in the title's tooltip, so the three lists line up at the same height.
            st.markdown(f"**{title}**", help=CAPTIONS[m])
            if m in shown_methods:
                top = top_comps(ranked, m)
                ids = top["pitcher"].astype(int).tolist()
                key = list_key(m, season, target_id)
                st.dataframe(comp_list(top, m), hide_index=True, width="stretch", height=LIST_HEIGHT, key=key,
                             on_select=partial(remember_click, key, ids, m), selection_mode=["single-row", "single-cell"],
                             column_config={"#": st.column_config.NumberColumn(width="small"),
                                            "Similarity": st.column_config.NumberColumn(format="%.1f",
                                                                                        help=SIM_HELP)})
                shown += ids
            elif m == "tjstats":
                st.info("TJStats needs all 6 of its measurements, so it runs only with full Statcast.")
            elif mode == "An MLB pitcher":
                st.warning("HZB tables are missing. Rebuild them: python -m src.app.build_artifacts")
            else:
                st.info("HZB compares pitch-by-pitch data, so it runs only for MLB pitchers.")
    st.caption(f"Similarity: {SIM_HELP} Compared with {len(p)} other {HAND[hand]} {role}s from {season} ({ms:.0f} ms).")

    twin_methods = [m for m in TWIN_METHODS if m in shown_methods]
    if twin_methods:
        whose = f"{last_name(actual['player_name'])}'s" if actual is not None else "your arsenal's"
        st.markdown(f"**Same measurements, different results:** {' or '.join(METHODS[m] for m in twin_methods)} "
                    f"ranks these pitchers among {whose} {TWIN_TOP} closest; our model ranks them in the bottom half.")
        flags = twin_flags(ranked, twin_methods)
        if len(flags):
            st.dataframe(flags, hide_index=True, width="stretch")
        else:
            st.caption("None for this pitcher.")

    st.divider()
    picked = st.session_state.get("picked")
    comp = ranked[ranked["pitcher"] == (picked if picked in shown else shown[0])].iloc[0]
    if actual is not None:
        compare(actual, comp, arsenal, meta)
    gameplan(comp, arsenal, usage, locs, meta, actual)


main()
