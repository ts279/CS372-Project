"""
Evaluate every similarity method on one season with the held-out-pitcher retrieval test.

Pipeline stage 4 of 7 (evaluate). Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

Methods (D4, D27): raw Euclidean, TJStats (rebuilt to the tool's defaults, D26), HZB (Healey, Zhao &
Brooks 2017, D27), SEAM, the learned embedding (T1), the T3a embedding (for the slices), and the
alpha blend of learned + TJStats (alpha = 0 is TJStats, 1 is learned).
All comps are same season, same hand, same role (D20, D23).

Headline (D28): learned vs. the strongest published baseline (TJStats, HZB, or SEAM: the highest
2025 RV/100 r), chosen on --split val and read back on --split test. Every delta vs. a baseline
(method comparison, per-stat breakdown, slices) uses that baseline; the deltas vs. all three
published baselines are also written, each with paired-bootstrap CIs.

Outputs, one set per split (docs/results/, docs/figures/; --sample writes under models/sample/):
    errors_<split>.csv              one row per query x stat x method (the table everything else groups)
    method_comparison_<split>.csv   RV/100: r and MAE with paired-bootstrap 95% CIs, deltas vs. the headline baseline (D11, D28)
    baseline_deltas_<split>.csv     RV/100: every method vs. each published baseline (TJStats, HZB, SEAM), paired CIs
    alpha_sweep_<split>.csv         r and MAE by alpha, deltas vs. alpha = 0; figure alpha_sweep_<split>.png
    stat_breakdown_<split>.csv      7 stats x methods, deltas vs. the headline baseline (D22); figure stat_breakdown_<split>.png
    slices_<split>.csv              velocity band (bottom FB-velo quartile) and seen vs. new pitchers
    twins_<split>.csv, twins_summary_<split>.csv   physical twins / functional strangers, and the reverse
    calibration_<split>.csv, reliability_bins_<split>.csv; figure reliability_<split>.png (D12)
    stat_reliability_<split>.csv    split-half reliability of each stat: the ceiling on any method's r
    alpha_star.json                 alpha* and the headline baseline, chosen on 2025 (val only; test reads them)

Data use (D5): --split val (2025) is where alpha* is chosen and anything may be iterated on.
--split test (2026) is scored once, at the end, with alpha* read from the val run.

Usage (from the repo root):
    python -m src.eval.run_eval --split val [--sample]
    python -m src.eval.run_eval --split test          # once

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # figures are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from src.eval.bootstrap import paired_bootstrap
from src.eval.calibration import MIN_BIN_COUNT, calibration_report, predict_proba
from src.eval.retrieval import (HEADLINE, STATS, candidate_pool, eligible, embedding_similarity, knn_errors)
from src.models import inputs as I
from src.models.hzb_similarity import hzb_similarity, load_pitches
from src.models.physical_similarity import SEAM, TJSTATS, arsenal_similarity, raw_euclidean_similarity
from src.models.train_embedding import PITCH_COLS, Paths, load_model

ROOT = Path(__file__).resolve().parents[2]
SEASON_OF = {"val": 2025, "test": 2026}
ALPHAS = [round(a, 1) for a in np.linspace(0.0, 1.0, 11)]
BASE_METHODS = ["raw", "tjstats", "hzb", "seam", "learned"]
PUBLISHED = ["tjstats", "hzb", "seam"]  # the published hand-built baselines (D28 picks the strongest on 2025)
LABELS = {"raw": "Raw Euclidean", "tjstats": "TJStats", "hzb": "HZB (EMD)", "seam": "SEAM", "learned": "Learned (T1)",
          "learned_t3a": "Learned (T3a)"}
PHYS_KEYS = ["pitcher", "game_year", "p_throws"]


# ---------------------------------------------------------------------------
# Similarity matrices, aligned to one season's pitcher-season table
# ---------------------------------------------------------------------------
def season_table(seasons: pd.DataFrame, year: int) -> pd.DataFrame:
    return seasons[seasons["game_year"] == year].sort_values(I.KEYS).reset_index(drop=True)


def align(ps: pd.DataFrame, pitchers: pd.DataFrame, S: np.ndarray, out: np.ndarray) -> None:
    """Write a per-hand matrix (rows/cols = `pitchers`) into the season matrix aligned with `ps`."""
    row_of = {k: i for i, k in enumerate(ps[PHYS_KEYS].itertuples(index=False, name=None))}
    rows = np.array([row_of.get(k, -1) for k in pitchers[PHYS_KEYS].itertuples(index=False, name=None)])
    keep = np.flatnonzero(rows >= 0)
    out[np.ix_(rows[keep], rows[keep])] = S[np.ix_(keep, keep)]


def physical_matrix(ps: pd.DataFrame, by_type: pd.DataFrame, pitches: pd.DataFrame, year: int,
                    method: str) -> np.ndarray:
    """One hand-built method's season matrix. HZB reads pitches (per batter hand); the others read by_type."""
    out = np.full((len(ps), len(ps)), -np.inf)
    for hand in ["R", "L"]:
        if method == "raw":
            pitchers, S = raw_euclidean_similarity(by_type, year, hand)
        elif method == "hzb":
            pitchers, S = hzb_similarity(pitches, year, hand)
        else:
            pitchers, S = arsenal_similarity(by_type, TJSTATS if method == "tjstats" else SEAM, year, hand)
        align(ps, pitchers, S, out)
    return out


def learned_matrix(ps: pd.DataFrame, emb: pd.DataFrame) -> np.ndarray:
    cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    e = ps[I.KEYS].merge(emb[I.KEYS + cols], on=I.KEYS, how="left")[cols].to_numpy("float64")
    return embedding_similarity(e)


def pool_zscore(ps: pd.DataFrame, sim: np.ndarray) -> np.ndarray:
    """Z-score each query's similarities over its own candidate pool, so two methods can be blended."""
    ok = eligible(ps)
    z = np.full(sim.shape, -np.inf)
    for i in np.flatnonzero(ok):
        cand = candidate_pool(ps, i, ok)
        if len(cand) < 2:
            continue
        v = sim[i, cand]
        z[i, cand] = (v - v.mean()) / (v.std() or 1.0)
    return z


def similarity_suite(ps: pd.DataFrame, by_type: pd.DataFrame, pitches: pd.DataFrame,
                     embs: Dict[str, pd.DataFrame], year: int) -> Dict[str, np.ndarray]:
    """Every method's similarity matrix plus the alpha blends of learned (T1) and TJStats.

    pitches: that season's pitches with the columns HZB needs (hzb_similarity.load_pitches).
    """
    sims = {m: physical_matrix(ps, by_type, pitches, year, m) for m in ["raw", "tjstats", "hzb", "seam"]}
    for name, emb in embs.items():
        sims[name] = learned_matrix(ps, emb)
    # why (D4): blend on z-scores within each query's pool. The two raw scales aren't comparable (kernel
    # values vs. embedding distances), and z-scoring preserves each method's own ranking, so
    # alpha = 0 reproduces TJStats' comps exactly and alpha = 1 the learned comps.
    zt, zl = pool_zscore(ps, sims["tjstats"]), pool_zscore(ps, sims["learned"])
    both = np.isfinite(zl) & np.isfinite(zt)   # pairs both methods can score; the rest stay -inf (never a comp)
    for a in ALPHAS:
        blend = np.full(zl.shape, -np.inf)
        blend[both] = a * zl[both] + (1.0 - a) * zt[both]
        sims[f"blend_{a:.1f}"] = blend
    return sims


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def all_errors(ps: pd.DataFrame, sims: Dict[str, np.ndarray]) -> pd.DataFrame:
    frames = []
    for name, sim in sims.items():
        e = knn_errors(ps, sim, stats=list(STATS))
        e.insert(0, "method", name)
        frames.append(e)
    return pd.concat(frames, ignore_index=True)


def wide(errors: pd.DataFrame, stat: str, methods: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Predictions per method aligned on the same queries (the pairing), plus actuals and query info."""
    e = errors[(errors["stat"] == stat) & errors["method"].isin(methods)]
    key = ["pitcher", "p_throws"]  # unique within one season (a switch-pitcher has one row per hand)
    p = e.pivot_table(index=key, columns="method", values="predicted").dropna()
    info = e[e["method"] == methods[0]].set_index(key).loc[p.index]
    return {m: p[m].to_numpy() for m in methods}, info["actual"].to_numpy(), info.reset_index()


def compare(errors: pd.DataFrame, stat: str, methods: List[str], reference: str,
            mask: Optional[Callable[[pd.DataFrame], np.ndarray]] = None) -> pd.DataFrame:
    preds, actual, info = wide(errors, stat, methods)
    if mask is not None:
        m = mask(info)
        preds, actual = {k: v[m] for k, v in preds.items()}, actual[m]
    out = paired_bootstrap(preds, actual, reference)
    base = errors[(errors["stat"] == stat) & (errors["method"] == reference)]
    out.insert(1, "stat", stat)
    out["baseline_mae"] = float(base["baseline_abs_error"].mean()) if mask is None else np.nan
    return out


def slices(errors: pd.DataFrame, ps: pd.DataFrame, train_pitchers: set, methods: List[str],
           reference: str) -> pd.DataFrame:
    """Velocity band (bottom FB-velo quartile of queries vs. the rest) and seen vs. new pitchers."""
    # why: these are the two slices the project's claims depend on. The soft-tossing quartile is the MLB
    # group closest to the lower-level pitchers that motivate the tool. "New" pitchers never appeared in
    # 2023-24 training, so the model has never seen their outcomes, which is what separates a similarity
    # metric that generalizes from one that remembers the pitchers it was trained on.
    velo = ps.drop_duplicates("pitcher").set_index("pitcher")["fb_velo"]
    q = errors[(errors["stat"] == HEADLINE) & (errors["method"] == "tjstats")]["pitcher"]
    cut = velo.reindex(q).quantile(0.25)
    groups = {
        f"FB velo bottom quartile (< {cut:.1f} mph)": lambda d: velo.reindex(d["pitcher"]).to_numpy() < cut,
        f"FB velo top 3 quartiles (>= {cut:.1f} mph)": lambda d: velo.reindex(d["pitcher"]).to_numpy() >= cut,
        "seen in 2023-24 training": lambda d: d["pitcher"].isin(train_pitchers).to_numpy(),
        "new (not in 2023-24 training)": lambda d: ~d["pitcher"].isin(train_pitchers).to_numpy(),
    }
    frames = []
    for name, mask in groups.items():
        c = compare(errors, HEADLINE, methods, reference, mask)
        c.insert(0, "slice", name)
        frames.append(c)
    return pd.concat(frames, ignore_index=True).drop(columns=["baseline_mae"])


def twins(ps: pd.DataFrame, sims: Dict[str, np.ndarray], top: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pairs one method calls near-identical while the other ranks them in the bottom half of the pool.

    physical twin, functional stranger: in TJStats' top 3, but in the learned bottom half.
    functional twin, physical stranger: in the learned top 3, but in TJStats' bottom half.
    TJStats only: the rule was set before HZB was added (D27). The app also flags HZB's top 3 (D31).
    """
    ok = eligible(ps)
    stats = list(STATS)
    rows = []
    for i in np.flatnonzero(ok):
        cand = candidate_pool(ps, i, ok)
        if len(cand) < 10:
            continue
        pct = {}
        for m in ["tjstats", "learned"]:
            order = cand[np.argsort(-sims[m][i, cand], kind="stable")]
            pct[m] = pd.Series(np.arange(len(order)) / (len(order) - 1), index=order)
        for kind, a, b in [("physical twin, functional stranger", "tjstats", "learned"),
                           ("functional twin, physical stranger", "learned", "tjstats"),
                           ("TJStats top 3 (all)", "tjstats", None), ("learned top 3 (all)", "learned", None)]:
            for j in pct[a].index[:top]:
                if b is not None and pct[b][j] <= 0.5:
                    continue
                row = {"kind": kind, "query": ps.at[i, "player_name"], "comp": ps.at[j, "player_name"],
                       "role": ps.at[i, "role"], "hand": ps.at[i, "p_throws"],
                       "tjstats_pct": round(float(pct["tjstats"][j]), 3), "learned_pct": round(float(pct["learned"][j]), 3),
                       "query_fb_velo": ps.at[i, "fb_velo"], "comp_fb_velo": ps.at[j, "fb_velo"]}
                for s in stats:
                    row[f"query_{s}"], row[f"comp_{s}"] = ps.at[i, s], ps.at[j, s]
                    row[f"absdiff_{s}"] = abs(ps.at[i, s] - ps.at[j, s])
                rows.append(row)
    pairs = pd.DataFrame(rows)
    summ = pairs.groupby("kind")[[f"absdiff_{s}" for s in stats]].mean()
    summ.insert(0, "n_pairs", pairs.groupby("kind").size())
    return pairs[pairs["kind"].str.contains("twin")], summ.reset_index()


def stat_reliability(data_dir: Path, split: str, seasons: pd.DataFrame, seed: int = 372) -> pd.DataFrame:
    """Split-half reliability of each stat across pitcher-seasons with >= 300 pitches.

    Each pitch goes to a random half; the stat is computed on both halves and correlated across
    pitchers, then stepped up to the full season (Spearman-Brown: 2r / (1 + r)). A stat that is
    mostly noise at the pitcher level caps how well *any* comps can predict it.
    """
    from src.features.build_arsenal import KEYS as BKEYS, outcome_stats
    cols = BKEYS + ["events", "description", "zone", "bb_type", "label7", "delta_run_exp"]
    p = pd.read_parquet(data_dir / "pitches.parquet", columns=cols, filters=[("split", "==", split)])
    p["half"] = np.random.default_rng(seed).integers(0, 2, len(p))
    ok = seasons[seasons["n_pitches"] >= I.MIN_PITCHES][BKEYS]
    halves = []
    for h in [0, 1]:
        q = p[p["half"] == h]
        st_ = outcome_stats(q)
        rv = q.groupby(BKEYS, observed=True)["delta_run_exp"].mean().mul(-100).rename("pitcher_rv100").reset_index()
        halves.append(ok.merge(st_, on=BKEYS).merge(rv, on=BKEYS))
    m = halves[0].merge(halves[1], on=BKEYS, suffixes=("_a", "_b"))
    rows = []
    for stat in STATS:
        r = float(m[[f"{stat}_a", f"{stat}_b"]].corr().iloc[0, 1])
        rows.append({"stat": stat, "n_pitcher_seasons": len(m), "split_half_r": r, "reliability": 2 * r / (1 + r),
                     "max_r_vs_truth": float(np.sqrt(max(2 * r / (1 + r), 0.0)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures (reference palette: slots 1-3 for the three published/learned methods, gray for raw)
# ---------------------------------------------------------------------------
COLORS = {"tjstats": "#2a78d6", "seam": "#eb6834", "learned": "#1baf7a", "hzb": "#eda100", "raw": "#8a8984"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _axes_style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)


def plot_alpha(sweep: pd.DataFrame, alpha_star: float, path: Path, season: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    _axes_style(ax)
    ax.fill_between(sweep["alpha"], sweep["r_lo"], sweep["r_hi"], color=COLORS["learned"], alpha=0.15, linewidth=0)
    ax.plot(sweep["alpha"], sweep["r"], color=COLORS["learned"], linewidth=2, marker="o", markersize=5)
    star = sweep[np.isclose(sweep["alpha"], alpha_star)].iloc[0]
    ax.plot([alpha_star], [star["r"]], marker="o", markersize=10, color=COLORS["learned"],
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"alpha* = {alpha_star:.1f}", (alpha_star, star["r"]), xytext=(0, 12), textcoords="offset points",
                ha="center", color=INK_2, fontsize=9)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0\nTJStats", "0.2", "0.4", "0.6", "0.8", "1\nlearned"])
    ax.set_xlabel("Blend weight on learned similarity (alpha)", color=INK_2)
    ax.set_ylabel("r: comps' RV/100 vs. his", color=INK_2)
    ax.set_title(f"Retrieval quality along the blend, {season} (95% paired-bootstrap band)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[save] {path}")


def plot_stats(breakdown: pd.DataFrame, path: Path, season: int) -> None:
    methods = BASE_METHODS
    stats = list(STATS)
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=SURFACE)
    _axes_style(ax)
    offsets = np.linspace(-0.3, 0.3, len(methods))
    for off, m in zip(offsets, methods):
        d = breakdown[breakdown["method"] == m].set_index("stat").reindex(stats)
        y = np.arange(len(stats)) + off
        ax.errorbar(d["r"], y, xerr=[d["r"] - d["r_lo"], d["r_hi"] - d["r"]], fmt="o", color=COLORS[m],
                    markersize=6, elinewidth=1.5, capsize=0, label=LABELS[m])
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(np.arange(len(stats)))
    ax.set_yticklabels([STATS[s] for s in stats], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("r: comps' mean vs. his actual value (95% CI)", color=INK_2)
    ax.set_title(f"Which outcomes do comps predict? {season}, same hand + role", color=INK, fontsize=11, loc="left", pad=28)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              borderaxespad=0.2, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[save] {path}")


def plot_reliability(bins: pd.DataFrame, summary: pd.DataFrame, path: Path, season: int) -> None:
    curves = list(bins["curve"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), facecolor=SURFACE)
    for ax, c in zip(axes.ravel(), curves):
        _axes_style(ax)
        d = bins[(bins["curve"] == c) & (bins["count"] >= MIN_BIN_COUNT)]
        lim = max(0.05, float(max(d["mean_pred"].max(), d["obs_freq"].max())) * 1.1)
        ax.plot([0, lim], [0, lim], color=INK_2, linewidth=1, linestyle="--")
        ax.plot(d["mean_pred"], d["obs_freq"], color=COLORS["tjstats"], linewidth=2, marker="o", markersize=5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        e = summary.loc[summary["curve"] == c, "ece"].iloc[0]
        ax.set_title(f"{c}  (ECE {e:.3f})", color=INK, fontsize=10, loc="left")
    for ax in axes.ravel()[len(curves):]:
        ax.set_visible(False)
    fig.supxlabel("Predicted probability (bin mean)", color=INK_2)
    fig.supylabel("Observed frequency", color=INK_2)
    fig.suptitle(f"Outcome-head calibration, {season} pitches (dashed = perfect)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[save] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Held-out retrieval eval, bootstrap, slices, calibration.")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--sample", action="store_true", help="June data under data/processed/sample/")
    ap.add_argument("--skip-calibration", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    paths = Paths.make(args.sample)
    year = SEASON_OF[args.split]
    res, fig = paths.results, paths.figures
    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    star_path = res / "alpha_star.json"
    if args.split == "test":
        # why: 2026 is scored once (D5). Check that val's choices exist before anything on 2026 is computed.
        choices = json.loads(star_path.read_text()) if star_path.exists() else {}
        if not {"alpha_star", "headline_baseline"} <= set(choices):
            raise SystemExit(f"{star_path} lacks alpha_star/headline_baseline: run --split val first (D28).")

    seasons = pd.read_parquet(paths.data / "pitcher_seasons.parquet")
    by_type = pd.read_parquet(paths.data / "arsenal_pitch_type.parquet")
    embs = {"learned": pd.read_parquet(paths.models / "embeddings_t1.parquet")}
    if (paths.models / "embeddings_t3a.parquet").exists():
        embs["learned_t3a"] = pd.read_parquet(paths.models / "embeddings_t3a.parquet")
    ps = season_table(seasons, year)
    print(f"[eval] {args.split} ({year}): {len(ps):,} pitcher-seasons, {int(eligible(ps).sum()):,} with >= {I.MIN_PITCHES} pitches; "
          f"roles {ps[eligible(ps)].groupby(['p_throws', 'role']).size().to_dict()}")

    sims = similarity_suite(ps, by_type, load_pitches(paths.data, year), embs, year)
    errors = all_errors(ps, sims)
    errors.to_csv(res / f"errors_{args.split}.csv", index=False)
    print(f"[eval] errors: {len(errors):,} rows ({errors['method'].nunique()} methods x {len(STATS)} stats)")

    # alpha and the headline baseline: chosen on 2025 only (D5, D28)
    blends = [f"blend_{a:.1f}" for a in ALPHAS]
    sweep = compare(errors, HEADLINE, blends, "blend_0.0")
    sweep.insert(0, "alpha", ALPHAS)
    if args.split == "val":
        alpha_star = float(sweep.sort_values("r", ascending=False).iloc[0]["alpha"])
        pub = compare(errors, HEADLINE, PUBLISHED, "tjstats")
        # why (D28): the headline compares against the best published method, not the one we happen to
        # blend with, so a win can't come from picking a weak opponent. Chosen on 2025, frozen for 2026.
        ref = str(pub.sort_values("r", ascending=False).iloc[0]["method"])
        star_path.write_text(json.dumps({"alpha_star": alpha_star, "headline_baseline": ref,
                                         "chosen_on": "2025 (val)"}) + "\n")
    else:
        alpha_star, ref = float(choices["alpha_star"]), str(choices["headline_baseline"])
    sweep.to_csv(res / f"alpha_sweep_{args.split}.csv", index=False)
    plot_alpha(sweep, alpha_star, fig / f"alpha_sweep_{args.split}.png", year)

    methods = BASE_METHODS + (["learned_t3a"] if "learned_t3a" in sims else []) + [f"blend_{alpha_star:.1f}"]
    comp = compare(errors, HEADLINE, methods, ref)
    comp.to_csv(res / f"method_comparison_{args.split}.csv", index=False)
    print(f"\n[eval] RV/100, {year} (alpha* = {alpha_star:.1f}; deltas vs. {LABELS[ref]}, the strongest published baseline):\n"
          f"{comp[['method', 'r', 'r_lo', 'r_hi', 'mae', 'delta_r', 'delta_r_lo', 'delta_r_hi']].round(3).to_string(index=False)}")

    deltas = pd.concat([compare(errors, HEADLINE, methods, b) for b in PUBLISHED], ignore_index=True)
    deltas = deltas[deltas["method"] != deltas["reference"]]
    deltas.to_csv(res / f"baseline_deltas_{args.split}.csv", index=False)
    learned_rows = deltas[deltas["method"] == "learned"]
    print(f"\n[eval] learned (T1) vs. each published baseline, RV/100:\n"
          f"{learned_rows[['reference', 'delta_r', 'delta_r_lo', 'delta_r_hi', 'delta_mae', 'delta_mae_lo', 'delta_mae_hi']].round(3).to_string(index=False)}")

    breakdown = pd.concat([compare(errors, s, BASE_METHODS, ref) for s in STATS], ignore_index=True)
    breakdown.to_csv(res / f"stat_breakdown_{args.split}.csv", index=False)
    plot_stats(breakdown, fig / f"stat_breakdown_{args.split}.png", year)
    print(f"\n[eval] per-stat r:\n{breakdown.pivot_table(index='stat', columns='method', values='r').reindex(list(STATS)).round(3).to_string()}")

    rel = stat_reliability(paths.data, args.split, seasons)
    rel.to_csv(res / f"stat_reliability_{args.split}.csv", index=False)
    print(f"\n[eval] split-half reliability ({year}):\n{rel.round(3).to_string(index=False)}")

    train_pitchers = set(seasons.loc[seasons["split"] == "train", "pitcher"])
    sl = slices(errors, ps, train_pitchers, [m for m in methods if not m.startswith("blend")], ref)
    sl.to_csv(res / f"slices_{args.split}.csv", index=False)
    print(f"\n[eval] slices (RV/100 r):\n{sl.pivot_table(index='slice', columns='method', values='r').round(3).to_string()}")

    pairs, summ = twins(ps, sims)
    pairs.to_csv(res / f"twins_{args.split}.csv", index=False)
    summ.to_csv(res / f"twins_summary_{args.split}.csv", index=False)
    print(f"\n[eval] twin pairs, mean |difference| between query and comp:\n"
          f"{summ[['kind', 'n_pairs', 'absdiff_pitcher_rv100', 'absdiff_k_pct', 'absdiff_damage_pct']].round(3).to_string(index=False)}")

    if not args.skip_calibration:
        model, ckpt = load_model(paths.models / "arsenal_net_t1.pt")
        pitches = pd.read_parquet(paths.data / "pitches.parquet", columns=PITCH_COLS, filters=[("split", "==", args.split)])
        tier = I.TIERS[ckpt["tier"]]
        train_y = I.pitch_labels(pd.read_parquet(paths.data / "pitches.parquet", columns=[tier.label],
                                                 filters=[("split", "==", "train")]), tier)
        prior = np.bincount(train_y[train_y >= 0], minlength=len(tier.classes)) / (train_y >= 0).sum()
        probs, y = predict_proba(model, ckpt, pitches, by_type, seasons)
        summary, bins = calibration_report(probs, y, tier.classes, prior)
        summary.to_csv(res / f"calibration_{args.split}.csv", index=False)
        bins.to_csv(res / f"reliability_bins_{args.split}.csv", index=False)
        plot_reliability(bins, summary, fig / f"reliability_{args.split}.png", year)
        print(f"\n[eval] calibration on {len(y):,} pitches:\n{summary.round(4).to_string(index=False)}")
    print(f"[done] {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
