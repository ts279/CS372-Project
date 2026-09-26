"""
Train the arsenal embedding, run the hyperparameter sweep, and train the ablation models.

Pipeline stage 3 of 7 (train). Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

Inputs:  data/processed/{pitches,arsenal_pitch_type,pitcher_seasons}.parquet (src/features/build_arsenal.py)
Outputs (full run; --sample writes everything under models/sample/ instead):
    models/sweep/<run>/model.pt, embeddings.parquet     every run (not committed)
    models/arsenal_net_{t1,t3a}.pt, models/embeddings_{t1,t3a}.parquet, models/published.json
                                        the shipped models (committed; the eval and the app load them)
    docs/results/sweep.csv, sweep_summary.csv          per run / per config (mean +- SD over 3 seeds)
    docs/results/ablation.csv, ablation_summary.csv    T1 vs. T3a x aux target (off / profile / profile + RV/100)
    docs/results/training_curves.csv    per-epoch losses and 2025 retrieval r for every run
    docs/figures/training_curves.png    the sweep's curves (seed 372)

Selection (D21): each run keeps the epoch with the best 3-epoch rolling mean of 2025 retrieval r
(training itself ends on pitch log loss). Configs are compared on the mean over 3 seeds. The shipped
model of a config is its seed-372 run, fixed in advance. Configs whose mean r is within one seed-SD of
the top config's count as tied, and the lowest mean 2025 retrieval MAE breaks the tie (D25).

Data use (D5): the weights learn from 2023-24 pitches only. Early stopping, the learning-rate
schedule, and model selection use 2025 only. 2026 arsenals are embedded at the end (inference on
inputs only) so the eval can score them once; no 2026 outcome is used here.

Usage (from the repo root):
    python -m src.models.train_embedding --sample --mode all   # smoke test on June data, 2 epochs
    python -m src.models.train_embedding --mode sweep          # 3 configs, select on 2025
    python -m src.models.train_embedding --mode ablation       # T1 vs T3a x aux on/off (needs the sweep)
    python -m src.models.train_embedding --mode all            # both
    python -m src.models.train_embedding --mode publish        # re-select + re-publish from saved runs, no training

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

from src.eval.retrieval import embedding_similarity, headline, knn_errors
from src.models import inputs as I
from src.models.arsenal_net import ArsenalNet, aux_loss

ROOT = Path(__file__).resolve().parents[2]
SEED = 372
PITCH_COLS = (I.KEYS + list(I.SHAPE_T1) + ["plate_x", "plate_z", "sz_top", "sz_bot", "stand",
                                           "balls", "strikes", "family", "attack_zone", "label7", "label5"])


AUX_TARGETS = ("off", "profile", "profile_rv")
# why (D21): MPS kernels on the aux path aren't deterministic (one seed gave 2025 r 0.174 and 0.212 on two
# runs), so configs are compared on the mean of 3 seeds; the shipped model is the seed-372 run, fixed in advance.
SEEDS = (372, 373, 374)


@dataclass(frozen=True)
class RunConfig:
    """One training run. The sweep varies d_emb; the ablation varies tier and the aux target."""
    name: str
    tier: str = "t1"
    aux: str = "profile_rv"   # off / profile (outcome mix) / profile_rv (outcome mix + RV/100), D10
    seed: int = SEED
    d_emb: int = 16
    d_hidden: int = 64
    d_head: int = 128
    dropout: float = 0.1
    lam: float = 1.0
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    max_epochs: int = 30
    patience: int = 3


# why (D21): the sweep varies the size of the retrieval space (embedding dimension), the one knob that
# directly shapes what "near" means. The aux target is held at the author's spec (profile + RV/100)
# because aux variants are the ablation, not a tuning knob, so the two experiments stay separate.
SWEEP_DIMS = (8, 16, 32)


def config_name(cfg: RunConfig) -> str:
    """Name of a configuration without its seed, e.g. t1_emb16_profile_rv."""
    return f"{cfg.tier}_emb{cfg.d_emb}_{cfg.aux}"


def make_run(tier: str, d_emb: int, aux: str, seed: int, max_epochs: Optional[int]) -> RunConfig:
    cfg = RunConfig(name="", tier=tier, d_emb=d_emb, aux=aux, seed=seed)
    cfg = dataclasses.replace(cfg, name=f"{config_name(cfg)}_s{seed}")
    return dataclasses.replace(cfg, max_epochs=max_epochs) if max_epochs else cfg


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")
    return device


@dataclass
class Paths:
    data: Path
    models: Path
    results: Path
    figures: Path

    @classmethod
    def make(cls, sample: bool) -> "Paths":
        if sample:
            root = ROOT / "models" / "sample"
            return cls(ROOT / "data" / "processed" / "sample", root, root / "results", root / "figures")
        return cls(ROOT / "data" / "processed", ROOT / "models", ROOT / "docs" / "results", ROOT / "docs" / "figures")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Prepared:
    """Everything one tier needs, as arrays. Pitch and arsenal rows are split into train / val."""
    tier: I.Tier
    seasons: pd.DataFrame                    # all pitcher-seasons (row order = arsenal arrays)
    ars_x: np.ndarray                        # [n_seasons, max_types, d_arsenal]
    ars_w: np.ndarray                        # [n_seasons, max_types]
    split_rows: Dict[str, np.ndarray]        # season rows per split
    pitch_x: Dict[str, np.ndarray]           # per split: [n_pitches, d_pitch]
    pitch_y: Dict[str, np.ndarray]           # per split: class index
    pitch_ps: Dict[str, np.ndarray]          # per split: row within that split's season rows
    profile: np.ndarray
    rv_z: np.ndarray
    aux_ok: np.ndarray
    pitch_scaler: I.Scaler
    arsenal_scaler: I.Scaler
    rv_mean: float
    rv_std: float


def load_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train + validation pitches (2026 pitches are never loaded here), all arsenals, all pitcher-seasons."""
    pitches = pd.read_parquet(data_dir / "pitches.parquet", columns=PITCH_COLS,
                              filters=[("split", "in", ["train", "val"])])
    by_type = pd.read_parquet(data_dir / "arsenal_pitch_type.parquet")
    seasons = pd.read_parquet(data_dir / "pitcher_seasons.parquet").sort_values(I.KEYS).reset_index(drop=True)
    print(f"[data] pitches (train + val) {pitches.shape}; arsenal rows {by_type.shape}; pitcher-seasons {seasons.shape}")
    return pitches, by_type, seasons


def prepare(tier: I.Tier, pitches: pd.DataFrame, by_type: pd.DataFrame, seasons: pd.DataFrame) -> Prepared:
    """Scale inputs with training-season statistics and build the arrays for one tier."""
    train_p = pitches[pitches["split"] == "train"]
    pitch_scaler = I.Scaler.fit(I.pitch_continuous(train_p, tier))
    train_types = by_type[(by_type["split"] == "train") & by_type["in_arsenal"]]
    arsenal_scaler = I.Scaler.fit(I.arsenal_continuous(train_types, tier))
    ars_x, ars_w = I.arsenal_sets(by_type, seasons, tier, arsenal_scaler)

    rv_mean, rv_std = I.rv_stats(seasons)
    profile, rv_z, aux_ok = I.aux_targets(seasons, tier, rv_mean, rv_std)
    aux_ok = aux_ok & seasons["split"].isin(["train", "val"]).to_numpy()  # never a 2026 outcome

    row_of = pd.Series(np.arange(len(seasons)), index=pd.MultiIndex.from_frame(seasons[I.KEYS]))
    split_rows, px, py, pps = {}, {}, {}, {}
    for split in ["train", "val"]:
        rows = np.flatnonzero(seasons["split"].to_numpy() == split)
        local = np.full(len(seasons), -1)
        local[rows] = np.arange(len(rows))
        p = pitches[pitches["split"] == split]
        y = I.pitch_labels(p, tier)
        keep = y >= 0  # unlabeled pitches (in play with no contact code) are left out of the loss
        p = p[keep]
        glob = row_of.reindex(pd.MultiIndex.from_frame(p[I.KEYS])).to_numpy()
        split_rows[split] = rows
        px[split] = I.pitch_matrix(p, tier, pitch_scaler)
        py[split] = y[keep]
        pps[split] = local[glob]
        print(f"[data] {tier.name} {split}: {len(p):,} labeled pitches x {px[split].shape[1]} features; "
              f"{len(rows):,} pitcher-seasons, {int(aux_ok[rows].sum()):,} with >= {I.MIN_PITCHES} pitches")
    print(f"[data] {tier.name} arsenal sets {ars_x.shape}; RV/100 train mean {rv_mean:.2f}, SD {rv_std:.2f}")
    return Prepared(tier, seasons, ars_x, ars_w, split_rows, px, py, pps, profile, rv_z, aux_ok,
                    pitch_scaler, arsenal_scaler, rv_mean, rv_std)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class SplitTensors:
    """One split's arsenal and auxiliary tensors, on the device."""

    def __init__(self, prep: Prepared, split: str, device: torch.device):
        rows = prep.split_rows[split]
        self.ars_x = torch.from_numpy(prep.ars_x[rows]).to(device)
        self.ars_w = torch.from_numpy(prep.ars_w[rows]).to(device)
        ok = prep.aux_ok[rows]
        self.aux_idx = torch.from_numpy(np.flatnonzero(ok)).to(device)
        self.profile = torch.from_numpy(prep.profile[rows][ok]).to(device)
        self.rv_z = torch.from_numpy(prep.rv_z[rows][ok]).to(device)


@torch.no_grad()
def evaluate(model: ArsenalNet, prep: Prepared, val: SplitTensors, device: torch.device,
             chunk: int = 65536) -> Dict[str, float]:
    """Validation pitch log loss and auxiliary losses (2025)."""
    model.eval()
    emb = model.embed(val.ars_x, val.ars_w)
    x, y, ps = prep.pitch_x["val"], prep.pitch_y["val"], prep.pitch_ps["val"]
    total = 0.0
    for s in range(0, len(y), chunk):
        xb = torch.from_numpy(x[s:s + chunk]).to(device)
        yb = torch.from_numpy(y[s:s + chunk]).to(device)
        pb = torch.from_numpy(ps[s:s + chunk]).to(device)
        total += F.cross_entropy(model(xb, emb[pb]), yb, reduction="sum").item()
    prof, rv = aux_loss(model, emb[val.aux_idx], val.profile, val.rv_z)
    # 2025 retrieval r (headline: RV/100, same hand + role), tracked every epoch; picks the checkpoint.
    ps = prep.seasons.iloc[prep.split_rows["val"]].reset_index(drop=True)
    r = headline(knn_errors(ps, embedding_similarity(emb.cpu().numpy().astype("float64"))))["r"]
    return {"val_pitch_ce": total / len(y), "val_profile_ce": prof.item(), "val_rv_mse": rv.item(), "val_retrieval_r": r}


def train_run(cfg: RunConfig, prep: Prepared, device: torch.device) -> Tuple[ArsenalNet, Dict[str, float], List[Dict]]:
    """Train one run; end on 2025 pitch log loss; keep the epoch with the best rolling 2025 retrieval r."""
    set_seed(cfg.seed)
    tier = prep.tier
    model = ArsenalNet(prep.ars_x.shape[2], prep.pitch_x["train"].shape[1], len(tier.classes),
                       cfg.d_emb, cfg.d_hidden, cfg.d_head, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # why (D21): halve the learning rate when validation loss stalls, before early stopping gives up.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    train, val = SplitTensors(prep, "train", device), SplitTensors(prep, "val", device)

    ds = TensorDataset(torch.from_numpy(prep.pitch_x["train"]), torch.from_numpy(prep.pitch_y["train"]),
                       torch.from_numpy(prep.pitch_ps["train"]))
    gen = torch.Generator().manual_seed(cfg.seed)
    # Batches of shuffled row indices; TensorDataset slices all three tensors at once.
    loader = DataLoader(ds, sampler=BatchSampler(RandomSampler(ds, generator=gen), cfg.batch_size, drop_last=False),
                        batch_size=None)

    best_ce, ce_epoch, bad = float("inf"), 0, 0
    states: List[Dict[str, torch.Tensor]] = []   # every epoch's weights (small model), for checkpoint selection
    curves: List[Dict] = []
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        sums = {"pitch_ce": 0.0, "aux": 0.0, "n": 0}
        te = time.time()
        for xb, yb, pb in loader:
            xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
            # All training arsenals are embedded every step (about 1,600 sets: cheap), so every
            # pitcher-season with enough pitches gets the same weight in the auxiliary loss.
            emb = model.embed(train.ars_x, train.ars_w)
            # why: plain (unweighted) cross-entropy. Class weights would distort the probabilities,
            # and expected run value multiplies those probabilities directly (D12).
            loss_p = F.cross_entropy(model(xb, emb[pb]), yb)
            loss = loss_p
            if cfg.aux != "off":
                prof, rv = aux_loss(model, emb[train.aux_idx], train.profile, train.rv_z)
                # why (D10, D25): profile_rv adds the RV/100 error. The shipped models use profile only:
                # RV/100 is too noisy a target (reliability 0.33), and it pulled retrieval r down.
                loss_a = prof + rv if cfg.aux == "profile_rv" else prof
                loss = loss_p + cfg.lam * loss_a
                sums["aux"] += loss_a.item() * len(yb)
            opt.zero_grad()
            loss.backward()
            # why: gradient clipping caps each step's size, so one noisy batch can't throw the weights far
            # (D21 counts it among this model's regularization).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sums["pitch_ce"] += loss_p.item() * len(yb)
            sums["n"] += len(yb)

        v = evaluate(model, prep, val, device)
        # why (D21): stop and schedule on the pitch log loss alone. It averages ~700k validation pitches,
        # while the RV/100 error averages ~500 pitcher-seasons and swings from noise. The same
        # criterion for every run also keeps aux on/off comparable.
        val_total = v["val_pitch_ce"]
        sched.step(val_total)
        row = {"run": cfg.name, "epoch": epoch, "train_pitch_ce": sums["pitch_ce"] / sums["n"],
               "train_aux": sums["aux"] / sums["n"] if cfg.aux != "off" else np.nan, **v,
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - te}
        curves.append(row)
        print(f"[{cfg.name}] epoch {epoch:>2}  train CE {row['train_pitch_ce']:.4f}  val CE {v['val_pitch_ce']:.4f}  "
              f"val profile {v['val_profile_ce']:.4f}  val RV mse {v['val_rv_mse']:.3f}  val r {v['val_retrieval_r']:.3f}  "
              f"lr {row['lr']:.1e}  {row['seconds']:.0f}s")

        states.append({k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
        # Early stopping on pitch log loss decides when training *ends*.
        if val_total < best_ce - 1e-4:
            best_ce, ce_epoch, bad = val_total, epoch, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[{cfg.name}] early stop: pitch log loss flat for {cfg.patience} epochs (best epoch {ce_epoch})")
                break

    # why: which epoch to *keep* is chosen on the project's objective. Retrieval r peaks early and
    # then drifts down while pitch log loss keeps improving (D21). A 3-epoch rolling mean damps
    # epoch-to-epoch noise in r (~450 validation pitchers).
    r = pd.Series([c["val_retrieval_r"] for c in curves]).rolling(3, center=True, min_periods=1).mean()
    best_epoch = int(r.idxmax()) + 1
    model.load_state_dict(states[best_epoch - 1])
    best_row = curves[best_epoch - 1]
    print(f"[{cfg.name}] kept epoch {best_epoch} (best rolling 2025 retrieval r {r.max():.3f}); "
          f"pitch log loss was best at epoch {ce_epoch}")
    metrics = {"best_epoch": best_epoch, "ce_best_epoch": ce_epoch, "epochs_run": len(curves),
               "train_minutes": (time.time() - t0) / 60,
               "val_pitch_logloss": best_row["val_pitch_ce"],
               "val_profile_ce": best_row["val_profile_ce"] if cfg.aux != "off" else np.nan,
               "val_rv_mse": best_row["val_rv_mse"] if cfg.aux == "profile_rv" else np.nan}
    return model, metrics, curves


# ---------------------------------------------------------------------------
# After training: embeddings, validation retrieval, checkpoint
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_all(model: ArsenalNet, prep: Prepared, device: torch.device) -> pd.DataFrame:
    """Embedding of every pitcher-season (inputs only; 2026 included so the eval can score it once)."""
    model.eval()
    emb = model.embed(torch.from_numpy(prep.ars_x).to(device), torch.from_numpy(prep.ars_w).to(device)).cpu().numpy()
    out = prep.seasons[I.KEYS + ["player_name", "n_pitches"]].copy()
    for j in range(emb.shape[1]):
        out[f"e{j}"] = emb[:, j]
    return out


def validation_retrieval(prep: Prepared, emb: pd.DataFrame) -> Dict[str, float]:
    """Held-out-pitcher k-NN retrieval on 2025 in the learned space (src/eval/retrieval.py)."""
    rows = prep.split_rows["val"]
    ps = prep.seasons.iloc[rows].reset_index(drop=True)
    e = emb.iloc[rows][[c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]].to_numpy("float64")
    s = headline(knn_errors(ps, embedding_similarity(e)))
    return {"val_retrieval_r": s["r"], "val_retrieval_mae": s["mae"], "val_baseline_mae": s["baseline_mae"],
            "n_val_queries": int(s["n_queries"])}


def prior_logloss(prep: Prepared) -> float:
    """Baseline: predict the training class frequencies for every 2025 pitch."""
    freq = np.bincount(prep.pitch_y["train"], minlength=len(prep.tier.classes)) / len(prep.pitch_y["train"])
    return float(-np.log(freq[prep.pitch_y["val"]]).mean())


def save_run(model: ArsenalNet, cfg: RunConfig, prep: Prepared, emb: pd.DataFrame, out_dir: Path) -> None:
    """Checkpoint = weights + everything needed to encode a new query the same way."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": dataclasses.asdict(cfg),
        "tier": prep.tier.name,
        "classes": prep.tier.classes,
        "d_arsenal": int(prep.ars_x.shape[2]),
        "d_pitch": int(prep.pitch_x["train"].shape[1]),
        "pitch_scaler": prep.pitch_scaler.to_dict(),
        "arsenal_scaler": prep.arsenal_scaler.to_dict(),
        "rv_mean": prep.rv_mean, "rv_std": prep.rv_std,
        "min_pitches": I.MIN_PITCHES, "coarse_movement_in": I.COARSE_MOVEMENT_IN,
        "state_dict": {k: t.detach().cpu() for k, t in model.state_dict().items()},
    }, out_dir / "model.pt")
    emb.to_parquet(out_dir / "embeddings.parquet", index=False)


def load_model(path: Path) -> Tuple[ArsenalNet, Dict]:
    """Rebuild a trained ArsenalNet (CPU) and return it with its checkpoint metadata."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = ArsenalNet(ckpt["d_arsenal"], ckpt["d_pitch"], len(ckpt["classes"]), c["d_emb"], c["d_hidden"],
                       c["d_head"], c["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def run_and_record(cfg: RunConfig, prep: Prepared, device: torch.device, paths: Paths) -> Tuple[Dict, List[Dict]]:
    """Train, embed, score on 2025, save the run. Returns (metrics row, curves)."""
    print(f"\n[run] {cfg.name}: tier {cfg.tier}, d_emb {cfg.d_emb}, aux {cfg.aux}, seed {cfg.seed}, dropout {cfg.dropout}, lr {cfg.lr}")
    model, m, curves = train_run(cfg, prep, device)
    emb = embed_all(model, prep, device)
    m.update(validation_retrieval(prep, emb))
    m["val_prior_logloss"] = prior_logloss(prep)
    save_run(model, cfg, prep, emb, paths.models / "sweep" / cfg.name)
    row = {"run": cfg.name, "config": config_name(cfg), "tier": cfg.tier, "aux": cfg.aux, "seed": cfg.seed,
           "lam": cfg.lam, "d_emb": cfg.d_emb, "dropout": cfg.dropout, "lr": cfg.lr,
           "n_classes": len(prep.tier.classes), **m}
    print(f"[{cfg.name}] best epoch {m['best_epoch']}: val log loss {m['val_pitch_logloss']:.4f} "
          f"(class-prior baseline {m['val_prior_logloss']:.4f}); 2025 retrieval r {m['val_retrieval_r']:.3f}, "
          f"MAE {m['val_retrieval_mae']:.3f} (no-similarity baseline {m['val_baseline_mae']:.3f}); "
          f"{m['train_minutes']:.1f} min")
    return row, curves


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated as a set for any pairing)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_curves(curves: pd.DataFrame, runs: List[str], path: Path) -> None:
    """Three panels (training log loss, validation log loss, 2025 retrieval r), one line per sweep config."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=SURFACE)
    axes[1].sharey(axes[0])
    for ax, col, title in zip(axes, ["train_pitch_ce", "val_pitch_ce", "val_retrieval_r"],
                              ["Training log loss (2023-24)", "Validation log loss (2025)",
                               "Retrieval r, comps' RV/100 vs. his (2025)"]):
        ax.set_facecolor(SURFACE)
        for color, run in zip(SERIES, runs):
            c = curves[curves["run"] == run]
            ax.plot(c["epoch"], c[col], color=color, linewidth=2, label=run)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Epoch", color=INK_2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_2)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("Pitch outcome log loss (nats)", color=INK_2)
    axes[2].set_ylabel("Pearson r", color=INK_2)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[save] {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def write_curves(curves: List[Dict], paths: Paths) -> pd.DataFrame:
    """Append this invocation's curves to training_curves.csv (replacing rows of re-run configs)."""
    path = paths.results / "training_curves.csv"
    new = pd.DataFrame(curves)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["run"].isin(new["run"].unique())], new], ignore_index=True)
    paths.results.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    print(f"[save] {path.relative_to(ROOT)}  ({len(new)} rows)")
    return new


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over seeds, per configuration."""
    metrics = ["val_retrieval_r", "val_retrieval_mae", "val_baseline_mae", "val_pitch_logloss", "val_rv_mse", "best_epoch"]
    g = runs.groupby(["config", "tier", "aux", "d_emb"], sort=False)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{a}" for m, a in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def select_config(summary: pd.DataFrame) -> str:
    """Top mean 2025 retrieval r; configs within one seed-SD of the top are tied, lowest mean MAE wins (D25)."""
    # why: r gaps smaller than seed-to-seed noise are coin flips. MAE is the other retrieval
    # metric (D20), scored on the same 2025 queries, so it breaks the tie without new data.
    top = summary.sort_values("val_retrieval_r_mean", ascending=False).iloc[0]
    sd = 0.0 if pd.isna(top["val_retrieval_r_std"]) else float(top["val_retrieval_r_std"])
    tied = summary[summary["val_retrieval_r_mean"] >= top["val_retrieval_r_mean"] - sd]
    best = tied.sort_values("val_retrieval_mae_mean").iloc[0]["config"]
    print(f"[select] top r: {top['config']} ({top['val_retrieval_r_mean']:.4f} ± {sd:.4f}); "
          f"tied: {', '.join(tied['config'])}; lowest MAE -> {best}")
    return best


def select_by_tier(summary: pd.DataFrame) -> Dict[str, str]:
    """The selected config of each tier in a summary table."""
    return {tier: select_config(g) for tier, g in summary.groupby("tier", sort=False)}


def publish(runs: pd.DataFrame, config: str, tier: str, paths: Paths) -> str:
    """Copy the seed-372 run of the selected config to paths.models as the tier's shipped model."""
    run = runs[(runs["config"] == config) & (runs["seed"] == SEEDS[0])].iloc[0]["run"]
    src = paths.models / "sweep" / run
    dest = paths.models / f"arsenal_net_{tier}.pt"
    shutil.copyfile(src / "model.pt", dest)
    shutil.copyfile(src / "embeddings.parquet", paths.models / f"embeddings_{tier}.parquet")
    print(f"[publish] {tier}: {config} (seed {SEEDS[0]} run {run}) -> {dest.relative_to(ROOT)}")
    return run


def write_published(published: Dict[str, str], paths: Paths) -> None:
    path = paths.models / "published.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(published)
    path.write_text(json.dumps(old, indent=2) + "\n")


def run_sweep(tables, device: torch.device, paths: Paths, max_epochs: Optional[int]) -> pd.DataFrame:
    """Embedding size x 3 seeds at T1; compare configs on mean 2025 retrieval r; publish the winner."""
    prep = prepare(I.TIERS["t1"], *tables)
    rows, curves = [], []
    for d in SWEEP_DIMS:
        for seed in SEEDS:
            row, c = run_and_record(make_run("t1", d, "profile_rv", seed, max_epochs), prep, device, paths)
            rows.append(row)
            curves += c
    runs = pd.DataFrame(rows)
    summary = summarize_runs(runs)
    best = select_config(summary)
    summary["selected"] = summary["config"] == best
    paths.results.mkdir(parents=True, exist_ok=True)
    runs.to_csv(paths.results / "sweep.csv", index=False)
    summary.to_csv(paths.results / "sweep_summary.csv", index=False)
    print(f"\n[sweep] selected {best} (mean 2025 retrieval r over {len(SEEDS)} seeds; ties broken by MAE, D25)\n"
          f"{summary[['config', 'val_retrieval_r_mean', 'val_retrieval_r_std', 'val_retrieval_mae_mean', 'val_pitch_logloss_mean']].to_string(index=False)}")
    all_curves = write_curves(curves, paths)
    plot_curves(all_curves, [r for r in runs["run"] if r.endswith(f"_s{SEEDS[0]}")], paths.figures / "training_curves.png")
    write_published({"t1": publish(runs, best, "t1", paths)}, paths)
    return summary


def run_ablation(tables, device: torch.device, paths: Paths, max_epochs: Optional[int]) -> pd.DataFrame:
    """T1 vs. T3a x aux target (off / profile / profile + RV/100), 3 seeds each, at the sweep's size (D6, D10)."""
    sweep_runs = pd.read_csv(paths.results / "sweep.csv")
    sel = pd.read_csv(paths.results / "sweep_summary.csv").query("selected").iloc[0]
    d_emb = int(sel["d_emb"])
    reused = sweep_runs[sweep_runs["config"] == sel["config"]]      # T1 + profile_rv already trained
    rows: List[Dict] = reused.to_dict("records")
    curves: List[Dict] = []
    for tier in ["t1", "t3a"]:
        prep = prepare(I.TIERS[tier], *tables)
        for aux in AUX_TARGETS:
            if tier == "t1" and aux == "profile_rv":
                continue
            for seed in SEEDS:
                row, c = run_and_record(make_run(tier, d_emb, aux, seed, max_epochs), prep, device, paths)
                rows.append(row)
                curves += c
    runs = pd.DataFrame(rows)
    runs.to_csv(paths.results / "ablation.csv", index=False)
    write_curves(curves, paths)
    return finish_ablation(runs, paths)


def finish_ablation(runs: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """Summarize the ablation runs, mark each tier's selected config, save the summary, publish both tiers."""
    summary = summarize_runs(runs)
    selected = select_by_tier(summary)
    summary["selected"] = summary["config"].isin(list(selected.values()))
    summary.to_csv(paths.results / "ablation_summary.csv", index=False)
    print(f"\n[ablation] mean over {len(SEEDS)} seeds\n"
          f"{summary[['config', 'val_retrieval_r_mean', 'val_retrieval_r_std', 'val_retrieval_mae_mean', 'val_pitch_logloss_mean', 'selected']].to_string(index=False)}")
    write_published({t: publish(runs, selected[t], t, paths) for t in ["t1", "t3a"]}, paths)
    return summary


def run_publish(paths: Paths) -> pd.DataFrame:
    """Re-select and re-publish from the saved runs (docs/results + models/sweep/); no training."""
    sweep = pd.read_csv(paths.results / "sweep_summary.csv")
    saved = sweep.loc[sweep["selected"], "config"].iloc[0]
    rule = select_config(sweep)
    print(f"[publish] sweep choice under D25: {rule} (saved: {saved})")
    if rule != saved:
        raise SystemExit("[publish] the sweep's pick changed under D25; rerun --mode all instead")
    runs = pd.read_csv(paths.results / "ablation.csv")
    print(f"[load] {(paths.results / 'ablation.csv').relative_to(ROOT)}  ({len(runs)} runs)")
    return finish_ablation(runs, paths)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the arsenal embedding: sweep and/or ablation.")
    ap.add_argument("--mode", choices=["sweep", "ablation", "all", "publish"], default="sweep",
                    help="publish: re-select and re-copy the shipped models from saved runs, no training")
    ap.add_argument("--sample", action="store_true", help="June data, 2 epochs, outputs under models/sample/")
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()

    paths = Paths.make(args.sample)
    t0 = time.time()
    if args.mode == "publish":
        run_publish(paths)
    else:
        device = pick_device()
        max_epochs = args.max_epochs or (2 if args.sample else None)
        tables = load_tables(paths.data)
        if args.mode in ("sweep", "all"):
            run_sweep(tables, device, paths, max_epochs)
        if args.mode in ("ablation", "all"):
            run_ablation(tables, device, paths, max_epochs)
    print(f"[done] {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
