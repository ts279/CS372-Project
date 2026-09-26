"""
Calibration of the pitch-outcome head (D12): do its probabilities mean what they say?

Expected run value multiplies each class probability by that class's run value, so a head that
is too confident (or too timid) biases it even when its top choice is right. Reported: per-class
reliability (binned predicted probability vs. observed frequency), expected calibration error
(ECE, per class and for the top choice), and log loss vs. a class-prior baseline. The temperature
fit was cut (D12); miscalibration is reported as a finding, not corrected.

Pipeline: used by stage 4 (evaluate). Stages: fetch -> features -> train -> evaluate -> comps report ->
app tables -> app.

Inputs:  a checkpoint (src/models/train_embedding.py), one split's pitches, the arsenal and
         pitcher-season tables
Outputs: per-pitch class probabilities; reliability bins and a summary table

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.models import inputs as I
from src.models.arsenal_net import ArsenalNet

N_BINS = 10
MIN_BIN_COUNT = 100  # bins with fewer pitches are too noisy to plot


@torch.no_grad()
def predict_proba(model: ArsenalNet, ckpt: Dict, pitches: pd.DataFrame, by_type: pd.DataFrame,
                  seasons: pd.DataFrame, chunk: int = 65536) -> Tuple[np.ndarray, np.ndarray]:
    """Class probabilities and labels for every labeled pitch, encoded exactly as in training."""
    tier = I.TIERS[ckpt["tier"]]
    ars_x, ars_w = I.arsenal_sets(by_type, seasons, tier, I.Scaler.from_dict(ckpt["arsenal_scaler"]))
    y = I.pitch_labels(pitches, tier)
    p = pitches[y >= 0]
    row_of = pd.Series(np.arange(len(seasons)), index=pd.MultiIndex.from_frame(seasons[I.KEYS]))
    ps_row = row_of.reindex(pd.MultiIndex.from_frame(p[I.KEYS])).to_numpy()
    x = I.pitch_matrix(p, tier, I.Scaler.from_dict(ckpt["pitch_scaler"]))

    model.eval()
    emb = model.embed(torch.from_numpy(ars_x), torch.from_numpy(ars_w))
    probs = []
    for s in range(0, len(x), chunk):
        logits = model(torch.from_numpy(x[s:s + chunk]), emb[torch.from_numpy(ps_row[s:s + chunk])])
        probs.append(torch.softmax(logits, dim=-1).numpy())
    return np.concatenate(probs), y[y >= 0]


def reliability_bins(prob: np.ndarray, hit: np.ndarray) -> pd.DataFrame:
    """Equal-count bins on predicted probability: mean prediction, observed frequency, count.

    why: equal-count (quantile) bins, not uniform ones. A rare class like damage (~2.5%) has almost
    every prediction below 0.1, so uniform bins would put nearly all pitches in one bin. The ECE
    over equal-count bins is the "adaptive" ECE.
    """
    b = pd.qcut(prob, q=N_BINS, labels=False, duplicates="drop")
    df = pd.DataFrame({"bin": b, "pred": prob, "hit": hit.astype(float)})
    g = df.groupby("bin").agg(mean_pred=("pred", "mean"), obs_freq=("hit", "mean"), count=("pred", "size"))
    return g.reset_index()


def ece(bins: pd.DataFrame) -> float:
    """Expected calibration error: count-weighted mean |observed - predicted| over bins."""
    w = bins["count"] / bins["count"].sum()
    return float((w * (bins["obs_freq"] - bins["mean_pred"]).abs()).sum())


def calibration_report(probs: np.ndarray, y: np.ndarray, classes: List[str],
                       prior: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per class + top-label + log loss, reliability bins per class)."""
    summary, all_bins = [], []
    for k, name in enumerate(classes):
        bins = reliability_bins(probs[:, k], y == k)
        bins.insert(0, "curve", name)
        all_bins.append(bins)
        summary.append({"curve": name, "ece": ece(bins), "mean_pred": float(probs[:, k].mean()),
                        "obs_freq": float((y == k).mean())})
    conf = probs.max(axis=1)
    bins = reliability_bins(conf, probs.argmax(axis=1) == y)
    bins.insert(0, "curve", "top choice")
    all_bins.append(bins)
    summary.append({"curve": "top choice", "ece": ece(bins), "mean_pred": float(conf.mean()),
                    "obs_freq": float((probs.argmax(axis=1) == y).mean())})
    eps = 1e-12
    summary.append({"curve": "log loss (model)", "ece": np.nan,
                    "mean_pred": float(-np.log(probs[np.arange(len(y)), y] + eps).mean()), "obs_freq": np.nan})
    summary.append({"curve": "log loss (class prior)", "ece": np.nan,
                    "mean_pred": float(-np.log(prior[y] + eps).mean()), "obs_freq": np.nan})
    return pd.DataFrame(summary), pd.concat(all_bins, ignore_index=True)
