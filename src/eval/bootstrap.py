"""
Paired bootstrap confidence intervals for retrieval results (D11).

Resample the *query pitchers* with replacement, B times, using the same resampled indices for
every method. Each method's r and MAE are recomputed on each resample, and so is the difference
between methods. Pairing cancels what's shared across methods (how hard a given pitcher is to
predict), so small consistent differences become visible.

Why pitchers, not pitches: the held-out pitcher is the unit of the test, and one pitcher's pitches
are correlated, so resampling pitches would make the intervals look narrower than they are (D11).

These intervals cover query-sampling uncertainty (which pitchers happened to be tested), not
uncertainty in the index or in training.

Pipeline: used by stage 4 (evaluate). Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

Inputs:  per-query predictions for several methods, aligned to the same queries, plus actual values
Outputs: one row per method (r, MAE, 95% CIs) and per method vs. a reference (difference CIs)

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

B_DEFAULT = 2000
SEED = 372


def resample_indices(n: int, b: int = B_DEFAULT, seed: int = SEED) -> np.ndarray:
    """[b, n] query indices drawn with replacement; shared by every method (the pairing)."""
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def rowwise_r(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Pearson r of each row pair ([b, n] x [b, n] -> [b])."""
    p = pred - pred.mean(axis=1, keepdims=True)
    a = actual - actual.mean(axis=1, keepdims=True)
    return (p * a).sum(axis=1) / np.sqrt((p ** 2).sum(axis=1) * (a ** 2).sum(axis=1))


def paired_bootstrap(preds: Dict[str, np.ndarray], actual: np.ndarray, reference: str,
                     b: int = B_DEFAULT, seed: int = SEED) -> pd.DataFrame:
    """r and MAE with 95% CIs per method, plus paired differences vs. `reference`.

    Differences are (method - reference): positive delta_r and negative delta_mae favor the method.
    """
    idx = resample_indices(len(actual), b, seed)
    a = actual[idx]
    boot_r, boot_mae = {}, {}
    for name, p in preds.items():
        boot_r[name] = rowwise_r(p[idx], a)
        boot_mae[name] = np.abs(p[idx] - a).mean(axis=1)

    # Percentile bootstrap: each 95% CI is the 2.5th to 97.5th percentile of the B resampled values.
    rows = []
    for name, p in preds.items():
        dr = boot_r[name] - boot_r[reference]
        dm = boot_mae[name] - boot_mae[reference]
        rows.append({
            "method": name, "n_queries": len(actual),
            "r": float(np.corrcoef(p, actual)[0, 1]),
            "r_lo": float(np.percentile(boot_r[name], 2.5)), "r_hi": float(np.percentile(boot_r[name], 97.5)),
            "mae": float(np.abs(p - actual).mean()),
            "mae_lo": float(np.percentile(boot_mae[name], 2.5)), "mae_hi": float(np.percentile(boot_mae[name], 97.5)),
            "reference": reference,
            "delta_r": float(np.corrcoef(p, actual)[0, 1] - np.corrcoef(preds[reference], actual)[0, 1]),
            "delta_r_lo": float(np.percentile(dr, 2.5)), "delta_r_hi": float(np.percentile(dr, 97.5)),
            "delta_mae": float(np.abs(p - actual).mean() - np.abs(preds[reference] - actual).mean()),
            "delta_mae_lo": float(np.percentile(dm, 2.5)), "delta_mae_hi": float(np.percentile(dm, 97.5)),
        })
    return pd.DataFrame(rows)
