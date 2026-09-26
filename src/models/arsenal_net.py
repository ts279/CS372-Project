"""
The Arsenal Match network: a set encoder that turns a pitcher-season's arsenal into an embedding,
a pitch-outcome head, and an auxiliary head that reads the embedding alone.

    arsenal (set of pitch types) --> SetEncoder --> embedding e  (the retrieval space)
    pitch features + e ------------> outcome head --> class logits      (per pitch)
    e -----------------------------> aux head -----> profile logits + RV/100 z-score (per pitcher-season)

Inputs:  tensors built by src/models/inputs.py
Outputs: logits and embeddings (training lives in src/models/train_embedding.py)

Design: set encoder = Deep Sets (Zaheer et al. 2017; D18). Auxiliary head = D10.

Pipeline: the model trained in stage 3 (train) and loaded by stage 4 (evaluate) and stage 7 (app).
Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SetEncoder(nn.Module):
    """Deep Sets: the same small network (phi) on every pitch type, a usage-weighted mean, then rho.

    why: an arsenal is a *set*. There's no natural order to a pitcher's pitches, and arsenals have
    2 to 9 types. Sharing phi across types and pooling makes the embedding identical under any
    ordering, and lets a pitch the pitcher throws 40% of the time count more than one he throws 5%.
    """

    def __init__(self, d_in: int, d_hidden: int, d_emb: int, dropout: float):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
        )
        self.rho = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_emb))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x [B, T, d_in] pitch types, w [B, T] usage weights (sum to 1; 0 on padding) -> [B, d_emb]."""
        h = self.phi(x)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        return self.rho(pooled)


class ArsenalNet(nn.Module):
    """Set encoder + pitch-outcome head + auxiliary head."""

    def __init__(self, d_arsenal: int, d_pitch: int, n_classes: int, d_emb: int = 16,
                 d_hidden: int = 64, d_head: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = SetEncoder(d_arsenal, d_hidden, d_emb, dropout)
        # why (D2): the embedding learns by helping predict each pitch's outcome class (a softmax over
        # 7 classes, D13). The head also sees the pitch itself: its shape, location, and count.
        self.head = nn.Sequential(
            nn.Linear(d_pitch + d_emb, d_head), nn.LayerNorm(d_head), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_head, d_head), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_head, n_classes),
        )
        # why: the aux head sees ONLY the embedding. It can lower its loss only if the embedding
        # itself carries outcome information, so the retrieval space can't be bypassed (D10).
        self.aux = nn.Sequential(nn.Linear(d_emb, d_hidden), nn.GELU(), nn.Linear(d_hidden, n_classes + 1))

    def embed(self, ars_x: torch.Tensor, ars_w: torch.Tensor) -> torch.Tensor:
        return self.encoder(ars_x, ars_w)

    def forward(self, pitch_x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Outcome logits for each pitch, given its pitcher-season's embedding."""
        return self.head(torch.cat([pitch_x, emb], dim=-1))

    def aux_outputs(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(profile logits [B, n_classes], RV/100 z-score prediction [B])."""
        out = self.aux(emb)
        return out[:, :-1], out[:, -1]


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a probability vector (the observed outcome mix), averaged over rows."""
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def aux_loss(model: ArsenalNet, emb: torch.Tensor, profile: torch.Tensor,
             rv_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(profile soft cross-entropy, RV/100 squared error) for the given embeddings."""
    prof_logits, rv_pred = model.aux_outputs(emb)
    return soft_cross_entropy(prof_logits, profile), F.mse_loss(rv_pred, rv_z)
