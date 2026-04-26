from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn


class LSTMSummaryPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        region_k: int,
        fcs_n: int,
        summary_dim: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.summary_head = nn.Linear(int(hidden_dim), int(summary_dim))
        self.region_head = nn.Linear(int(summary_dim), int(region_k))
        self.congestion_head = nn.Linear(int(summary_dim), int(fcs_n))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, L, input_dim]
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        summary = self.summary_head(h)
        req_pred = self.region_head(summary)
        cong_pred = self.congestion_head(summary)
        return req_pred, cong_pred, summary


@dataclass
class PredictiveSummaryBundle:
    model: LSTMSummaryPredictor
    device: torch.device
    seq_len: int
    region_k: int
    fcs_n: int
    summary_dim: int
    region_centers: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    req_mean: np.ndarray
    req_std: np.ndarray
    cong_mean: np.ndarray
    cong_std: np.ndarray

    def encode_summary(self, seq_features: np.ndarray) -> np.ndarray:
        req_pred, cong_pred, summary = self.predict_outputs(seq_features)
        _ = req_pred
        _ = cong_pred
        return summary

    def predict_outputs(self, seq_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(seq_features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("seq_features must be [L, input_dim].")

        input_dim = int(self.region_k + self.fcs_n)
        if x.shape[1] != input_dim:
            raise ValueError(f"input_dim mismatch: got {x.shape[1]}, expected {input_dim}")

        if x.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - x.shape[0], input_dim), dtype=np.float32)
            x = np.concatenate([pad, x], axis=0)
        elif x.shape[0] > self.seq_len:
            x = x[-self.seq_len :]

        x = (x - self.input_mean) / self.input_std
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            req_n, cong_n, summary = self.model(xt)

        req = req_n.squeeze(0).cpu().numpy().astype(np.float32) * self.req_std + self.req_mean
        cong = cong_n.squeeze(0).cpu().numpy().astype(np.float32) * self.cong_std + self.cong_mean
        req = np.maximum(req, 0.0)
        cong = np.maximum(cong, 0.0)
        summ = summary.squeeze(0).cpu().numpy().astype(np.float32)
        return req, cong, summ


def kmeans_latlon(points: np.ndarray, rng: np.random.Generator, k: int, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    if n <= 0 or k <= 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=np.int64)

    if n >= k:
        centers = pts[rng.choice(n, size=k, replace=False)].copy()
    else:
        centers = pts[rng.choice(n, size=k, replace=True)].copy()

    labels = np.zeros((n,), dtype=np.int64)
    for it in range(max(1, int(max_iter))):
        dist2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist2, axis=1).astype(np.int64)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = pts[mask].mean(axis=0)
            else:
                centers[j] = pts[int(rng.integers(0, n))]

    counts = np.bincount(labels, minlength=k).astype(np.int64)
    return centers, counts


def assign_points_to_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    c = np.asarray(centers, dtype=float)
    if pts.size == 0 or c.size == 0:
        return np.zeros((0,), dtype=np.int64)
    dist2 = ((pts[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dist2, axis=1).astype(np.int64)


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = np.asarray(x, dtype=np.float32)
    s = np.where(np.abs(s) < 1e-6, 1.0, s)
    return s


def load_predictive_summary_bundle(path: Path | str, device: str = "cpu") -> PredictiveSummaryBundle:
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location="cpu")

    region_k = int(payload["region_k"])
    fcs_n = int(payload["fcs_n"])
    summary_dim = int(payload["summary_dim"])
    seq_len = int(payload["seq_len"])
    hidden_dim = int(payload["hidden_dim"])
    num_layers = int(payload["num_layers"])
    input_dim = int(region_k + fcs_n)

    model = LSTMSummaryPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        region_k=region_k,
        fcs_n=fcs_n,
        summary_dim=summary_dim,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(dev)

    return PredictiveSummaryBundle(
        model=model,
        device=dev,
        seq_len=seq_len,
        region_k=region_k,
        fcs_n=fcs_n,
        summary_dim=summary_dim,
        region_centers=np.asarray(payload["region_centers"], dtype=np.float32),
        input_mean=np.asarray(payload["input_mean"], dtype=np.float32),
        input_std=_safe_std(np.asarray(payload["input_std"], dtype=np.float32)),
        req_mean=np.asarray(payload["req_mean"], dtype=np.float32),
        req_std=_safe_std(np.asarray(payload["req_std"], dtype=np.float32)),
        cong_mean=np.asarray(payload["cong_mean"], dtype=np.float32),
        cong_std=_safe_std(np.asarray(payload["cong_std"], dtype=np.float32)),
    )
