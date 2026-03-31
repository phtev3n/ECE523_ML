from __future__ import annotations
import numpy as np


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    if len(a) == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def binary_f1(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.5) -> float:
    pred_bin = (pred >= thresh).astype(np.int32)
    gt_bin = gt.astype(np.int32)
    tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())
    if tp == 0:
        return 0.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def smoothness(track_uv: np.ndarray) -> float:
    if len(track_uv) < 3:
        return 0.0
    second_diff = track_uv[2:] - 2 * track_uv[1:-1] + track_uv[:-2]
    return float(np.mean(np.linalg.norm(second_diff, axis=1)))
