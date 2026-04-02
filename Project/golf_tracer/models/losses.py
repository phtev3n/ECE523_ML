from __future__ import annotations

import torch
import torch.nn.functional as F


def detector_losses(pred: dict, target: dict, weights: dict) -> dict:
    heatmap_pred = pred["heatmap"]
    offset_pred = pred["offset"]
    visible_logit = pred["visible_logit"]

    heatmap_tgt = target["heatmap"]
    offset_tgt = target["offset"]
    visible_tgt = target["visible"].float().view_as(visible_logit)

    # Better than plain MSE for sparse small-object heatmaps
    heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap_tgt)

    # Only supervise offsets near positive target regions
    pos_mask = (heatmap_tgt > 0.1).float()
    pos_mask_2 = pos_mask.repeat(1, offset_pred.shape[1], 1, 1)

    offset_err = F.smooth_l1_loss(offset_pred, offset_tgt, reduction="none")
    offset_loss = (offset_err * pos_mask_2).sum() / pos_mask_2.sum().clamp_min(1.0)

    pos_weight = torch.tensor(
        [weights.get("visible_pos_weight", 2.0)],
        device=visible_logit.device,
        dtype=visible_logit.dtype,
    )
    visible_loss = F.binary_cross_entropy_with_logits(
        visible_logit,
        visible_tgt,
        pos_weight=pos_weight,
    )

    if "log_var" in pred:
        unc_err = torch.exp(-pred["log_var"]) * (offset_pred - offset_tgt) ** 2 + pred["log_var"]
        uncertainty_loss = (unc_err * pos_mask_2).sum() / pos_mask_2.sum().clamp_min(1.0)
    else:
        uncertainty_loss = torch.zeros((), device=heatmap_pred.device, dtype=heatmap_pred.dtype)

    total = (
        weights["heatmap_weight"] * heatmap_loss
        + weights["offset_weight"] * offset_loss
        + weights["visible_weight"] * visible_loss
        + weights.get("uncertainty_weight", 0.0) * uncertainty_loss
    )

    return {
        "loss": total,
        "heatmap_loss": heatmap_loss.detach(),
        "offset_loss": offset_loss.detach(),
        "visible_loss": visible_loss.detach(),
        "uncertainty_loss": uncertainty_loss.detach(),
    }


def weighted_bce(pred: torch.Tensor, target: torch.Tensor, gamma: float = 0.75) -> torch.Tensor:
    eps = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)
    loss = -(gamma * target * torch.log(pred) + (1.0 - gamma) * (1.0 - target) * torch.log(1.0 - pred))
    return loss.mean()


def trajectory_losses(pred: dict, target: dict, weights: dict) -> dict:
    eot_loss = weighted_bce(pred["eot_prob"], target["eot"])
    recon3d_loss = F.mse_loss(pred["xyz"], target["xyz"])
    below = torch.clamp(-pred["xyz"][..., 1], min=0.0)
    below_ground_loss = torch.mean(below ** 2)

    total = (
        weights["eot_weight"] * eot_loss
        + weights["recon3d_weight"] * recon3d_loss
        + weights["below_ground_weight"] * below_ground_loss
    )

    return {
        "loss": total,
        "eot_loss": eot_loss.detach(),
        "recon3d_loss": recon3d_loss.detach(),
        "xyz_loss": recon3d_loss.detach(),
        "below_ground_loss": below_ground_loss.detach(),
    }