from __future__ import annotations
import torch
import torch.nn.functional as F


def detector_losses(pred: dict, target: dict, weights: dict) -> dict:
    heatmap_loss = F.mse_loss(pred["heatmap"], target["heatmap"])
    offset_loss = F.smooth_l1_loss(pred["offset"], target["offset"])
    visible_loss = F.binary_cross_entropy_with_logits(pred["visible_logit"], target["visible"])
    uncertainty_loss = torch.mean(torch.exp(-pred["log_var"]) * (pred["offset"] - target["offset"]) ** 2 + pred["log_var"])
    total = (
        weights["heatmap_weight"] * heatmap_loss
        + weights["offset_weight"] * offset_loss
        + weights["visible_weight"] * visible_loss
        + weights["uncertainty_weight"] * uncertainty_loss
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
        "below_ground_loss": below_ground_loss.detach(),
    }
