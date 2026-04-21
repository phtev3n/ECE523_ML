from __future__ import annotations

import torch
import torch.nn.functional as F


def detector_losses(pred: dict, target: dict, weights: dict) -> dict:
    """Compute the combined detector loss for a single-frame detection batch.

    Loss components
    ---------------
    heatmap_loss : Binary cross-entropy on the Gaussian heatmap.
        BCE is preferred over MSE for sparse small-object maps because MSE
        heavily down-weights the near-zero background (which dominates by area)
        while BCE naturally handles the imbalance via log probabilities.

    offset_loss  : Smooth-L1 (Huber) loss on sub-pixel offsets.
        Only supervised at cells where the heatmap target is > 0.1 (i.e. within
        the Gaussian blob).  Applying loss elsewhere would train the offset head
        on meaningless background cells.  Smooth-L1 is robust to the occasional
        large offset error caused by a misdetected background cell.

    visible_loss : Binary cross-entropy on the visibility logit.
        pos_weight > 1 compensates for the label imbalance when more frames
        are occluded than visible.

    uncertainty_loss : Learned-uncertainty (Kendall & Gal) loss on offset.
        Trains the log-variance head to predict its own uncertainty:
            L = exp(-log_var) * |offset_err| + log_var
        Only active when 'log_var' is present in pred (it always is for
        MultiScaleBallDetector) and weighted by 'uncertainty_weight' in config.
    """
    heatmap_logit = pred["heatmap_logit"]   # raw pre-sigmoid logit
    offset_pred   = pred["offset"]
    visible_logit = pred["visible_logit"]

    heatmap_tgt = target["heatmap"]
    offset_tgt  = target["offset"]
    visible_tgt = target["visible"].float().view_as(visible_logit)

    # bce_with_logits: numerically stable, AMP-safe, and correct because
    # heatmap_logit is the raw pre-sigmoid output (no double-sigmoid).
    heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_logit, heatmap_tgt)
    # Probability map needed for the offset mask — apply sigmoid once here.
    heatmap_pred = torch.sigmoid(heatmap_logit)

    # Mask offset loss to cells inside the Gaussian blob only
    pos_mask   = (heatmap_tgt > 0.1).float()
    pos_mask_2 = pos_mask.repeat(1, offset_pred.shape[1], 1, 1)

    offset_err  = F.smooth_l1_loss(offset_pred, offset_tgt, reduction="none")
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
        # Heteroscedastic aleatoric uncertainty loss (Kendall & Gal 2017).
        # The model learns to predict higher variance where its offset
        # predictions are less reliable (e.g. occluded or far-away balls).
        unc_err = torch.exp(-pred["log_var"]) * (offset_pred - offset_tgt) ** 2 + pred["log_var"]
        uncertainty_loss = (unc_err * pos_mask_2).sum() / pos_mask_2.sum().clamp_min(1.0)
    else:
        uncertainty_loss = torch.zeros((), device=heatmap_pred.device, dtype=heatmap_pred.dtype)

    total = (
        weights["heatmap_weight"]  * heatmap_loss
        + weights["offset_weight"] * offset_loss
        + weights["visible_weight"] * visible_loss
        + weights.get("uncertainty_weight", 0.0) * uncertainty_loss
    )

    return {
        "loss":             total,
        "heatmap_loss":     heatmap_loss.detach(),
        "offset_loss":      offset_loss.detach(),
        "visible_loss":     visible_loss.detach(),
        "uncertainty_loss": uncertainty_loss.detach(),
    }


def weighted_bce(pred: torch.Tensor, target: torch.Tensor, gamma: float = 0.75) -> torch.Tensor:
    """Asymmetric binary cross-entropy that down-weights false-negative errors.

    gamma controls the positive/negative weighting:
      gamma = 0.5 → standard BCE
      gamma > 0.5 → penalises missing a positive more than a false alarm

    Used for the end-of-trajectory head where failing to detect landing is
    worse than a slightly early stop signal.
    """
    eps  = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)
    loss = -(gamma * target * torch.log(pred)
             + (1.0 - gamma) * (1.0 - target) * torch.log(1.0 - pred))
    return loss.mean()


def trajectory_losses(pred: dict, target: dict, weights: dict) -> dict:
    """Compute the combined trajectory-lifter loss for a sequence batch.

    Loss components
    ---------------
    eot_loss          : Asymmetric BCE on end-of-trajectory probability.

    recon3d_loss      : MSE between predicted and ground-truth 3D positions.
        Provides dense supervision at every frame, driving the LSTM to
        maintain a physically consistent 3D trajectory throughout the clip.

    below_ground_loss : Physics constraint penalising y < 0 (ball below ground).
        The ground truth may occasionally clip y=0 due to rounding, but the
        model should never predict underground positions.  This soft constraint
        supplements recon3d_loss without requiring explicit y≥0 clamping in
        the model output.

    spin_loss         : Normalised MSE between predicted and labelled spin.
        Only active when spin_weight > 0 AND ground-truth spin is available
        (i.e. synthetic data with Magnus-force simulation labels).  Targets
        are divided by [5000, 1000] rpm so both backspin and sidespin
        contribute at similar magnitude regardless of their different scales.
        Gracefully degrades to zero on real data (where spin labels are absent).
    """
    eot_loss      = weighted_bce(pred["eot_prob"], target["eot"])
    recon3d_loss  = F.mse_loss(pred["xyz"], target["xyz"])

    # Penalise any predicted position below the ground plane (y < 0)
    below             = torch.clamp(-pred["xyz"][..., 1], min=0.0)
    below_ground_loss = torch.mean(below ** 2)

    total = (
        weights["eot_weight"]          * eot_loss
        + weights["recon3d_weight"]    * recon3d_loss
        + weights["below_ground_weight"] * below_ground_loss
    )

    out = {
        "loss":              total,
        "eot_loss":          eot_loss.detach(),
        "recon3d_loss":      recon3d_loss.detach(),
        "xyz_loss":          recon3d_loss.detach(),
        "below_ground_loss": below_ground_loss.detach(),
    }

    spin_weight = weights.get("spin_weight", 0.0)
    if spin_weight > 0.0 and "spin" in pred and "spin" in target:
        # Normalise by typical spin magnitudes before computing MSE so both
        # backspin (~thousands of rpm) and sidespin (~hundreds of rpm) are
        # weighted equally in the loss rather than backspin dominating.
        spin_scale = torch.tensor(
            [5000.0, 1000.0], device=pred["spin"].device, dtype=pred["spin"].dtype
        )
        spin_loss = F.mse_loss(pred["spin"] / spin_scale, target["spin"] / spin_scale)
        out["loss"]      = out["loss"] + spin_weight * spin_loss
        out["spin_loss"] = spin_loss.detach()
    else:
        # No spin labels available (real data) — zero loss, no gradient
        out["spin_loss"] = torch.zeros((), device=pred["xyz"].device)

    return out
