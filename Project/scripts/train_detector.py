from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from golf_tracer.data.real_dataset import RealGolfSequenceDataset
from golf_tracer.data.synthetic_dataset import SyntheticGolfTrajectoryDataset
from golf_tracer.models.detector import MultiScaleBallDetector
from golf_tracer.models.losses import detector_losses
from golf_tracer.utils.config import load_config
from golf_tracer.utils.io import ensure_dir
from golf_tracer.utils.train import resolve_device, set_seed


def freeze_batchnorm(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_root", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    if args.dataset_root:
        dataset = RealGolfSequenceDataset(
            args.dataset_root,
            sequence_length=cfg["sequence_length"],
            mode="detector",
        )
    else:
        dataset = SyntheticGolfTrajectoryDataset(
            num_sequences=cfg["synthetic"]["num_sequences"],
            sequence_length=cfg["sequence_length"],
            image_size=tuple(cfg["image_size"]),
            fps=cfg["synthetic"]["fps"],
            mode="detector",
        )

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 samples for train/val split.")

    n_train = max(1, int(len(dataset) * cfg["train_split"]))
    n_val = len(dataset) - n_train
    if n_val < 1:
        n_val = 1
        n_train = len(dataset) - 1

    split_gen = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_gen)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )

    model = MultiScaleBallDetector(cfg["model"]["backbone"]).to(device)
    model.apply(freeze_batchnorm)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=3,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val = float("inf")
    save_path = Path(cfg["save_path"])
    ensure_dir(save_path.parent)
    last_path = save_path
    best_path = save_path.with_name(save_path.stem + "_best.pt")

    for epoch in range(cfg["epochs"]):
        model.train()

        train_loss = 0.0
        train_heatmap = 0.0
        train_offset = 0.0
        train_visible = 0.0
        train_uncertainty = 0.0
        train_samples = 0

        train_pbar = tqdm(train_loader, desc=f"detector train epoch {epoch + 1}/{cfg['epochs']}")
        for batch in train_pbar:
            image = batch["image"].to(device, non_blocking=pin_memory)
            target = {
                "heatmap": batch["heatmap"].to(device, non_blocking=pin_memory),
                "offset": batch["offset"].to(device, non_blocking=pin_memory),
                "visible": batch["visible"].to(device, non_blocking=pin_memory),
            }
            if "uncertainty" in batch:
                target["uncertainty"] = batch["uncertainty"].to(device, non_blocking=pin_memory)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(image)
                losses = detector_losses(pred, target, cfg["loss"])

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            bs = image.size(0)
            train_samples += bs
            train_loss += losses["loss"].item() * bs
            train_heatmap += losses.get("heatmap_loss", torch.tensor(0.0)).item() * bs
            train_offset += losses.get("offset_loss", torch.tensor(0.0)).item() * bs
            train_visible += losses.get("visible_loss", torch.tensor(0.0)).item() * bs
            train_uncertainty += losses.get("uncertainty_loss", torch.tensor(0.0)).item() * bs

            train_pbar.set_postfix(
                loss=f"{losses['loss'].item():.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

        model.eval()

        val_loss = 0.0
        val_heatmap = 0.0
        val_offset = 0.0
        val_visible = 0.0
        val_uncertainty = 0.0
        val_samples = 0

        val_visible_prob_sum = 0.0
        val_visible_pred_pos = 0.0
        val_visible_true_pos = 0.0
        val_visible_correct = 0.0
        val_visible_count = 0.0

        val_pos_prob_sum = 0.0
        val_neg_prob_sum = 0.0
        val_pos_count = 0.0
        val_neg_count = 0.0

        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=pin_memory)
                target = {
                    "heatmap": batch["heatmap"].to(device, non_blocking=pin_memory),
                    "offset": batch["offset"].to(device, non_blocking=pin_memory),
                    "visible": batch["visible"].to(device, non_blocking=pin_memory),
                }
                if "uncertainty" in batch:
                    target["uncertainty"] = batch["uncertainty"].to(device, non_blocking=pin_memory)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(image)
                    losses = detector_losses(pred, target, cfg["loss"])

                bs = image.size(0)
                val_samples += bs
                val_loss += losses["loss"].item() * bs
                val_heatmap += losses.get("heatmap_loss", torch.tensor(0.0)).item() * bs
                val_offset += losses.get("offset_loss", torch.tensor(0.0)).item() * bs
                val_visible += losses.get("visible_loss", torch.tensor(0.0)).item() * bs
                val_uncertainty += losses.get("uncertainty_loss", torch.tensor(0.0)).item() * bs

                visible_logit = pred.get("visible_logit")
                if visible_logit is not None:
                    visible_prob = torch.sigmoid(visible_logit).view(-1)
                    visible_true = target["visible"].float().view(-1)
                    visible_pred = (visible_prob > 0.35).float()

                    val_visible_prob_sum += visible_prob.sum().item()
                    val_visible_pred_pos += visible_pred.sum().item()
                    val_visible_true_pos += visible_true.sum().item()
                    val_visible_correct += (visible_pred == visible_true).float().sum().item()
                    val_visible_count += visible_true.numel()

                    pos_mask = visible_true > 0.5
                    neg_mask = ~pos_mask

                    if pos_mask.any():
                        val_pos_prob_sum += visible_prob[pos_mask].sum().item()
                        val_pos_count += pos_mask.sum().item()

                    if neg_mask.any():
                        val_neg_prob_sum += visible_prob[neg_mask].sum().item()
                        val_neg_count += neg_mask.sum().item()

        train_loss /= max(train_samples, 1)
        train_heatmap /= max(train_samples, 1)
        train_offset /= max(train_samples, 1)
        train_visible /= max(train_samples, 1)
        train_uncertainty /= max(train_samples, 1)

        val_loss /= max(val_samples, 1)
        val_heatmap /= max(val_samples, 1)
        val_offset /= max(val_samples, 1)
        val_visible /= max(val_samples, 1)
        val_uncertainty /= max(val_samples, 1)

        scheduler.step(val_loss)

        if val_visible_count > 0:
            mean_visible_prob = val_visible_prob_sum / val_visible_count
            pred_positive_rate = val_visible_pred_pos / val_visible_count
            true_positive_rate = val_visible_true_pos / val_visible_count
            visible_accuracy = val_visible_correct / val_visible_count
        else:
            mean_visible_prob = 0.0
            pred_positive_rate = 0.0
            true_positive_rate = 0.0
            visible_accuracy = 0.0

        pos_prob_mean = val_pos_prob_sum / max(val_pos_count, 1.0)
        neg_prob_mean = val_neg_prob_sum / max(val_neg_count, 1.0)

        print(
            f"[detector] epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_heatmap={train_heatmap:.6f} train_offset={train_offset:.6f} "
            f"train_visible={train_visible:.6f} train_unc={train_uncertainty:.6f} "
            f"val_heatmap={val_heatmap:.6f} val_offset={val_offset:.6f} "
            f"val_visible={val_visible:.6f} val_unc={val_uncertainty:.6f} "
            f"val_vis_prob_mean={mean_visible_prob:.4f} "
            f"val_vis_pred_pos_rate={pred_positive_rate:.4f} "
            f"val_vis_true_pos_rate={true_positive_rate:.4f} "
            f"val_vis_acc={visible_accuracy:.4f} "
            f"val_vis_prob_pos_mean={pos_prob_mean:.4f} "
            f"val_vis_prob_neg_mean={neg_prob_mean:.4f} "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": cfg,
            "epoch": epoch + 1,
            "best_val": best_val,
        }

        torch.save(ckpt, last_path)
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_path)


if __name__ == "__main__":
    main()