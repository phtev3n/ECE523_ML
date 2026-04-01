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
from golf_tracer.models.losses import trajectory_losses
from golf_tracer.models.trajectory_lifter import TrajectoryLifter
from golf_tracer.utils.config import load_config
from golf_tracer.utils.io import ensure_dir
from golf_tracer.utils.train import resolve_device, set_seed


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
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
            mode="trajectory",
        )
    else:
        dataset = SyntheticGolfTrajectoryDataset(
            num_sequences=cfg["synthetic"]["num_sequences"],
            sequence_length=cfg["sequence_length"],
            fps=cfg["synthetic"]["fps"],
            mode="trajectory",
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

    # Clamp workers to a conservative value for cluster stability
    num_workers = int(cfg.get("num_workers", 0))
    num_workers = max(0, min(num_workers, 1))

    train_loader = make_loader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = TrajectoryLifter(
        input_size=5,
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    save_path = Path(cfg["save_path"])
    ensure_dir(save_path.parent)
    last_path = save_path
    best_path = save_path.with_name(save_path.stem + "_best.pt")

    for epoch in range(cfg["epochs"]):
        model.train()

        train_loss = 0.0
        train_xyz = 0.0
        train_eot = 0.0
        train_bg = 0.0
        train_samples = 0

        train_pbar = tqdm(train_loader, desc=f"trajectory train epoch {epoch + 1}/{cfg['epochs']}")
        for batch in train_pbar:
            x = batch["features"].to(device, non_blocking=pin_memory)
            target = {
                "xyz": batch["xyz"].to(device, non_blocking=pin_memory),
                "eot": batch["eot"].to(device, non_blocking=pin_memory),
            }

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                losses = trajectory_losses(pred, target, cfg["loss"])

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            train_samples += bs
            train_loss += losses["loss"].item() * bs
            train_xyz += losses.get("xyz_loss", losses.get("recon3d_loss", torch.tensor(0.0))).item() * bs
            train_eot += losses.get("eot_loss", torch.tensor(0.0)).item() * bs
            train_bg += losses.get("below_ground_loss", torch.tensor(0.0)).item() * bs

            train_pbar.set_postfix(
                loss=f"{losses['loss'].item():.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

        model.eval()

        val_loss = 0.0
        val_xyz = 0.0
        val_eot = 0.0
        val_bg = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device, non_blocking=pin_memory)
                target = {
                    "xyz": batch["xyz"].to(device, non_blocking=pin_memory),
                    "eot": batch["eot"].to(device, non_blocking=pin_memory),
                }

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x)
                    losses = trajectory_losses(pred, target, cfg["loss"])

                bs = x.size(0)
                val_samples += bs
                val_loss += losses["loss"].item() * bs
                val_xyz += losses.get("xyz_loss", losses.get("recon3d_loss", torch.tensor(0.0))).item() * bs
                val_eot += losses.get("eot_loss", torch.tensor(0.0)).item() * bs
                val_bg += losses.get("below_ground_loss", torch.tensor(0.0)).item() * bs

        train_loss /= max(train_samples, 1)
        train_xyz /= max(train_samples, 1)
        train_eot /= max(train_samples, 1)
        train_bg /= max(train_samples, 1)

        val_loss /= max(val_samples, 1)
        val_xyz /= max(val_samples, 1)
        val_eot /= max(val_samples, 1)
        val_bg /= max(val_samples, 1)

        scheduler.step(val_loss)

        print(
            f"[trajectory] epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_xyz={train_xyz:.6f} train_eot={train_eot:.6f} train_bg={train_bg:.6f} "
            f"val_xyz={val_xyz:.6f} val_eot={val_eot:.6f} val_bg={val_bg:.6f} "
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