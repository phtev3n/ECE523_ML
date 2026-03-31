from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_root", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    if args.dataset_root:
        dataset = RealGolfSequenceDataset(args.dataset_root, sequence_length=cfg["sequence_length"], mode="trajectory")
    else:
        dataset = SyntheticGolfTrajectoryDataset(
            num_sequences=cfg["synthetic"]["num_sequences"],
            sequence_length=cfg["sequence_length"],
            fps=cfg["synthetic"]["fps"],
            mode="trajectory",
        )

    n_train = int(len(dataset) * cfg["train_split"])
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = TrajectoryLifter(
        input_size=5,
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    best_val = float("inf")
    save_path = Path(cfg["save_path"])
    ensure_dir(save_path.parent)

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"trajectory train epoch {epoch+1}/{cfg['epochs']}"):
            x = batch["features"].to(device)
            target = {
                "xyz": batch["xyz"].to(device),
                "eot": batch["eot"].to(device),
            }
            pred = model(x)
            losses = trajectory_losses(pred, target, cfg["loss"])
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            train_loss += losses["loss"].item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                target = {
                    "xyz": batch["xyz"].to(device),
                    "eot": batch["eot"].to(device),
                }
                pred = model(x)
                losses = trajectory_losses(pred, target, cfg["loss"])
                val_loss += losses["loss"].item() * x.size(0)

        train_loss /= max(len(train_ds), 1)
        val_loss /= max(len(val_ds), 1)
        print(f"[trajectory] epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        ckpt = {
            "model_state": model.state_dict(),
            "config": cfg,
            "epoch": epoch + 1,
        }
        torch.save(ckpt, save_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_path.with_name(save_path.stem + "_best.pt"))


if __name__ == "__main__":
    main()
