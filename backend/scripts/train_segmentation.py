"""Train the segmentation model on dataset samples."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from backend.app.datasets.segmentation import WildfireSegmentationDataset
from backend.app.models.unet import WildfireUNet


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    val_split: float
    device: str
    dropout: float
    base_filters: int
    seed: int
    max_samples: Optional[int]
    num_workers: int
    threshold: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wildfire segmentation U-Net")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory containing NPZ samples"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level), format="[%(levelname)s] %(name)s: %(message)s"
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> Mapping[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets_bin = (targets > 0.5).float()

    intersection = (preds * targets_bin).sum().item()
    pred_sum = preds.sum().item()
    target_sum = targets_bin.sum().item()
    union = pred_sum + target_sum - intersection

    dice = (
        (2 * intersection) / (pred_sum + target_sum + 1e-6)
        if (pred_sum + target_sum) > 0
        else 0.0
    )
    iou = intersection / (union + 1e-6) if union > 0 else 0.0
    bce = nn.functional.binary_cross_entropy(probs, targets_bin)
    return {"dice": dice, "iou": iou, "bce": float(bce.item())}


def create_datasets(
    config: TrainConfig,
) -> tuple[WildfireSegmentationDataset, WildfireSegmentationDataset]:
    files = sorted(config.data_dir.glob("*.npz"))
    if config.max_samples:
        files = files[: config.max_samples]
    if len(files) < 2:
        raise ValueError("Not enough samples for training")

    val_size = max(1, int(len(files) * config.val_split))
    train_files = files[:-val_size]
    val_files = files[-val_size:]

    train_dataset = WildfireSegmentationDataset(
        train_files, augment=True, seed=config.seed
    )
    val_dataset = WildfireSegmentationDataset(
        val_files, augment=False, seed=config.seed
    )
    return train_dataset, val_dataset


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
) -> Mapping[str, float]:
    model.train()
    bce_loss = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_dice = 0.0
    count = 0

    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = bce_loss(logits, targets) + dice_loss(logits, targets)
        loss.backward()
        optimizer.step()

        metrics = compute_metrics(logits.detach(), targets, threshold=threshold)
        total_loss += float(loss.item())
        total_dice += metrics["dice"]
        count += 1

    return {"loss": total_loss / count, "dice": total_dice / count}


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device, threshold: float
) -> Mapping[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    count = 0
    bce_loss = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = bce_loss(logits, targets) + dice_loss(logits, targets)
            metrics = compute_metrics(logits, targets, threshold=threshold)
            total_loss += float(loss.item())
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            count += 1

    return {
        "loss": total_loss / count,
        "dice": total_dice / count,
        "iou": total_iou / count,
    }


def run(args: argparse.Namespace) -> Mapping[str, object]:
    configure_logging(args.log_level)
    set_seed(args.seed)

    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        device=args.device,
        dropout=args.dropout,
        base_filters=args.base_filters,
        seed=args.seed,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = config.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = create_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    in_channels = train_dataset.num_channels
    device = torch.device(config.device)
    model = WildfireUNet(
        in_channels=in_channels,
        base_filters=config.base_filters,
        dropout=config.dropout,
    )
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: List[Dict[str, float]] = []
    best_dice = 0.0
    best_checkpoint: Optional[Path] = None
    best_metrics: Mapping[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config.threshold
        )
        val_metrics = evaluate(model, val_loader, device, config.threshold)
        history.append(
            {
                "epoch": epoch,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        LOGGER.info(
            "Epoch %d/%d - train_loss: %.4f train_dice: %.4f val_loss: %.4f val_dice: %.4f val_iou: %.4f",
            epoch,
            config.epochs,
            train_metrics["loss"],
            train_metrics["dice"],
            val_metrics["loss"],
            val_metrics["dice"],
            val_metrics["iou"],
        )

        if val_metrics["dice"] >= best_dice:
            best_dice = val_metrics["dice"]
            best_checkpoint = checkpoints_dir / "best_model.pt"
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "in_channels": in_channels,
                        "base_filters": config.base_filters,
                        "dropout": config.dropout,
                    },
                    "val_metrics": val_metrics,
                },
                best_checkpoint,
            )

    metrics = {
        "best_dice": best_dice,
        "history": history,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
        "val_samples": len(val_dataset),
        "train_samples": len(train_dataset),
        "best_metrics": best_metrics,
    }

    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (config.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2)
    )
    return metrics


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
