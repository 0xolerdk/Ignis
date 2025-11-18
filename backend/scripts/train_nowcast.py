"""Train ConvLSTM/TCN-based wildfire perimeter nowcasting models."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from backend.app.datasets.nowcast import WildfireNowcastDataset
from backend.app.models.convlstm import WildfireConvLSTM


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
    seed: int
    max_samples: Optional[int]
    num_workers: int
    dropout: float
    hidden_dims: Sequence[int]
    horizons: Sequence[int]
    mc_samples: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wildfire nowcasting ConvLSTM")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dims", type=str, default="32,64")
    parser.add_argument("--horizons", type=str, default="6,12,24")
    parser.add_argument("--mc-samples", type=int, default=5)
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


def create_datasets(
    config: TrainConfig,
) -> tuple[WildfireNowcastDataset, WildfireNowcastDataset]:
    files = sorted(config.data_dir.glob("*.npz"))
    if config.max_samples:
        files = files[: config.max_samples]
    if len(files) < 4:
        raise ValueError("Not enough samples for training nowcast model")

    val_size = max(1, int(len(files) * config.val_split))
    train_files = files[:-val_size]
    val_files = files[-val_size:]

    train_dataset = WildfireNowcastDataset(train_files, augment=True, seed=config.seed)
    val_dataset = WildfireNowcastDataset(val_files, augment=False, seed=config.seed)
    return train_dataset, val_dataset


def brier_scores(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return ((probs - targets) ** 2).mean(dim=(-2, -1))


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    horizons: Sequence[int],
) -> Mapping[str, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_brier = np.zeros(len(horizons), dtype=np.float64)
    count = 0

    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits.detach())
        brier = brier_scores(probs, targets).cpu().numpy().mean(axis=0)
        total_loss += float(loss.item())
        total_brier += brier
        count += 1

    return {
        "loss": total_loss / count,
        "brier_mean": float(total_brier.mean() / count),
        **{
            f"brier_{h}": float(total_brier[idx] / count)
            for idx, h in enumerate(horizons)
        },
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    horizons: Sequence[int],
    mc_samples: int,
) -> Mapping[str, object]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_brier = np.zeros(len(horizons), dtype=np.float64)
    count = 0
    probs_accumulator = defaultdict(list)
    targets_accumulator = defaultdict(list)

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = _mc_forward(model, inputs, mc_samples)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)
            brier = brier_scores(probs, targets).cpu().numpy().mean(axis=0)
            total_loss += float(loss.item())
            total_brier += brier
            count += 1

            probs_np = probs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            for idx, horizon in enumerate(horizons):
                probs_accumulator[horizon].append(probs_np[:, idx, ...].reshape(-1))
                targets_accumulator[horizon].append(targets_np[:, idx, ...].reshape(-1))

    mean_brier = total_brier / count
    metrics: Dict[str, object] = {
        "loss": total_loss / count,
        "brier_mean": float(mean_brier.mean()),
        "brier_scores": {
            str(h): float(mean_brier[idx]) for idx, h in enumerate(horizons)
        },
        "probs": {h: np.concatenate(probs_accumulator[h]) for h in horizons},
        "targets": {h: np.concatenate(targets_accumulator[h]) for h in horizons},
    }
    return metrics


def _mc_forward(
    model: nn.Module, inputs: torch.Tensor, mc_samples: int
) -> torch.Tensor:
    if mc_samples <= 1:
        return model(inputs)
    predictions = []
    model.train()
    with torch.no_grad():
        for _ in range(mc_samples):
            predictions.append(model(inputs))
    model.eval()
    return torch.stack(predictions).mean(dim=0)


def plot_reliability(
    probs: np.ndarray, targets: np.ndarray, horizon: int, output_dir: Path
) -> None:
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_true = []
    bin_conf = []
    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        bin_true.append(targets[mask].mean())
        bin_conf.append(probs[mask].mean())

    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    if bin_conf:
        plt.plot(bin_conf, bin_true, marker="o", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(f"Reliability Horizon {horizon}h")
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"reliability_h{horizon}.png")
    plt.close()


def run(args: argparse.Namespace) -> Mapping[str, object]:
    configure_logging(args.log_level)
    set_seed(args.seed)

    hidden_dims = tuple(
        int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()
    )
    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        device=args.device,
        seed=args.seed,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        dropout=args.dropout,
        hidden_dims=hidden_dims,
        horizons=horizons,
        mc_samples=args.mc_samples,
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

    # Ensure horizons align with dataset metadata if provided
    dataset_horizons = tuple(
        int(h) for h in getattr(train_dataset, "horizons", config.horizons)
    )
    horizons = dataset_horizons

    device = torch.device(config.device)
    model = WildfireConvLSTM(
        in_channels=1,
        hidden_dims=config.hidden_dims,
        horizons=horizons,
        dropout=config.dropout,
    )
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: List[Dict[str, float]] = []
    best_score = float("inf")
    best_checkpoint: Optional[Path] = None
    best_metrics: Mapping[str, object] = {}

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, horizons)
        val_metrics = evaluate(model, val_loader, device, horizons, config.mc_samples)
        LOGGER.info(
            "Epoch %d/%d - train_loss %.4f train_brier %.4f val_loss %.4f val_brier %.4f",
            epoch,
            config.epochs,
            train_metrics["loss"],
            train_metrics["brier_mean"],
            val_metrics["loss"],
            val_metrics["brier_mean"],
        )
        history.append(
            {
                "epoch": epoch,
                **train_metrics,
                **{
                    f"val_{k}": v
                    for k, v in val_metrics.items()
                    if isinstance(v, (int, float))
                },
            }
        )

        if cast(float, val_metrics["brier_mean"]) <= best_score:
            best_score = cast(float, val_metrics["brier_mean"])
            best_checkpoint = checkpoints_dir / "best_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "in_channels": 1,
                        "hidden_dims": config.hidden_dims,
                        "dropout": config.dropout,
                        "horizons": horizons,
                    },
                    "val_metrics": val_metrics,
                },
                best_checkpoint,
            )
            best_metrics = val_metrics

        # Reliability plots for each horizon
        for horizon in horizons:
            probs = cast(np.ndarray, val_metrics["probs"])[horizon]
            targets = cast(np.ndarray, val_metrics["targets"])[horizon]
            plot_reliability(probs, targets, horizon, config.output_dir)

    metrics_summary = {
        "best_brier_mean": best_score,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "horizons": horizons,
        "history": history,
        "best_metrics": {
            "brier_scores": best_metrics.get("brier_scores", {}),
            "loss": best_metrics.get("loss", 0.0),
        },
    }

    (config.output_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2)
    )
    (config.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2)
    )
    return metrics_summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
