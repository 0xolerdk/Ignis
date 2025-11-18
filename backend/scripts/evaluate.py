"""Evaluate segmentation model checkpoints and generate reliability diagrams."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Mapping, Optional, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.app.datasets.segmentation import WildfireSegmentationDataset
from backend.app.models.unet import WildfireUNet


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
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
        level=getattr(logging, level), format="[%(levelname)s] %(message)s"
    )


def load_model(
    checkpoint_path: Path, device: torch.device, in_channels: int
) -> WildfireUNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = WildfireUNet(
        in_channels=config.get("in_channels", in_channels),
        base_filters=config.get("base_filters", 32),
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model


def mc_predict(
    model: torch.nn.Module, inputs: torch.Tensor, mc_samples: int, device: torch.device
) -> torch.Tensor:
    if mc_samples <= 1:
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(inputs.to(device)))

    predictions = []
    model.train()  # enable dropout
    with torch.no_grad():
        for _ in range(mc_samples):
            preds = torch.sigmoid(model(inputs.to(device)))
            predictions.append(preds)
    model.eval()
    return torch.stack(predictions).mean(dim=0)


def evaluate(
    dataset: WildfireSegmentationDataset,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    mc_samples: int,
    dataloader: DataLoader,
) -> Mapping[str, float]:
    total_intersection = 0.0
    total_union = 0.0
    total_pred = 0.0
    total_target = 0.0
    brier_sum = 0.0
    total_pixels = 0
    probs_all = []
    targets_all = []

    for inputs, targets, _ in dataloader:
        probs = mc_predict(model, inputs, mc_samples, device).cpu()
        preds = (probs > threshold).float()
        targets_bin = (targets > 0.5).float()

        intersection = (preds * targets_bin).sum().item()
        pred_sum = preds.sum().item()
        target_sum = targets_bin.sum().item()
        union = pred_sum + target_sum - intersection

        total_intersection += intersection
        total_union += union
        total_pred += pred_sum
        total_target += target_sum
        brier_sum += ((probs - targets_bin) ** 2).sum().item()
        total_pixels += targets_bin.numel()

        probs_all.append(probs.view(-1).numpy())
        targets_all.append(targets_bin.view(-1).numpy())

    dice = (
        (2 * total_intersection) / (total_pred + total_target + 1e-6)
        if (total_pred + total_target) > 0
        else 0.0
    )
    iou = total_intersection / (total_union + 1e-6) if total_union > 0 else 0.0
    brier = brier_sum / (total_pixels + 1e-6)

    probs_flat = np.concatenate(probs_all)
    targets_flat = np.concatenate(targets_all)

    return {
        "dice": dice,
        "iou": iou,
        "brier": brier,
        "probs": probs_flat,
        "targets": targets_flat,
    }


def plot_reliability(
    probs: np.ndarray, targets: np.ndarray, output_path: Path, bins: int = 10
) -> None:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_true = []
    bin_conf = []

    for b in range(bins):
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
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def run(args: argparse.Namespace) -> Mapping[str, float]:
    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.data_dir.glob("*.npz"))
    if args.max_samples:
        files = files[: args.max_samples]
    dataset = WildfireSegmentationDataset(files, augment=False)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, dataset.num_channels)

    metrics = evaluate(
        dataset, model, device, args.threshold, args.mc_samples, dataloader
    )
    results = {k: float(v) for k, v in metrics.items() if k not in {"probs", "targets"}}

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    plot_reliability(
        cast(np.ndarray, metrics["probs"]),
        cast(np.ndarray, metrics["targets"]),
        args.output_dir / "reliability_curve.png",
    )

    return results


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
