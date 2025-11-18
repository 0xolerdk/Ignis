"""Inference pipeline orchestrating segmentation and nowcasting models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from backend.app.models.unet import WildfireUNet
from backend.app.models.convlstm import WildfireConvLSTM


@dataclass
class SegmentationModel:
    model: WildfireUNet
    checkpoint: Path
    device: torch.device
    threshold: float


@dataclass
class NowcastModel:
    model: WildfireConvLSTM
    checkpoint: Path
    device: torch.device
    horizons: Sequence[int]
    mc_samples: int


def load_segmentation_model(
    checkpoint_path: Path, device: str = "cpu"
) -> SegmentationModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = WildfireUNet(
        in_channels=config.get("in_channels", 1),
        base_filters=config.get("base_filters", 32),
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    threshold = checkpoint.get("val_metrics", {}).get("threshold", 0.5)
    return SegmentationModel(
        model=model,
        checkpoint=checkpoint_path,
        device=torch.device(device),
        threshold=threshold,
    )


def load_nowcast_model(
    checkpoint_path: Path, device: str = "cpu", mc_samples: int = 10
) -> NowcastModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    horizons = tuple(config.get("horizons", [6, 12, 24]))
    model = WildfireConvLSTM(
        in_channels=config.get("in_channels", 1),
        hidden_dims=tuple(config.get("hidden_dims", [32, 64])),
        dropout=config.get("dropout", 0.2),
        horizons=horizons,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return NowcastModel(
        model=model,
        checkpoint=checkpoint_path,
        device=torch.device(device),
        horizons=horizons,
        mc_samples=mc_samples,
    )


def preprocess_chip(chip: Mapping[str, np.ndarray]) -> torch.Tensor:
    """Convert chip dictionary into tensor for U-Net."""
    channel_arrays: List[np.ndarray] = []
    for key in ("rgb", "thermal", "vegetation"):
        if key in chip and chip[key] is not None:
            arr = chip[key]
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)
            elif arr.ndim == 2:
                arr = arr[None, ...]
            channel_arrays.append(
                arr.astype(np.float32) / 255.0
                if key != "vegetation"
                else arr.astype(np.float32)
            )
    if not channel_arrays:
        raise ValueError("No chip data available for inference.")
    stacked = np.concatenate(channel_arrays, axis=0)
    return torch.from_numpy(stacked).unsqueeze(0)  # (1, C, H, W)


def postprocess_mask(mask: np.ndarray, transform) -> Dict[str, object]:
    """Convert binary mask into GeoJSON polygon output."""
    from rasterio.features import shapes

    shapes_geojson = []
    for geom, value in shapes(
        mask.astype(np.uint8), mask=(mask > 0), transform=transform
    ):
        if value == 1:
            shapes_geojson.append(shape(geom))
    if not shapes_geojson:
        return {"type": "FeatureCollection", "features": []}
    merged: Polygon = unary_union(shapes_geojson)
    if merged.geom_type == "Polygon":
        polygons = [merged]
    else:
        polygons = list(merged.geoms)
    features = [
        {
            "type": "Feature",
            "geometry": json.loads(json.dumps(polygon.__geo_interface__)),
            "properties": {},
        }
        for polygon in polygons
    ]
    return {"type": "FeatureCollection", "features": features}


def run_segmentation(
    segmentation: SegmentationModel,
    chip: Mapping[str, np.ndarray],
    transform=None,
    mc_samples: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Run segmentation with MC-dropout and return mean mask + polygon GeoJSON."""
    inputs = preprocess_chip(chip).to(segmentation.device)
    model = segmentation.model
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    predictions = []
    model.train()
    with torch.no_grad():
        for _ in range(mc_samples):
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())
    model.eval()

    mean_probs = np.mean(np.stack(predictions, axis=0), axis=0)[0, 0]
    mask = (mean_probs >= segmentation.threshold).astype(np.uint8)

    polygons = postprocess_mask(mask, transform) if transform is not None else {}

    return {
        "mask": mean_probs,
        "binary_mask": mask,
        "polygons": polygons,
        "mean_prob": float(mean_probs.mean()),
        "threshold": segmentation.threshold,
    }


def run_nowcasting(
    nowcast: NowcastModel,
    sequence: np.ndarray,
    mc_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Produce probabilistic nowcasts for requested horizons."""
    if seed is not None:
        torch.manual_seed(seed + 1)
        np.random.seed(seed + 1)

    seq_tensor = (
        torch.from_numpy(sequence[:, None, ...].astype(np.float32))
        .unsqueeze(0)
        .to(nowcast.device)
    )
    model = nowcast.model
    mc = mc_samples if mc_samples is not None else nowcast.mc_samples
    if mc <= 1:
        with torch.no_grad():
            logits = model(seq_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return {str(h): probs[idx] for idx, h in enumerate(nowcast.horizons)}

    predictions = []
    model.train()
    with torch.no_grad():
        for _ in range(mc):
            logits = model(seq_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            predictions.append(probs)
    model.eval()
    mean_probs = np.mean(np.stack(predictions, axis=0), axis=0)
    return {str(h): mean_probs[idx] for idx, h in enumerate(nowcast.horizons)}


def run_pipeline(
    segmentation_model: SegmentationModel,
    nowcast_model: NowcastModel,
    chip: Mapping[str, np.ndarray],
    transform,
    sequence: np.ndarray,
    event_geom,
    mc_samples_seg: int = 10,
    mc_samples_nowcast: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Execute segmentation then nowcasting pipeline."""
    segmentation_result = run_segmentation(
        segmentation_model,
        chip,
        transform=transform,
        mc_samples=mc_samples_seg,
        seed=seed,
    )

    nowcast_probs = run_nowcasting(
        nowcast_model,
        sequence,
        mc_samples=mc_samples_nowcast,
        seed=seed,
    )

    return {
        "segmentation": segmentation_result,
        "nowcast": nowcast_probs,
    }


__all__ = [
    "SegmentationModel",
    "NowcastModel",
    "load_segmentation_model",
    "load_nowcast_model",
    "run_segmentation",
    "run_nowcasting",
    "run_pipeline",
]
