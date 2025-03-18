"""Dataset helpers for wildfire segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as data:
        features: List[np.ndarray] = []

        if "rgb" in data:
            rgb = data["rgb"]
            if rgb.size:
                rgb = rgb.astype(np.float32) / 255.0
                features.append(np.transpose(rgb, (2, 0, 1)))

        if "thermal" in data:
            thermal = data["thermal"]
            if thermal.size and thermal.dtype != np.object_:
                if thermal.ndim == 3:
                    thermal = thermal[..., :1]
                thermal = thermal.astype(np.float32) / 255.0
                features.append(np.transpose(thermal, (2, 0, 1)))

        if "vegetation" in data:
            vegetation = data["vegetation"]
            if (
                vegetation.size
                and vegetation.ndim >= 2
                and vegetation.dtype != np.object_
            ):
                if vegetation.ndim == 2:
                    vegetation = vegetation[..., None]
                vegetation = vegetation.astype(np.float32)
                features.append(np.transpose(vegetation, (2, 0, 1)))

        if not features:
            raise ValueError(f"Sample {path} does not contain feature arrays")

        inputs = np.concatenate(features, axis=0)
        label = data["label"].astype(np.float32)
        metadata_raw = data["metadata"]
        if np.isscalar(metadata_raw):
            metadata_json = str(metadata_raw)
        else:
            metadata_json = (
                metadata_raw.item()
                if hasattr(metadata_raw, "item")
                else str(metadata_raw)
            )
        metadata = json.loads(metadata_json)

    return inputs, label, metadata


def _augment(
    rng: np.random.Generator, image: np.ndarray, label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = image[:, :, ::-1]
        label = label[:, ::-1]
    if rng.random() < 0.5:
        image = image[:, ::-1, :]
        label = label[::-1, :]
    # Random 90-degree rotations
    k = rng.integers(0, 4)
    if k:
        image = np.rot90(image, k=k, axes=(-2, -1))
        label = np.rot90(label, k=k, axes=(-2, -1))
    return image, label


class WildfireSegmentationDataset(Dataset):
    """Dataset backed by NPZ samples generated from tile chips."""

    def __init__(
        self,
        files: Sequence[Path],
        *,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        if not files:
            raise ValueError("Dataset requires at least one sample file")
        self.files = list(files)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        sample_inputs, sample_label, _ = _load_npz(self.files[0])
        self.num_channels = sample_inputs.shape[0]
        self.input_shape = sample_inputs.shape
        self.label_shape = sample_label.shape

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        inputs, label, metadata = _load_npz(path)

        if self.augment:
            inputs, label = _augment(self.rng, inputs, label)

        inputs_tensor = torch.from_numpy(inputs.copy()).float()
        label_tensor = torch.from_numpy(label[None, ...].copy()).float()
        return inputs_tensor, label_tensor, metadata


__all__ = ["WildfireSegmentationDataset"]
