"""Dataset helpers for wildfire perimeter nowcasting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    with np.load(path, allow_pickle=True) as data:
        inputs = data["inputs"].astype(np.float32)  # (seq_len, H, W)
        targets = data["targets"].astype(np.float32)  # (horizons, H, W)
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
    return inputs, targets, metadata


def _augment_sequence(
    rng: np.random.Generator, sequence: np.ndarray, targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Horizontal flip
    if rng.random() < 0.5:
        sequence = sequence[..., ::-1]
        targets = targets[..., ::-1]
    # Vertical flip
    if rng.random() < 0.5:
        sequence = sequence[..., ::-1, :]
        targets = targets[..., ::-1, :]
    # 90-degree rotation
    k = rng.integers(0, 4)
    if k:
        sequence = np.rot90(sequence, k=k, axes=(-2, -1))
        targets = np.rot90(targets, k=k, axes=(-2, -1))
    return sequence, targets


class WildfireNowcastDataset(Dataset):
    """Dataset providing perimeter probability sequences and horizon targets."""

    def __init__(
        self,
        files: Sequence[Path],
        *,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        if not files:
            raise ValueError("Dataset requires at least one sample file.")
        self.files = list(files)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        inputs, targets, metadata = _load_npz(self.files[0])
        self.sequence_length = inputs.shape[0]
        self.horizons = metadata.get("horizons", [6, 12, 24])
        self.input_shape = inputs.shape
        self.target_shape = targets.shape

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        inputs, targets, metadata = _load_npz(path)

        if self.augment:
            inputs, targets = _augment_sequence(self.rng, inputs, targets)

        inputs_tensor = torch.from_numpy(inputs[:, None, ...].copy())  # (seq, 1, H, W)
        targets_tensor = torch.from_numpy(targets.copy())  # (horizon, H, W)
        return inputs_tensor, targets_tensor, metadata


__all__ = ["WildfireNowcastDataset"]
