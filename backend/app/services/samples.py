"""Utility helpers for loading offline sample bundle assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_BUNDLE_PATH = Path(__file__).resolve().parents[3] / "sample_bundle"


def read_events_fixture() -> dict[str, Any]:
    events_file = SAMPLE_BUNDLE_PATH / "data" / "events.json"
    if not events_file.exists():
        raise FileNotFoundError(events_file)
    return json.loads(events_file.read_text())


def load_nowcast_sample() -> dict[str, Any]:
    sample_file = SAMPLE_BUNDLE_PATH / "data" / "samples" / "demo_sample.npz"
    if not sample_file.exists():
        raise FileNotFoundError(sample_file)
    with np.load(sample_file, allow_pickle=True) as data:
        inputs = data["inputs"]
        targets = data["targets"]
        metadata = json.loads(str(data["metadata"].tolist()))
    return {"inputs": inputs, "targets": targets, "metadata": metadata}


__all__ = ["SAMPLE_BUNDLE_PATH", "read_events_fixture", "load_nowcast_sample"]
