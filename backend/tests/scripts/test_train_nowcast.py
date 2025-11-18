"""Integration test for the nowcast training script."""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from backend.scripts import train_nowcast


def _generate_nowcast_samples(output_dir: Path, num_samples: int = 48) -> Path:
    rng = np.random.default_rng(123)
    sequence_length = 4
    horizons = [6, 12, 24]
    height, width = 32, 32
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(num_samples):
        base = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
        sequence = []
        for t in range(sequence_length):
            decay = 0.7 ** (sequence_length - t)
            noise = rng.normal(scale=0.1, size=(height, width)).astype(np.float32)
            frame = np.clip(base * decay + noise + 0.5, 0.0, 1.0)
            sequence.append(frame)
        sequence_arr = np.stack(sequence, axis=0)

        targets = []
        for h_idx, _ in enumerate(horizons, start=1):
            growth = base + rng.normal(scale=0.15 * h_idx, size=(height, width))
            target = np.clip(growth + 0.5, 0.0, 1.0)
            targets.append(target)
        targets_arr = np.stack(targets, axis=0).astype(np.float32)

        metadata = {"horizons": horizons, "sequence_length": sequence_length}
        sample_path = output_dir / f"sample_{idx:04d}.npz"
        np.savez_compressed(
            sample_path,
            inputs=sequence_arr.astype(np.float32),
            targets=targets_arr,
            metadata=json_dumps(metadata),
        )
    return output_dir


def json_dumps(payload):
    import json

    return json.dumps(payload)


def test_train_nowcast_produces_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "nowcast_samples"
    _generate_nowcast_samples(data_dir)

    output_dir = tmp_path / "nowcast_training"
    args = argparse.Namespace(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        val_split=0.2,
        device="cpu",
        seed=42,
        max_samples=48,
        num_workers=0,
        dropout=0.1,
        hidden_dims="16,32",
        horizons="6,12,24",
        mc_samples=2,
        log_level="WARNING",
    )

    metrics = train_nowcast.run(args)

    metrics_path = output_dir / "metrics.json"
    assert metrics_path.exists()
    summary = metrics_path.read_text()
    assert "best_brier_mean" in summary
    for horizon in [6, 12, 24]:
        assert (output_dir / f"reliability_h{horizon}.png").exists()
        assert str(horizon) in summary
    assert metrics["best_checkpoint"], "Expected best checkpoint path"
