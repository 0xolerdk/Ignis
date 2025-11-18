"""Integration tests for training and evaluation scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.scripts import build_dataset, evaluate, train_segmentation


def _build_fixture_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    args = argparse.Namespace(
        output_dir=dataset_dir,
        limit=1,
        zoom_levels="8,9",
        time_offsets="-24,-12,-6,0",
        patch_size=64,
        stride=32,
        min_positive_ratio=0.01,
        fixtures=True,
        fixtures_count=1,
        seed=123,
        log_level="WARNING",
    )
    build_dataset.run(args)
    return dataset_dir / "samples"


def test_training_and_evaluation_workflow(tmp_path: Path) -> None:
    samples_dir = _build_fixture_dataset(tmp_path)

    train_output = tmp_path / "training"
    train_args = argparse.Namespace(
        data_dir=samples_dir,
        output_dir=train_output,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        val_split=0.2,
        device="cpu",
        dropout=0.2,
        base_filters=16,
        seed=123,
        max_samples=64,
        num_workers=0,
        threshold=0.5,
        log_level="WARNING",
    )

    metrics = train_segmentation.run(train_args)
    metrics_path = train_output / "metrics.json"
    assert metrics_path.exists()
    assert metrics["best_checkpoint"]

    eval_output = tmp_path / "evaluation"
    eval_args = argparse.Namespace(
        data_dir=samples_dir,
        checkpoint=Path(str(metrics["best_checkpoint"])),
        output_dir=eval_output,
        threshold=0.5,
        mc_samples=2,
        device="cpu",
        max_samples=32,
        num_workers=0,
        log_level="WARNING",
    )

    results = evaluate.run(eval_args)
    assert (eval_output / "metrics.json").exists()
    assert (eval_output / "reliability_curve.png").exists()
    assert "dice" in results
