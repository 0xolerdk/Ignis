"""Tests for inference and explainability services."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from affine import Affine

from backend.app.models.convlstm import WildfireConvLSTM
from backend.app.models.unet import WildfireUNet
from backend.app.services import (
    SegmentationModel,
    NowcastModel,
    load_segmentation_model,
    load_nowcast_model,
    run_pipeline,
    run_explainability,
)


def _create_segmentation_checkpoint(path: Path) -> Path:
    model = WildfireUNet(in_channels=7, base_filters=4, dropout=0.0)
    for param in model.parameters():
        param.data.zero_()
    if model.outc.conv.bias is not None:
        model.outc.conv.bias.data.fill_(4.0)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {"in_channels": 7, "base_filters": 4, "dropout": 0.0},
        "val_metrics": {"threshold": 0.5},
    }
    torch.save(checkpoint, path)
    return path


def _create_nowcast_checkpoint(path: Path, horizons: list[int]) -> Path:
    model = WildfireConvLSTM(
        in_channels=1, hidden_dims=(8,), horizons=horizons, dropout=0.0
    )
    for param in model.parameters():
        param.data.zero_()
    head_bias = None
    if hasattr(model.head, "bias") and model.head.bias is not None:
        head_bias = model.head.bias
    elif hasattr(model.head, "modules"):
        modules = list(model.head.modules())
        for module in reversed(modules):
            if isinstance(module, torch.nn.Conv2d) and module.bias is not None:
                head_bias = module.bias
                break
    if head_bias is not None:
        for idx, _ in enumerate(horizons):
            head_bias.data[idx] = -1.0 + idx
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "in_channels": 1,
            "hidden_dims": (8,),
            "dropout": 0.0,
            "horizons": horizons,
        },
        "val_metrics": {},
    }
    torch.save(checkpoint, path)
    return path


def test_run_pipeline_and_explainability(tmp_path: Path) -> None:
    seg_ckpt = _create_segmentation_checkpoint(tmp_path / "seg.pt")
    now_ckpt = _create_nowcast_checkpoint(tmp_path / "now.pt", [6, 12, 24])

    segmentation_model: SegmentationModel = load_segmentation_model(
        seg_ckpt, device="cpu"
    )
    nowcast_model: NowcastModel = load_nowcast_model(
        now_ckpt, device="cpu", mc_samples=3
    )

    height = width = 32
    thermal = np.ones((height, width, 4), dtype=np.uint8) * 200
    rgb = np.ones((height, width, 3), dtype=np.uint8) * 150
    chip = {"thermal": thermal, "rgb": rgb}
    transform = Affine.identity()

    sequence = np.ones((4, height, width), dtype=np.float32) * 0.3

    result = run_pipeline(
        segmentation_model,
        nowcast_model,
        chip,
        transform,
        sequence,
        event_geom=None,
        mc_samples_seg=3,
        mc_samples_nowcast=3,
        seed=123,
    )

    segmentation = cast(dict, result["segmentation"])
    assert cast(np.ndarray, segmentation["binary_mask"]).mean() == 1.0
    assert cast(dict, segmentation["polygons"])["features"], "Expected non-empty polygons"

    nowcast = cast(dict, result["nowcast"])
    assert set(nowcast.keys()) == {6, 12, 24}
    for horizon, array in nowcast.items():
        expected = torch.sigmoid(torch.tensor(-1.0 + [6, 12, 24].index(horizon))).item()
        assert np.allclose(array, expected, atol=1e-3)

    explain_output = run_explainability(
        segmentation_model.model,
        chip,
        device=torch.device("cpu"),
        target_layer="inc",
        output_path=tmp_path / "explain.png",
    )
    assert explain_output["output_path"]
    assert Path(cast(str, explain_output["output_path"])).exists()
