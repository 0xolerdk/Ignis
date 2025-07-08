"""Tests for the dataset build script."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.app.services.eonet import EventGeometry, WildfireEvent
from backend.app.services.gibs import ChipSet, normalize_time
from backend.app.services.tiling import compute_tile_grid
from backend.scripts import build_dataset


def _synthetic_chip_set(event: WildfireEvent, times, zooms) -> dict[int, ChipSet]:
    rng = np.random.default_rng(123)
    chip_sets = {}
    for zoom in zooms:
        tile_grid = compute_tile_grid(event.bbox, zoom)
        height = tile_grid.height_tiles * 256
        width = tile_grid.width_tiles * 256
        arrays = {}
        for time_iso in times:
            time_key = normalize_time(time_iso)
            rgb = np.clip(rng.normal(120, 40, size=(height, width, 3)), 0, 255).astype(
                np.uint8
            )
            thermal = np.zeros((height, width, 4), dtype=np.uint8)
            thermal[height // 4 : height // 2, width // 4 : width // 2, 0] = 255
            thermal[..., 3] = 255
            arrays[time_key] = {
                "MODIS_Terra_CorrectedReflectance_TrueColor": rgb,
                "MODIS_Terra_Thermal_Anomalies_Night": thermal,
            }
        chip_sets[zoom] = ChipSet(
            zoom=zoom,
            arrays=arrays,
            tile_grid=tile_grid,
            bbox_4326=tile_grid.bbox_4326,
            bbox_3857=tile_grid.bbox_3857,
            transform=tile_grid.transform,
        )
    return chip_sets


def _fixture_events() -> tuple[WildfireEvent, ...]:
    geometry = EventGeometry(
        type="Polygon",
        coordinates=[
            [
                (-101.0, 40.0),
                (-100.6, 40.0),
                (-100.6, 40.3),
                (-101.0, 40.3),
                (-101.0, 40.0),
            ]
        ],
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    event = WildfireEvent(
        event_id="fixture-event",
        title="Fixture Event",
        description=None,
        updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
        categories=("wildfires",),
        geometry=(geometry,),
        sources=(),
        bbox=(-101.2, 39.9, -100.4, 40.4),
    )
    return (event,)


def test_run_build_dataset_with_fixtures(monkeypatch, tmp_path: Path) -> None:
    events = _fixture_events()
    zooms = (8, 9)

    monkeypatch.setattr(
        build_dataset.eonet, "fetch_open_wildfire_events", lambda **_: events
    )
    monkeypatch.setattr(
        build_dataset.gibs,
        "build_chips",
        lambda event, times, zooms, **_: _synthetic_chip_set(event, times, zooms),
    )

    args = SimpleNamespace(
        output_dir=tmp_path,
        limit=1,
        zoom_levels=",".join(str(z) for z in zooms),
        time_offsets="-24,-12,-6,0",
        patch_size=64,
        stride=32,
        min_positive_ratio=0.01,
        fixtures=False,
        fixtures_count=1,
        seed=42,
        log_level="INFO",
    )

    summary = build_dataset.run(args)

    sample_files = list((tmp_path / "samples").glob("*.npz"))
    assert len(sample_files) >= 100
    assert summary["num_samples"] >= 100
    assert summary["positive_ratio_mean"] > 0
    assert summary["hotspot_overlap_mean"] >= 0
