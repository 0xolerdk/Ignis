"""Tests for GIBS tile fetching and chip assembly."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.app.services import gibs
from backend.app.services.eonet import EventGeometry, WildfireEvent
from backend.tests.conftest import vcr_instance

TEST_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
TEST_TIME = "2013-06-16"
TEST_ZOOM = 9


def _make_event() -> WildfireEvent:
    geometry = EventGeometry(
        type="Point",
        coordinates=[-100.9, 26.1],
        date=datetime(2013, 6, 16, tzinfo=timezone.utc),
    )
    return WildfireEvent(
        event_id="test-event",
        title="Test Event",
        description=None,
        updated=datetime(2013, 6, 16, tzinfo=timezone.utc),
        categories=("wildfires",),
        geometry=(geometry,),
        sources=(),
        bbox=(-101.25, 25.8, -100.55, 26.43),
    )


@vcr_instance.use_cassette("gibs_build_chips.yaml")
def test_build_chips_returns_mosaics(tmp_path: Path) -> None:
    event = _make_event()
    zooms = [TEST_ZOOM]
    times = [TEST_TIME]

    chips = gibs.build_chips(
        event,
        times,
        zooms,
        layers=[TEST_LAYER],
        cache_dir=tmp_path / "cache",
    )

    assert zooms[0] in chips
    chip_set = chips[zooms[0]]
    time_key = gibs.normalize_time(TEST_TIME)
    array = chip_set.arrays[time_key][TEST_LAYER]
    assert isinstance(array, np.ndarray)
    assert array.ndim == 3
    grid = chip_set.tile_grid
    expected_height = grid.height_tiles * 256
    expected_width = grid.width_tiles * 256
    assert array.shape[0] == expected_height
    assert array.shape[1] == expected_width
    assert chip_set.transform == grid.transform
    # Ensure transform aligns with bbox
    west, south, east, north = chip_set.bbox_4326
    if event.bbox is not None:
        assert west <= event.bbox[0] <= east
        assert south <= event.bbox[1] <= north
    # Cache files should exist
    cached_files = list((tmp_path / "cache").rglob("*.png"))
    assert len(cached_files) == len(grid.tiles)


def test_build_chips_uses_cache(tmp_path: Path) -> None:
    event = _make_event()
    zooms = [TEST_ZOOM]
    times = [TEST_TIME]

    with vcr_instance.use_cassette("gibs_build_chips_cache.yaml"):
        gibs.build_chips(
            event,
            times,
            zooms,
            layers=[TEST_LAYER],
            cache_dir=tmp_path / "cache",
        )

    with patch("backend.app.services.gibs._download_tile") as mocked_download:
        gibs.build_chips(
            event,
            times,
            zooms,
            layers=[TEST_LAYER],
            cache_dir=tmp_path / "cache",
        )
        mocked_download.assert_not_called()
