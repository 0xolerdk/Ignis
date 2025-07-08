"""Tests for tiling utilities."""

import numpy as np

from backend.app.services import tiling


def test_compute_tile_grid_for_small_bbox() -> None:
    bbox = (-122.5, 37.7, -122.3, 37.85)
    zoom = 6

    grid = tiling.compute_tile_grid(bbox, zoom)

    assert grid.zoom == zoom
    assert grid.width_tiles >= 1
    assert grid.height_tiles >= 1
    west, south, east, north = grid.bbox_4326
    assert west <= bbox[0] <= east
    assert south <= bbox[1] <= north
    assert west <= bbox[2] <= east
    assert south <= bbox[3] <= north


def test_stitch_tiles_shapes() -> None:
    bbox = (-122.5, 37.7, -122.3, 37.85)
    zoom = 5
    grid = tiling.compute_tile_grid(bbox, zoom)

    tile_arrays = {
        tile: np.ones((256, 256, 1), dtype=np.uint8) * 255 for tile in grid.tiles
    }
    mosaic = tiling.stitch_tiles(tile_arrays, grid)
    assert mosaic.shape[0] == grid.height_tiles * 256
    assert mosaic.shape[1] == grid.width_tiles * 256
    assert mosaic.shape[2] == 1
