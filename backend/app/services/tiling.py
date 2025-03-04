"""Spatial tiling and chip assembly helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np
from affine import Affine
from pyproj import Transformer

# Web Mercator latitude limits to avoid infinite projections
MAX_LAT = 85.0511287798066
MIN_LAT = -85.0511287798066

_WGS84_TO_WEBMERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


@dataclass(frozen=True)
class TileCoordinate:
    """Tile index in the WMTS GoogleMapsCompatible pyramid."""

    x: int
    y: int
    z: int


@dataclass(frozen=True)
class TileGrid:
    """Description of the tiles required to cover a bounding box."""

    zoom: int
    tiles: Tuple[TileCoordinate, ...]
    width_tiles: int
    height_tiles: int
    bbox_4326: Tuple[float, float, float, float]
    bbox_3857: Tuple[float, float, float, float]
    transform: Affine


def compute_tile_grid(
    bbox_lonlat: Sequence[float], zoom: int, tile_size: int = 256
) -> TileGrid:
    """
    Compute the WMTS tile grid covering a longitude/latitude bounding box.

    Args:
        bbox_lonlat: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
        zoom: GoogleMapsCompatible zoom level.
        tile_size: Tile pixel dimension (usually 256).

    Returns:
        TileGrid describing the tiles and geospatial metadata.
    """
    if len(bbox_lonlat) != 4:
        raise ValueError(
            "Bounding box must contain four values (min_lon, min_lat, max_lon, max_lat)."
        )

    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid bounding box ordering.")

    min_lat = max(min_lat, MIN_LAT)
    max_lat = min(max_lat, MAX_LAT)

    min_x = _lon_to_tile(min_lon, zoom, is_max=False)
    max_x = _lon_to_tile(max_lon, zoom, is_max=True)
    min_y = _lat_to_tile(max_lat, zoom, is_max=False)  # northern edge
    max_y = _lat_to_tile(min_lat, zoom, is_max=True)  # southern edge

    tiles = [
        TileCoordinate(x=x, y=y, z=zoom)
        for y in range(min_y, max_y + 1)
        for x in range(min_x, max_x + 1)
    ]

    width_tiles = max_x - min_x + 1
    height_tiles = max_y - min_y + 1

    bbox4326 = tile_grid_bounds_lonlat(min_x, min_y, max_x, max_y, zoom)
    bbox3857 = _to_3857_bounds(bbox4326)

    width_px = width_tiles * tile_size
    height_px = height_tiles * tile_size
    res_x = (bbox3857[2] - bbox3857[0]) / float(width_px)
    res_y = (bbox3857[3] - bbox3857[1]) / float(height_px)
    transform = Affine.translation(bbox3857[0], bbox3857[3]) * Affine.scale(
        res_x, -res_y
    )

    return TileGrid(
        zoom=zoom,
        tiles=tuple(tiles),
        width_tiles=width_tiles,
        height_tiles=height_tiles,
        bbox_4326=bbox4326,
        bbox_3857=bbox3857,
        transform=transform,
    )


def tile_grid_bounds_lonlat(
    min_x: int, min_y: int, max_x: int, max_y: int, zoom: int
) -> Tuple[float, float, float, float]:
    """Return the longitude/latitude bounds of a tile grid."""
    west = tile_x_to_lon(min_x, zoom)
    east = tile_x_to_lon(max_x + 1, zoom)
    north = tile_y_to_lat(min_y, zoom)
    south = tile_y_to_lat(max_y + 1, zoom)
    return (west, south, east, north)


def tile_bounds(tile: TileCoordinate) -> Tuple[float, float, float, float]:
    """Return the longitude/latitude bounds of a tile."""
    west = tile_x_to_lon(tile.x, tile.z)
    east = tile_x_to_lon(tile.x + 1, tile.z)
    north = tile_y_to_lat(tile.y, tile.z)
    south = tile_y_to_lat(tile.y + 1, tile.z)
    return (west, south, east, north)


def stitch_tiles(
    tile_arrays: Mapping[TileCoordinate, np.ndarray],
    grid: TileGrid,
    tile_size: int = 256,
) -> np.ndarray:
    """
    Stitch individual tile arrays into a single mosaic.

    Args:
        tile_arrays: Mapping of TileCoordinate to numpy array (H, W[, C]).
        grid: TileGrid describing output dimensions.
        tile_size: Expected per-tile pixel dimension.

    Returns:
        Numpy array of shape (grid_height_px, grid_width_px, channels).
    """
    if not tile_arrays:
        raise ValueError("No tile arrays provided for stitching.")

    sample_tile = next(iter(tile_arrays.values()))
    sample_array = _ensure_channels(sample_tile)
    channels = sample_array.shape[2]
    dtype = sample_array.dtype

    mosaic_height = grid.height_tiles * tile_size
    mosaic_width = grid.width_tiles * tile_size
    mosaic = np.zeros((mosaic_height, mosaic_width, channels), dtype=dtype)

    x_indices = sorted({tile.x for tile in grid.tiles})
    y_indices = sorted({tile.y for tile in grid.tiles})

    tile_lookup = {
        (tile.x, tile.y): _ensure_channels(arr) for tile, arr in tile_arrays.items()
    }

    for row_idx, tile_y in enumerate(y_indices):
        for col_idx, tile_x in enumerate(x_indices):
            key = (tile_x, tile_y)
            if key not in tile_lookup:
                raise ValueError(f"Missing tile array for ({tile_x}, {tile_y}).")
            arr = tile_lookup[key]
            if arr.shape[0] != tile_size or arr.shape[1] != tile_size:
                raise ValueError(
                    "Tile array dimensions do not match expected tile size."
                )
            y_start = row_idx * tile_size
            y_end = y_start + tile_size
            x_start = col_idx * tile_size
            x_end = x_start + tile_size
            mosaic[y_start:y_end, x_start:x_end, :] = arr

    return mosaic


def _ensure_channels(array: np.ndarray) -> np.ndarray:
    """Ensure input array has shape (H, W, C)."""
    if array.ndim == 2:
        return np.expand_dims(array, axis=-1)
    if array.ndim == 3:
        return array
    raise ValueError(f"Unsupported array dimensions: {array.shape}")


def _lon_to_tile(lon: float, zoom: int, *, is_max: bool) -> int:
    n = 1 << zoom
    x = (lon + 180.0) / 360.0 * n
    if is_max:
        x = math.nextafter(x, math.inf)
    return int(math.floor(x))


def _lat_to_tile(lat: float, zoom: int, *, is_max: bool) -> int:
    lat = min(max(lat, MIN_LAT), MAX_LAT)
    lat_rad = math.radians(lat)
    n = 1 << zoom
    y = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    if is_max:
        y = math.nextafter(y, math.inf)
    return int(math.floor(y))


def tile_x_to_lon(x: int, zoom: int) -> float:
    """Convert tile X index to longitude."""
    n = 1 << zoom
    return x / n * 360.0 - 180.0


def tile_y_to_lat(y: int, zoom: int) -> float:
    """Convert tile Y index to latitude."""
    n = 1 << zoom
    rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(rad)


def _to_3857_bounds(bbox4326: Sequence[float]) -> Tuple[float, float, float, float]:
    west, south, east, north = bbox4326
    x_min, y_min = _WGS84_TO_WEBMERCATOR.transform(west, south)
    x_max, y_max = _WGS84_TO_WEBMERCATOR.transform(east, north)
    return (x_min, y_min, x_max, y_max)


__all__ = [
    "TileCoordinate",
    "TileGrid",
    "compute_tile_grid",
    "tile_bounds",
    "stitch_tiles",
]
