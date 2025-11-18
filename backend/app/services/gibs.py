"""GIBS WMTS tile fetching and chip assembly utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx
import numpy as np
from PIL import Image
from affine import Affine

from .eonet import WildfireEvent
from .tiling import TileCoordinate, TileGrid, compute_tile_grid, stitch_tiles

GIBS_BASE_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"
DEFAULT_LAYERS = (
    "MODIS_Terra_CorrectedReflectance_TrueColor",
    "MODIS_Terra_Thermal_Anomalies_Night",
)
DEFAULT_TILE_SIZE = 256
DEFAULT_CACHE_DIR = Path("data/cache/gibs")


class GIBSServiceError(RuntimeError):
    """Raised when WMTS tiles cannot be fetched."""


@dataclass(frozen=True)
class ChipSet:
    """Container for mosaicked chips and georeferencing metadata."""

    zoom: int
    arrays: Mapping[str, Mapping[str, np.ndarray]]
    tile_grid: TileGrid
    bbox_4326: Tuple[float, float, float, float]
    bbox_3857: Tuple[float, float, float, float]
    transform: Affine


def build_tile_url(layer: str, time_iso: str, tile: TileCoordinate) -> str:
    """Construct a WMTS URL for the requested tile."""
    date_path = time_iso
    matrix_set = f"GoogleMapsCompatible_Level{tile.z}"
    return (
        f"{GIBS_BASE_URL}/{layer}/default/{date_path}/{matrix_set}/"
        f"{tile.z}/{tile.y}/{tile.x}.png"
    )


def fetch_tile(
    layer: str,
    time_iso: str,
    tile: TileCoordinate,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    client: Optional[httpx.Client] = None,
) -> np.ndarray:
    """
    Fetch a single WMTS tile, caching on disk and returning as a numpy array.

    Args:
        layer: GIBS layer identifier.
        time_iso: ISO date string supported by GIBS (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
        tile: Tile index.
        cache_dir: Base directory for cached tiles.
        client: Optional httpx client for re-use/testing.
    """
    cache_path = _tile_cache_path(cache_dir, layer, time_iso, tile)
    if cache_path.exists():
        return _read_tile(cache_path)

    url = build_tile_url(layer, time_iso, tile)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    session = client or httpx.Client(timeout=20.0)
    try:
        _download_tile(session, url, cache_path)
    except httpx.HTTPError as exc:
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)
        raise GIBSServiceError(f"Failed to fetch tile {url}") from exc
    finally:
        if client is None:
            session.close()

    return _read_tile(cache_path)


def build_chips(
    event: WildfireEvent,
    times: Sequence[str],
    zooms: Sequence[int],
    *,
    layers: Optional[Sequence[str]] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    client: Optional[httpx.Client] = None,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> Mapping[int, ChipSet]:
    """
    Build mosaicked chips for a wildfire event across times and zoom levels.

    Args:
        event: Wildfire event metadata.
        times: Iterable of ISO timestamps/dates compatible with GIBS.
        zooms: Sequence of zoom levels to fetch.
        layers: Iterable of GIBS layer identifiers. Defaults to canonical wildfire set.
        cache_dir: Base directory for tile cache.
        client: Optional httpx client to reuse connections.
        tile_size: Tile dimension in pixels.

    Returns:
        Mapping of zoom level to ChipSet containing arrays and georeferencing.
    """
    if not times or not zooms:
        raise ValueError("Times and zooms must be non-empty.")

    layer_list = tuple(layers or DEFAULT_LAYERS)
    bbox4326 = _event_bbox(event)

    results: Dict[int, ChipSet] = {}
    for zoom in zooms:
        tile_grid = compute_tile_grid(bbox4326, zoom, tile_size=tile_size)
        arrays: Dict[str, Dict[str, np.ndarray]] = {}

        for time_iso in times:
            layer_arrays: Dict[str, np.ndarray] = {}
            for layer in layer_list:
                tile_data: Dict[TileCoordinate, np.ndarray] = {}
                normalized_time = normalize_time(time_iso)
                for tile in tile_grid.tiles:
                    tile_array = fetch_tile(
                        layer,
                        normalized_time,
                        tile,
                        cache_dir=cache_dir,
                        client=client,
                    )
                    tile_data[tile] = tile_array

                mosaic = stitch_tiles(tile_data, tile_grid, tile_size=tile_size)
                layer_arrays[layer] = mosaic
            arrays[normalized_time] = layer_arrays

        chip_set = ChipSet(
            zoom=zoom,
            arrays=arrays,
            tile_grid=tile_grid,
            bbox_4326=tile_grid.bbox_4326,
            bbox_3857=tile_grid.bbox_3857,
            transform=tile_grid.transform,
        )
        results[zoom] = chip_set

    return results


def normalize_time(time_value: str) -> str:
    """Normalize timestamps to GIBS default format."""
    if "T" in time_value:
        try:
            dt = datetime.fromisoformat(time_value.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return time_value
    return time_value


def _event_bbox(event: WildfireEvent) -> Tuple[float, float, float, float]:
    if event.bbox:
        return event.bbox

    min_lon = min_lat = float("inf")
    max_lon = max_lat = float("-inf")
    for geometry in event.geometry:
        coords = _flatten_coordinates(geometry.coordinates)
        for lon, lat in coords:
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)

    if not np.isfinite([min_lon, max_lon, min_lat, max_lat]).all():
        raise ValueError("Unable to derive bounding box from event geometry.")

    return (min_lon, min_lat, max_lon, max_lat)


def _flatten_coordinates(coordinates: Sequence) -> Iterable[Tuple[float, float]]:
    """Flatten GeoJSON coordinate arrays to (lon, lat) pairs."""
    if not coordinates:
        return []

    if isinstance(coordinates[0], (float, int)) and len(coordinates) >= 2:
        lon, lat = float(coordinates[0]), float(coordinates[1])
        return [(lon, lat)]

    pairs: List[Tuple[float, float]] = []
    for sub in coordinates:
        pairs.extend(_flatten_coordinates(sub))
    return pairs


def _tile_cache_path(
    cache_dir: Path, layer: str, time_iso: str, tile: TileCoordinate
) -> Path:
    safe_layer = _sanitize(layer)
    safe_time = _sanitize(time_iso)
    return (
        cache_dir / safe_layer / safe_time / str(tile.z) / str(tile.y) / f"{tile.x}.png"
    )


def _sanitize(value: str) -> str:
    """Sanitize path components while preserving uniqueness."""
    cleaned = "".join(c for c in value if c.isalnum() or c in ("-", "_"))
    if cleaned:
        return cleaned
    # Fallback to hashed representation if string contains only special chars
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return digest


def _download_tile(client: httpx.Client, url: str, destination: Path) -> None:
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)


def _read_tile(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGBA"))


__all__ = [
    "ChipSet",
    "GIBSServiceError",
    "build_chips",
    "build_tile_url",
    "fetch_tile",
    "normalize_time",
]
