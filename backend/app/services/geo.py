"""Geospatial helpers for coordinate transforms and label generation."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Tuple

import numpy as np
import rasterio.features
from affine import Affine
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from skimage.morphology import binary_dilation, disk

# WildfireEvent only used for typing convenience in signatures.
from .eonet import WildfireEvent


def geometries_from_event(event: WildfireEvent) -> List[BaseGeometry]:
    """Convert event geometries into Shapely objects."""
    geometries: List[BaseGeometry] = []
    for geom in event.geometry:
        try:
            geometries.append(
                shape({"type": geom.type, "coordinates": geom.coordinates})
            )
        except (ValueError, TypeError):
            continue
    return [geom for geom in geometries if not geom.is_empty]


def rasterize_geometries(
    geometries: Iterable[BaseGeometry],
    out_shape: Tuple[int, int],
    transform: Affine,
    *,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize geometries into a binary mask."""
    shapes = ((geom, 1) for geom in geometries if not geom.is_empty)
    mask = rasterio.features.rasterize(
        shapes=list(shapes),
        out_shape=out_shape,
        transform=transform,
        all_touched=all_touched,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def hotspot_mask_from_thermal(thermal: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Derive hotspot mask from thermal anomaly imagery."""
    if thermal.ndim == 3:
        red_channel = thermal[..., 0]
    else:
        red_channel = thermal
    mask = red_channel.astype(np.int16) > threshold
    return mask.astype(np.uint8)


def fuse_hotspots_and_polygons(
    hotspot_mask: np.ndarray,
    polygon_mask: np.ndarray,
    hotspot_dilation: int = 1,
    polygon_dilation: int = 0,
) -> np.ndarray:
    """Combine hotspot and polygon masks into a pseudo-perimeter label."""
    hotspot = hotspot_mask.astype(bool)
    polygon = polygon_mask.astype(bool)

    if hotspot_dilation > 0:
        footprint = disk(max(1, hotspot_dilation))
        hotspot = binary_dilation(hotspot, footprint=footprint)

    if polygon_dilation > 0:
        footprint = disk(max(1, polygon_dilation))
        polygon = binary_dilation(polygon, footprint=footprint)

    fused = np.logical_or(hotspot, polygon)
    return fused.astype(np.uint8)


def label_quality_stats(
    label_mask: np.ndarray,
    hotspot_mask: np.ndarray,
    polygon_mask: np.ndarray,
) -> Mapping[str, float]:
    """Compute basic quality metrics for generated labels."""
    total_pixels = float(label_mask.size)
    positives = float(label_mask.sum())
    hotspot_overlap = float(np.logical_and(label_mask, hotspot_mask).sum())
    polygon_overlap = float(np.logical_and(label_mask, polygon_mask).sum())

    return {
        "positive_ratio": positives / total_pixels if total_pixels else 0.0,
        "hotspot_overlap_ratio": hotspot_overlap / positives if positives else 0.0,
        "polygon_overlap_ratio": polygon_overlap / positives if positives else 0.0,
    }


__all__ = [
    "geometries_from_event",
    "rasterize_geometries",
    "hotspot_mask_from_thermal",
    "fuse_hotspots_and_polygons",
    "label_quality_stats",
]
