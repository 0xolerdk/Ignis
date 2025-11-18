"""Dataset construction pipeline for wildfire nowcasting."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    cast,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from affine import Affine

from backend.app.services import (
    ChipSet,
    WildfireEvent,
    eonet,
    gibs,
    geo,
    tiling,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/processed/dataset")
DEFAULT_ZOOMS = (8, 9)
DEFAULT_TIME_OFFSETS = (-24, -12, -6, 0)
DEFAULT_PATCH_SIZE = 256


@dataclass
class SampleMeta:
    """Metadata stored with each dataset sample."""

    event_id: str
    event_title: str
    time_iso: str
    zoom: int
    x: int
    y: int
    patch_size: int
    transform: Tuple[float, float, float, float, float, float]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wildfire nowcasting dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of events to process"
    )
    parser.add_argument(
        "--zoom-levels",
        type=str,
        default=",".join(str(z) for z in DEFAULT_ZOOMS),
        help="Comma-separated list of zoom levels",
    )
    parser.add_argument(
        "--time-offsets",
        type=str,
        default=",".join(str(t) for t in DEFAULT_TIME_OFFSETS),
        help="Comma-separated hour offsets (e.g. -24,-12,-6,0)",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--min-positive-ratio", type=float, default=0.001)
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Use synthetic fixtures for tests instead of live services",
    )
    parser.add_argument(
        "--fixtures-count",
        type=int,
        default=3,
        help="Number of synthetic fixture events to generate when using fixtures",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for fixture generation"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def run(args: argparse.Namespace) -> Mapping[str, float]:
    configure_logging(args.log_level)

    output_dir: Path = args.output_dir
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    zoom_levels = tuple(
        int(z.strip()) for z in args.zoom_levels.split(",") if z.strip()
    )
    time_offsets = tuple(
        int(t.strip()) for t in args.time_offsets.split(",") if t.strip()
    )

    rng = np.random.default_rng(args.seed)

    if args.fixtures:
        events = _generate_fixture_events(args.fixtures_count, rng)

        def fetch_fixtures(**_: object) -> Sequence[WildfireEvent]:
            return events

        def build_fixtures(
            event: WildfireEvent,
            times: Sequence[str],
            zooms: Sequence[int],
            **_: object,
        ) -> Mapping[int, ChipSet]:
            return _generate_fixture_chips(event, times, zooms, rng)

        fetch_fn = cast(Callable, fetch_fixtures)
        build_fn = cast(Callable, build_fixtures)
    else:
        fetch_fn = eonet.fetch_open_wildfire_events
        build_fn = gibs.build_chips

    events = fetch_fn(limit=args.limit)
    if not events:
        LOGGER.warning("No events returned; dataset will be empty")
        return {}

    stats_tracker: Dict[str, List[float]] = {
        "positive_ratios": [],
        "hotspot_overlap": [],
        "polygon_overlap": [],
    }

    sample_count = 0
    for event in events[: args.limit]:
        LOGGER.info("Processing event %s", event.event_id)
        event_times = _resolve_times(event, time_offsets)
        chip_sets = build_fn(event, event_times, zoom_levels)

        for zoom, chip_set in chip_sets.items():
            for time_iso in event_times:
                normalized_time = gibs.normalize_time(time_iso)
                layer_map = chip_set.arrays.get(normalized_time)
                if not layer_map:
                    LOGGER.debug(
                        "No data for time %s at zoom %s", normalized_time, zoom
                    )
                    continue

                thermal_layer = _select_layer(layer_map, thermal=True)

                hotspot_mask = geo.hotspot_mask_from_thermal(thermal_layer)
                geometries = geo.geometries_from_event(event)
                polygon_mask = geo.rasterize_geometries(
                    geometries,
                    out_shape=(hotspot_mask.shape[0], hotspot_mask.shape[1]),
                    transform=chip_set.transform,
                )
                label_mask = geo.fuse_hotspots_and_polygons(hotspot_mask, polygon_mask)
                label_stats = geo.label_quality_stats(
                    label_mask, hotspot_mask, polygon_mask
                )

                stats_tracker["positive_ratios"].append(label_stats["positive_ratio"])
                stats_tracker["hotspot_overlap"].append(
                    label_stats["hotspot_overlap_ratio"]
                )
                stats_tracker["polygon_overlap"].append(
                    label_stats["polygon_overlap_ratio"]
                )

                patches = _extract_patches(
                    layer_map,
                    label_mask,
                    patch_size=args.patch_size,
                    stride=args.stride,
                )

                for patch_idx, (coords, patch_layers, label_patch) in enumerate(
                    patches
                ):
                    positive_ratio = label_patch.mean()
                    if positive_ratio < args.min_positive_ratio:
                        continue

                    x, y = coords
                    meta = SampleMeta(
                        event_id=event.event_id,
                        event_title=event.title,
                        time_iso=normalized_time,
                        zoom=zoom,
                        x=x,
                        y=y,
                        patch_size=args.patch_size,
                        transform=_patch_transform(
                            chip_set.transform, x, y, args.patch_size
                        ),
                    )

                    sample_path = (
                        samples_dir
                        / f"sample_{event.event_id}_{zoom}_{normalized_time}_{patch_idx}.npz"
                    )
                    np.savez(
                        sample_path,
                        label=label_patch.astype(np.uint8),
                        rgb=(
                            patch_layers.get("rgb")
                            if patch_layers.get("rgb") is not None
                            else np.array([])
                        ),
                        thermal=(
                            patch_layers.get("thermal")
                            if patch_layers.get("thermal") is not None
                            else np.array([])
                        ),
                        vegetation=(
                            patch_layers.get("vegetation")
                            if patch_layers.get("vegetation") is not None
                            else np.array([])
                        ),
                        metadata=json.dumps(asdict(meta)),
                    )
                    sample_count += 1

    summary = _summarize(stats_tracker, sample_count, len(events))
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Dataset build complete: %s", summary)
    return summary


def _resolve_times(event: WildfireEvent, offsets: Sequence[int]) -> Sequence[str]:
    geometry_dates = [geom.date for geom in event.geometry if geom.date]
    if geometry_dates:
        anchor = max(geometry_dates)
    else:
        anchor = event.updated
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    times = []
    for offset in offsets:
        target = anchor + timedelta(hours=offset)
        times.append(target.isoformat())
    return times


def _select_layer(layer_map: Mapping[str, np.ndarray], thermal: bool) -> np.ndarray:
    if thermal:
        for key in layer_map:
            if "thermal" in key.lower() or "hotspot" in key.lower():
                return layer_map[key]
        raise ValueError("No thermal layer found in layer map")
    else:
        for key in layer_map:
            if "correctedreflectance" in key.lower() or "truecolor" in key.lower():
                return layer_map[key]
        # default to first layer if no RGB
        return next(iter(layer_map.values()))


def _extract_patches(
    layer_map: Mapping[str, np.ndarray],
    label_mask: np.ndarray,
    *,
    patch_size: int,
    stride: int,
) -> Iterable[Tuple[Tuple[int, int], Mapping[str, np.ndarray], np.ndarray]]:
    height, width = label_mask.shape
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            label_patch = label_mask[y : y + patch_size, x : x + patch_size]
            patch_layers: Dict[str, np.ndarray] = {}
            for layer_name, array in layer_map.items():
                patch = array[y : y + patch_size, x : x + patch_size]
                canonical = _canonical_layer_name(layer_name)
                patch_layers[canonical] = patch
            yield (x, y), patch_layers, label_patch


def _canonical_layer_name(layer_name: str) -> str:
    lname = layer_name.lower()
    if "thermal" in lname:
        return "thermal"
    if "vegetation" in lname or "ndvi" in lname:
        return "vegetation"
    return "rgb"


def _patch_transform(
    transform: Affine, x: int, y: int, patch_size: int
) -> Tuple[float, float, float, float, float, float]:
    patch_transform = transform * Affine.translation(x, y)
    return tuple(patch_transform)  # type: ignore[return-value]


def _summarize(
    stats: Mapping[str, List[float]], samples: int, events: int
) -> Mapping[str, float]:
    summary = {
        "num_samples": int(samples),
        "num_events": int(events),
        "positive_ratio_mean": (
            float(np.mean(stats["positive_ratios"]))
            if stats["positive_ratios"]
            else 0.0
        ),
        "hotspot_overlap_mean": (
            float(np.mean(stats["hotspot_overlap"]))
            if stats["hotspot_overlap"]
            else 0.0
        ),
        "polygon_overlap_mean": (
            float(np.mean(stats["polygon_overlap"]))
            if stats["polygon_overlap"]
            else 0.0
        ),
    }
    return summary


def _generate_fixture_events(
    count: int, rng: np.random.Generator
) -> Sequence[WildfireEvent]:
    from backend.app.services.eonet import EventGeometry, WildfireEvent

    events: List[WildfireEvent] = []
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for idx in range(count):
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
            date=base_time,
        )
        event = WildfireEvent(
            event_id=f"fixture-{idx}",
            title=f"Fixture Event {idx}",
            description=None,
            updated=base_time,
            categories=("wildfires",),
            geometry=(geometry,),
            sources=(),
            bbox=(-101.2, 39.9, -100.4, 40.4),
        )
        events.append(event)
    return tuple(events)


def _generate_fixture_chips(
    event: WildfireEvent,
    times: Sequence[str],
    zooms: Sequence[int],
    rng: np.random.Generator,
) -> Mapping[int, ChipSet]:
    chip_sets: Dict[int, ChipSet] = {}
    for zoom in zooms:
        tile_grid = tiling.compute_tile_grid(
            event.bbox or (-101.2, 39.9, -100.4, 40.4), zoom
        )
        height = tile_grid.height_tiles * 256
        width = tile_grid.width_tiles * 256
        arrays: Dict[str, Dict[str, np.ndarray]] = {}
        for time_iso in times:
            normalized = gibs.normalize_time(time_iso)
            rgb = np.clip(
                rng.normal(loc=120, scale=40, size=(height, width, 3)), 0, 255
            ).astype(np.uint8)
            thermal = np.zeros((height, width, 4), dtype=np.uint8)
            hotspot_region = (
                slice(height // 4, height // 4 + height // 6),
                slice(width // 4, width // 4 + width // 6),
            )
            thermal[hotspot_region + (0,)] = 255
            thermal[..., 3] = 255
            arrays[normalized] = {
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
