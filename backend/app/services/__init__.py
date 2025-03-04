"""Data access and processing service layer."""

from .eonet import (
    EONETServiceError,
    EventGeometry,
    WildfireEvent,
    clear_cache as clear_eonet_cache,
    events_to_feature_collection,
    fetch_open_wildfire_events,
)
from .epic import (
    EPICServiceError,
    EpicImage,
    clear_cache as clear_epic_cache,
    fetch_epic_metadata,
)
from .donki import (
    DONKIServiceError,
    DonkiNotification,
    clear_cache as clear_donki_cache,
    fetch_recent_activity,
)
from .gibs import (
    ChipSet,
    GIBSServiceError,
    build_chips,
    build_tile_url,
    fetch_tile,
    normalize_time,
)
from .geo import (
    fuse_hotspots_and_polygons,
    geometries_from_event,
    hotspot_mask_from_thermal,
    label_quality_stats,
    rasterize_geometries,
)
from .inference import (
    SegmentationModel,
    NowcastModel,
    load_segmentation_model,
    load_nowcast_model,
    run_pipeline,
    run_segmentation,
    run_nowcasting,
)
from .explain import run_explainability
from .tiling import TileCoordinate, TileGrid, compute_tile_grid, stitch_tiles

__all__ = [
    "EONETServiceError",
    "EventGeometry",
    "WildfireEvent",
    "fetch_open_wildfire_events",
    "events_to_feature_collection",
    "clear_eonet_cache",
    "EPICServiceError",
    "EpicImage",
    "fetch_epic_metadata",
    "clear_epic_cache",
    "DONKIServiceError",
    "DonkiNotification",
    "fetch_recent_activity",
    "clear_donki_cache",
    "ChipSet",
    "GIBSServiceError",
    "build_chips",
    "build_tile_url",
    "fetch_tile",
    "normalize_time",
    "SegmentationModel",
    "NowcastModel",
    "load_segmentation_model",
    "load_nowcast_model",
    "run_pipeline",
    "run_segmentation",
    "run_nowcasting",
    "run_explainability",
    "geometries_from_event",
    "rasterize_geometries",
    "hotspot_mask_from_thermal",
    "fuse_hotspots_and_polygons",
    "label_quality_stats",
    "TileCoordinate",
    "TileGrid",
    "compute_tile_grid",
    "stitch_tiles",
]
