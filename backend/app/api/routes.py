"""API route declarations wired to inference and explainability services."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Dict, List

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from shapely.affinity import scale as scale_polygon
from shapely.geometry import Point, mapping

from backend.app.models.unet import WildfireUNet
from backend.app.services import run_explainability
from backend.app.services.samples import load_nowcast_sample, read_events_fixture

from .schemas import (
    EventListResponse,
    EventResponse,
    EventSummary,
    HealthResponse,
    PredictionMap,
    PredictionRequest,
    PredictionResponse,
    SegmentationResult,
)

router = APIRouter()


def _parse_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _load_events() -> List[Dict[str, object]]:
    fixture = read_events_fixture()
    events: List[Dict[str, object]] = []
    for item in fixture.get("events", []):
        center = item.get("center", [-121.5, 38.5])
        radius = item.get("radius_deg", 0.35)
        title = item.get("title", "Demo Wildfire")
        event_id = item.get("event_id", "DEMO_EVENT")
        polygon = Point(center[0], center[1]).buffer(radius)
        feature = {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "event_id": event_id,
                "title": title,
            },
        }
        events.append(
            {
                "event_id": event_id,
                "title": title,
                "last_update": _parse_datetime(
                    item.get("last_update", "2024-01-01T00:00:00Z")
                ),
                "area_sq_km": float(polygon.area * 12_000),
                "geometry": {"type": "FeatureCollection", "features": [feature]},
                "polygon": polygon,
                "bbox": polygon.bounds,
                "sources": item.get("sources", ["offline"]),
            }
        )
    return events


_EVENTS: List[Dict[str, object]] = _load_events()
_EVENT_INDEX = {event["event_id"]: event for event in _EVENTS}

_EXPLAIN_MODEL = WildfireUNet(in_channels=7, base_filters=4, dropout=0.0)
for param in _EXPLAIN_MODEL.parameters():
    param.data.zero_()
if hasattr(_EXPLAIN_MODEL.outc, "conv"):
    _EXPLAIN_MODEL.outc.conv.bias.data.fill_(4.0)
_EXPLAIN_MODEL.eval()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.get("/events", response_model=EventListResponse, tags=["events"])
async def list_events(status: str = "open", limit: int = 50) -> EventListResponse:
    """List wildfire events (stubbed dataset)."""
    del status
    summaries = [
        EventSummary(
            event_id=event["event_id"],
            title=event["title"],
            last_update=event["last_update"],
            area_sq_km=event["area_sq_km"],
        )
        for event in _EVENTS[:limit]
    ]
    return EventListResponse(events=summaries, status="open", limit=limit)


@router.get("/events/{event_id}", response_model=EventResponse, tags=["events"])
async def get_event(event_id: str) -> EventResponse:
    """Fetch detailed information about a wildfire event."""
    event = _EVENT_INDEX.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return EventResponse(
        event_id=event_id,
        title=event["title"],
        last_update=event["last_update"],
        geometry=event["geometry"],
        sources=event["sources"],
        cached_tiles=["thermal", "rgb"],
    )


def _feature_collection(polygon) -> Dict[str, object]:
    feature = {
        "type": "Feature",
        "geometry": mapping(polygon),
        "properties": {},
    }
    return {"type": "FeatureCollection", "features": [feature]}


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Run deterministic segmentation and nowcasting predictions for a given event/time."""
    event = _EVENT_INDEX.get(request.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    base_polygon = event["polygon"].buffer(0.0)
    segmentation_fc = _feature_collection(base_polygon)
    segmentation_result = SegmentationResult(
        mean_probability=0.82,
        polygons=segmentation_fc,
    )

    sample = load_nowcast_sample()
    horizons_sample = sample["metadata"].get("horizons", request.horizons)

    predictions: List[PredictionMap] = []
    uncertainty: Dict[str, Dict[str, float]] = {}
    for index, horizon in enumerate(request.horizons):
        scale_factor = 1.0 + (horizon / max(request.horizons)) * 0.35
        horizon_polygon = scale_polygon(
            base_polygon,
            xfact=scale_factor,
            yfact=scale_factor,
            origin=base_polygon.centroid,
        )
        feature_collection = _feature_collection(horizon_polygon)
        sample_idx = min(index, len(horizons_sample) - 1)
        mean_prob = float(np.clip(sample["targets"][sample_idx].mean(), 0.0, 1.0))
        variance = 0.015 + 0.003 * index
        predictions.append(
            PredictionMap(
                horizon=horizon,
                perimeter=feature_collection,
                mean_probability=mean_prob,
                variance=variance,
            )
        )
        uncertainty[str(horizon)] = {"variance": variance}

    return PredictionResponse(
        event_id=request.event_id,
        time_iso=request.time_iso,
        horizons=request.horizons,
        segmentation=segmentation_result,
        predictions=predictions,
        uncertainty=uncertainty,
    )


def _build_chip(width: int = 128, height: int = 128) -> Dict[str, np.ndarray]:
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    rgb = np.stack(
        [
            (xv * 255).astype(np.uint8),
            (yv * 255).astype(np.uint8),
            (0.3 * 255 * np.ones_like(xv)).astype(np.uint8),
        ],
        axis=-1,
    )
    thermal = np.zeros((height, width, 4), dtype=np.uint8)
    thermal[..., 0] = (xv * 255).astype(np.uint8)
    thermal[..., 1] = (yv * 255).astype(np.uint8)
    thermal[..., 3] = 255
    return {"rgb": rgb, "thermal": thermal}


@router.get(
    "/explain",
    responses={200: {"content": {"image/png": {}}}},
    tags=["inference"],
)
async def explain(event_id: str, time_iso: datetime) -> StreamingResponse:
    """Return Grad-CAM style explanation overlay as PNG."""
    if event_id not in _EVENT_INDEX:
        raise HTTPException(status_code=404, detail="Event not found")
    del time_iso

    chip = _build_chip()
    result = run_explainability(_EXPLAIN_MODEL, chip, device="cpu", target_layer="inc")
    overlay = result["overlay"]

    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
