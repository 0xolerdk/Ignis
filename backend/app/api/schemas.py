"""Pydantic schemas for the API surface."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """API health response payload."""

    status: str = Field(..., description="Overall health status of the service.")


class EventSummary(BaseModel):
    """Summary representation of a wildfire event."""

    event_id: str
    title: str
    last_update: Optional[datetime] = None
    area_sq_km: Optional[float] = None


class EventListResponse(BaseModel):
    """Response container for event listing."""

    events: List[EventSummary]
    status: str
    limit: int


class EventResponse(BaseModel):
    """Detailed event response."""

    event_id: str
    title: str
    last_update: Optional[datetime]
    geometry: Optional[dict]
    sources: Optional[List[str]] = None
    cached_tiles: Optional[List[str]] = None


class PredictionRequest(BaseModel):
    """Prediction request payload."""

    event_id: str
    time_iso: datetime
    horizons: List[int] = Field(default_factory=lambda: [6, 12, 24])
    mc_samples: int = 20


class PredictionMap(BaseModel):
    """Model prediction artifact metadata."""

    horizon: int
    perimeter: Dict[str, Any]
    mean_probability: float
    variance: float


class SegmentationResult(BaseModel):
    """Segmentation output summary."""

    mean_probability: float
    polygons: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Prediction response structure."""

    event_id: str
    time_iso: datetime
    horizons: List[int]
    segmentation: SegmentationResult
    predictions: List[PredictionMap]
    uncertainty: Dict[str, Dict[str, float]]


class ExplainResponse(BaseModel):
    """Explainability payload."""

    event_id: str
    time_iso: datetime
    summary: Optional[str] = None
    overlay_url: Optional[str] = None
