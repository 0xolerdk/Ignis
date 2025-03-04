"""Service helpers for interacting with the NASA EONET API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from time import monotonic, sleep
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import httpx


EONET_BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3"
WILDFIRE_CATEGORY = "wildfires"
DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.8


class EONETServiceError(RuntimeError):
    """Raised when the EONET service cannot be reached."""


@dataclass(frozen=True)
class EventGeometry:
    """Normalized representation of an event geometry entry."""

    type: str
    coordinates: Any
    date: datetime
    magnitude_value: Optional[float] = None
    magnitude_unit: Optional[str] = None


@dataclass(frozen=True)
class WildfireEvent:
    """Normalized structure for a wildfire event."""

    event_id: str
    title: str
    description: Optional[str]
    updated: datetime
    categories: Tuple[str, ...]
    geometry: Tuple[EventGeometry, ...]
    sources: Tuple[str, ...]
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_feature_collection(self) -> Dict[str, Any]:
        """Convert the event geometry into a GeoJSON FeatureCollection."""
        return events_to_feature_collection(self)


CacheKey = Tuple[str, int, Optional[int], Optional[str]]
CacheEntry = Tuple[float, Tuple[WildfireEvent, ...]]

_EONET_CACHE: MutableMapping[CacheKey, CacheEntry] = {}
_CACHE_LOCK = Lock()


def _cache_lookup(
    key: CacheKey, cache_seconds: int
) -> Optional[Tuple[WildfireEvent, ...]]:
    if cache_seconds <= 0:
        return None
    now = monotonic()
    with _CACHE_LOCK:
        entry = _EONET_CACHE.get(key)
        if entry and entry[0] > now:
            return entry[1]
    return None


def _cache_store(
    key: CacheKey, cache_seconds: int, value: Tuple[WildfireEvent, ...]
) -> None:
    if cache_seconds <= 0:
        return
    expires = monotonic() + cache_seconds
    with _CACHE_LOCK:
        _EONET_CACHE[key] = (expires, value)


def clear_cache() -> None:
    """Clear the in-memory EONET cache (useful for tests)."""
    with _CACHE_LOCK:
        _EONET_CACHE.clear()


def fetch_open_wildfire_events(
    status: str = "open",
    limit: int = 50,
    days: Optional[int] = None,
    cache_seconds: int = 300,
    session: Optional[httpx.Client] = None,
) -> Tuple[WildfireEvent, ...]:
    """
    Fetch wildfire events from EONET, normalizing the payload.

    Args:
        status: EONET event status filter (defaults to ``open``).
        limit: Maximum number of events to return.
        days: Optional look-back window in days.
        cache_seconds: Cache lifetime for identical requests; set to ``0`` to disable caching.
        session: Optional httpx client for advanced usage/testing.

    Returns:
        Tuple of :class:`WildfireEvent` objects.
    """
    base_url = f"{EONET_BASE_URL}/events"
    params: Dict[str, Any] = {
        "status": status,
        "limit": limit,
        "category": WILDFIRE_CATEGORY,
    }
    if days is not None:
        params["days"] = days

    cache_key: CacheKey = (status, limit, days, None)
    cached = _cache_lookup(cache_key, cache_seconds)
    if cached is not None:
        return cached

    client = session or httpx.Client(timeout=DEFAULT_TIMEOUT)
    try:
        payload = _get_with_retries(client, base_url, params=params)
    finally:
        if session is None:
            client.close()

    events = tuple(_normalize_event(event) for event in payload.get("events", []))
    _cache_store(cache_key, cache_seconds, events)
    return events


def events_to_feature_collection(event: WildfireEvent) -> Dict[str, Any]:
    """Convert a :class:`WildfireEvent` into a GeoJSON FeatureCollection."""
    features: List[Dict[str, Any]] = []
    for geometry in event.geometry:
        feature: Dict[str, Any] = {
            "type": "Feature",
            "id": f"{event.event_id}:{geometry.date.isoformat()}",
            "properties": {
                "event_id": event.event_id,
                "title": event.title,
                "updated": event.updated.isoformat(),
                "geometry_date": geometry.date.isoformat(),
                "magnitude_value": geometry.magnitude_value,
                "magnitude_unit": geometry.magnitude_unit,
            },
            "geometry": {
                "type": geometry.type,
                "coordinates": geometry.coordinates,
            },
        }
        features.append(feature)

    feature_collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }
    if event.bbox:
        feature_collection["bbox"] = list(event.bbox)
    return feature_collection


def _get_with_retries(
    client: httpx.Client,
    url: str,
    params: Optional[Mapping[str, Any]] = None,
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF,
) -> Mapping[str, Any]:
    attempt = 0
    while True:
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            attempt += 1
            if attempt >= retries:
                raise EONETServiceError(
                    f"Failed to fetch data from EONET: {exc}"
                ) from exc
            sleep(backoff_factor * (2 ** (attempt - 1)))


def _normalize_event(raw_event: Mapping[str, Any]) -> WildfireEvent:
    event_id = str(raw_event.get("id", ""))
    title = str(raw_event.get("title", ""))
    description = raw_event.get("description")
    updated = _parse_datetime(raw_event.get("updated"))
    categories = tuple(cat.get("title", "") for cat in raw_event.get("categories", []))
    sources = tuple(
        src.get("url", "") for src in raw_event.get("sources", []) if src.get("url")
    )
    geometry_entries = tuple(
        _normalize_geometry(geometry) for geometry in raw_event.get("geometry", [])
    )
    bbox = _normalize_bbox(raw_event.get("bbox"))

    return WildfireEvent(
        event_id=event_id,
        title=title,
        description=description,
        updated=updated,
        categories=categories,
        geometry=geometry_entries,
        sources=sources,
        bbox=bbox,
    )


def _normalize_geometry(entry: Mapping[str, Any]) -> EventGeometry:
    geometry_type = str(entry.get("type", "Point"))
    coordinates = _normalize_coordinates(entry.get("coordinates"))
    geometry_date = _parse_datetime(entry.get("date"))
    magnitude_value = _optional_float(entry.get("magnitudeValue"))
    magnitude_unit = entry.get("magnitudeUnit")

    return EventGeometry(
        type=geometry_type,
        coordinates=coordinates,
        date=geometry_date,
        magnitude_value=magnitude_value,
        magnitude_unit=magnitude_unit,
    )


def _normalize_coordinates(data: Any) -> Any:
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, (list, tuple)):
        return [_normalize_coordinates(item) for item in data]
    raise ValueError(f"Unsupported coordinate format: {data!r}")


def _normalize_bbox(
    bbox: Optional[Sequence[Any]],
) -> Optional[Tuple[float, float, float, float]]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        return tuple(float(coord) for coord in bbox)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bbox format: {bbox}") from exc


def _parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "EONETServiceError",
    "EventGeometry",
    "WildfireEvent",
    "fetch_open_wildfire_events",
    "events_to_feature_collection",
    "clear_cache",
]
