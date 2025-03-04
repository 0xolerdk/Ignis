"""Optional EPIC API integration for cloud cover context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as DateType, datetime, timezone
from threading import Lock
from time import monotonic, sleep
from typing import (
    Any,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import httpx


EPIC_BASE_URL = "https://api.nasa.gov/EPIC/api/natural/date"
EPIC_ARCHIVE_BASE_URL = "https://api.nasa.gov/EPIC/archive/natural"
DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.8
DEFAULT_CACHE_SECONDS = 3600


class EPICServiceError(RuntimeError):
    """Raised when EPIC metadata cannot be retrieved."""


@dataclass(frozen=True)
class EpicImage:
    """Normalized representation of an EPIC image metadata record."""

    identifier: str
    caption: str
    image: str
    version: str
    date: datetime
    centroid_coordinates: Mapping[str, float]
    dscovr_j2000_position: Mapping[str, float]
    lunar_j2000_position: Mapping[str, float]
    sun_j2000_position: Mapping[str, float]
    attitude_quaternions: Mapping[str, float]

    def asset_url(self, api_key: Optional[str] = None, variant: str = "png") -> str:
        """Generate the direct asset URL for the image."""
        api_key = api_key or _default_api_key()
        dt = self.date.astimezone(timezone.utc)
        return (
            f"{EPIC_ARCHIVE_BASE_URL}/{dt:%Y/%m/%d}/{variant}/{self.image}.{variant}"
            f"?api_key={api_key}"
        )


CacheKey = Tuple[str, str]
CacheEntry = Tuple[float, Tuple[EpicImage, ...]]
_EPIC_CACHE: MutableMapping[CacheKey, CacheEntry] = {}
_CACHE_LOCK = Lock()


def clear_cache() -> None:
    """Clear the EPIC request cache."""
    with _CACHE_LOCK:
        _EPIC_CACHE.clear()


def fetch_epic_metadata(
    target_date: Union[str, datetime, DateType],
    api_key: Optional[str] = None,
    cache_seconds: int = DEFAULT_CACHE_SECONDS,
    session: Optional[httpx.Client] = None,
) -> Tuple[EpicImage, ...]:
    """
    Fetch EPIC metadata for a given date, returning normalized records.

    Args:
        target_date: Date to request imagery for.
        api_key: NASA API key; defaults to environment ``NASA_API_KEY`` or ``DEMO_KEY``.
        cache_seconds: Cache retention; set to ``0`` to disable caching.
        session: Optional httpx client for testing.
    """
    formatted_date = _format_date(target_date)
    api_key = api_key or _default_api_key()
    cache_key = (formatted_date, api_key)
    cached = _cache_lookup(cache_key, cache_seconds)
    if cached is not None:
        return cached

    client = session or httpx.Client(timeout=DEFAULT_TIMEOUT)
    url = f"{EPIC_BASE_URL}/{formatted_date}"
    params = {"api_key": api_key}
    try:
        payload = _get_with_retries(client, url, params=params)
    finally:
        if session is None:
            client.close()

    records = tuple(_normalize_record(record) for record in payload)
    _cache_store(cache_key, cache_seconds, records)
    return records


def _format_date(target_date: Union[str, datetime, DateType]) -> str:
    if isinstance(target_date, str):
        return target_date
    if isinstance(target_date, datetime):
        return target_date.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return target_date.strftime("%Y-%m-%d")


def _default_api_key() -> str:
    import os

    return os.getenv("NASA_API_KEY", "DEMO_KEY")


def _cache_lookup(key: CacheKey, cache_seconds: int) -> Optional[Tuple[EpicImage, ...]]:
    if cache_seconds <= 0:
        return None
    now = monotonic()
    with _CACHE_LOCK:
        entry = _EPIC_CACHE.get(key)
        if entry and entry[0] > now:
            return entry[1]
    return None


def _cache_store(
    key: CacheKey, cache_seconds: int, value: Tuple[EpicImage, ...]
) -> None:
    if cache_seconds <= 0:
        return
    expires = monotonic() + cache_seconds
    with _CACHE_LOCK:
        _EPIC_CACHE[key] = (expires, value)


def _get_with_retries(
    client: httpx.Client,
    url: str,
    params: Optional[Mapping[str, Any]] = None,
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF,
) -> List[Mapping[str, Any]]:
    attempt = 0
    while True:
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise EPICServiceError("Unexpected EPIC response format.")
            return payload
        except httpx.HTTPError as exc:
            attempt += 1
            if attempt >= retries:
                raise EPICServiceError(f"Failed to fetch EPIC data: {exc}") from exc
            sleep(backoff_factor * (2 ** (attempt - 1)))


def _normalize_record(record: Mapping[str, Any]) -> EpicImage:
    timestamp = record.get("date") or record.get("identifier")
    dt = _parse_datetime(timestamp)
    return EpicImage(
        identifier=str(record.get("identifier", "")),
        caption=str(record.get("caption", "")),
        image=str(record.get("image", "")),
        version=str(record.get("version", "")),
        date=dt,
        centroid_coordinates=_ensure_mapping(record.get("centroid_coordinates", {})),
        dscovr_j2000_position=_ensure_mapping(record.get("dscovr_j2000_position", {})),
        lunar_j2000_position=_ensure_mapping(record.get("lunar_j2000_position", {})),
        sun_j2000_position=_ensure_mapping(record.get("sun_j2000_position", {})),
        attitude_quaternions=_ensure_mapping(record.get("attitude_quaternions", {})),
    )


def _ensure_mapping(data: Any) -> Mapping[str, float]:
    if isinstance(data, Mapping):
        return {key: float(value) for key, value in data.items()}
    raise ValueError(f"Expected mapping for EPIC coordinates, got {type(data)}")


def _parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


__all__ = ["EPICServiceError", "EpicImage", "fetch_epic_metadata", "clear_cache"]
