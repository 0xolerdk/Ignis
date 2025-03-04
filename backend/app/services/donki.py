"""Space weather notes via NASA DONKI (optional stretch)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as DateType, datetime, timedelta, timezone
from threading import Lock
from time import monotonic, sleep
from typing import Any, List, Mapping, MutableMapping, Optional, Tuple, Union

import httpx


DONKI_BASE_URL = "https://api.nasa.gov/DONKI/notifications"
DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.8
DEFAULT_LOOKBACK_DAYS = 5
DEFAULT_CACHE_SECONDS = 1800


class DONKIServiceError(RuntimeError):
    """Raised when DONKI notifications cannot be retrieved."""


@dataclass(frozen=True)
class DonkiNotification:
    """Normalized DONKI notification payload."""

    message_id: str
    message_type: str
    message_issue_time: datetime
    message_body: str
    link: Optional[str]
    source: Optional[str]


CacheKey = Tuple[str, str, str]
CacheEntry = Tuple[float, Tuple[DonkiNotification, ...]]
_DONKI_CACHE: MutableMapping[CacheKey, CacheEntry] = {}
_CACHE_LOCK = Lock()


def clear_cache() -> None:
    """Clear the DONKI cache."""
    with _CACHE_LOCK:
        _DONKI_CACHE.clear()


def fetch_recent_activity(
    start_date: Optional[Union[str, datetime, DateType]] = None,
    end_date: Optional[Union[str, datetime, DateType]] = None,
    api_key: Optional[str] = None,
    cache_seconds: int = DEFAULT_CACHE_SECONDS,
    session: Optional[httpx.Client] = None,
) -> Tuple[DonkiNotification, ...]:
    """
    Fetch recent DONKI notifications within the provided window.

    Args:
        start_date: Inclusive lower bound; defaults to ``today - DEFAULT_LOOKBACK_DAYS``.
        end_date: Inclusive upper bound; defaults to ``today``.
        api_key: NASA API key; falls back to environment or ``DEMO_KEY``.
        cache_seconds: Cache retention; ``0`` disables caching.
        session: Optional reusable httpx client.
    """
    start = _format_date(
        start_date
        or (datetime.now(tz=timezone.utc) - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    )
    end = _format_date(end_date or datetime.now(tz=timezone.utc))
    api_key = api_key or _default_api_key()
    cache_key = (start, end, api_key)
    cached = _cache_lookup(cache_key, cache_seconds)
    if cached is not None:
        return cached

    params = {"startDate": start, "endDate": end, "type": "all", "api_key": api_key}
    client = session or httpx.Client(timeout=DEFAULT_TIMEOUT)
    try:
        payload = _get_with_retries(client, DONKI_BASE_URL, params=params)
    finally:
        if session is None:
            client.close()

    notifications = tuple(_normalize_notification(item) for item in payload)
    _cache_store(cache_key, cache_seconds, notifications)
    return notifications


def _format_date(value: Union[str, datetime, DateType]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return value.strftime("%Y-%m-%d")


def _default_api_key() -> str:
    import os

    return os.getenv("NASA_API_KEY", "DEMO_KEY")


def _cache_lookup(
    key: CacheKey, cache_seconds: int
) -> Optional[Tuple[DonkiNotification, ...]]:
    if cache_seconds <= 0:
        return None
    now = monotonic()
    with _CACHE_LOCK:
        entry = _DONKI_CACHE.get(key)
        if entry and entry[0] > now:
            return entry[1]
    return None


def _cache_store(
    key: CacheKey, cache_seconds: int, value: Tuple[DonkiNotification, ...]
) -> None:
    if cache_seconds <= 0:
        return
    expires = monotonic() + cache_seconds
    with _CACHE_LOCK:
        _DONKI_CACHE[key] = (expires, value)


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
                raise DONKIServiceError("Unexpected DONKI response format.")
            return payload
        except httpx.HTTPError as exc:
            attempt += 1
            if attempt >= retries:
                raise DONKIServiceError(f"Failed to fetch DONKI data: {exc}") from exc
            sleep(backoff_factor * (2 ** (attempt - 1)))


def _normalize_notification(entry: Mapping[str, Any]) -> DonkiNotification:
    issue_time = _parse_datetime(entry.get("messageIssueTime"))
    return DonkiNotification(
        message_id=str(entry.get("messageID", "")),
        message_type=str(entry.get("messageType", "")),
        message_issue_time=issue_time,
        message_body=str(entry.get("messageBody", "")),
        link=entry.get("link"),
        source=entry.get("source"),
    )


def _parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


__all__ = [
    "DonkiNotification",
    "DONKIServiceError",
    "fetch_recent_activity",
    "clear_cache",
]
