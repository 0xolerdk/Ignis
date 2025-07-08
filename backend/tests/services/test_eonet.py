"""Tests for the EONET service integration."""

from unittest.mock import patch

from backend.app.services import eonet
from backend.tests.conftest import vcr_instance


@vcr_instance.use_cassette("eonet_open_events.yaml")
def test_fetch_open_wildfire_events_returns_events() -> None:
    eonet.clear_cache()
    events = eonet.fetch_open_wildfire_events(limit=3, cache_seconds=3600)
    assert events, "Expected to receive at least one wildfire event"

    first_event = events[0]
    feature_collection = eonet.events_to_feature_collection(first_event)

    assert feature_collection["type"] == "FeatureCollection"
    assert feature_collection[
        "features"
    ], "Feature collection should include geometries"


def test_fetch_open_wildfire_events_uses_cache() -> None:
    eonet.clear_cache()

    with vcr_instance.use_cassette("eonet_open_events_cache.yaml"):
        events_initial = eonet.fetch_open_wildfire_events(limit=2, cache_seconds=3600)

    with patch("backend.app.services.eonet._get_with_retries") as mocked_get:
        events_cached = eonet.fetch_open_wildfire_events(limit=2, cache_seconds=3600)
        mocked_get.assert_not_called()
    assert events_cached == events_initial
