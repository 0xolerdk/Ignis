"""Tests for EPIC metadata integration."""

from datetime import date
from unittest.mock import patch

from backend.app.services import epic
from backend.tests.conftest import vcr_instance


@vcr_instance.use_cassette("epic_metadata.yaml")
def test_fetch_epic_metadata_returns_records() -> None:
    epic.clear_cache()
    records = epic.fetch_epic_metadata(
        date(2024, 3, 1), api_key="DEMO_KEY", cache_seconds=3600
    )
    assert records, "Expected EPIC metadata records for the given date"

    first = records[0]
    asset_url = first.asset_url(api_key="DEMO_KEY")
    assert asset_url.endswith(".png?api_key=DEMO_KEY")
    assert "/2024/03/01/" in asset_url


def test_fetch_epic_metadata_uses_cache() -> None:
    epic.clear_cache()

    with vcr_instance.use_cassette("epic_metadata_cache.yaml"):
        epic.fetch_epic_metadata("2024-03-01", api_key="DEMO_KEY", cache_seconds=3600)

    with patch("backend.app.services.epic._get_with_retries") as mocked_get:
        epic.fetch_epic_metadata("2024-03-01", api_key="DEMO_KEY", cache_seconds=3600)
        mocked_get.assert_not_called()
