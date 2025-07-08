"""Tests for the DONKI service integration."""

from datetime import date
from unittest.mock import patch

from backend.app.services import donki
from backend.tests.conftest import vcr_instance


@vcr_instance.use_cassette("donki_notifications.yaml")
def test_fetch_recent_activity_returns_notifications() -> None:
    donki.clear_cache()
    notifications = donki.fetch_recent_activity(
        start_date=date(2024, 3, 1),
        end_date=date(2024, 3, 3),
        api_key="DEMO_KEY",
        cache_seconds=1800,
    )
    assert notifications, "Expected DONKI notifications for sample window"
    first = notifications[0]
    assert first.message_id
    assert first.message_body


def test_fetch_recent_activity_uses_cache() -> None:
    donki.clear_cache()

    with vcr_instance.use_cassette("donki_notifications_cache.yaml"):
        donki.fetch_recent_activity(
            start_date="2024-03-01",
            end_date="2024-03-03",
            api_key="DEMO_KEY",
            cache_seconds=1800,
        )

    with patch("backend.app.services.donki._get_with_retries") as mocked_get:
        donki.fetch_recent_activity(
            start_date="2024-03-01",
            end_date="2024-03-03",
            api_key="DEMO_KEY",
            cache_seconds=1800,
        )
        mocked_get.assert_not_called()
