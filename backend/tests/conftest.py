"""Shared pytest fixtures for service tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import vcr

CASSETTE_DIR = Path(__file__).parent / "cassettes"
CASSETTE_DIR.mkdir(parents=True, exist_ok=True)

vcr_instance = vcr.VCR(
    serializer="yaml",
    cassette_library_dir=str(CASSETTE_DIR),
    record_mode="once",
    filter_headers=["api_key"],
)


@pytest.fixture(scope="session")
def vcr_cassette_dir() -> Path:
    """Expose cassette directory for tests that need to inspect files."""
    return CASSETTE_DIR
