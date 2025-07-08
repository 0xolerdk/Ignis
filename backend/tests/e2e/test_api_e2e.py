"""End-to-end API smoke tests using canned data."""

from io import BytesIO

from datetime import datetime
from fastapi.testclient import TestClient

from backend.app.api.schemas import (
    EventListResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.main import create_app

client = TestClient(create_app())


def test_api_e2e_predict_and_explain(tmp_path):
    events_response = client.get("/api/events")
    assert events_response.status_code == 200
    events_payload = events_response.json()
    parsed_events = EventListResponse.model_validate(events_payload)
    assert parsed_events.events, "Expected at least one event from API"

    event = parsed_events.events[0]
    if isinstance(event.last_update, datetime):
        time_iso = event.last_update.isoformat()
    else:
        time_iso = event.last_update or "2024-01-01T00:00:00Z"

    predict_request = PredictionRequest(event_id=event.event_id, time_iso=time_iso)
    prediction_response = client.post(
        "/api/predict", json=predict_request.model_dump(mode="json")
    )
    assert prediction_response.status_code == 200
    prediction_payload = prediction_response.json()
    parsed_prediction = PredictionResponse.model_validate(prediction_payload)
    assert parsed_prediction.predictions, "Expected nowcast predictions"
    assert parsed_prediction.segmentation.polygons[
        "features"
    ], "Segmentation polygons missing"

    explain_response = client.get(
        "/api/explain",
        params={"event_id": event.event_id, "time_iso": time_iso},
    )
    assert explain_response.status_code == 200
    assert explain_response.headers["content-type"] == "image/png"
    content = explain_response.content
    assert len(content) > 0

    from PIL import Image

    image = Image.open(BytesIO(content))
    image.verify()
    Image.open(BytesIO(content)).save(tmp_path / "explain.png")
