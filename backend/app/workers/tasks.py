"""Celery task definitions for long-running jobs."""

from celery import Celery

from backend.app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "wildfire_nowcasting",
    broker=settings.redis_url,
    backend=settings.redis_url,
)


@celery_app.task
def queue_prediction(event_id: str, time_iso: str) -> str:
    """Placeholder Celery task to queue a prediction job."""
    return f"queued:{event_id}:{time_iso}"
