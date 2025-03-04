"""Configuration models and settings for the backend."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration via environment variables."""

    nasa_api_key: str = Field(default="", alias="NASA_API_KEY")
    gibs_base_url: str = Field(
        default="https://gibs.earthdata.nasa.gov", alias="GIBS_BASE"
    )
    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
