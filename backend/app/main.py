"""Entrypoint for the FastAPI application."""

import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router


def _allowed_origins() -> List[str]:
    origins = os.getenv("FRONTEND_ORIGINS", "*")
    if origins.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


def create_app() -> FastAPI:
    """Create a configured FastAPI instance."""
    app = FastAPI(
        title="Wildfire Nowcasting & Impact Explorer",
        version="0.1.0",
        description=(
            "API for ingesting wildfire data, running segmentation and nowcasting models, "
            "and serving interactive visualizations."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
