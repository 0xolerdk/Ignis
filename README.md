# Wildfire Nowcasting & Impact Explorer

An end-to-end research platform that ingests NASA wildfire feeds, builds segmentation and nowcasting models, and serves interactive forecasts through a React + Leaflet UI.

## Features
- **Data ingestion & preprocessing** – Scripts to fetch EONET wildfire events, stitch GIBS tiles, and build pseudo-labels.
- **Model training** – U-Net segmentation and ConvLSTM nowcasting pipelines with MC-dropout for uncertainty quantification.
- **Inference & explainability** – FastAPI endpoints for prediction and Grad-CAM overlays, returning GeoJSON polygons and heatmap PNGs.
- **Interactive frontend** – React + TypeScript dashboard with Leaflet layers, horizon scrubbing, explain overlays, and downloads.
- **CI-ready** – GitHub Actions workflow covering linting, tests, type checking, and Docker builds.

## Prerequisites
- Python 3.11+
- Node.js 20+
- GDAL system libraries (`gdal-bin`, `libgdal-dev`) for rasterio
- Docker (optional, for container builds)

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd frontend
npm install
```
Copy the environment template and fill in secrets:
```bash
cp .env.example .env
# edit .env to include NASA_API_KEY, etc.
```

## Dataset generation
Synthetic fixtures are available for quick experiments:
```bash
make dataset
```
This downloads/creates sample chips at `data/processed/dataset/samples`. Customize pipeline parameters via `DATASET_ARGS` in the `Makefile`.

## Training
Segmentation (U-Net) and nowcasting (ConvLSTM) models can be trained independently:
```bash
make train-seg     # trains segmentation model, saves checkpoints to runs/segmentation
make train-nowcast # trains nowcasting model, saves checkpoints to runs/nowcast
```
Metrics and reliability plots are written alongside checkpoints. Adjust hyperparameters with CLI flags in `backend/scripts/train_segmentation.py` and `backend/scripts/train_nowcast.py`.

## Serving
Run the backend locally with auto-reload:
```bash
make serve
```
Launch the frontend in a separate terminal:
```bash
make frontend
```
The React app (default `http://localhost:5173`) calls the FastAPI backend (`http://localhost:8000`). Layer toggles, time scrubber, and explain overlays update automatically after selecting an event.

## Testing & linting
```bash
make lint  # Ruff + Black + ESLint
make test  # Pytest suite
```
Type checking:
```bash
env PYTHONPATH=backend mypy backend
```

## Docker builds
```bash
docker compose up --build
```
This builds the backend, frontend, worker, and Redis services for an integrated stack.

### Offline demo bundle

To run without external downloads, the repository ships with a synthetic bundle under `sample_bundle/` containing:
- canned events (`data/events.json`)
- a mini perimeter sequence (`data/samples/demo_sample.npz`)
- tiny checkpoints (`models/segmentation_demo.pt`, `models/nowcast_demo.pt`)

The FastAPI service automatically falls back to these assets. You can regenerate the synthetic sample with:

```bash
python sample_bundle/data/samples/generate_sample.py
```

After launching `docker compose up --build`, open the frontend and select the demo event to visualise predictions immediately.

## Notebooks
Sample notebooks under `notebooks/` demonstrate:
- `01_event_eda.ipynb` – synthetic event feature exploration.
- `02_model_ablation.ipynb` – comparing segmentation ablations with mock metrics.

Both notebooks run without external data and can be executed via `jupyter notebook notebooks/` or `jupyter nbconvert --execute`.

## CI pipeline
`/.github/workflows/ci.yml` runs on every push/PR:
1. Installs Python & Node dependencies.
2. Lints backend/frontend sources.
3. Runs pytest & mypy.
4. Builds backend and frontend Docker images.

## Project layout
```
backend/    FastAPI app, data pipelines, models, tests
frontend/   React + Vite dashboard (Tailwind, Leaflet)
data/       Generated datasets (gitignored)
models/     Saved checkpoints (gitignored)
notebooks/  EDA and ablation examples
```

## Environment variables
| Variable | Description |
|----------|-------------|
| `NASA_API_KEY` | Required for live NASA APIs |
| `FRONTEND_ORIGINS` | Optional comma-separated list to tighten CORS |
| `REDIS_URL` | Celery/Redis connection URL |

## Contributing
1. Fork and clone.
2. Create a feature branch.
3. Run `make lint` and `make test` before pushing.
4. Submit a pull request targeting `main`.

## License
MIT
