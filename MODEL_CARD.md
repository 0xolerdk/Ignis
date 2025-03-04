# Wildfire Nowcasting Model Card

## Model details
- **Authors:** Ignis project contributors
- **Version:** 0.1.0 (research prototype)
- **Architectures:**
  - Segmentation: WildfireUNet (U-Net with dropout, BCE + Dice objective)
  - Nowcasting: ConvLSTM stack with MC-Dropout heads predicting 6/12/24 hour horizons
- **Frameworks:** PyTorch 2.x, FastAPI for serving

## Intended use
- **Purpose:** Provide situational awareness of ongoing wildfires by estimating current perimeters and near-term spread with uncertainty bounds.
- **Primary users:** Emergency managers, wildfire analysts, and researchers evaluating fire-behavior models.
- **Usage notes:** Predictions are decision-support aids, not authoritative wildfire boundaries; human validation is required before acting.

## Data
- **Inputs:** NASA EONET wildfire events, MODIS/VIIRS GIBS tiles (RGB, thermal anomaly, vegetation proxy), synthetic fixtures for development.
- **Preprocessing:** Tile stitching in EPSG:3857, normalization per channel, pseudo-label perimeter generation via buffered polygons + thermal hotspots.
- **Dataset pipeline:** `backend/scripts/build_dataset.py` orchestrates fetching, tiling, label fusion, and patch creation. Fixtures can be produced via `make dataset`.

## Training
- **Segmentation:** `make train-seg` (or `backend/scripts/train_segmentation.py`) trains U-Net with BCE+Dice, MC-dropout, and Grad-CAM support. Metrics recorded in `runs/segmentation`.
- **Nowcasting:** `make train-nowcast` trains the ConvLSTM using historical perimeter sequences; Brier scores and reliability curves saved in `runs/nowcast`.
- **Hyperparameters:** Configurable via CLI flags (learning rate, dropout, horizons, MC samples, etc.).

## Evaluation metrics
- **Segmentation:** Dice coefficient, IoU, pixel-wise accuracy.
- **Nowcasting:** Brier score per horizon, reliability diagrams, calibration curves.
- **Operational:** Inference latency, polygon vertex counts, MC variance summaries.

## Uncertainty & explainability
- **MC-Dropout:** Segmentation and nowcasting both support configurable MC sampling (`mc_samples`), returning mean/variance per horizon.
- **Explainability:** `backend/app/services/explain.py` exposes Grad-CAM overlays aligned to input tiles via `/api/explain`.
- **Confidence communication:** API responses include mean probability and variance; frontend visualizes uncertainty heatmaps with adjustable opacity.

## Limitations & ethical considerations
- Synthetic fixtures are not representative of real wildfire complexity; production deployment requires training on curated historical events.
- Sensor coverage gaps, clouds, and thermal anomalies from non-fire sources can degrade accuracy.
- Predictions can influence emergency response—clearly communicate uncertainty and avoid automated decision making without expert oversight.
- Monitor for data drift (seasonal changes, sensor updates) and retrain periodically.

## Maintenance & versioning
- Checkpoints stored in `models/` (gitignored). Document model provenance, training date, and dataset version before releasing.
- CI (`.github/workflows/ci.yml`) ensures linting, tests, mypy, and docker builds stay green.
- Future work: incorporate meteorological covariates, retraining automation, and human-in-the-loop corrections.
