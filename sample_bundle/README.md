# Sample Bundle

This directory contains a small offline dataset that allows the demo stack (backend + frontend) to run without network access.

Contents:
- `data/events.json` — list of canned events consumed by the API during fixture mode.
- `data/samples/demo_sample.npz` — synthetic perimeter sequence used by the nowcasting inference pipeline.
- `models/segmentation_demo.pt` — tiny U-Net checkpoint with constant logits.
- `models/nowcast_demo.pt` — tiny ConvLSTM checkpoint producing deterministic forecasts.

To regenerate the synthetic sample NPZ file:
```bash
python sample_bundle/data/samples/generate_sample.py
```
