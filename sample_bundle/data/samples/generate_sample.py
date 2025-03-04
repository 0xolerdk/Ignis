import json
from pathlib import Path

import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)
    height = width = 64
    sequence_len = 4
    horizons = [6, 12, 24]

    base = np.zeros((height, width), dtype=np.float32)
    rr, cc = np.ogrid[:height, :width]
    mask = (rr - height / 2) ** 2 + (cc - width / 2) ** 2 <= (height / 4) ** 2
    base[mask] = 1.0

    inputs = []
    for t in range(sequence_len):
        frame = np.clip(
            base * (0.7 + 0.1 * t) + rng.normal(scale=0.05, size=base.shape), 0.0, 1.0
        )
        inputs.append(frame)
    inputs = np.stack(inputs, axis=0).astype(np.float32)

    targets = []
    for idx, _ in enumerate(horizons):
        frame = np.clip(
            base * (0.8 + 0.1 * idx) + rng.normal(scale=0.05, size=base.shape), 0.0, 1.0
        )
        targets.append(frame)
    targets = np.stack(targets, axis=0).astype(np.float32)

    metadata = {"event_id": "DEMO_001", "horizons": horizons}

    out_path = Path(__file__).with_suffix(".npz")
    np.savez_compressed(
        out_path, inputs=inputs, targets=targets, metadata=json.dumps(metadata)
    )


if __name__ == "__main__":
    main()
