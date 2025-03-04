"""Explainability utilities for segmentation models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import cv2
import numpy as np
import torch

from backend.app.models.unet import WildfireUNet


def grad_cam(
    model: WildfireUNet,
    inputs: torch.Tensor,
    target_layer: str,
    device: torch.device,
) -> np.ndarray:
    gradients = {}
    activations = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    layer = dict(model.named_modules())[target_layer]
    hook_forward = layer.register_forward_hook(forward_hook)
    hook_backward = layer.register_backward_hook(backward_hook)

    model.zero_grad()
    inputs = inputs.to(device)
    output = model(inputs)
    prob = torch.sigmoid(output)
    loss = prob.mean()
    loss.backward()

    hook_forward.remove()
    hook_backward.remove()

    acts = activations["value"].cpu().numpy()[0]
    grads = gradients["value"].cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for w, activation in zip(weights, acts):
        cam += w * activation
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (inputs.shape[-1], inputs.shape[-2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    return cam


def overlay_heatmap(
    base_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    if base_image.ndim == 2:
        base_image = np.stack([base_image] * 3, axis=-1)
    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(
        base_image.astype(np.uint8), 1 - alpha, heatmap_color, alpha, 0
    )
    return overlay


def run_explainability(
    model: WildfireUNet,
    chip: Mapping[str, np.ndarray],
    *,
    device: str = "cpu",
    target_layer: str = "inc",
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    device = torch.device(device)
    model.eval()
    channel_arrays = []
    for key in ("rgb", "thermal", "vegetation"):
        if key in chip and chip[key] is not None:
            arr = chip[key]
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)
            elif arr.ndim == 2:
                arr = arr[None, ...]
            channel_arrays.append(arr.astype(np.float32) / 255.0)
    if not channel_arrays:
        raise ValueError("Chip does not contain suitable feature arrays.")
    inputs = torch.from_numpy(np.concatenate(channel_arrays, axis=0)).unsqueeze(0)

    heatmap = grad_cam(model, inputs, target_layer, device)
    base_image = (
        chip.get("rgb") if chip.get("rgb") is not None else inputs[0, 0].cpu().numpy()
    )
    overlay = overlay_heatmap((base_image * 255).astype(np.uint8), heatmap)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "output_path": str(output_path) if output_path else "",
    }


__all__ = ["run_explainability"]
