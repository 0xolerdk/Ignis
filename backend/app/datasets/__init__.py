"""Data loading utilities."""

from .segmentation import WildfireSegmentationDataset
from .nowcast import WildfireNowcastDataset

__all__ = ["WildfireSegmentationDataset", "WildfireNowcastDataset"]
