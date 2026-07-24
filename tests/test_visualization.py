"""
tests/test_visualization.py
────────────────────────────
Unit tests for src.utils.visualization module.
"""

import numpy as np
from PIL import Image
import pytest

from src.utils.visualization import (
    draw_detections,
    plot_uncertainty_distribution,
    create_side_by_side_comparison,
)


@pytest.fixture
def sample_image():
    """Return a 300x300 RGB synthetic image."""
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, :] = [40, 50, 60]
    return Image.fromarray(arr)


@pytest.fixture
def sample_detections():
    """Return sample detections containing both known and unknown objects."""
    return [
        {
            "box": [20, 20, 100, 100],
            "conf": 0.92,
            "cls": 0,
            "name": "Pedestrian",
            "uncertainty": 0.15,
            "is_unknown": False,
        },
        {
            "box": [120, 120, 220, 200],
            "conf": 0.85,
            "cls": 1,
            "name": "Vehicle",
            "uncertainty": 0.22,
            "is_unknown": False,
        },
        {
            "box": [210, 50, 280, 150],
            "conf": 0.65,
            "cls": -1,
            "name": "unknown",
            "uncertainty": 0.78,
            "is_unknown": True,
        },
    ]


def test_draw_detections(sample_image, sample_detections):
    annotated = draw_detections(sample_image, sample_detections, show_uncertainty=True)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == sample_image.size


def test_plot_uncertainty_distribution(sample_detections):
    fig = plot_uncertainty_distribution(sample_detections, threshold=0.6, metric_name="entropy")
    assert fig is not None


def test_create_side_by_side_comparison(sample_image, sample_detections):
    annotated = draw_detections(sample_image, sample_detections)
    composite = create_side_by_side_comparison(sample_image, annotated)
    assert isinstance(composite, Image.Image)
    w, h = composite.size
    assert w > sample_image.width
