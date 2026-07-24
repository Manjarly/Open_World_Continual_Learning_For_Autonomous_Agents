"""
src/utils/visualization.py
────────────────────────────
Visualization utilities for Open-World Continual Learning.

Provides functions to:
  1. Draw bounding boxes and labels on images with distinct color-coding
     for known vs. unknown (high-entropy) objects.
  2. Plot uncertainty/entropy distributions across detected objects.
  3. Generate side-by-side comparison images for UI and evaluation reports.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Default known class mappings (Waymo / nuScenes standard)
DEFAULT_CLASS_NAMES = {
    0: "Pedestrian",
    1: "Vehicle",
    2: "Cyclist",
    3: "Sign",
    -1: "UNKNOWN",
}

# Color palette (RGB format for PIL)
COLOR_KNOWN_DEFAULT = (0, 230, 118)      # Emerald / Bright Green
COLOR_UNKNOWN_DEFAULT = (255, 23, 68)     # Crimson / Bright Red
COLOR_TEXT_BG = (20, 24, 33)              # Dark Navy/Slate background for label tags


def draw_detections(
    image: Union[Image.Image, np.ndarray, str],
    detections: List[Dict],
    class_names: Optional[Dict[int, str]] = None,
    known_color: Tuple[int, int, int] = COLOR_KNOWN_DEFAULT,
    unknown_color: Tuple[int, int, int] = COLOR_UNKNOWN_DEFAULT,
    box_thickness: int = 3,
    show_uncertainty: bool = True,
) -> Image.Image:
    """
    Draw bounding boxes, labels, confidence scores, and uncertainty flags on an image.

    Args:
        image: PIL Image, NumPy array (HWC RGB/BGR), or file path string.
        detections: List of detection dicts with 'box', 'conf', 'cls', and optional 'is_unknown', 'uncertainty'.
        class_names: Dict mapping class_id -> class label name.
        known_color: RGB tuple for known class bounding boxes.
        unknown_color: RGB tuple for flagged unknown object bounding boxes.
        box_thickness: Line thickness for bounding box outlines.
        show_uncertainty: Whether to include uncertainty score in label text.

    Returns:
        Annotated PIL Image (RGB).
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Load / convert to PIL Image in RGB mode
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            raise ValueError(f"Invalid image array shape: {image.shape}")
    elif isinstance(image, Image.Image):
        pil_img = image.copy().convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    draw = ImageDraw.Draw(pil_img)
    width, height = pil_img.size

    for det in detections:
        box = det.get("box", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in box]

        # Clip box coordinates within image boundaries
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        is_unknown = det.get("is_unknown", False)
        conf = det.get("conf", 0.0)
        cls_id = det.get("cls", 0)
        uncertainty = det.get("uncertainty", None)

        if is_unknown or cls_id == -1:
            color = unknown_color
            class_str = class_names.get(-1, "UNKNOWN")
        else:
            color = known_color
            class_str = class_names.get(cls_id, f"Class {cls_id}")

        # Label construction
        if show_uncertainty and uncertainty is not None:
            label_text = f"{class_str} {conf:.2f} (u:{uncertainty:.2f})"
        else:
            label_text = f"{class_str} {conf:.2f}"

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)

        # Estimate text dimensions
        char_w, char_h = 7, 14
        text_w = len(label_text) * char_w + 6
        text_h = char_h + 4

        # Position label above box if space permits, otherwise inside top
        label_y1 = max(0, y1 - text_h) if y1 - text_h >= 0 else y1
        label_y2 = label_y1 + text_h
        label_x2 = min(width - 1, x1 + text_w)

        # Draw label background tag
        draw.rectangle([x1, label_y1, label_x2, label_y2], fill=COLOR_TEXT_BG, outline=color)
        draw.text((x1 + 3, label_y1 + 2), label_text, fill=color)

    return pil_img


def plot_uncertainty_distribution(
    detections: List[Dict],
    threshold: Optional[float] = None,
    metric_name: str = "entropy",
) -> Optional[object]:
    """
    Generate a matplotlib histogram figure showing uncertainty score distribution.

    Args:
        detections: List of detection dicts containing 'uncertainty' keys.
        threshold: Optional threshold line to display on plot.
        metric_name: Name of metric ('entropy', 'max_softmax', 'energy').

    Returns:
        matplotlib.figure.Figure instance or None if no detections.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib is not installed. Skipping plot generation.")
        return None

    scores = [d.get("uncertainty", 0.0) for d in detections if "uncertainty" in d]
    if not scores:
        return None

    is_unknown = [d.get("is_unknown", False) for d in detections if "uncertainty" in d]
    known_scores = [s for s, unk in zip(scores, is_unknown) if not unk]
    unknown_scores = [s for s, unk in zip(scores, is_unknown) if unk]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#161B22")

    bins = np.linspace(0.0, 1.0, 25)

    if known_scores:
        ax.hist(
            known_scores,
            bins=bins,
            alpha=0.7,
            color="#00E676",
            label=f"Known Detections (n={len(known_scores)})",
            edgecolor="#00C853",
        )

    if unknown_scores:
        ax.hist(
            unknown_scores,
            bins=bins,
            alpha=0.7,
            color="#FF1744",
            label=f"Flagged Unknowns (n={len(unknown_scores)})",
            edgecolor="#D50000",
        )

    if threshold is not None:
        ax.axvline(
            threshold,
            color="#FFD600",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold:.2f})",
        )

    ax.set_title(f"Open-Set Uncertainty Distribution ({metric_name})", color="#FFFFFF", fontsize=12, pad=10)
    ax.set_xlabel("Uncertainty Score", color="#C9D1D9", fontsize=10)
    ax.set_ylabel("Detection Count", color="#C9D1D9", fontsize=10)
    ax.tick_params(colors="#8B949E")
    ax.spines['bottom'].set_color('#30363D')
    ax.spines['left'].set_color('#30363D')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#F0F6FC")

    fig.tight_layout()
    return fig


def create_side_by_side_comparison(
    raw_image: Union[Image.Image, np.ndarray, str],
    annotated_image: Image.Image,
    title1: str = "Input Frame",
    title2: str = "OWCL Model Predictions",
) -> Image.Image:
    """
    Combine original image and annotated prediction image side-by-side.

    Args:
        raw_image: Input raw image.
        annotated_image: Annotated prediction PIL image.
        title1: Title banner for raw frame.
        title2: Title banner for annotated frame.

    Returns:
        Combined side-by-side PIL Image.
    """
    if isinstance(raw_image, str):
        img1 = Image.open(raw_image).convert("RGB")
    elif isinstance(raw_image, np.ndarray):
        img1 = Image.fromarray(raw_image.astype(np.uint8)).convert("RGB")
    else:
        img1 = raw_image.convert("RGB")

    img2 = annotated_image.convert("RGB")

    # Resize img2 to match img1 dimensions if needed
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.BILINEAR)

    w, h = img1.size
    banner_h = 30
    composite_w = w * 2 + 10
    composite_h = h + banner_h

    composite = Image.new("RGB", (composite_w, composite_h), color=(14, 17, 23))
    draw = ImageDraw.Draw(composite)

    # Paste images
    composite.paste(img1, (0, banner_h))
    composite.paste(img2, (w + 10, banner_h))

    # Banners
    draw.rectangle([0, 0, w, banner_h], fill=(22, 27, 34))
    draw.rectangle([w + 10, 0, composite_w, banner_h], fill=(22, 27, 34))

    draw.text((10, 8), title1, fill=(240, 246, 252))
    draw.text((w + 20, 8), title2, fill=(0, 230, 118))

    return composite
