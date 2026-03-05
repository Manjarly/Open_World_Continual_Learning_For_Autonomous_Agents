"""
src/data/transforms.py
──────────────────────
Shared augmentation pipeline for both Waymo and nuScenes.

Wraps Albumentations transforms with a unified API that works
for bounding box regression tasks (YOLO format).

Usage:
    from src.data.transforms import get_train_transforms, get_val_transforms

    train_tfm = get_train_transforms(img_size=640)
    result = train_tfm(image=np_img, bboxes=bboxes, class_labels=labels)
"""

from typing import List, Tuple

import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def get_train_transforms(img_size: int = 640):
    """
    Full augmentation pipeline for training.

    Includes geometric + photometric augmentations suitable for
    autonomous driving data (weather, lighting variation, camera jitter).
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return _fallback_transform(img_size)

    return A.Compose(
        [
            # ── Geometric ────────────────────────────────────────────────
            A.RandomResizedCrop(
                height=img_size, width=img_size,
                scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=5, border_mode=0, p=0.4
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.2),

            # ── Photometric ───────────────────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.2),

            # ── Weather simulation ────────────────────────────────────────
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=15, drop_width=1,
                    rain_type=None, p=1.0
                ),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=1.0),
            ], p=0.15),

            # ── Normalize & tensorize ─────────────────────────────────────
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_val_transforms(img_size: int = 640):
    """
    Minimal deterministic transforms for validation / inference.
    Only resize + normalize — no augmentation.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return _fallback_transform(img_size)

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size,
                border_mode=0, value=(114, 114, 114)
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.1,
        ),
    )


def _fallback_transform(img_size: int):
    """Simple numpy-based transform when Albumentations is not installed."""

    class FallbackTransform:
        def __init__(self, size):
            self.size = size

        def __call__(self, image: np.ndarray, bboxes: List, class_labels: List):
            from PIL import Image
            pil = Image.fromarray(image).resize((self.size, self.size))
            arr = np.array(pil, dtype=np.float32) / 255.0
            return {"image": arr, "bboxes": bboxes, "class_labels": class_labels}

    return FallbackTransform(img_size)
