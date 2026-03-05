"""
src/data/waymo_loader.py
────────────────────────
Waymo Open Dataset loader.

Reads raw TFRecord files, extracts camera frames and 2D bounding box labels,
and exports them into a YOLO-compatible flat directory structure for training.

Usage (CLI):
    python -m src.data.waymo_loader \
        --raw_dir data/waymo/ \
        --out_dir data/processed/waymo/ \
        --splits 0.8 0.1 0.1

Requires:
    pip install waymo-open-dataset-tf-2-12-0
"""

import os
import io
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Waymo class ID → unified label name ─────────────────────────────────────
WAYMO_CLASS_MAP: Dict[int, str] = {
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "sign",
}

LABEL_TO_IDX: Dict[str, int] = {v: i for i, v in enumerate(WAYMO_CLASS_MAP.values())}


# ── Try importing the Waymo SDK ──────────────────────────────────────────────
try:
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils
    WAYMO_SDK_AVAILABLE = True
except ImportError:
    WAYMO_SDK_AVAILABLE = False
    logger.warning(
        "waymo-open-dataset SDK not found. "
        "Install with: pip install waymo-open-dataset-tf-2-12-0\n"
        "Loader will run in MOCK mode for development/testing."
    )


class WaymoLoader:
    """
    Parses Waymo TFRecord files and exports YOLO-format labels.

    Output structure:
        out_dir/
          images/train/  *.jpg
          images/val/    *.jpg
          images/test/   *.jpg
          labels/train/  *.txt   (YOLO format: cls cx cy w h, normalized)
          labels/val/    *.txt
          labels/test/   *.txt
    """

    def __init__(
        self,
        raw_dir: str,
        out_dir: str,
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        camera_name: int = 1,   # FRONT camera
        seed: int = 42,
    ):
        assert abs(sum(splits) - 1.0) < 1e-6, "Splits must sum to 1.0"
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.splits = splits
        self.camera_name = camera_name
        self.rng = np.random.default_rng(seed)

        self._create_dirs()

    def _create_dirs(self):
        for split in ("train", "val", "test"):
            (self.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Split assignment ─────────────────────────────────────────────────────

    def _assign_split(self, frame_id: str) -> str:
        """Deterministically assign a frame to a split via hashing."""
        h = int(hashlib.md5(frame_id.encode()).hexdigest(), 16) / (16 ** 32)
        if h < self.splits[0]:
            return "train"
        elif h < self.splits[0] + self.splits[1]:
            return "val"
        return "test"

    # ── YOLO label formatting ────────────────────────────────────────────────

    @staticmethod
    def _bbox_to_yolo(
        cx: float, cy: float, w: float, h: float,
        img_w: int, img_h: int
    ) -> Tuple[float, float, float, float]:
        """Convert absolute pixel bbox → normalized YOLO format."""
        return (
            cx / img_w,
            cy / img_h,
            w  / img_w,
            h  / img_h,
        )

    # ── Real SDK parsing ─────────────────────────────────────────────────────

    def _parse_tfrecord(self, tfrecord_path: Path) -> List[Dict]:
        """Parse a single TFRecord file and return frame dicts."""
        if not WAYMO_SDK_AVAILABLE:
            raise RuntimeError("Waymo SDK not installed.")

        frames = []
        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

        for raw_bytes in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(raw_bytes.numpy())

            # Find the requested camera image
            camera_image = None
            for img in frame.images:
                if img.name == self.camera_name:
                    camera_image = img
                    break
            if camera_image is None:
                continue

            # Decode JPEG
            pil_img = Image.open(io.BytesIO(camera_image.image))
            img_w, img_h = pil_img.size

            # Collect 2D labels for this camera
            labels = []
            for cam_label in frame.camera_labels:
                if cam_label.name != self.camera_name:
                    continue
                for label in cam_label.labels:
                    cls_name = WAYMO_CLASS_MAP.get(label.type)
                    if cls_name is None:
                        continue
                    b = label.box
                    cx_n, cy_n, w_n, h_n = self._bbox_to_yolo(
                        b.center_x, b.center_y, b.length, b.width,
                        img_w, img_h
                    )
                    labels.append({
                        "class_idx": LABEL_TO_IDX[cls_name],
                        "cx": cx_n, "cy": cy_n, "w": w_n, "h": h_n,
                    })

            frame_id = f"{frame.context.name}_{frame.timestamp_micros}"
            frames.append({
                "frame_id": frame_id,
                "image": pil_img,
                "labels": labels,
            })

        return frames

    # ── Mock mode (no SDK) ───────────────────────────────────────────────────

    def _generate_mock_frames(self, n: int = 100) -> List[Dict]:
        """Generate synthetic frames for development without the Waymo SDK."""
        logger.info(f"[MOCK] Generating {n} synthetic Waymo frames.")
        frames = []
        for i in range(n):
            img_w, img_h = 640, 640  # Use smaller size for faster training
            # Generate random RGB array and create valid PIL image
            arr = self.rng.integers(0, 255, (img_h, img_w, 3), dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            num_boxes = self.rng.integers(1, 6)
            labels = []
            for _ in range(num_boxes):
                cls_idx = self.rng.integers(0, len(LABEL_TO_IDX))
                cx = float(self.rng.uniform(0.2, 0.8))
                cy = float(self.rng.uniform(0.2, 0.8))
                w  = float(self.rng.uniform(0.05, 0.25))
                h  = float(self.rng.uniform(0.05, 0.2))
                labels.append({"class_idx": int(cls_idx), "cx": cx, "cy": cy, "w": w, "h": h})
            frames.append({
                "frame_id": f"mock_waymo_{i:05d}",
                "image": img,
                "labels": labels,
            })
        return frames

    # ── Public API ───────────────────────────────────────────────────────────

    def process(self) -> Dict[str, int]:
        """
        Process all TFRecords (or mock data) and write images + labels.

        Returns:
            counts: {"train": N, "val": N, "test": N}
        """
        counts = {"train": 0, "val": 0, "test": 0}

        if WAYMO_SDK_AVAILABLE:
            tfrecords = sorted(self.raw_dir.glob("**/*.tfrecord"))
            if not tfrecords:
                logger.warning("No TFRecords found — falling back to mock mode.")
                all_frames = self._generate_mock_frames()
            else:
                all_frames = []
                for tfr in tqdm(tfrecords, desc="Parsing TFRecords"):
                    all_frames.extend(self._parse_tfrecord(tfr))
        else:
            all_frames = self._generate_mock_frames()

        logger.info(f"Total frames extracted: {len(all_frames)}")

        for frame in tqdm(all_frames, desc="Writing YOLO data"):
            split = self._assign_split(frame["frame_id"])
            stem  = frame["frame_id"].replace("/", "_")

            # Save image — write via BytesIO to guarantee valid JPEG bytes
            img_path = self.out_dir / "images" / split / f"{stem}.jpg"
            buf = io.BytesIO()
            frame["image"].convert("RGB").save(buf, format="JPEG", quality=90)
            img_path.write_bytes(buf.getvalue())

            # Save YOLO label
            lbl_path = self.out_dir / "labels" / split / f"{stem}.txt"
            with open(lbl_path, "w") as f:
                for lbl in frame["labels"]:
                    f.write(
                        f"{lbl['class_idx']} "
                        f"{lbl['cx']:.6f} {lbl['cy']:.6f} "
                        f"{lbl['w']:.6f} {lbl['h']:.6f}\n"
                    )

            counts[split] += 1

        # Write dataset YAML for Ultralytics
        self._write_dataset_yaml()
        logger.info(f"Processing complete. Split counts: {counts}")
        return counts

    def _write_dataset_yaml(self):
        yaml_path = self.out_dir / "dataset.yaml"
        names = list(WAYMO_CLASS_MAP.values())
        with open(yaml_path, "w") as f:
            f.write(f"path: {self.out_dir.resolve()}\n")
            f.write(f"train: images/train\n")
            f.write(f"val:   images/val\n")
            f.write(f"test:  images/test\n\n")
            f.write(f"nc: {len(names)}\n")
            f.write(f"names: {names}\n")
        logger.info(f"Dataset YAML written → {yaml_path}")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Waymo → YOLO data preprocessor")
    p.add_argument("--raw_dir", default="data/waymo/",      help="Raw TFRecord directory")
    p.add_argument("--out_dir", default="data/processed/waymo/", help="Output directory")
    p.add_argument("--splits",  nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"), help="Train/val/test split ratios")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    loader = WaymoLoader(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        splits=tuple(args.splits),
        seed=args.seed,
    )
    counts = loader.process()
    print(f"\n✅ Done. Split counts: {counts}")
