"""
src/data/nuscenes_loader.py
───────────────────────────
nuScenes dataset loader.

Reads the nuScenes annotation JSON, extracts front-camera frames and 2D
projected bounding boxes, and writes a YOLO-compatible directory layout.

Usage (CLI):
    python -m src.data.nuscenes_loader \
        --raw_dir data/nuscenes/ \
        --out_dir data/processed/nuscenes/ \
        --version v1.0-trainval

Requires:
    pip install nuscenes-devkit
"""

import os
import io
import shutil
import logging
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Unified label map (nuScenes → shared class space) ───────────────────────
NUSCENES_CLASS_MAP: Dict[str, str] = {
    "car":              "vehicle",
    "truck":            "vehicle",
    "bus":              "vehicle",
    "construction_vehicle": "vehicle",
    "trailer":          "vehicle",
    "motorcycle":       "cyclist",
    "bicycle":          "cyclist",
    "pedestrian":       "pedestrian",
    "traffic_cone":     "sign",
    "barrier":          "unknown",    # open-set candidate
}

SHARED_CLASSES = ["vehicle", "pedestrian", "cyclist", "sign", "unknown"]
LABEL_TO_IDX   = {c: i for i, c in enumerate(SHARED_CLASSES)}

# Only front camera for 2D detection
FRONT_CAMERA = "CAM_FRONT"

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
    from pyquaternion import Quaternion
    NUSCENES_SDK_AVAILABLE = True
except ImportError:
    NUSCENES_SDK_AVAILABLE = False
    logger.warning(
        "nuscenes-devkit not found. "
        "Install with: pip install nuscenes-devkit\n"
        "Loader will run in MOCK mode."
    )


class NuScenesLoader:
    """
    Converts nuScenes annotations into YOLO-format training data.

    Output mirrors WaymoLoader for a unified training pipeline:
        out_dir/
          images/{train,val,test}/*.jpg
          labels/{train,val,test}/*.txt
          dataset.yaml
    """

    def __init__(
        self,
        raw_dir: str,
        out_dir: str,
        version: str = "v1.0-trainval",
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.version = version
        self.splits  = splits
        self.rng     = np.random.default_rng(seed)
        self._create_dirs()

    def _create_dirs(self):
        for split in ("train", "val", "test"):
            (self.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _assign_split(self, token: str) -> str:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16) / (16 ** 32)
        if h < self.splits[0]:
            return "train"
        elif h < self.splits[0] + self.splits[1]:
            return "val"
        return "test"

    # ── Real SDK path ────────────────────────────────────────────────────────

    def _project_3d_box_to_2d(self, nusc, sample_data_token: str, ann_token: str):
        """
        Project a 3D annotation box into the front camera image plane
        and return (cx, cy, w, h) in pixel coordinates.
        Returns None if box is not visible.
        """
        sd_record  = nusc.get("sample_data", sample_data_token)
        cs_record  = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
        ann_record = nusc.get("sample_annotation", ann_token)

        # Build the 3D box
        from nuscenes.utils.data_classes import Box
        box = Box(
            ann_record["translation"],
            ann_record["size"],
            Quaternion(ann_record["rotation"]),
        )

        # Transform: global → ego → sensor
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        # Check visibility
        intrinsic = np.array(cs_record["camera_intrinsic"])
        imsize    = (sd_record["width"], sd_record["height"])
        if not box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY):
            return None

        # Project corners and compute 2D bounding box
        corners_2d = view_points(box.corners(), intrinsic, normalize=True)[:2]
        x_min, x_max = corners_2d[0].min(), corners_2d[0].max()
        y_min, y_max = corners_2d[1].min(), corners_2d[1].max()

        # Clip to image bounds
        x_min = max(0.0, x_min); y_min = max(0.0, y_min)
        x_max = min(imsize[0], x_max); y_max = min(imsize[1], y_max)

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w  = x_max - x_min
        h  = y_max - y_min

        if w <= 0 or h <= 0:
            return None

        # Normalize
        return (
            cx / imsize[0],
            cy / imsize[1],
            w  / imsize[0],
            h  / imsize[1],
        )

    def _process_real(self) -> List[Dict]:
        """Use nuscenes-devkit to build frame list."""
        nusc = NuScenes(version=self.version, dataroot=str(self.raw_dir), verbose=False)
        frames = []

        for sample in tqdm(nusc.sample, desc="Processing nuScenes samples"):
            cam_token = sample["data"][FRONT_CAMERA]
            sd_record = nusc.get("sample_data", cam_token)
            img_path  = self.raw_dir / sd_record["filename"]

            labels = []
            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                raw_cls = ann["category_name"].split(".")[0]
                unified_cls = NUSCENES_CLASS_MAP.get(raw_cls)
                if unified_cls is None:
                    continue

                bbox = self._project_3d_box_to_2d(nusc, cam_token, ann_token)
                if bbox is None:
                    continue

                labels.append({
                    "class_idx": LABEL_TO_IDX[unified_cls],
                    "cx": bbox[0], "cy": bbox[1],
                    "w":  bbox[2], "h":  bbox[3],
                })

            frames.append({
                "frame_id": sample["token"],
                "img_path": img_path,
                "labels":   labels,
            })

        return frames

    # ── Mock mode ────────────────────────────────────────────────────────────

    def _generate_mock_frames(self, n: int = 100) -> List[Dict]:
        logger.info(f"[MOCK] Generating {n} synthetic nuScenes frames.")
        frames = []
        for i in range(n):
            img_w, img_h = 640, 640  # Smaller size for faster training
            arr = self.rng.integers(0, 255, (img_h, img_w, 3), dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            num_boxes = self.rng.integers(1, 8)
            labels = []
            for _ in range(num_boxes):
                cls_idx = self.rng.integers(0, len(LABEL_TO_IDX))
                labels.append({
                    "class_idx": int(cls_idx),
                    "cx": float(self.rng.uniform(0.2, 0.8)),
                    "cy": float(self.rng.uniform(0.2, 0.8)),
                    "w":  float(self.rng.uniform(0.05, 0.3)),
                    "h":  float(self.rng.uniform(0.05, 0.25)),
                })
            frames.append({
                "frame_id": f"mock_nuscenes_{i:05d}",
                "img_path": None,
                "image":    img,
                "labels":   labels,
            })
        return frames

    # ── Public API ───────────────────────────────────────────────────────────

    def process(self) -> Dict[str, int]:
        counts = {"train": 0, "val": 0, "test": 0}

        if NUSCENES_SDK_AVAILABLE and (self.raw_dir / "v1.0-trainval").exists():
            all_frames = self._process_real()
        else:
            logger.warning("nuScenes SDK or data not found — using mock mode.")
            all_frames = self._generate_mock_frames()

        for frame in tqdm(all_frames, desc="Writing YOLO data"):
            split = self._assign_split(frame["frame_id"])
            stem  = frame["frame_id"].replace("/", "_")

            img_out = self.out_dir / "images" / split / f"{stem}.jpg"
            img_to_save = frame.get("image")
            if img_to_save is None and frame.get("img_path") and Path(frame["img_path"]).exists():
                img_to_save = Image.open(frame["img_path"]).convert("RGB")
            if img_to_save is None:
                img_to_save = Image.new("RGB", (640, 640), color=(128, 128, 128))
            buf = io.BytesIO()
            img_to_save.convert("RGB").save(buf, format="JPEG", quality=90)
            img_out.write_bytes(buf.getvalue())

            lbl_out = self.out_dir / "labels" / split / f"{stem}.txt"
            with open(lbl_out, "w") as f:
                for lbl in frame["labels"]:
                    f.write(
                        f"{lbl['class_idx']} "
                        f"{lbl['cx']:.6f} {lbl['cy']:.6f} "
                        f"{lbl['w']:.6f} {lbl['h']:.6f}\n"
                    )

            counts[split] += 1

        self._write_dataset_yaml()
        logger.info(f"Done. Split counts: {counts}")
        return counts

    def _write_dataset_yaml(self):
        yaml_path = self.out_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"path: {self.out_dir.resolve()}\n")
            f.write("train: images/train\n")
            f.write("val:   images/val\n")
            f.write("test:  images/test\n\n")
            f.write(f"nc: {len(SHARED_CLASSES)}\n")
            f.write(f"names: {SHARED_CLASSES}\n")
        logger.info(f"Dataset YAML written → {yaml_path}")


def parse_args():
    p = argparse.ArgumentParser(description="nuScenes → YOLO data preprocessor")
    p.add_argument("--raw_dir", default="data/nuscenes/")
    p.add_argument("--out_dir", default="data/processed/nuscenes/")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--splits", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    loader = NuScenesLoader(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        version=args.version,
        splits=tuple(args.splits),
        seed=args.seed,
    )
    counts = loader.process()
    print(f"\n✅ Done. Split counts: {counts}")
