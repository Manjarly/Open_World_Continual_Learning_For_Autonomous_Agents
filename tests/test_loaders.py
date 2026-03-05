"""
tests/test_loaders.py
─────────────────────
Unit tests for Waymo and nuScenes data loaders (mock mode — no SDK required).
"""

import sys
from pathlib import Path
import tempfile
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.waymo_loader    import WaymoLoader, WAYMO_CLASS_MAP, LABEL_TO_IDX
from src.data.nuscenes_loader import NuScenesLoader, NUSCENES_CLASS_MAP, SHARED_CLASSES


# ── WaymoLoader tests ─────────────────────────────────────────────────────────

class TestWaymoLoader:

    def test_mock_processing(self, tmp_path):
        """Loader creates correct output structure in mock mode."""
        loader = WaymoLoader(
            raw_dir=str(tmp_path / "raw"),
            out_dir=str(tmp_path / "processed"),
            splits=(0.8, 0.1, 0.1),
            seed=42,
        )
        counts = loader.process()

        # Check split counts sum to total
        assert sum(counts.values()) > 0
        assert set(counts.keys()) == {"train", "val", "test"}

    def test_output_directories(self, tmp_path):
        """Images and labels directories are created for all splits."""
        loader = WaymoLoader(
            raw_dir=str(tmp_path / "raw"),
            out_dir=str(tmp_path / "processed"),
        )
        loader.process()

        out = tmp_path / "processed"
        for split in ("train", "val", "test"):
            assert (out / "images" / split).is_dir()
            assert (out / "labels" / split).is_dir()

    def test_dataset_yaml_created(self, tmp_path):
        """dataset.yaml is written after processing."""
        loader = WaymoLoader(
            raw_dir=str(tmp_path / "raw"),
            out_dir=str(tmp_path / "processed"),
        )
        loader.process()
        yaml_path = tmp_path / "processed" / "dataset.yaml"
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "nc:" in content
        assert "vehicle" in content

    def test_label_format(self, tmp_path):
        """Labels are YOLO format: 5 space-separated values per line."""
        loader = WaymoLoader(
            raw_dir=str(tmp_path / "raw"),
            out_dir=str(tmp_path / "processed"),
        )
        loader.process()

        label_files = list((tmp_path / "processed" / "labels").rglob("*.txt"))
        assert len(label_files) > 0

        for lf in label_files[:5]:
            lines = lf.read_text().strip().splitlines()
            for line in lines:
                parts = line.split()
                assert len(parts) == 5, f"Expected 5 parts, got {len(parts)} in {lf}"
                cls_idx = int(parts[0])
                coords  = [float(x) for x in parts[1:]]
                assert cls_idx >= 0
                assert all(0.0 <= c <= 1.0 for c in coords), \
                    f"Coordinates out of [0,1]: {coords}"

    def test_split_determinism(self, tmp_path):
        """Same frame_id always maps to the same split."""
        loader = WaymoLoader(
            raw_dir=str(tmp_path / "raw"),
            out_dir=str(tmp_path / "processed"),
        )
        splits1 = [loader._assign_split(f"frame_{i}") for i in range(100)]
        splits2 = [loader._assign_split(f"frame_{i}") for i in range(100)]
        assert splits1 == splits2

    def test_class_map_coverage(self):
        """All Waymo class IDs have entries in the label index."""
        for cls_id, cls_name in WAYMO_CLASS_MAP.items():
            assert cls_name in LABEL_TO_IDX, \
                f"Class '{cls_name}' missing from LABEL_TO_IDX"

    def test_bbox_to_yolo_normalization(self):
        """Bounding box normalization produces values in [0, 1]."""
        cx, cy, w, h = WaymoLoader._bbox_to_yolo(960, 640, 200, 100, 1920, 1280)
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 <= w  <= 1.0
        assert 0.0 <= h  <= 1.0
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6


# ── NuScenesLoader tests ──────────────────────────────────────────────────────

class TestNuScenesLoader:

    def test_mock_processing(self, tmp_path):
        loader = NuScenesLoader(
            raw_dir=str(tmp_path / "raw"),
            out_dir=str(tmp_path / "processed"),
        )
        counts = loader.process()
        assert sum(counts.values()) > 0

    def test_shared_class_space(self):
        """All mapped nuScenes classes are in the shared class list."""
        for raw_cls, unified in NUSCENES_CLASS_MAP.items():
            assert unified in SHARED_CLASSES, \
                f"'{unified}' (from '{raw_cls}') not in SHARED_CLASSES"

    def test_dataset_yaml_created(self, tmp_path):
        loader = NuScenesLoader(
            raw_dir=str(tmp_path / "raw"),
            out_dir=str(tmp_path / "processed"),
        )
        loader.process()
        yaml_path = tmp_path / "processed" / "dataset.yaml"
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "pedestrian" in content
        assert "unknown" in content    # open-set class present

    def test_split_ratios_approximate(self, tmp_path):
        """Generated splits should roughly respect the requested ratios."""
        loader = NuScenesLoader(
            raw_dir=str(tmp_path / "raw"),
            out_dir=str(tmp_path / "processed"),
            splits=(0.7, 0.15, 0.15),
        )
        counts = loader.process()
        total  = sum(counts.values())
        if total > 0:
            train_ratio = counts["train"] / total
            assert 0.5 < train_ratio < 0.9, \
                f"Unexpected train ratio: {train_ratio:.2f}"
