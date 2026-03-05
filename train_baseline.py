"""
train_baseline.py
─────────────────
Phase 1: Train a YOLOv8 baseline detector on the Waymo Open Dataset.

This script orchestrates the full Phase 1 pipeline:
  1. Load and validate the Waymo processed dataset.
  2. Initialize YOLOv8 (pretrained on COCO).
  3. Fine-tune on Waymo classes: vehicle, pedestrian, cyclist, sign.
  4. Log metrics and checkpoints to MLflow.
  5. Save the best checkpoint for Phase 2 continual learning.

Usage:
    python train_baseline.py --config configs/waymo_config.yaml

    # Quick smoke test (mock data, 2 epochs):
    python train_baseline.py --config configs/waymo_config.yaml --smoke_test

    # Override any config value via CLI:
    python train_baseline.py --config configs/waymo_config.yaml \
                             --epochs 100 --batch_size 32
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.waymo_loader  import WaymoLoader
from src.models.yolo_detector import YOLODetector
from src.utils.mlflow_utils import setup_mlflow, log_phase1_results, log_config
from src.utils.metrics      import compute_forgetting

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_baseline")


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> dict:
    # Load base config first
    base_path = Path(config_path).parent / "base_config.yaml"
    cfg = {}
    if base_path.exists():
        with open(base_path) as f:
            cfg = yaml.safe_load(f) or {}
    # Deep-merge the task-specific config on top
    with open(config_path, "r") as f:
        task_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(cfg, task_cfg)
    # Apply CLI overrides
    for k, v in overrides.items():
        if v is not None:
            _set_nested(cfg, k, v)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base without clobbering nested keys."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_nested(d: dict, key: str, value):
    """Set a dotted key like 'training.epochs' in a nested dict."""
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_waymo_data(cfg: dict, smoke_test: bool = False) -> str:
    """
    Run Waymo loader if processed data doesn't exist yet.
    Returns path to dataset.yaml.
    """
    dataset_yaml = Path(cfg["dataset"]["processed_path"]) / "dataset.yaml"

    if dataset_yaml.exists():
        logger.info(f"Processed Waymo data found at {dataset_yaml} — skipping re-processing.")
        return str(dataset_yaml)

    logger.info("Processed data not found — running WaymoLoader...")
    splits = (
        cfg["dataset"]["train_ratio"],
        cfg["dataset"]["val_ratio"],
        cfg["dataset"]["test_ratio"],
    )
    loader = WaymoLoader(
        raw_dir=cfg["dataset"]["raw_path"],
        out_dir=cfg["dataset"]["processed_path"],
        splits=splits,
        seed=cfg["training"]["seed"],
    )
    if smoke_test:
        # Monkey-patch mock mode for smoke test
        loader._generate_mock_frames = lambda: loader._generate_mock_frames.__wrapped__(n=50) \
            if hasattr(loader._generate_mock_frames, "__wrapped__") else \
            type(loader)._generate_mock_frames(loader, n=50)

    counts = loader.process()
    logger.info(f"Data preparation done. Counts: {counts}")
    return str(dataset_yaml)


# ── Main training ─────────────────────────────────────────────────────────────

def train(cfg: dict, smoke_test: bool = False):
    logger.info("=" * 60)
    logger.info("  Phase 1 — Baseline YOLOv8 Training on Waymo")
    logger.info("=" * 60)

    # ── 1. Setup MLflow ────────────────────────────────────────────────────
    setup_mlflow(
        experiment_name=cfg["project"]["mlflow_experiment"],
        tracking_uri=cfg["project"]["mlflow_tracking_uri"],
    )

    # ── 2. Prepare data ────────────────────────────────────────────────────
    dataset_yaml = prepare_waymo_data(cfg, smoke_test=smoke_test)

    # ── 3. Initialize detector ─────────────────────────────────────────────
    num_classes = len(cfg["dataset"]["class_map"])
    detector = YOLODetector(
        model_size=cfg["model"]["architecture"],
        num_classes=num_classes,
        device=cfg["training"]["device"],
    )

    # ── 4. Train ───────────────────────────────────────────────────────────
    epochs      = 2 if smoke_test else cfg["training"]["epochs"]
    batch_size  = 2 if smoke_test else cfg["training"]["batch_size"]
    checkpoint_name = cfg["training"]["checkpoint_name"]

    logger.info(
        f"Starting training | model={cfg['model']['architecture']} | "
        f"epochs={epochs} | batch={batch_size} | device={cfg['training']['device']}"
    )

    t0 = time.time()
    try:
        import mlflow
        with mlflow.start_run(run_name=f"phase1_{checkpoint_name}"):
            log_config(cfg)
            metrics = detector.train(
                dataset_yaml=dataset_yaml,
                epochs=epochs,
                batch_size=batch_size,
                img_size=cfg["training"]["img_size"],
                lr=cfg["training"]["lr"],
                checkpoint_name=checkpoint_name,
                mlflow_run_name=f"phase1_{checkpoint_name}",
            )
            log_phase1_results(metrics, cfg)
    except Exception:
        # MLflow not available — train without tracking
        metrics = detector.train(
            dataset_yaml=dataset_yaml,
            epochs=epochs,
            batch_size=batch_size,
            img_size=cfg["training"]["img_size"],
            lr=cfg["training"]["lr"],
            checkpoint_name=checkpoint_name,
        )

    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed/60:.1f} min")

    # ── 5. Report results ──────────────────────────────────────────────────
    best_ckpt = Path("runs") / checkpoint_name / "weights" / "best.pt"

    logger.info("\n" + "=" * 60)
    logger.info("  Phase 1 Results")
    logger.info("=" * 60)
    logger.info(f"  mAP@50:      {metrics.get('mAP50',    0.0):.4f}")
    logger.info(f"  mAP@50-95:   {metrics.get('mAP50_95', 0.0):.4f}")
    logger.info(f"  Precision:   {metrics.get('precision', 0.0):.4f}")
    logger.info(f"  Recall:      {metrics.get('recall',    0.0):.4f}")
    logger.info(f"  Checkpoint → {best_ckpt}")
    logger.info("=" * 60)

    if not best_ckpt.exists():
        logger.warning(
            "Best checkpoint not found. "
            "Ensure training ran with sufficient data and epochs."
        )

    return metrics, str(best_ckpt)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 — Baseline YOLOv8 training on Waymo")
    p.add_argument("--config",     default="configs/waymo_config.yaml", help="Config YAML path")
    p.add_argument("--smoke_test", action="store_true",                  help="2-epoch mock run")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--device",     type=str,   default=None, help="cuda / cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    overrides = {
        "training.epochs":     args.epochs,
        "training.batch_size": args.batch_size,
        "training.lr":         args.lr,
        "training.device":     args.device,
    }

    cfg = load_config(args.config, overrides)
    metrics, checkpoint_path = train(cfg, smoke_test=args.smoke_test)

    print(f"\n✅  Phase 1 complete.")
    print(f"    mAP@50:    {metrics.get('mAP50', 0.0):.4f}")
    print(f"    mAP@50-95: {metrics.get('mAP50_95', 0.0):.4f}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"\n▶  Next: python train_continual.py --checkpoint {checkpoint_path}")
