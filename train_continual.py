"""
train_continual.py
──────────────────
Phase 2: Continual learning on nuScenes with EWC + Open-Set Recognition.

This script:
  1. Loads the Waymo-trained YOLOv8 checkpoint (from Phase 1).
  2. Estimates the Fisher Information Matrix on a Waymo sample → builds EWC.
  3. Fine-tunes on nuScenes with EWC penalty to prevent catastrophic forgetting.
  4. Applies entropy-based open-set flagging on the nuScenes val set.
  5. Measures forgetting: mAP on Waymo before vs. after nuScenes training.
  6. Logs all metrics, EWC stats, and open-set stats to MLflow.

Usage:
    python train_continual.py \
        --config configs/nuscenes_config.yaml \
        --checkpoint runs/waymo_baseline/weights/best.pt

    # Smoke test:
    python train_continual.py \
        --config configs/nuscenes_config.yaml \
        --checkpoint runs/waymo_baseline/weights/best.pt \
        --smoke_test

    # Tune EWC lambda:
    python train_continual.py \
        --config configs/nuscenes_config.yaml \
        --checkpoint runs/waymo_baseline/weights/best.pt \
        --ewc_lambda 0.8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data.nuscenes_loader  import NuScenesLoader
from src.models.yolo_detector  import YOLODetector
from src.continual.ewc         import EWC
from src.openset.uncertainty   import UncertaintyDetector
from src.utils.mlflow_utils    import (
    setup_mlflow, log_phase2_results, log_config
)
from src.utils.metrics import compute_forgetting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_continual")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> dict:
    base_path = Path(config_path).parent / "base_config.yaml"
    cfg = {}
    if base_path.exists():
        with open(base_path) as f:
            cfg = yaml.safe_load(f) or {}
    with open(config_path, "r") as f:
        task_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(cfg, task_cfg)
    for k, v in overrides.items():
        if v is not None:
            _nested_set(cfg, k, v)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _nested_set(d, key, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


# ── Waymo Fisher DataLoader ───────────────────────────────────────────────────

def build_fisher_dataloader(waymo_processed_path: str, n_samples: int, img_size: int):
    """
    Build a minimal DataLoader over Waymo images for Fisher estimation.
    Uses a simple glob-based dataset — no Waymo SDK needed at this stage.
    """
    import glob
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class SimpleImageDataset(Dataset):
        def __init__(self, img_dir, size, max_n):
            self.paths = sorted(glob.glob(f"{img_dir}/**/*.jpg", recursive=True))[:max_n]
            self.size  = size

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB").resize((self.size, self.size))
                arr = np.array(img, dtype=np.float32) / 255.0
                return torch.tensor(arr).permute(2, 0, 1)   # (3, H, W)
            except Exception:
                return torch.zeros(3, self.size, self.size)

    img_dir = Path(waymo_processed_path) / "images" / "train"
    dataset = SimpleImageDataset(str(img_dir), img_size, n_samples * 4)

    if len(dataset) == 0:
        logger.warning("No Waymo images found for Fisher estimation — using mock tensors.")

        class MockDataset(Dataset):
            def __len__(self): return n_samples
            def __getitem__(self, i):
                return torch.rand(3, img_size, img_size)

        dataset = MockDataset()

    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


# ── Data ──────────────────────────────────────────────────────────────────────

def prepare_nuscenes_data(cfg: dict, smoke_test: bool = False) -> str:
    dataset_yaml = Path(cfg["dataset"]["processed_path"]) / "dataset.yaml"

    if dataset_yaml.exists():
        logger.info(f"Processed nuScenes data found — skipping re-processing.")
        return str(dataset_yaml)

    logger.info("Processing nuScenes data...")
    loader = NuScenesLoader(
        raw_dir=cfg["dataset"]["raw_path"],
        out_dir=cfg["dataset"]["processed_path"],
        version=cfg["dataset"].get("version", "v1.0-trainval"),
        seed=cfg["training"]["seed"],
    )
    counts = loader.process()
    logger.info(f"nuScenes processing done. Counts: {counts}")
    return str(dataset_yaml)


# ── Open-set evaluation ───────────────────────────────────────────────────────

def run_openset_evaluation(
    detector: YOLODetector,
    dataset_processed_path: str,
    cfg: dict,
    smoke_test: bool = False,
) -> dict:
    """
    Run the uncertainty detector on val images and return open-set stats.
    """
    import glob, random
    from PIL import Image
    import numpy as np

    openset_cfg = cfg.get("openset", {})
    uncertainty_detector = UncertaintyDetector(
        metric=openset_cfg.get("uncertainty_metric", "entropy"),
        threshold=openset_cfg.get("threshold", 0.6),
        num_classes=len(cfg["dataset"]["class_map"]),
    )

    val_dir  = Path(dataset_processed_path) / "images" / "val"
    img_paths = sorted(glob.glob(f"{val_dir}/*.jpg"))

    if smoke_test:
        img_paths = img_paths[:10]
    elif len(img_paths) > 200:
        img_paths = random.sample(img_paths, 200)

    if not img_paths:
        logger.warning("No val images found for open-set evaluation.")
        return {}

    logger.info(f"Running open-set evaluation on {len(img_paths)} val images...")
    all_detections = []

    for img_path in img_paths:
        try:
            dets = detector.predict(
                img_path,
                conf_threshold=cfg["model"]["confidence_threshold"],
                return_probs=True,
            )
            flagged = uncertainty_detector.flag_unknowns(dets)
            all_detections.extend(flagged)
        except Exception as e:
            logger.debug(f"Skipped {img_path}: {e}")

    stats = uncertainty_detector.compute_stats(all_detections)
    logger.info(f"Open-set stats: {stats}")
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: dict, checkpoint: str, smoke_test: bool = False):
    logger.info("=" * 60)
    logger.info("  Phase 2 — Continual Learning (EWC) on nuScenes")
    logger.info("=" * 60)

    setup_mlflow(
        experiment_name=cfg["project"]["mlflow_experiment"],
        tracking_uri=cfg["project"]["mlflow_tracking_uri"],
    )

    # ── 1. Prepare nuScenes data ───────────────────────────────────────────
    dataset_yaml = prepare_nuscenes_data(cfg, smoke_test=smoke_test)

    # ── 2. Load Phase 1 checkpoint ─────────────────────────────────────────
    if not Path(checkpoint).exists():
        logger.warning(
            f"Checkpoint not found at {checkpoint}. "
            "Initializing from scratch (Phase 2 results may be suboptimal)."
        )
        checkpoint = None

    num_classes = len(cfg["dataset"]["class_map"])
    detector = YOLODetector(
        model_size=cfg["model"]["architecture"],
        num_classes=num_classes,
        checkpoint=checkpoint,
        device=cfg["training"]["device"],
    )

    # ── 3. Measure Task A (Waymo) performance BEFORE Task B ────────────────
    waymo_yaml = Path("data/processed/waymo/dataset.yaml")
    map50_before = 0.0
    if waymo_yaml.exists() and not smoke_test:
        logger.info("Measuring Waymo mAP50 BEFORE nuScenes training...")
        try:
            waymo_metrics_before = detector.validate(
                dataset_yaml=str(waymo_yaml),
                img_size=cfg["training"]["img_size"],
            )
            map50_before = waymo_metrics_before.get("mAP50", 0.0)
            logger.info(f"  Waymo mAP50 (before): {map50_before:.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate on Waymo before training: {e}")

    # ── 4. Build EWC ───────────────────────────────────────────────────────
    ewc_cfg     = cfg.get("ewc", {})
    ewc_lambda  = ewc_cfg.get("lambda", 0.4)
    n_fisher    = 10 if smoke_test else ewc_cfg.get("fisher_samples", 200)
    img_size    = cfg["training"]["img_size"]

    pytorch_model = detector.get_pytorch_model()
    device        = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    pytorch_model = pytorch_model.to(device)

    logger.info(f"Building EWC (lambda={ewc_lambda}, fisher_samples={n_fisher})...")
    fisher_loader = build_fisher_dataloader(
        waymo_processed_path="data/processed/waymo/",
        n_samples=n_fisher,
        img_size=img_size,
    )
    ewc = EWC(
        model=pytorch_model,
        dataloader=fisher_loader,
        device=device,
        n_samples=n_fisher,
        ewc_lambda=ewc_lambda,
    )
    ewc_summary = ewc.summary()
    logger.info(f"EWC summary: {ewc_summary}")

    # ── 5. Train on nuScenes ───────────────────────────────────────────────
    epochs      = 2 if smoke_test else cfg["training"]["epochs"]
    batch_size  = 2 if smoke_test else cfg["training"]["batch_size"]
    checkpoint_name = cfg["training"]["checkpoint_name"]

    logger.info(
        f"Starting nuScenes training | epochs={epochs} | "
        f"batch={batch_size} | EWC λ={ewc_lambda}"
    )

    t0 = time.time()

    try:
        import mlflow
        try:
            mlflow.end_run()   # Close any stale active run first
        except Exception:
            pass
        with mlflow.start_run(run_name=f"phase2_{checkpoint_name}", nested=False):
            log_config(cfg)
            mlflow.log_param("ewc_lambda", ewc_lambda)
            mlflow.log_param("phase1_checkpoint", str(checkpoint))

            metrics = detector.train(
                dataset_yaml=dataset_yaml,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                lr=cfg["training"]["lr"],
                checkpoint_name=checkpoint_name,
            )

            # ── 6. Open-set evaluation ─────────────────────────────────────
            openset_stats = {}
            if cfg.get("openset", {}).get("enabled", False):
                openset_stats = run_openset_evaluation(
                    detector, cfg["dataset"]["processed_path"],
                    cfg, smoke_test=smoke_test
                )

            # ── 7. Measure forgetting ──────────────────────────────────────
            map50_after = 0.0
            if waymo_yaml.exists() and not smoke_test:
                logger.info("Measuring Waymo mAP50 AFTER nuScenes training...")
                try:
                    waymo_metrics_after = detector.validate(
                        dataset_yaml=str(waymo_yaml),
                        img_size=img_size,
                    )
                    map50_after = waymo_metrics_after.get("mAP50", 0.0)
                except Exception as e:
                    logger.warning(f"Could not evaluate on Waymo after training: {e}")

            forgetting = compute_forgetting(map50_before, map50_after)
            metrics.update({"forgetting": forgetting.get("forgetting", 0.0)})

            log_phase2_results(metrics, ewc_summary, openset_stats, cfg)

            mlflow.log_metrics({
                "forgetting/absolute": forgetting.get("forgetting",    0.0),
                "forgetting/retention": forgetting.get("retention",    0.0),
                "forgetting/relative_drop": forgetting.get("relative_drop", 0.0),
            })

    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}. Training without tracking.")
        metrics = detector.train(
            dataset_yaml=dataset_yaml,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=cfg["training"]["lr"],
            checkpoint_name=checkpoint_name,
        )
        openset_stats = {}
        forgetting    = {}

    elapsed = time.time() - t0
    best_ckpt = Path("runs") / checkpoint_name / "weights" / "best.pt"

    # ── 8. Summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Phase 2 Results")
    logger.info("=" * 60)
    logger.info(f"  nuScenes mAP@50:    {metrics.get('mAP50',    0.0):.4f}")
    logger.info(f"  nuScenes mAP@50-95: {metrics.get('mAP50_95', 0.0):.4f}")
    logger.info(f"  EWC Lambda:         {ewc_lambda}")
    logger.info(f"  Forgetting (Δ mAP): {forgetting.get('forgetting', 'N/A')}")
    logger.info(f"  Checkpoint → {best_ckpt}")
    if openset_stats:
        logger.info(f"  Unknown rate:       {openset_stats.get('unknown_rate', 0.0):.3f}")
        logger.info(f"  Uncertainty mean:   {openset_stats.get('uncertainty_mean', 0.0):.3f}")
    logger.info(f"  Total time: {elapsed/60:.1f} min")
    logger.info("=" * 60)

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 — Continual learning with EWC on nuScenes")
    p.add_argument("--config",     default="configs/nuscenes_config.yaml")
    p.add_argument("--checkpoint", default="runs/waymo_baseline/weights/best.pt",
                   help="Phase 1 Waymo checkpoint path")
    p.add_argument("--ewc_lambda", type=float, default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--device",     type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    overrides = {
        "training.epochs":     args.epochs,
        "training.batch_size": args.batch_size,
        "training.device":     args.device,
        "ewc.lambda":          args.ewc_lambda,
    }

    cfg = load_config(args.config, overrides)
    metrics = train(cfg, checkpoint=args.checkpoint, smoke_test=args.smoke_test)

    print(f"\n✅  Phase 2 complete.")
    print(f"    nuScenes mAP@50: {metrics.get('mAP50', 0.0):.4f}")
    print(f"\n▶  View results: mlflow ui --port 5000")
