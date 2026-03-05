"""
evaluate.py
───────────
Unified evaluation script for Phase 1 and Phase 2 checkpoints.

Runs:
  - mAP50 / mAP50-95 on any YOLO-format dataset.
  - Forgetting measurement (Task A vs Task B performance).
  - Open-set evaluation with AUROC, AUPR, FPR95.

Usage:
    # Evaluate Phase 1 checkpoint on Waymo test set:
    python evaluate.py --checkpoint runs/waymo_baseline/weights/best.pt \
                       --dataset waymo

    # Evaluate Phase 2 checkpoint on nuScenes with open-set:
    python evaluate.py --checkpoint runs/continual_ewc/weights/best.pt \
                       --dataset nuscenes \
                       --open_set

    # Full forgetting analysis (both datasets):
    python evaluate.py --checkpoint runs/continual_ewc/weights/best.pt \
                       --dataset both
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.yolo_detector import YOLODetector
from src.openset.uncertainty  import UncertaintyDetector
from src.utils.metrics        import compute_forgetting, compute_openset_metrics
from src.utils.mlflow_utils   import setup_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

DATASET_YAML_MAP = {
    "waymo":    "data/processed/waymo/dataset.yaml",
    "nuscenes": "data/processed/nuscenes/dataset.yaml",
}


def load_config(dataset: str) -> dict:
    cfg_map = {
        "waymo":    "configs/waymo_config.yaml",
        "nuscenes": "configs/nuscenes_config.yaml",
    }
    cfg_path = cfg_map.get(dataset, "configs/base_config.yaml")
    if Path(cfg_path).exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


def evaluate_detection(
    detector: YOLODetector,
    dataset_yaml: str,
    img_size: int = 640,
    dataset_name: str = "",
) -> dict:
    logger.info(f"Evaluating detection on {dataset_name or dataset_yaml}...")
    if not Path(dataset_yaml).exists():
        logger.warning(f"Dataset YAML not found: {dataset_yaml}")
        return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}
    return detector.validate(dataset_yaml=dataset_yaml, img_size=img_size)


def evaluate_openset(
    detector: YOLODetector,
    dataset_processed_path: str,
    threshold: float = 0.6,
    metric: str = "entropy",
    num_classes: int = 4,
    max_images: int = 200,
) -> dict:
    """Run open-set detection on val images and return stats + AUROC/AUPR."""
    import glob, random
    from PIL import Image
    import numpy as np

    unc_detector = UncertaintyDetector(
        metric=metric,
        threshold=threshold,
        num_classes=num_classes,
    )

    val_dir   = Path(dataset_processed_path) / "images" / "val"
    img_paths = sorted(glob.glob(f"{val_dir}/*.jpg"))
    if len(img_paths) > max_images:
        img_paths = random.sample(img_paths, max_images)

    if not img_paths:
        logger.warning("No validation images found for open-set evaluation.")
        return {}

    all_dets = []
    for p in img_paths:
        try:
            dets = detector.predict(p, return_probs=True)
            flagged = unc_detector.flag_unknowns(dets)
            all_dets.extend(flagged)
        except Exception as e:
            logger.debug(f"Skipped {p}: {e}")

    stats = unc_detector.compute_stats(all_dets)

    # Compute AUROC/AUPR using uncertainty score as open-set discriminator
    if all_dets:
        scores = [d.get("uncertainty", 0.0) for d in all_dets]
        # For evaluation, label barrier/unknown class as truly unknown
        is_unknown_gt = [d.get("cls", 0) in (-1, 4) for d in all_dets]
        if any(is_unknown_gt):
            oc_metrics = compute_openset_metrics(scores, is_unknown_gt)
            stats.update(oc_metrics)

    return stats


def print_report(results: dict):
    print("\n" + "═" * 55)
    print("  EVALUATION REPORT — Team Delaware")
    print("═" * 55)
    for section, data in results.items():
        print(f"\n  [{section}]")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, float):
                    print(f"    {k:<25} {v:.4f}")
                else:
                    print(f"    {k:<25} {v}")
        else:
            print(f"    {data}")
    print("\n" + "═" * 55)


def run_evaluation(args):
    cfg = load_config(args.dataset if args.dataset != "both" else "nuscenes")
    img_size = cfg.get("training", {}).get("img_size", 640)

    detector = YOLODetector(
        model_size=cfg.get("model", {}).get("architecture", "yolov8m"),
        checkpoint=args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    results = {}

    # ── Detection metrics ──────────────────────────────────────────────────
    if args.dataset in ("waymo", "both"):
        metrics = evaluate_detection(
            detector, DATASET_YAML_MAP["waymo"], img_size, "Waymo (Task A)"
        )
        results["Waymo (Task A) Detection"] = metrics

    if args.dataset in ("nuscenes", "both"):
        metrics = evaluate_detection(
            detector, DATASET_YAML_MAP["nuscenes"], img_size, "nuScenes (Task B)"
        )
        results["nuScenes (Task B) Detection"] = metrics

    # ── Forgetting analysis ────────────────────────────────────────────────
    if args.dataset == "both" and args.baseline_checkpoint:
        baseline = YOLODetector(
            model_size=cfg.get("model", {}).get("architecture", "yolov8m"),
            checkpoint=args.baseline_checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        before = evaluate_detection(baseline, DATASET_YAML_MAP["waymo"], img_size)
        after  = results.get("Waymo (Task A) Detection", {})
        forgetting = compute_forgetting(
            before.get("mAP50", 0.0),
            after.get("mAP50", 0.0),
        )
        results["Catastrophic Forgetting Analysis"] = forgetting

    # ── Open-set evaluation ────────────────────────────────────────────────
    if args.open_set and args.dataset in ("nuscenes", "both"):
        openset_stats = evaluate_openset(
            detector,
            dataset_processed_path="data/processed/nuscenes/",
            threshold=args.threshold,
            metric=args.uncertainty_metric,
            num_classes=len(cfg.get("dataset", {}).get("class_map", {}) or range(4)),
        )
        results["Open-Set Recognition"] = openset_stats

    # ── Output ─────────────────────────────────────────────────────────────
    print_report(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Unified evaluation for OWCL project")
    p.add_argument("--checkpoint",          required=True,  help="Model checkpoint (.pt)")
    p.add_argument("--dataset",             default="nuscenes",
                   choices=["waymo", "nuscenes", "both"],   help="Dataset to evaluate on")
    p.add_argument("--open_set",            action="store_true", help="Run open-set evaluation")
    p.add_argument("--uncertainty_metric",  default="entropy",
                   choices=["entropy", "max_softmax", "energy"])
    p.add_argument("--threshold",           type=float, default=0.6)
    p.add_argument("--baseline_checkpoint", default=None,
                   help="Phase 1 checkpoint for forgetting comparison")
    p.add_argument("--output",              default=None,   help="Save results to JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
