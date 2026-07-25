"""
run_all_metrics.py
──────────────────
Standalone script to compute and print all Computer Vision & Machine Learning metrics
used in the Open-World Continual Learning (OWCL) project:

1. Object Detection Metrics (mAP50, mAP50-95, Precision, Recall)
2. Catastrophic Forgetting Metrics (Absolute Forgetting, Retention Rate, Relative Drop %)
3. Open-Set Recognition Metrics (AUROC, AUPR, FPR95, Unknown Detection Rate, Entropy Stats)

Usage:
    # Run evaluation on a trained checkpoint:
    python run_all_metrics.py --checkpoint yolov8m.pt

    # Run full forgetting comparison with baseline checkpoint:
    python run_all_metrics.py --checkpoint runs/continual_ewc/weights/best.pt \
                             --baseline_checkpoint runs/waymo_baseline/weights/best.pt \
                             --output_json results_summary.json
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import torch

# Project imports
sys.path.insert(0, str(Path(__file__).parent))

from src.models.yolo_detector import YOLODetector
from src.openset.uncertainty import UncertaintyDetector
from src.utils.metrics import (
    compute_map,
    compute_forgetting,
    compute_openset_metrics,
)
from evaluate import evaluate_detection, evaluate_openset, DATASET_YAML_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all_metrics")


def print_formatted_metrics_table(all_metrics: dict):
    """Prints a structured summary table of all computed metrics."""
    print("\n" + "═" * 70)
    print(" 📊 OWCL COMPLETE MODEL EVALUATION METRICS REPORT")
    print("═" * 70)

    for category, metrics in all_metrics.items():
        print(f"\n  ▶ [{category.upper()}]")
        print("  " + "─" * 66)
        if isinstance(metrics, dict):
            for metric_name, val in metrics.items():
                if isinstance(val, float):
                    print(f"    • {metric_name:<35} : {val:.4f}")
                elif isinstance(val, dict):
                    print(f"    • {metric_name:<35} : {val}")
                else:
                    print(f"    • {metric_name:<35} : {val}")
        else:
            print(f"    • {metrics}")
    print("\n" + "═" * 70 + "\n")


def run_metrics_evaluation(
    checkpoint: str,
    baseline_checkpoint: str = None,
    uncertainty_metric: str = "entropy",
    threshold: float = 0.6,
    output_json: str = None,
):
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading evaluation model from: {checkpoint} (Device: {device})")
    detector = YOLODetector(
        model_size="yolov8m",
        checkpoint=checkpoint,
        device=device,
    )

    # 1. Detection Metrics on nuScenes / Target Domain
    logger.info("Computing Detection Metrics (mAP50, mAP50-95, Precision, Recall)...")
    nuscenes_yaml = DATASET_YAML_MAP.get("nuscenes", "data/processed/nuscenes/dataset.yaml")
    det_metrics = evaluate_detection(
        detector=detector,
        dataset_yaml=nuscenes_yaml,
        img_size=640,
        dataset_name="nuScenes Target",
    )
    results["1. Detection Performance (nuScenes)"] = det_metrics

    # 2. Catastrophic Forgetting & Domain Transfer Metrics
    if baseline_checkpoint and Path(baseline_checkpoint).exists():
        logger.info("Computing Forgetting Metrics (Task A vs Task B retention)...")
        waymo_yaml = DATASET_YAML_MAP.get("waymo", "data/processed/waymo/dataset.yaml")
        baseline_detector = YOLODetector(
            model_size="yolov8m",
            checkpoint=baseline_checkpoint,
            device=device,
        )
        before_metrics = evaluate_detection(baseline_detector, waymo_yaml, 640, "Waymo Baseline")
        after_metrics = evaluate_detection(detector, waymo_yaml, 640, "Waymo Continual")
        
        forgetting_stats = compute_forgetting(
            task_a_before=before_metrics.get("mAP50", 0.0),
            task_a_after=after_metrics.get("mAP50", 0.0),
        )
        results["2. Continual Learning & Forgetting"] = forgetting_stats
    else:
        results["2. Continual Learning & Forgetting"] = {
            "Note": "Pass --baseline_checkpoint to evaluate Task A (Waymo) retention drop."
        }

    # 3. Open-Set Recognition & OOD Uncertainty Metrics
    logger.info(f"Computing Open-Set Metrics (AUROC, AUPR, FPR95, Unknown Rate using {uncertainty_metric})...")
    openset_stats = evaluate_openset(
        detector=detector,
        dataset_processed_path="data/processed/nuscenes/",
        threshold=threshold,
        metric=uncertainty_metric,
        num_classes=4,
        conf_threshold=0.05,
    )
    results["3. Open-Set Recognition & Uncertainty"] = openset_stats


    # Print summary report
    print_formatted_metrics_table(results)

    # Save output to JSON if requested
    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved to JSON → {out_p.resolve()}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all Computer Vision and ML metrics for OWCL project")
    p.add_argument("--checkpoint", default="yolov8m.pt", help="Path to evaluation model checkpoint (.pt)")
    p.add_argument("--baseline_checkpoint", default=None, help="Path to Task A baseline checkpoint for forgetting measure")
    p.add_argument("--uncertainty_metric", default="entropy", choices=["entropy", "max_softmax", "energy"])
    p.add_argument("--threshold", type=float, default=0.6, help="Uncertainty threshold for open-set detection")
    p.add_argument("--output_json", default=None, help="Optional path to save JSON metrics report")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_metrics_evaluation(
        checkpoint=args.checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        uncertainty_metric=args.uncertainty_metric,
        threshold=args.threshold,
        output_json=args.output_json,
    )
