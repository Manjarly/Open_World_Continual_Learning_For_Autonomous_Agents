"""
src/utils/metrics.py
────────────────────
Evaluation metrics for detection, continual learning, and open-set recognition.

Includes:
  - compute_map()           : mAP50 / mAP50-95 from raw predictions.
  - compute_forgetting()    : Backward transfer / catastrophic forgetting measure.
  - compute_openset_metrics(): AUROC, FPR95, AUPR for open-set evaluation.
  - DetectionEvaluator      : Stateful evaluator that accumulates per-batch results.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SKLEARN_AVAILABLE = False

def _try_import_sklearn():
    global SKLEARN_AVAILABLE
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score  # noqa
        SKLEARN_AVAILABLE = True
        return True
    except Exception:
        return False


# ── IoU utilities ─────────────────────────────────────────────────────────────

def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    Inputs can be 1D arrays of length 4.
    """
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ── Per-class AP ──────────────────────────────────────────────────────────────

def compute_ap(
    tp: np.ndarray,
    fp: np.ndarray,
    n_gt: int,
    method: str = "interp101",
) -> float:
    """
    Compute Average Precision from sorted TP/FP arrays.

    Args:
        tp:     Cumulative true positives (sorted by decreasing confidence).
        fp:     Cumulative false positives.
        n_gt:   Total number of ground truth boxes for this class.
        method: "interp101" (COCO-style) or "voc" (11-point interpolation).

    Returns:
        AP as a float in [0, 1].
    """
    if n_gt == 0:
        return 0.0

    recall    = tp / (n_gt + 1e-16)
    precision = tp / (tp + fp + 1e-16)

    # Prepend sentinel values
    recall    = np.concatenate([[0.0], recall,    [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    if method == "interp101":
        thresholds = np.linspace(0, 1, 101)
        ap = np.mean([
            precision[np.searchsorted(recall, t, side="left")]
            for t in thresholds
        ])
    else:
        # 11-point VOC interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            ap  += precision[mask].max() / 11.0 if mask.any() else 0.0

    return float(ap)


# ── mAP ───────────────────────────────────────────────────────────────────────

def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute mAP50 and mAP50-95 over a dataset.

    Args:
        predictions:   List of {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "labels": [...]}.
        ground_truths: List of {"boxes": [[x1,y1,x2,y2],...], "labels": [...]}.
        num_classes:   Number of known classes.
        iou_thresholds: IoU thresholds to evaluate at. Defaults to [0.5] and 0.5:0.95:0.05.

    Returns:
        {"mAP50": float, "mAP50_95": float, "per_class_AP": {cls: float}}
    """
    iou_thresholds = iou_thresholds or [0.5] + list(np.arange(0.5, 1.0, 0.05))
    aps_per_thresh = {t: [] for t in iou_thresholds}

    for cls_idx in range(num_classes):
        cls_gts   = []
        cls_preds = []

        for img_gt, img_pred in zip(ground_truths, predictions):
            gt_mask  = [i for i, l in enumerate(img_gt["labels"])  if l == cls_idx]
            pred_mask= [i for i, l in enumerate(img_pred["labels"]) if l == cls_idx]

            cls_gts.append({
                "boxes": [img_gt["boxes"][i] for i in gt_mask],
            })
            cls_preds.append({
                "boxes":  [img_pred["boxes"][i]  for i in pred_mask],
                "scores": [img_pred["scores"][i] for i in pred_mask],
            })

        for iou_t in iou_thresholds:
            ap = _compute_class_ap(cls_preds, cls_gts, iou_t)
            aps_per_thresh[iou_t].append(ap)

    map50    = float(np.mean(aps_per_thresh[0.5]))
    map50_95 = float(np.mean([np.mean(v) for v in aps_per_thresh.values()]))

    per_class = {
        cls_idx: float(np.mean([aps_per_thresh[t][cls_idx] for t in iou_thresholds]))
        for cls_idx in range(num_classes)
    }

    return {
        "mAP50":         map50,
        "mAP50_95":      map50_95,
        "per_class_AP":  per_class,
    }


def _compute_class_ap(cls_preds, cls_gts, iou_threshold: float) -> float:
    """Compute AP for a single class at a given IoU threshold."""
    # Gather all predictions sorted by score
    all_preds = []
    for img_idx, pred in enumerate(cls_preds):
        for box, score in zip(pred["boxes"], pred["scores"]):
            all_preds.append((score, img_idx, box))

    if not all_preds:
        return 0.0

    all_preds.sort(key=lambda x: -x[0])
    n_gt = sum(len(g["boxes"]) for g in cls_gts)

    tp_arr = np.zeros(len(all_preds))
    fp_arr = np.zeros(len(all_preds))
    matched = [set() for _ in cls_gts]

    for k, (score, img_idx, pred_box) in enumerate(all_preds):
        gt_boxes = cls_gts[img_idx]["boxes"]
        best_iou, best_j = 0.0, -1

        for j, gt_box in enumerate(gt_boxes):
            if j in matched[img_idx]:
                continue
            iou = box_iou(np.array(pred_box), np.array(gt_box))
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j >= 0:
            tp_arr[k] = 1
            matched[img_idx].add(best_j)
        else:
            fp_arr[k] = 1

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    return compute_ap(tp_cum, fp_cum, n_gt)


# ── Forgetting metric ─────────────────────────────────────────────────────────

def compute_forgetting(
    task_a_before: float,
    task_a_after:  float,
) -> Dict[str, float]:
    """
    Measure catastrophic forgetting on Task A after learning Task B.

    Args:
        task_a_before: mAP50 on Task A (Waymo) *before* Task B training.
        task_a_after:  mAP50 on Task A (Waymo) *after*  Task B training.

    Returns:
        {
            "forgetting": drop in mAP (positive = forgot),
            "retention":  fraction of Task A performance retained,
            "relative_drop": percentage drop,
        }
    """
    forgetting    = task_a_before - task_a_after
    retention     = task_a_after  / (task_a_before + 1e-8)
    relative_drop = 100.0 * forgetting / (task_a_before + 1e-8)

    result = {
        "forgetting":    round(forgetting,    4),
        "retention":     round(retention,     4),
        "relative_drop": round(relative_drop, 2),
    }
    logger.info(
        f"Forgetting: {forgetting:.4f} | "
        f"Retention: {retention:.4f} | "
        f"Relative drop: {relative_drop:.2f}%"
    )
    return result


# ── Open-set metrics ──────────────────────────────────────────────────────────

def compute_openset_metrics(
    uncertainty_scores: List[float],
    is_unknown_gt:      List[bool],
) -> Dict[str, float]:
    """
    Compute open-set recognition metrics.

    Args:
        uncertainty_scores: Scalar uncertainty score per detection.
        is_unknown_gt:      Ground-truth boolean (True = unknown object).

    Returns:
        {
            "AUROC":  Area under the ROC curve.
            "AUPR":   Area under the Precision-Recall curve (unknown = positive).
            "FPR95":  False Positive Rate at 95% True Positive Rate.
        }
    """
    scores = np.array(uncertainty_scores)
    labels = np.array(is_unknown_gt, dtype=int)   # 1 = unknown, 0 = known

    if labels.sum() == 0 or labels.sum() == len(labels):
        logger.warning("Cannot compute open-set metrics: all samples have same class.")
        return {"AUROC": 0.0, "AUPR": 0.0, "FPR95": 1.0}

    if not _try_import_sklearn():
        logger.warning("scikit-learn not installed — returning placeholder metrics.")
        return {"AUROC": 0.0, "AUPR": 0.0, "FPR95": 1.0}

    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    auroc = float(roc_auc_score(labels, scores))
    aupr  = float(average_precision_score(labels, scores))

    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx    = np.searchsorted(tpr, 0.95)
    fpr95  = float(fpr[min(idx, len(fpr) - 1)])

    return {
        "AUROC": round(auroc, 4),
        "AUPR":  round(aupr,  4),
        "FPR95": round(fpr95, 4),
    }


# ── Stateful accumulator ──────────────────────────────────────────────────────

class DetectionEvaluator:
    """
    Accumulates per-image predictions/GTs across an epoch, then computes mAP.

    Usage:
        evaluator = DetectionEvaluator(num_classes=4)
        for batch in val_loader:
            preds, gts = model(batch)
            evaluator.update(preds, gts)
        metrics = evaluator.compute()
        evaluator.reset()
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self._predictions   = []
        self._ground_truths = []

    def update(self, predictions: List[Dict], ground_truths: List[Dict]):
        self._predictions.extend(predictions)
        self._ground_truths.extend(ground_truths)

    def compute(self) -> Dict[str, float]:
        return compute_map(
            self._predictions,
            self._ground_truths,
            num_classes=self.num_classes,
        )
