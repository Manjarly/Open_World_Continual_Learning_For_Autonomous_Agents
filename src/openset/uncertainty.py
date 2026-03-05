"""
src/openset/uncertainty.py
──────────────────────────
Entropy-based Open-Set Recognition for YOLOv8 detections.

An object detector trained on known classes (vehicle, pedestrian, cyclist, sign)
will produce a probability distribution over those classes for every detection.
When the detector encounters an "unknown" object (e.g., a novel obstacle in
Singapore that it never saw in Waymo training), the distribution tends to be
more uniform — i.e., high entropy.

We exploit this to flag detections as "unknown" when their prediction
entropy exceeds a calibrated threshold.

Three uncertainty metrics are supported:
  • entropy      — Shannon entropy of the class probability vector (default)
  • max_softmax  — 1 - max(softmax), i.e. low confidence = high uncertainty
  • energy       — Energy score (Liu et al., 2020): -log Σ exp(logit_i)

Reference:
    Liu et al., "Energy-based Out-of-Distribution Detection", NeurIPS 2020.
    https://arxiv.org/abs/2010.03759

Usage:
    from src.openset.uncertainty import UncertaintyDetector

    detector = UncertaintyDetector(metric="entropy", threshold=0.6)

    # detections from YOLODetector.predict(return_probs=True)
    flagged = detector.flag_unknowns(detections)
    calibrated_thresh = detector.calibrate(val_detections, val_labels, target_fpr=0.05)
"""

import logging
import numpy as np
from typing import Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

UncertaintyMetric = Literal["entropy", "max_softmax", "energy"]

# Unknown class label used when a detection is flagged
UNKNOWN_CLASS_IDX = -1
UNKNOWN_CLASS_NAME = "unknown"


class UncertaintyDetector:
    """
    Wraps a set of YOLOv8 detections and flags those with high uncertainty
    as "unknown" objects for open-set recognition.

    Args:
        metric:    Uncertainty metric to use ("entropy", "max_softmax", "energy").
        threshold: Detections with uncertainty >= threshold are flagged unknown.
                   Can be calibrated via calibrate() on a validation set.
        num_classes: Number of known classes (used for entropy normalisation).
    """

    def __init__(
        self,
        metric:      UncertaintyMetric = "entropy",
        threshold:   float = 0.6,
        num_classes: int   = 4,
    ):
        assert metric in ("entropy", "max_softmax", "energy"), \
            f"Invalid metric '{metric}'. Choose from: entropy, max_softmax, energy."
        self.metric      = metric
        self.threshold   = threshold
        self.num_classes = num_classes

    # ── Uncertainty scoring ───────────────────────────────────────────────────

    def score(self, probs: np.ndarray) -> float:
        """
        Compute a scalar uncertainty score for a single detection.

        Args:
            probs: Class probability vector (shape: [num_classes]).
                   Should be post-softmax. For energy score, raw logits are
                   expected — pass logits=True in that case.

        Returns:
            Scalar uncertainty in [0, 1] (entropy, max_softmax) or (−∞, 0] (energy).
        """
        probs = np.asarray(probs, dtype=np.float64)

        if self.metric == "entropy":
            return self._entropy(probs)
        elif self.metric == "max_softmax":
            return self._max_softmax(probs)
        elif self.metric == "energy":
            return self._energy_score(probs)

    def score_batch(self, probs_batch: np.ndarray) -> np.ndarray:
        """Compute uncertainty scores for a batch of probability vectors."""
        return np.array([self.score(p) for p in probs_batch])

    # ── Metric implementations ────────────────────────────────────────────────

    def _entropy(self, probs: np.ndarray) -> float:
        """
        Normalised Shannon entropy H(p) / log(K), in [0, 1].
        Higher = more uncertain.
        """
        probs  = np.clip(probs, 1e-10, 1.0)
        probs /= probs.sum()   # re-normalise for safety
        H = -np.sum(probs * np.log(probs))
        H_max = np.log(self.num_classes)   # maximum possible entropy
        return float(H / H_max) if H_max > 0 else 0.0

    @staticmethod
    def _max_softmax(probs: np.ndarray) -> float:
        """
        1 − max(softmax).
        Higher = lower max confidence = more uncertain.
        """
        probs = np.clip(probs, 0.0, 1.0)
        return float(1.0 - np.max(probs))

    @staticmethod
    def _energy_score(logits: np.ndarray) -> float:
        """
        Negative energy score: -log Σ exp(logit_i).
        Out-of-distribution samples tend to have higher (less negative) energy.
        Returns value in (−∞, 0] after negation.

        Note: pass raw logits (pre-softmax) for correct energy computation.
        """
        # Numerically stable log-sum-exp
        max_logit = np.max(logits)
        lse = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
        return float(-lse)

    # ── Flagging ──────────────────────────────────────────────────────────────

    def flag_unknowns(self, detections: List[Dict]) -> List[Dict]:
        """
        Process a list of detections and flag uncertain ones as unknown.

        Each detection dict should have either:
          - "probs": np.ndarray of class probabilities (preferred)
          - "conf":  float confidence (fallback — uses max_softmax heuristic)

        Args:
            detections: List of detection dicts from YOLODetector.predict().

        Returns:
            Same list with added fields:
              "uncertainty": float score
              "is_unknown":  bool
            Detections flagged as unknown have "cls" set to UNKNOWN_CLASS_IDX.
        """
        results = []
        for det in detections:
            det = dict(det)   # shallow copy

            if "probs" in det and det["probs"] is not None:
                uncertainty = self.score(det["probs"])
            else:
                # Fallback: treat 1 − conf as max_softmax uncertainty
                uncertainty = 1.0 - float(det.get("conf", 0.5))

            is_unknown = uncertainty >= self.threshold

            det["uncertainty"] = round(uncertainty, 4)
            det["is_unknown"]  = is_unknown

            if is_unknown:
                det["cls"]  = UNKNOWN_CLASS_IDX
                det["name"] = UNKNOWN_CLASS_NAME

            results.append(det)

        n_unknown = sum(1 for d in results if d["is_unknown"])
        logger.debug(
            f"Flagged {n_unknown}/{len(results)} detections as unknown "
            f"(threshold={self.threshold:.3f}, metric={self.metric})"
        )
        return results

    # ── Threshold calibration ─────────────────────────────────────────────────

    def calibrate(
        self,
        val_detections: List[Dict],
        val_labels:     List[int],
        target_fpr:     float = 0.05,
    ) -> float:
        """
        Calibrate threshold on a validation set to achieve a target False Positive Rate.

        A "false positive" here means a known object flagged as unknown.

        Args:
            val_detections: Detections with "probs" field.
            val_labels:     Ground truth class indices (−1 if truly unknown).
            target_fpr:     Desired FPR on known classes (default: 5%).

        Returns:
            Optimal threshold (also sets self.threshold).
        """
        assert len(val_detections) == len(val_labels), \
            "val_detections and val_labels must have the same length."

        # Compute uncertainty scores
        scores = np.array([
            self.score(d["probs"]) if "probs" in d else 1.0 - d.get("conf", 0.5)
            for d in val_detections
        ])
        labels = np.array(val_labels)

        # Known samples = label >= 0
        known_mask = labels >= 0
        known_scores = scores[known_mask]

        if len(known_scores) == 0:
            logger.warning("No known samples found for calibration.")
            return self.threshold

        # Find threshold such that FPR on known samples ≤ target_fpr
        threshold = float(np.quantile(known_scores, 1.0 - target_fpr))
        self.threshold = threshold

        # Compute resulting metrics
        unknown_mask = labels < 0
        if unknown_mask.sum() > 0:
            unknown_scores = scores[unknown_mask]
            tpr = float((unknown_scores >= threshold).mean())
            fpr = float((known_scores   >= threshold).mean())
            logger.info(
                f"Calibrated threshold = {threshold:.4f} | "
                f"TPR (unknown recall) = {tpr:.3f} | "
                f"FPR (known false alarms) = {fpr:.3f}"
            )
        else:
            logger.info(f"Calibrated threshold = {threshold:.4f} (no unknown samples in val set)")

        return threshold

    # ── Statistics ────────────────────────────────────────────────────────────

    def compute_stats(self, detections: List[Dict]) -> Dict:
        """Summarise uncertainty distribution over a set of detections."""
        if not detections:
            return {}
        scores = np.array([d.get("uncertainty", 0.0) for d in detections])
        is_unknown = np.array([d.get("is_unknown", False) for d in detections])
        return {
            "n_detections":      len(detections),
            "n_unknown":         int(is_unknown.sum()),
            "unknown_rate":      float(is_unknown.mean()),
            "uncertainty_mean":  float(scores.mean()),
            "uncertainty_std":   float(scores.std()),
            "uncertainty_p95":   float(np.percentile(scores, 95)),
            "threshold":         self.threshold,
            "metric":            self.metric,
        }
