"""
tests/test_uncertainty.py
──────────────────────────
Unit tests for the open-set uncertainty detection module.
No GPU required.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.openset.uncertainty import (
    UncertaintyDetector, UNKNOWN_CLASS_IDX, UNKNOWN_CLASS_NAME
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def uniform_probs(n=4):
    """Maximally uncertain: uniform distribution → high entropy."""
    return np.ones(n) / n


def peaked_probs(n=4):
    """Minimally uncertain: all mass on one class → low entropy."""
    p = np.zeros(n); p[0] = 1.0
    return p


def make_detections(probs_list, confs=None):
    """Build a list of detection dicts from probability vectors."""
    dets = []
    for i, probs in enumerate(probs_list):
        det = {
            "box":  [0.1, 0.1, 0.5, 0.5],
            "conf": float(np.max(probs)),
            "cls":  int(np.argmax(probs)),
            "probs": probs,
        }
        dets.append(det)
    return dets


# ── Entropy metric ────────────────────────────────────────────────────────────

class TestEntropy:

    def test_uniform_is_max(self):
        det = UncertaintyDetector(metric="entropy", num_classes=4)
        assert abs(det.score(uniform_probs()) - 1.0) < 1e-6

    def test_peaked_is_zero(self):
        det = UncertaintyDetector(metric="entropy", num_classes=4)
        assert det.score(peaked_probs()) < 1e-6

    def test_range(self):
        det = UncertaintyDetector(metric="entropy", num_classes=4)
        for _ in range(20):
            p = np.random.dirichlet(np.ones(4))
            s = det.score(p)
            assert 0.0 <= s <= 1.0, f"Entropy out of range: {s}"

    def test_monotone(self):
        """More peaked → lower entropy."""
        det = UncertaintyDetector(metric="entropy", num_classes=4)
        very_peaked = np.array([0.9, 0.05, 0.025, 0.025])
        mild_peaked  = np.array([0.5, 0.3, 0.15, 0.05])
        uniform      = np.array([0.25, 0.25, 0.25, 0.25])
        s1 = det.score(very_peaked)
        s2 = det.score(mild_peaked)
        s3 = det.score(uniform)
        assert s1 < s2 < s3


# ── Max softmax metric ────────────────────────────────────────────────────────

class TestMaxSoftmax:

    def test_uniform_is_max(self):
        det = UncertaintyDetector(metric="max_softmax", num_classes=4)
        assert abs(det.score(uniform_probs()) - 0.75) < 1e-6

    def test_peaked_is_zero(self):
        det = UncertaintyDetector(metric="max_softmax", num_classes=4)
        assert det.score(peaked_probs()) < 1e-6

    def test_range(self):
        det = UncertaintyDetector(metric="max_softmax", num_classes=4)
        for _ in range(20):
            p = np.random.dirichlet(np.ones(4))
            s = det.score(p)
            assert 0.0 <= s <= 1.0


# ── Energy score ──────────────────────────────────────────────────────────────

class TestEnergyScore:

    def test_large_logits_lower_energy(self):
        """Higher max logit (confident) → lower (more negative) energy."""
        det = UncertaintyDetector(metric="energy", num_classes=4)
        confident_logits = np.array([10.0, 0.1, 0.1, 0.1])
        uncertain_logits = np.array([0.25, 0.25, 0.25, 0.25])
        s_conf = det.score(confident_logits)
        s_unc  = det.score(uncertain_logits)
        # Energy is negative; confident → more negative → lower score
        assert s_conf < s_unc

    def test_returns_float(self):
        det = UncertaintyDetector(metric="energy", num_classes=4)
        s = det.score(np.array([1.0, 2.0, 0.5, 0.3]))
        assert isinstance(s, float)


# ── Flagging logic ────────────────────────────────────────────────────────────

class TestFlagging:

    def test_uniform_flagged_unknown(self):
        """Uniform-probability detections should be flagged as unknown."""
        det   = UncertaintyDetector(metric="entropy", threshold=0.6, num_classes=4)
        dets  = make_detections([uniform_probs()])
        flagged = det.flag_unknowns(dets)
        assert flagged[0]["is_unknown"] is True
        assert flagged[0]["cls"] == UNKNOWN_CLASS_IDX

    def test_peaked_not_flagged(self):
        """High-confidence detections should not be flagged."""
        det   = UncertaintyDetector(metric="entropy", threshold=0.6, num_classes=4)
        dets  = make_detections([peaked_probs()])
        flagged = det.flag_unknowns(dets)
        assert flagged[0]["is_unknown"] is False

    def test_mixed_batch(self):
        """Correct flagging in a mixed batch."""
        probs  = [uniform_probs(), peaked_probs(), uniform_probs(), peaked_probs()]
        dets   = make_detections(probs)
        det    = UncertaintyDetector(metric="entropy", threshold=0.6, num_classes=4)
        flagged = det.flag_unknowns(dets)
        assert flagged[0]["is_unknown"] is True
        assert flagged[1]["is_unknown"] is False
        assert flagged[2]["is_unknown"] is True
        assert flagged[3]["is_unknown"] is False

    def test_uncertainty_field_added(self):
        """All detections get an 'uncertainty' field after flagging."""
        det   = UncertaintyDetector(metric="entropy", threshold=0.5, num_classes=4)
        dets  = make_detections([uniform_probs(), peaked_probs()])
        flagged = det.flag_unknowns(dets)
        for d in flagged:
            assert "uncertainty" in d
            assert isinstance(d["uncertainty"], float)

    def test_fallback_to_conf_when_no_probs(self):
        """If 'probs' is absent, falls back to 1 - conf."""
        det  = UncertaintyDetector(metric="entropy", threshold=0.5, num_classes=4)
        dets = [{"box": [0, 0, 1, 1], "conf": 0.1, "cls": 0}]  # no probs
        flagged = det.flag_unknowns(dets)
        # conf=0.1 → uncertainty=0.9 → should be flagged
        assert flagged[0]["is_unknown"] is True

    def test_threshold_boundary(self):
        """Detection exactly at threshold should be flagged (>= threshold)."""
        det = UncertaintyDetector(metric="max_softmax", threshold=0.75, num_classes=4)
        # uniform 4-class → max_softmax = 1 - 0.25 = 0.75
        dets = make_detections([uniform_probs()])
        flagged = det.flag_unknowns(dets)
        assert flagged[0]["is_unknown"] is True


# ── Calibration ───────────────────────────────────────────────────────────────

class TestCalibration:

    def test_calibration_returns_float(self):
        np.random.seed(42)
        det = UncertaintyDetector(metric="entropy", threshold=0.5, num_classes=4)
        val_dets = make_detections([
            np.random.dirichlet(np.ones(4)) for _ in range(100)
        ])
        val_labels = [0] * 80 + [-1] * 20   # 20 unknowns
        thresh = det.calibrate(val_dets, val_labels, target_fpr=0.1)
        assert isinstance(thresh, float)
        assert 0.0 <= thresh <= 1.0

    def test_calibration_updates_threshold(self):
        np.random.seed(0)
        det = UncertaintyDetector(metric="entropy", threshold=0.5, num_classes=4)
        val_dets   = make_detections([np.random.dirichlet(np.ones(4)) for _ in range(50)])
        val_labels = [0] * 50
        det.calibrate(val_dets, val_labels, target_fpr=0.05)
        # threshold should have been updated
        assert det.threshold != 0.5


# ── Stats ─────────────────────────────────────────────────────────────────────

class TestStats:

    def test_stats_keys(self):
        det = UncertaintyDetector(metric="entropy", threshold=0.6, num_classes=4)
        probs = [np.random.dirichlet(np.ones(4)) for _ in range(20)]
        dets  = make_detections(probs)
        flagged = det.flag_unknowns(dets)
        stats   = det.compute_stats(flagged)
        for key in ("n_detections", "n_unknown", "unknown_rate",
                    "uncertainty_mean", "uncertainty_std", "threshold"):
            assert key in stats, f"Missing key: {key}"

    def test_empty_stats(self):
        det = UncertaintyDetector()
        stats = det.compute_stats([])
        assert stats == {}
