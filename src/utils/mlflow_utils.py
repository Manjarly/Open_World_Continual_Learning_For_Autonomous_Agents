"""
src/utils/mlflow_utils.py
─────────────────────────
MLflow helpers for experiment tracking across Phase 1 and Phase 2.

Provides:
  - setup_mlflow()       : Initialize tracking URI & experiment.
  - log_config()         : Log a YAML config dict as MLflow params.
  - log_phase1_results() : Log Phase 1 (baseline) metrics.
  - log_phase2_results() : Log Phase 2 (continual) metrics.
  - compare_runs()       : Pull a summary DataFrame across all runs.

Usage:
    from src.utils.mlflow_utils import setup_mlflow, log_phase1_results

    run_id = setup_mlflow(experiment_name="owcl_experiments")
    with mlflow.start_run(run_name="waymo_baseline"):
        log_phase1_results(metrics, config)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MLFLOW_AVAILABLE = False
mlflow = None

def _try_import_mlflow():
    global MLFLOW_AVAILABLE, mlflow
    if mlflow is not None:
        return mlflow
    try:
        import mlflow as _mlflow
        mlflow = _mlflow
        MLFLOW_AVAILABLE = True
        return mlflow
    except Exception:
        return None


def setup_mlflow(
    experiment_name: str = "owcl_experiments",
    tracking_uri:    str = "mlruns",
) -> Optional[str]:
    mlf = _try_import_mlflow()
    if not mlf:
        logger.warning("MLflow not available — skipping setup.")
        return None

    mlf.set_tracking_uri(tracking_uri)
    experiment = mlf.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlf.create_experiment(
            name=experiment_name,
            tags={"project": "OWCL", "team": "Delaware"},
        )
        logger.info(f"Created MLflow experiment '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{experiment_name}' (id={experiment_id})")

    mlf.set_experiment(experiment_name)
    return experiment_id


def log_config(config: Dict[str, Any], prefix: str = ""):
    mlf = _try_import_mlflow()
    if not mlf:
        return
    flat = _flatten_dict(config, prefix=prefix)
    flat_str = {k: str(v)[:500] for k, v in flat.items()}
    try:
        mlf.log_params(flat_str)
    except Exception as e:
        logger.warning(f"Failed to log some params: {e}")


def log_phase1_results(metrics: Dict[str, float], config: Optional[Dict] = None):
    mlf = _try_import_mlflow()
    if not mlf:
        logger.info(f"[Phase 1 metrics] {metrics}")
        return
    if config:
        log_config(config, prefix="phase1")
    mlf.log_metrics({
        "phase1/mAP50":     metrics.get("mAP50",     0.0),
        "phase1/mAP50_95":  metrics.get("mAP50_95",  0.0),
        "phase1/precision": metrics.get("precision",  0.0),
        "phase1/recall":    metrics.get("recall",     0.0),
    })
    mlf.set_tag("phase", "1_baseline")
    logger.info(f"Phase 1 metrics logged to MLflow: {metrics}")


def log_phase2_results(
    metrics:      Dict[str, float],
    ewc_summary:  Optional[Dict] = None,
    openset_stats: Optional[Dict] = None,
    config:       Optional[Dict] = None,
):
    mlf = _try_import_mlflow()
    if not mlf:
        logger.info(f"[Phase 2 metrics] {metrics}")
        return
    if config:
        log_config(config, prefix="phase2")
    mlf.log_metrics({
        "phase2/mAP50":     metrics.get("mAP50",     0.0),
        "phase2/mAP50_95":  metrics.get("mAP50_95",  0.0),
        "phase2/precision": metrics.get("precision",  0.0),
        "phase2/recall":    metrics.get("recall",     0.0),
    })
    if ewc_summary:
        mlf.log_metrics({
            "ewc/lambda":          ewc_summary.get("ewc_lambda",      0.0),
            "ewc/non_zero_fisher": ewc_summary.get("non_zero_fisher", 0),
        })
        mlf.set_tag("ewc_lambda", str(ewc_summary.get("ewc_lambda")))
    if openset_stats:
        mlf.log_metrics({
            "openset/unknown_rate":      openset_stats.get("unknown_rate",     0.0),
            "openset/uncertainty_mean":  openset_stats.get("uncertainty_mean", 0.0),
            "openset/uncertainty_p95":   openset_stats.get("uncertainty_p95",  0.0),
        })
        mlf.set_tag("openset_threshold", str(openset_stats.get("threshold")))
    mlf.set_tag("phase", "2_continual")
    logger.info("Phase 2 metrics logged to MLflow.")


def compare_runs(experiment_name: str = "owcl_experiments") -> Optional[object]:
    mlf = _try_import_mlflow()
    if not mlf:
        logger.warning("MLflow not available.")
        return None
    try:
        import pandas as pd
        client = mlf.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found.")
            return None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.phase2/mAP50 DESC"],
        )
        records = []
        for run in runs:
            records.append({
                "run_id":   run.info.run_id[:8],
                "run_name": run.data.tags.get("mlflow.runName", "—"),
                "phase":    run.data.tags.get("phase", "—"),
                **{k: round(v, 4) for k, v in run.data.metrics.items()},
            })
        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Failed to fetch run comparison: {e}")
        return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _flatten_dict(d: Dict, prefix: str = "", sep: str = "/") -> Dict:
    """Recursively flatten a nested dict."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out
