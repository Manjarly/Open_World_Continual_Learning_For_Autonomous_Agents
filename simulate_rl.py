"""
simulate_rl.py
──────────────
Autonomous Driving Simulation & Cross-City Scalability Benchmark.

Demonstrates:
  1. Base training on City A (rules, navigation, indicators).
  2. Zero-shot deployment to unknown path in City B:
     - Visualizes where, how, and why the naive baseline fails (crashes in dark tunnel / lane departure).
  3. Ultra-fast few-shot adaptation on City B using OWCL + EWC (10-15 episodes in minutes).
  4. Post-adaptation evaluation on City B (navigates safely with headlights & indicators).
  5. City A retention check (proves 0% catastrophic forgetting).
  6. Generates high-resolution failure showcase comparison graphic.

Usage:
    python simulate_rl.py
    python simulate_rl.py --smoke_test
    python simulate_rl.py --episodes_a 50 --few_shot_episodes 15 --device mps
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulate_rl")

from src.simulation.agent import DrivingPolicy, resolve_rl_device
from src.simulation.continual_rl import ContinualRLTrainer
from src.simulation.environment import build_city_a_track, build_city_b_track
from src.simulation.renderer import SimulationRenderer


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Driving RL Simulation & Scalability Transfer")
    parser.add_argument("--episodes_a", type=int, default=40, help="City A base training episodes")
    parser.add_argument("--few_shot_episodes", type=int, default=12, help="City B few-shot adaptation episodes")
    parser.add_argument("--ewc_lambda", type=float, default=80.0, help="EWC regularization weight")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'mps', 'cuda', 'cpu')")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--save_plot", type=str, default="runs/simulation_failure_showcase.png")
    return parser.parse_args()


def print_scalability_table(results: dict):
    """Print ASCII comparison table highlighting sample efficiency & scalability."""
    eval_a_before = results["eval_a_before"]
    eval_b_zero = results["eval_b_zero_shot"]
    eval_b_cautious = results["eval_b_cautious"]
    eval_b_adapted = results["eval_b_adapted"]
    eval_a_after = results["eval_a_after"]

    header = f"{'Phase':<28} | {'City':<8} | {'Episodes':<9} | {'Dist (m)':<9} | {'Compl %':<8} | {'City A Ret %':<12} | {'Status'}"
    sep = "-" * len(header)

    rows = [
        ("1. City A Baseline", "City A", str(results.get("episodes_a", 40)), f"{eval_a_before['distance_m']}", f"{eval_a_before['completion_pct']}%", "100.0%", eval_a_before["failure_reason"][:30]),
        ("2. Zero-Shot Naive Deploy", "City B", "0 (0s)", f"{eval_b_zero['distance_m']}", f"{eval_b_zero['completion_pct']}%", "100.0%", eval_b_zero["failure_reason"][:30]),
        ("3. Zero-Shot OWCL Reflex", "City B", "0 (0s)", f"{eval_b_cautious['distance_m']}", f"{eval_b_cautious['completion_pct']}%", "100.0%", eval_b_cautious["failure_reason"][:30]),
        ("4. OWCL + EWC (Few-Shot)", "City B", f"{results['few_shot_episodes']} ({results['time_city_b_adapt_sec']}s)", f"{eval_b_adapted['distance_m']}", f"{eval_b_adapted['completion_pct']}%", f"{results['retention_pct']}%", eval_b_adapted["failure_reason"][:30]),
        ("5. Post-Adaptation City A", "City A", "-", f"{eval_a_after['distance_m']}", f"{eval_a_after['completion_pct']}%", f"{results['retention_pct']}%", eval_a_after["failure_reason"][:30]),
    ]

    print("\n" + "=" * len(header))
    print("  AUTONOMOUS VEHICLE SCALABILITY BENCHMARK (CROSS-CITY TRANSFER)")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in rows:
        print(f"{r[0]:<28} | {r[1]:<8} | {r[2]:<9} | {r[3]:<9} | {r[4]:<8} | {r[5]:<12} | {r[6]}")
    print("=" * len(header) + "\n")


def main():
    args = parse_args()
    device = resolve_rl_device(args.device)
    logger.info(f"Using compute device: {device} (Apple MPS / CUDA memory caps enforced)")

    episodes_a = 4 if args.smoke_test else args.episodes_a
    few_shot_b = 2 if args.smoke_test else args.few_shot_episodes

    # Initialize policy and trainer
    policy = DrivingPolicy(device=device)
    trainer = ContinualRLTrainer(policy=policy, device=device)

    # Run the end-to-end city transfer experiment
    results = trainer.run_city_transfer_experiment(
        episodes_city_a=episodes_a,
        few_shot_episodes_b=few_shot_b,
        ewc_lambda=args.ewc_lambda,
    )
    results["episodes_a"] = episodes_a

    # Print summary table
    print_scalability_table(results)

    # Render and save high-impact failure showcase graphic
    track_b = build_city_b_track()
    plot_path = SimulationRenderer.render_failure_showcase(
        eval_zero_shot=results["eval_b_zero_shot"],
        eval_adapted=results["eval_b_adapted"],
        track=track_b,
        save_path=args.save_plot,
    )
    logger.info(f"Saved failure showcase comparison plot to: {plot_path}")

    # Save metrics JSON
    metrics_path = Path("runs/simulation_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        # Strip long arrays for clean json summary
        summary = {
            "retention_pct": results["retention_pct"],
            "forgetting_pct": results["forgetting_pct"],
            "time_city_a_sec": results["time_city_a_sec"],
            "time_city_b_adapt_sec": results["time_city_b_adapt_sec"],
            "city_b_zero_shot_distance_m": results["eval_b_zero_shot"]["distance_m"],
            "city_b_zero_shot_failure": results["eval_b_zero_shot"]["failure_reason"],
            "city_b_adapted_distance_m": results["eval_b_adapted"]["distance_m"],
            "city_b_adapted_completion_pct": results["eval_b_adapted"]["completion_pct"],
        }
        json.dump(summary, f, indent=2)
    logger.info(f"Saved simulation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
