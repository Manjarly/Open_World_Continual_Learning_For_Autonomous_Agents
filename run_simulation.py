"""
run_simulation.py
─────────────────
Interactive 4-Wheel Autonomous Vehicle Simulator using Real Waymo Open Dataset Paths.

Workflow:
  1. Select from 100+ real Waymo street paths extracted from Waymo Open Dataset TFRecords.
  2. Live Training Simulation: Starts immediately, displays each attempt, highlights
     red crash markers (X) and error diagnosis, and visually demonstrates learning.
  3. Obstacle Training: Re-trains on the same street with dynamic obstacles
     (vehicles, pedestrians, construction barrels).
  4. Generalization Testing: Select any unseen Waymo street from the library
     (with random, none, or heavy obstacles) and watch the trained model navigate live!

Usage:
    python run_simulation.py
    python run_simulation.py --auto (for non-interactive execution)
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Suppress noisy external logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_simulation")

from src.simulation.agent import DrivingPolicy, resolve_rl_device
from src.simulation.continual_rl import ContinualRLTrainer
from src.simulation.environment import (
    AutonomousDrivingEnv,
    build_waymo_track,
    load_waymo_streets,
)
from src.simulation.renderer import SimulationRenderer


# Check if GUI window is available (OpenCV or Matplotlib)
GUI_AVAILABLE = False
try:
    import cv2
    GUI_AVAILABLE = True
except ImportError:
    pass


def display_frame_live(frame_img, window_name: str = "Waymo Autonomous Simulation", wait_ms: int = 400):
    """Display frame in OpenCV GUI window if available, or save to disk."""
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    live_path = str(runs_dir / "live_simulation_frame.png")

    if isinstance(frame_img, str):
        # Already a path
        return
    
    # Save frame for browser / Streamlit
    frame_img.save(live_path)

    if GUI_AVAILABLE:
        try:
            # Convert PIL RGB to OpenCV BGR
            cv_img = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, cv_img)
            cv2.waitKey(wait_ms)
        except Exception:
            pass


def print_banner():
    print("""
================================================================================
  🚗 WAYMO OPEN DATASET: 4-WHEEL AUTONOMOUS VEHICLE TRAINING & SIMULATION
================================================================================
  Using real street coordinates, poses, and road geometry extracted from
  Waymo Open Dataset TFRecords (San Francisco & Phoenix).
""")


def select_street_interactive(streets: List[Dict], default_idx: int = 0) -> Dict:
    """Prompt user to choose a Waymo street path."""
    print(f"Library: {len(streets)} authentic Waymo street paths available.")
    print("\nSample Streets:")
    sample_indices = [0, 15, 30, 45, 60, 75, 90, 105]
    for idx in sample_indices:
        if idx < len(streets):
            s = streets[idx]
            print(f"  [{idx + 1:3d}] {s['name']} | Length: {s['length_m']}m")

    print("\nOptions:")
    print("  • Enter street number (1 - 111)")
    print("  • Enter 'r' for a random street")
    print("  • Press Enter to select Street #1")

    choice = input("\n👉 Select a street path: ").strip().lower()

    if choice == "r" or choice == "random":
        chosen_idx = random.randint(0, len(streets) - 1)
    elif choice.isdigit():
        chosen_idx = max(0, min(int(choice) - 1, len(streets) - 1))
    else:
        chosen_idx = default_idx

    selected = streets[chosen_idx]
    print(f"\n✅ Selected: {selected['name']} ({selected['length_m']}m | {selected['street_type']})")
    return selected


def run_stage_1_training(
    street: Dict,
    policy: DrivingPolicy,
    trainer: ContinualRLTrainer,
    max_episodes: int = 15,
) -> bool:
    """
    Stage 1: Live visual training on the chosen Waymo street.
    Demonstrates crashes (red X) and recovery until the car masters the path.
    """
    print("\n" + "=" * 80)
    print("  🏁 STAGE 1: LIVE VISUAL TRAINING (LEARNING THE WAYMO STREET)")
    print("=" * 80)
    print("  Watch the simulation window. The car attempts the street, logs errors,")
    print("  and updates its policy upon crashes until it completes the route!\n")

    track = build_waymo_track(street, obstacle_mode="none")
    env = AutonomousDrivingEnv(track, max_steps=200)

    street_mastered = False

    for ep in range(1, max_episodes + 1):
        obs = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0

        obs_buf = []
        m_acts_buf, i_acts_buf, h_acts_buf = [], [], []
        rewards_buf = []

        # Single episode rollout
        while not (terminated or truncated):
            actions, _, _, unc = policy.act(obs, deterministic=(ep > 8))
            next_obs, reward, terminated, truncated, info = env.step(
                actions[0], actions[1], actions[2], uncertainty_score=unc
            )

            obs_buf.append(obs)
            m_acts_buf.append(actions[0])
            i_acts_buf.append(actions[1])
            h_acts_buf.append(actions[2])
            rewards_buf.append(reward)

            obs = next_obs
            ep_reward += reward

        # Render visual frame with failure marker or success
        crash_reason = info["failure_reason"] if not info["route_completed"] else None
        frame = SimulationRenderer.render_episode_frame(
            track=track,
            history_x=env.history_x,
            history_y=env.history_y,
            car=env.car,
            crash_reason=crash_reason,
            episode_num=ep,
            stage_name="Stage 1: Learning Street Path",
        )
        display_frame_live(frame, wait_ms=450)

        # Print episode telemetry
        if info["route_completed"]:
            print(f"  [Ep {ep:2d}] 🏆 ROUTE COMPLETED! Distance: {info['progress_m']:.1f}m / {info['total_track_m']:.1f}m (100%)")
            street_mastered = True
            time.sleep(1.0)
            break
        else:
            print(f"  [Ep {ep:2d}] 💥 CRASH at {info['progress_m']:.1f}m ({info['completion_pct']:.1f}%) | Error: {info['failure_reason']}")

        # Train policy on the collected rollout
        if len(obs_buf) > 1:
            trainer.train_on_env(env, num_episodes=1, desc=f"Ep {ep} Update")

    if not street_mastered:
        print(f"\n  ℹ️ Street training reached max episodes. Progress reached: {info['progress_m']:.1f}m")
    else:
        print("\n  🎉 Optimal path learned! The car has mastered navigation on this street.")

    return street_mastered


def run_stage_2_obstacle_training(
    street: Dict,
    policy: DrivingPolicy,
    trainer: ContinualRLTrainer,
    num_episodes: int = 8,
):
    """
    Stage 2: Train on the same street with varied obstacles.
    """
    print("\n" + "=" * 80)
    print("  🚧 STAGE 2: OBSTACLE TRAINING (VEHICLES, PEDESTRIANS, BARRICADES)")
    print("=" * 80)
    print("  Now training on the same street with dynamic obstacles spawned along lanes.")
    print("  The car learns to avoid collisions, brake, and steer around hazards!\n")

    track_obs = build_waymo_track(street, obstacle_mode="random")
    print(f"  Spawned {len(track_obs.obstacles)} obstacles along the street:")
    for obs in track_obs.obstacles:
        print(f"    • {obs.obstacle_type} at ({obs.x:.1f}, {obs.y:.1f})")

    env = AutonomousDrivingEnv(track_obs, max_steps=200)

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        terminated = False
        truncated = False

        obs_buf, m_buf, i_buf, h_buf, r_buf = [], [], [], [], []
        while not (terminated or truncated):
            actions, _, _, unc = policy.act(obs, deterministic=(ep > 3))
            next_obs, reward, terminated, truncated, info = env.step(
                actions[0], actions[1], actions[2], uncertainty_score=unc
            )
            obs_buf.append(obs)
            m_buf.append(actions[0]); i_buf.append(actions[1]); h_buf.append(actions[2])
            r_buf.append(reward)
            obs = next_obs

        crash_reason = info["failure_reason"] if not info["route_completed"] else None
        frame = SimulationRenderer.render_episode_frame(
            track=track_obs,
            history_x=env.history_x,
            history_y=env.history_y,
            car=env.car,
            crash_reason=crash_reason,
            episode_num=ep,
            stage_name="Stage 2: Obstacle Avoidance Training",
        )
        display_frame_live(frame, wait_ms=450)

        if info["route_completed"]:
            print(f"  [Obstacle Ep {ep}] 🏆 COMPLETED WITH OBSTACLES! ({info['progress_m']:.1f}m)")
            time.sleep(1.0)
            break
        else:
            print(f"  [Obstacle Ep {ep}] 💥 {info['failure_reason']} at {info['progress_m']:.1f}m")

        if len(obs_buf) > 1:
            trainer.train_on_env(env, num_episodes=1, desc=f"Obs Ep {ep}")

    print("\n  ✅ Obstacle avoidance training completed!")


def run_stage_3_generalization_test(
    streets: List[Dict],
    policy: DrivingPolicy,
    auto_mode: bool = False,
):
    """
    Stage 3: Test on unseen Waymo streets with or without obstacles.
    """
    print("\n" + "=" * 80)
    print("  🌐 STAGE 3: GENERALIZATION TEST ON UNSEEN WAYMO STREETS")
    print("=" * 80)

    while True:
        if auto_mode:
            test_street = random.choice(streets)
            obs_mode = "random"
        else:
            print(f"\nSelect an unseen Waymo street to test (1 - {len(streets)}, 'r' for random, or 'q' to quit):")
            choice = input("👉 Test street: ").strip().lower()

            if choice in ("q", "quit", "exit"):
                break
            elif choice in ("r", "random"):
                test_street = random.choice(streets)
            elif choice.isdigit():
                idx = max(0, min(int(choice) - 1, len(streets) - 1))
                test_street = streets[idx]
            else:
                test_street = random.choice(streets)

            print("\nSelect obstacle mode:")
            print("  [1] Random Obstacles")
            print("  [2] Clean Street (No Obstacles)")
            print("  [3] Heavy Traffic & Barricades")
            obs_choice = input("👉 Obstacle mode [1-3, default=1]: ").strip()
            obs_map = {"1": "random", "2": "none", "3": "heavy"}
            obs_mode = obs_map.get(obs_choice, "random")

        print(f"\n🚗 Running test on {test_street['name']} with obstacle mode='{obs_mode}'...")
        track = build_waymo_track(test_street, obstacle_mode=obs_mode)
        env = AutonomousDrivingEnv(track, max_steps=250)

        obs = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            actions, _, _, unc = policy.act(obs, deterministic=True, enable_cautious_reflex=True)
            obs, reward, terminated, truncated, info = env.step(
                actions[0], actions[1], actions[2], uncertainty_score=unc
            )

        crash_reason = info["failure_reason"] if not info["route_completed"] else None
        frame = SimulationRenderer.render_episode_frame(
            track=track,
            history_x=env.history_x,
            history_y=env.history_y,
            car=env.car,
            crash_reason=crash_reason,
            episode_num=1,
            stage_name=f"Generalization Test ({obs_mode.capitalize()} Obstacles)",
        )
        display_frame_live(frame, wait_ms=1500)

        status_text = "🏆 SUCCESSFULLY COMPLETED UNSEEN STREET" if info["route_completed"] else f"💥 {info['failure_reason']}"
        print(f"\n  Result: {status_text}")
        print(f"  Distance Traveled: {info['progress_m']:.1f}m / {info['total_track_m']:.1f}m ({info['completion_pct']:.1f}%)")
        print(f"  Active Lights: Left={info['active_lights']['left']}, Right={info['active_lights']['right']}, Headlights={info['active_lights']['headlights']}")

        if auto_mode:
            break


def main():
    parser = argparse.ArgumentParser(description="Waymo Open Dataset Interactive Simulation")
    parser.add_argument("--auto", action="store_true", help="Run in non-interactive automatic mode")
    parser.add_argument("--street", type=int, default=None, help="Street index to train on (1-111)")
    args = parser.parse_args()

    print_banner()

    # Load 100+ Waymo streets
    streets = load_waymo_streets()
    if not streets:
        print("❌ Error: No Waymo streets found. Please run: python -m src.data.waymo_path_extractor")
        return

    # Select base street to learn
    if args.street is not None:
        street_idx = max(0, min(args.street - 1, len(streets) - 1))
        chosen_street = streets[street_idx]
        print(f"Selected Street #{args.street}: {chosen_street['name']}")
    elif args.auto:
        chosen_street = streets[0]
        print(f"Auto mode: selected {chosen_street['name']}")
    else:
        chosen_street = select_street_interactive(streets)

    # Initialize device, policy, and trainer
    device = resolve_rl_device("auto")
    print(f"Hardware compute device: {device} (MPS/CUDA GPU memory cap enforced)")

    policy = DrivingPolicy(device=device)
    trainer = ContinualRLTrainer(policy=policy, device=device)

    # ── Stage 1: Live Visual Training ─────────────────────────────────────────
    run_stage_1_training(chosen_street, policy, trainer, max_episodes=10 if args.auto else 15)

    # ── Stage 2: Obstacle Training on the Same Street ─────────────────────────
    run_stage_2_obstacle_training(chosen_street, policy, trainer, num_episodes=5 if args.auto else 8)

    # ── Stage 3: Generalization Testing on Other Waymo Streets ────────────────
    run_stage_3_generalization_test(streets, policy, auto_mode=args.auto)

    if GUI_AVAILABLE:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("  🏁 SIMULATION RUN COMPLETE. All frames saved to runs/live_simulation_frame.png")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
