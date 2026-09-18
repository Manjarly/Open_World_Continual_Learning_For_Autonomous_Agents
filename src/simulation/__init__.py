"""
src/simulation
──────────────
Autonomous driving simulation with 4-wheel car dynamics, vehicle signaling,
and open-world continual reinforcement learning.
"""

from src.simulation.environment import (
    AutonomousDrivingEnv,
    Car4Wheel,
    CityTrack,
    load_waymo_streets,
    build_waymo_track,
)
from src.simulation.agent import DrivingPolicy, resolve_rl_device
from src.simulation.continual_rl import ContinualRLTrainer
from src.simulation.renderer import SimulationRenderer
from src.simulation.renderer3d import SimulationRenderer3D

__all__ = [
    "AutonomousDrivingEnv",
    "Car4Wheel",
    "CityTrack",
    "load_waymo_streets",
    "build_waymo_track",
    "DrivingPolicy",
    "resolve_rl_device",
    "ContinualRLTrainer",
    "SimulationRenderer",
    "SimulationRenderer3D",
]
