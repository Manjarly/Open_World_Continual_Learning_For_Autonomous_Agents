"""
tests/test_simulation.py
────────────────────────
Unit tests for the Autonomous Driving RL Simulation:
  - 4-Wheel Car Dynamics (Ackerman geometry, braking, lights)
  - Environment stepping, signaling rules, and dark tunnel visibility reduction
  - Multi-Head DrivingPolicy on MPS/CPU with Open-World Cautious Reflex
  - PolicyEWC Fisher Information and penalty computation
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.agent import DrivingPolicy, resolve_rl_device
from src.simulation.continual_rl import PolicyEWC
from src.simulation.environment import (
    AutonomousDrivingEnv,
    Car4Wheel,
    Obstacle,
    build_city_a_track,
    build_city_b_track,
)


# ── 1. 4-Wheel Car Dynamics Tests ─────────────────────────────────────────────

class TestCar4Wheel:

    def test_car_initialization_and_geometry(self):
        car = Car4Wheel(x=10.0, y=5.0, yaw=0.0)
        assert car.length == 4.5
        assert car.width == 2.0
        assert car.wheelbase == 2.8
        assert car.track_width == 1.6

        corners = car.get_corners()
        assert corners.shape == (4, 2)
        # Check that front corners are ahead of x=10
        assert corners[0, 0] > 10.0
        assert corners[1, 0] > 10.0
        # Check that rear corners are behind x=10
        assert corners[2, 0] < 10.0
        assert corners[3, 0] < 10.0

    def test_wheel_positions_and_ackerman_steer(self):
        car = Car4Wheel(x=0.0, y=0.0, yaw=0.0, steer=math.radians(15.0))
        wheels = car.get_wheel_positions()
        assert "FL" in wheels and "FR" in wheels and "RL" in wheels and "RR" in wheels

        # Front wheels should have yaw equal to car.yaw + car.steer
        fl_x, fl_y, fl_yaw = wheels["FL"]
        assert pytest.approx(fl_yaw, abs=1e-4) == math.radians(15.0)

        # Rear wheels should have yaw equal to car.yaw (0)
        rl_x, rl_y, rl_yaw = wheels["RL"]
        assert pytest.approx(rl_yaw, abs=1e-4) == 0.0

    def test_braking_and_brake_lights(self):
        car = Car4Wheel(v=10.0)
        car.update_physics(throttle=0.0, brake=0.8, target_steer=0.0, dt=0.1)
        assert car.v < 10.0
        assert car.brake_lights is True

    def test_acceleration_and_drag(self):
        car = Car4Wheel(v=0.0)
        car.update_physics(throttle=1.0, brake=0.0, target_steer=0.0, dt=0.1)
        assert car.v > 0.0
        assert car.brake_lights is False


# ── 2. Environment Tests ──────────────────────────────────────────────────────

class TestAutonomousDrivingEnv:

    def test_city_a_track_generation(self):
        track_a = build_city_a_track()
        assert len(track_a.waypoints) > 20
        assert len(track_a.dark_zones) == 0  # City A has no dark tunnels
        assert len(track_a.signaling_zones) > 0

    def test_city_b_track_and_dark_tunnel(self):
        track_b = build_city_b_track()
        assert len(track_b.dark_zones) > 0
        tunnel_start, tunnel_end = track_b.dark_zones[0]

        # Illumination inside dark tunnel should be low (~0.08)
        mid_tunnel = (tunnel_start + tunnel_end) / 2.0
        assert track_b.get_ambient_illumination(mid_tunnel) < 0.2
        # Outside tunnel should be full daylight (1.0)
        assert track_b.get_ambient_illumination(0.0) == 1.0

    def test_observation_shape_and_stepping(self):
        env = AutonomousDrivingEnv(build_city_a_track())
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (25,)  # 11 LiDAR + telemetry + signals + lights + open_world

        # Step forward with acceleration and right indicator
        next_obs, reward, term, trunc, info = env.step(
            motion_act=6, indicator_act=2, headlight_act=0
        )
        assert next_obs.shape == (25,)
        assert isinstance(reward, float)
        assert "completion_pct" in info
        assert "speed_mps" in info
        assert info["active_lights"]["right"] is True

    def test_dark_zone_reduced_vision_without_headlights(self):
        env = AutonomousDrivingEnv(build_city_b_track())
        env.reset()
        # Place car in dark tunnel zone
        env.car.x = 95.0
        env.car.y = 12.0

        # Without headlights
        env.car.headlights = False
        obs_no_hl = env._get_observation()

        # With headlights
        env.car.headlights = True
        obs_hl = env._get_observation()

        # Headlight status should reflect in observation lights vector (index 22)
        assert obs_no_hl[22] == 0.0
        assert obs_hl[22] == 1.0

    def test_load_waymo_streets_and_build_track(self):
        from src.simulation.environment import load_waymo_streets, build_waymo_track
        streets = load_waymo_streets()
        assert len(streets) >= 100

        first_street = streets[0]
        assert "name" in first_street
        assert "waypoints" in first_street
        assert len(first_street["waypoints"]) > 5

        # Test track building with random obstacles
        track = build_waymo_track(first_street, obstacle_mode="random")
        assert track.name == first_street["name"]
        assert len(track.obstacles) > 0

        # Step environment on Waymo track
        env = AutonomousDrivingEnv(track)
        obs = env.reset()
        assert obs.shape == (25,)


# ── 3. RL Agent & Policy Tests ────────────────────────────────────────────────

class TestDrivingPolicy:

    def test_forward_pass_and_head_dimensions(self):
        device = resolve_rl_device("cpu")
        policy = DrivingPolicy(obs_dim=25, device=device)

        batch_obs = torch.randn(4, 25)
        m_logits, i_logits, h_logits, values, uncertainty = policy(batch_obs)

        assert m_logits.shape == (4, 7)
        assert i_logits.shape == (4, 3)
        assert h_logits.shape == (4, 2)
        assert values.shape == (4,)
        assert uncertainty.shape == (4,)

    def test_act_cautious_reflex(self):
        policy = DrivingPolicy(obs_dim=25, device=torch.device("cpu"))

        # Create dummy observation in pitch dark (ambient_light = 0.05 at index 11)
        obs = np.zeros(25, dtype=np.float32)
        obs[11] = 0.05  # dark tunnel

        # When uncertainty threshold is exceeded, cautious reflex should turn on headlights
        policy.uncertainty_threshold = -1.0  # Force uncertainty trigger
        actions, log_prob, val, unc = policy.act(obs, enable_cautious_reflex=True)

        assert len(actions) == 3
        # Headlight action should be turned ON (1) by the darkness safety reflex
        assert actions[2] == 1


# ── 4. Policy EWC Tests ───────────────────────────────────────────────────────

class TestPolicyEWC:

    def test_fisher_and_penalty_computation(self):
        env = AutonomousDrivingEnv(build_city_a_track(), max_steps=20)
        policy = DrivingPolicy(obs_dim=25, device=torch.device("cpu"))
        ewc = PolicyEWC(policy, ewc_lambda=50.0)

        ewc.snapshot_task_a(env, num_episodes=2)
        assert len(ewc.fisher) > 0
        assert len(ewc.params_task_a) > 0

        # At snapshot point, penalty should be zero
        penalty_initial = ewc.penalty().item()
        assert penalty_initial < 1e-5

        # Perturb policy parameters
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        penalty_perturbed = ewc.penalty().item()
        assert penalty_perturbed > penalty_initial


# ── 5. 3D WebGL Renderer & Trajectory Tests ───────────────────────────────────

class TestSimulationRenderer3D:

    def test_get_episode_trajectory_format(self):
        env = AutonomousDrivingEnv(build_city_a_track(), max_steps=10)
        env.reset()
        for _ in range(5):
            env.step(2, indicator_act=1, headlight_act=1)

        traj_data = env.get_episode_trajectory()
        assert "track_name" in traj_data
        assert "track_length" in traj_data
        assert "lane_width" in traj_data
        assert "waypoints" in traj_data
        assert "obstacles" in traj_data
        assert "trajectory" in traj_data
        assert len(traj_data["trajectory"]) == 6  # initial + 5 steps

        step0 = traj_data["trajectory"][0]
        assert "x" in step0 and "y" in step0 and "yaw" in step0 and "v" in step0
        assert "steer" in step0 and "progress" in step0 and "lights" in step0

    def test_build_3d_simulation_html(self):
        from src.simulation.renderer3d import SimulationRenderer3D

        env = AutonomousDrivingEnv(build_city_a_track(), max_steps=5)
        env.reset()
        env.step(2)
        traj_data = env.get_episode_trajectory()

        html = SimulationRenderer3D.build_3d_simulation_html(
            episodes_data=[traj_data],
            hud_image_b64="data:image/jpeg;base64,mock",
            auto_play=True,
        )

        assert "<!DOCTYPE html>" in html
        assert "three.min.js" in html
        assert "OrbitControls.js" in html
        assert "buildAutonomousCar" in html
        assert "generate3DTrack" in html
        assert "setCameraMode" in html
        assert "data:image/jpeg;base64,mock" in html
