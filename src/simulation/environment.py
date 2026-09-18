"""
src/simulation/environment.py
──────────────────────────────
Autonomous Driving Environment with:
  - Full 4-Wheel Car Dynamics (Ackerman steering, wheel geometry, braking, inertia)
  - Vehicle Signaling & Lighting (Left/Right indicators, brake lights, headlights)
  - City A (Known Domain): Structured grid, daylight, traffic lights, signaling zones
  - City B (Unknown Domain): Curved roads, dark tunnels (reduced visibility without headlights),
    novel open-world obstacles (construction zones, stopped vehicles)
  - Detailed failure mode classification and telemetry
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# ── 4-Wheel Car Model ─────────────────────────────────────────────────────────

@dataclass
class Car4Wheel:
    """
    4-Wheel Car Dynamics with Ackerman steering geometry.
    
    Dimensions:
      length: 4.5m, width: 2.0m, wheelbase: 2.8m, track_width: 1.6m
    """
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0          # Radians (0 = facing +X)
    v: float = 0.0            # Longitudinal velocity (m/s)
    steer: float = 0.0        # Front wheel steer angle (radians)

    # Vehicle specifications
    length: float = 4.5
    width: float = 2.0
    wheelbase: float = 2.8    # Distance between front and rear axle
    track_width: float = 1.6  # Distance between left and right wheels
    max_steer: float = math.radians(35.0)
    max_speed: float = 15.0   # ~54 km/h
    accel_max: float = 3.5    # m/s^2
    brake_max: float = 7.0    # m/s^2
    drag_coeff: float = 0.05

    # Signaling & Lighting state
    left_indicator: bool = False
    right_indicator: bool = False
    headlights: bool = False
    brake_lights: bool = False

    def get_corners(self) -> np.ndarray:
        """
        Return the 4 corner coordinates of the car body in world space.
        Order: Front-Left, Front-Right, Rear-Right, Rear-Left.
        Shape: (4, 2)
        """
        hl = self.length / 2.0
        hw = self.width / 2.0
        local_corners = np.array([
            [hl, hw],
            [hl, -hw],
            [-hl, -hw],
            [-hl, hw],
        ])
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        rot_matrix = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        world_corners = (local_corners @ rot_matrix.T) + np.array([self.x, self.y])
        return world_corners

    def get_wheel_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Return positions and steering orientations for all 4 wheels.
        Returns: {wheel_name: (x, y, wheel_yaw)}
        """
        half_wb = self.wheelbase / 2.0
        half_tw = self.track_width / 2.0
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)

        wheels = {
            "FL": (half_wb, half_tw, self.yaw + self.steer),
            "FR": (half_wb, -half_tw, self.yaw + self.steer),
            "RL": (-half_wb, half_tw, self.yaw),
            "RR": (-half_wb, -half_tw, self.yaw),
        }
        world_wheels = {}
        for k, (lx, ly, wyaw) in wheels.items():
            wx = self.x + lx * cos_y - ly * sin_y
            wy = self.y + lx * sin_y + ly * cos_y
            world_wheels[k] = (wx, wy, wyaw)
        return world_wheels

    def update_physics(
        self,
        throttle: float,
        brake: float,
        target_steer: float,
        dt: float = 0.1,
    ):
        """
        Update vehicle state using 4-wheel kinematic model with Ackerman geometry.
        """
        # Steering slew rate
        steer_rate = math.radians(60.0)  # max 60 deg/sec
        steer_diff = np.clip(target_steer - self.steer, -steer_rate * dt, steer_rate * dt)
        self.steer = float(np.clip(self.steer + steer_diff, -self.max_steer, self.max_steer))

        # Longitudinal acceleration
        accel = throttle * self.accel_max - brake * self.brake_max - self.drag_coeff * (self.v ** 2)
        self.v = float(np.clip(self.v + accel * dt, 0.0, self.max_speed))

        # Brake lights active when decelerating with brake command
        self.brake_lights = (brake > 0.15)

        # Ackerman yaw rate and kinematic update
        yaw_rate = (self.v / self.wheelbase) * math.tan(self.steer)
        self.yaw = float((self.yaw + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi)

        # Position update
        self.x += float(self.v * math.cos(self.yaw) * dt)
        self.y += float(self.v * math.sin(self.yaw) * dt)


# ── Obstacles & Traffic Entities ─────────────────────────────────────────────

@dataclass
class Obstacle:
    """Obstacle on or near the roadway."""
    x: float
    y: float
    radius: float = 1.2
    obstacle_type: str = "construction_barrel"  # "construction_barrel", "stopped_car", "barrier"
    is_open_world_novel: bool = False           # Out-of-distribution in City B


@dataclass
class TrafficLight:
    """Traffic light at intersection."""
    x: float
    y: float
    state: str = "GREEN"  # "GREEN", "YELLOW", "RED"
    stop_distance_threshold: float = 12.0


# ── City Track Definition ─────────────────────────────────────────────────────

class CityTrack:
    """
    Representation of a city driving circuit.
    """
    def __init__(
        self,
        name: str,
        waypoints: np.ndarray,
        lane_width: float = 8.0,
        speed_limit: float = 10.0,
        dark_zones: Optional[List[Tuple[float, float]]] = None,
        signaling_zones: Optional[List[Dict]] = None,
        obstacles: Optional[List[Obstacle]] = None,
        traffic_lights: Optional[List[TrafficLight]] = None,
        is_city_b: bool = False,
    ):
        self.name = name
        self.waypoints = waypoints  # (N, 2)
        self.lane_width = lane_width
        self.speed_limit = speed_limit
        self.dark_zones = dark_zones or []           # List of (start_dist, end_dist) along path
        self.signaling_zones = signaling_zones or [] # [{'start': s, 'end': e, 'type': 'LEFT'/'RIGHT'}]
        self.obstacles = obstacles or []
        self.traffic_lights = traffic_lights or []
        self.is_city_b = is_city_b

        # Precompute cumulative path distances
        diffs = np.diff(waypoints, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        self.cum_dist = np.insert(np.cumsum(seg_lens), 0, 0.0)
        self.total_length = float(self.cum_dist[-1])

    def get_progress_and_error(self, x: float, y: float) -> Tuple[float, float, float, int]:
        """
        Find closest point on centerline.
        Returns:
          progress: distance along track (m)
          lane_deviation: signed lateral distance from centerline (m)
          heading_error: angle between vehicle and track tangent (rad)
          closest_idx: index of nearest segment
        """
        pos = np.array([x, y])
        dists = np.linalg.norm(self.waypoints - pos, axis=1)
        idx = int(np.argmin(dists))

        # Select forward segment
        p0_idx = max(0, min(idx, len(self.waypoints) - 2))
        p0 = self.waypoints[p0_idx]
        p1 = self.waypoints[p0_idx + 1]
        v_seg = p1 - p0
        seg_len = np.linalg.norm(v_seg)
        if seg_len < 1e-6:
            tangent_angle = 0.0
            t_proj = 0.0
        else:
            v_dir = v_seg / seg_len
            tangent_angle = math.atan2(v_dir[1], v_dir[0])
            t_proj = np.clip(np.dot(pos - p0, v_dir), 0.0, seg_len)

        progress = self.cum_dist[p0_idx] + t_proj

        # Lateral deviation (cross product to determine sign: positive = left of centerline)
        if seg_len > 1e-6:
            cross = v_dir[0] * (y - p0[1]) - v_dir[1] * (x - p0[0])
            lane_deviation = float(cross)
        else:
            lane_deviation = float(dists[idx])

        return progress, lane_deviation, tangent_angle, p0_idx

    def get_ambient_illumination(self, progress: float) -> float:
        """
        Returns ambient light level: 1.0 (full daylight) down to 0.05 (pitch dark tunnel).
        """
        for start, end in self.dark_zones:
            if start <= progress <= end:
                return 0.08  # Dark tunnel / night zone
        return 1.0

    def get_required_signal(self, progress: float) -> str:
        """
        Returns required blinker: 'NONE', 'LEFT', 'RIGHT'.
        """
        for zone in self.signaling_zones:
            if zone["start"] <= progress <= zone["end"]:
                return zone["type"]
        return "NONE"


def build_city_a_track() -> CityTrack:
    """
    City A ('MetroGrid'): Known training domain.
    Structured rectangular street layout with clear 90° turns, daylight,
    standard traffic lights, and designated turn signaling zones.
    """
    pts = []
    # Segment 1: Straightaway (0 to 60m)
    for x in np.linspace(0, 60, 30):
        pts.append([x, 0.0])
    # Segment 2: 90° Right Turn into North
    for theta in np.linspace(-math.pi / 2, 0, 15):
        pts.append([60.0 + 15.0 * math.cos(theta), 15.0 + 15.0 * math.sin(theta)])
    # Segment 3: Northbound Straightaway (15m to 75m)
    for y in np.linspace(15, 75, 25):
        pts.append([75.0, y])
    # Segment 4: 90° Left Turn
    for theta in np.linspace(0, math.pi / 2, 15):
        pts.append([75.0 - 15.0 + 15.0 * math.cos(theta), 75.0 + 15.0 * math.sin(theta)])
    # Segment 5: Westbound Straightaway (60m to 0m)
    for x in np.linspace(60, 0, 25):
        pts.append([x, 90.0])

    waypoints = np.array(pts, dtype=np.float32)

    signaling_zones = [
        {"start": 45.0, "end": 65.0, "type": "RIGHT"},  # Signaling right turn
        {"start": 125.0, "end": 145.0, "type": "LEFT"},  # Signaling left turn
    ]

    traffic_lights = [
        TrafficLight(x=75.0, y=40.0, state="GREEN"),
    ]

    obstacles = [
        Obstacle(x=35.0, y=2.5, radius=0.8, obstacle_type="trash_can", is_open_world_novel=False),
        Obstacle(x=75.0, y=65.0, radius=1.0, obstacle_type="pothole_marker", is_open_world_novel=False),
    ]

    return CityTrack(
        name="City A (MetroGrid)",
        waypoints=waypoints,
        lane_width=8.0,
        speed_limit=10.0,
        dark_zones=[],  # Always daylight in City A
        signaling_zones=signaling_zones,
        obstacles=obstacles,
        traffic_lights=traffic_lights,
        is_city_b=False,
    )


def build_city_b_track() -> CityTrack:
    """
    City B ('NovelMetropolis - DarkHills'): Unknown transfer domain.
    Challenging winding S-curves, chicanes, unexpected dark tunnels
    where headlights are required, and novel open-world hazards.
    """
    pts = []
    # Segment 1: Entry straightaway (0 to 30m)
    for x in np.linspace(0, 30, 15):
        pts.append([x, 0.0])
    # Segment 2: Sharp S-Bend Curve (30m to 90m)
    for x in np.linspace(30, 90, 35):
        y = 12.0 * math.sin((x - 30.0) / 60.0 * 2 * math.pi)
        pts.append([x, y])
    # Segment 3: Deep curve into Dark Tunnel Zone (90m to 140m)
    for x in np.linspace(90, 140, 25):
        y = 12.0 * math.cos((x - 90.0) / 50.0 * math.pi)
        pts.append([x, y])
    # Segment 4: Chicane exit straightaway (140m to 180m)
    for x in np.linspace(140, 180, 20):
        pts.append([x, -12.0])

    waypoints = np.array(pts, dtype=np.float32)

    # Dark tunnel zone: 65m to 135m along path
    dark_zones = [
        (65.0, 135.0),
    ]

    signaling_zones = [
        {"start": 25.0, "end": 45.0, "type": "LEFT"},   # Sharp turn signal
        {"start": 130.0, "end": 145.0, "type": "RIGHT"}, # Exit tunnel signal
    ]

    traffic_lights = [
        TrafficLight(x=150.0, y=-12.0, state="GREEN"),
    ]

    # Novel Open-World hazards in City B!
    obstacles = [
        Obstacle(x=55.0, y=9.5, radius=1.3, obstacle_type="construction_barricade", is_open_world_novel=True),
        Obstacle(x=105.0, y=3.0, radius=1.5, obstacle_type="stopped_disabled_car", is_open_world_novel=True),
        Obstacle(x=120.0, y=-7.0, radius=1.0, obstacle_type="fallen_debris", is_open_world_novel=True),
    ]

    return CityTrack(
        name="City B (NovelMetropolis)",
        waypoints=waypoints,
        lane_width=8.0,
        speed_limit=9.0,
        dark_zones=dark_zones,
        signaling_zones=signaling_zones,
        obstacles=obstacles,
        traffic_lights=traffic_lights,
        is_city_b=True,
    )


# ── Waymo Open Dataset Dynamic Track Builders ─────────────────────────────────

def load_waymo_streets(json_path: str = "data/waymo_paths/waymo_streets.json") -> List[Dict]:
    """Load extracted Waymo streets library (100+ real streets), prioritizing longer realistic urban paths."""
    p = Path(json_path)
    if not p.exists():
        from src.data.waymo_path_extractor import generate_waymo_streets
        streets = generate_waymo_streets(out_json=str(p))
    else:
        with open(p) as f:
            streets = json.load(f)

    # Sort streets so dramatic long paths (>80m: 90° turns, s-bends, loops) are featured first
    def street_sort_key(s):
        length = float(s.get("length_m", 0.0))
        stype = s.get("street_type", "")
        # Prioritize 90° turns, S-bends, loops, and chicanes over straight stretches
        type_priority = 2 if ("Turn" in stype or "S-Bend" in stype or "Chicane" in stype or "Loop" in stype) else 1
        return (type_priority, length)

    return sorted(streets, key=street_sort_key, reverse=True)


def build_waymo_track(street_dict: Dict, obstacle_mode: str = "none") -> CityTrack:
    """
    Build a CityTrack directly from a Waymo street path dictionary.

    obstacle_mode:
      - 'none': clean street with no obstacles
      - 'random': 2-3 randomly placed obstacles (cars, pedestrians, construction)
      - 'heavy': 5-8 obstacles scattered along the roadway
    """
    waypoints = np.array(street_dict["waypoints"], dtype=np.float32)
    track_len = street_dict.get("length_m", 100.0)

    # Signaling zones for turns
    signaling_zones = []
    street_type = street_dict.get("street_type", "")
    if "Turn" in street_type or "S-Bend" in street_type or "Chicane" in street_type:
        signaling_zones.append({
            "start": max(5.0, track_len * 0.25),
            "end": min(track_len - 5.0, track_len * 0.65),
            "type": "RIGHT" if ("Right" in street_type or "90°" in street_type) else "LEFT",
        })

    # Obstacles injection
    obstacles = []
    if obstacle_mode != "none":
        diffs = np.diff(waypoints, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        dirs = diffs / np.maximum(seg_lens[:, None], 1e-6)
        dirs = np.vstack([dirs, dirs[-1]])
        normals = np.column_stack([-dirs[:, 1], dirs[:, 0]])

        num_obs = 3 if obstacle_mode == "random" else 6
        obs_types = ["stopped_vehicle", "pedestrian", "construction_barrel", "debris"]

        valid_indices = list(range(max(2, len(waypoints) // 6), len(waypoints) - 3))
        if len(valid_indices) > 0:
            chosen = np.random.choice(valid_indices, size=min(num_obs, len(valid_indices)), replace=False)
            for idx in sorted(chosen):
                pt = waypoints[idx]
                norm = normals[idx]
                side_offset = 2.0 if np.random.rand() > 0.5 else -2.0
                obs_pos = pt + norm * side_offset
                obs_t = np.random.choice(obs_types)
                radius = 1.4 if obs_t == "stopped_vehicle" else (0.8 if obs_t == "pedestrian" else 1.0)
                obstacles.append(Obstacle(
                    x=float(obs_pos[0]),
                    y=float(obs_pos[1]),
                    radius=radius,
                    obstacle_type=obs_t,
                    is_open_world_novel=(obs_t in ["construction_barrel", "debris"]),
                ))

    return CityTrack(
        name=street_dict["name"],
        waypoints=waypoints,
        lane_width=8.0,
        speed_limit=10.0,
        dark_zones=[],
        signaling_zones=signaling_zones,
        obstacles=obstacles,
        traffic_lights=[],
        is_city_b=True,
    )


# ── Environment ───────────────────────────────────────────────────────────────

class AutonomousDrivingEnv:
    """
    Autonomous Vehicle Navigation Environment.
    
    Observation Vector (dim = 25):
      - 11 LiDAR normalized distance beams: [-75° .. +75°]
      - Ambient illumination level [0.0 - 1.0]
      - Normalized speed v / v_max
      - Lane deviation d_lane / (half_lane_width)
      - Heading error Delta theta / pi
      - Speed limit ratio v / v_limit
      - Required signal encoding [None, Left, Right] (3 values)
      - Current vehicle light state [Left_Ind, Right_Ind, Brake_Light, Headlights] (4 values)
      - 2 Open-world domain context features (triggering uncertainty)

    Action Space:
      Composite actions:
        motion_act: [0: Hard Left, 1: Gentle Left, 2: Cruise Straight, 3: Gentle Right, 4: Hard Right, 5: Brake/Slow, 6: Accelerate]
        indicator_act: [0: Off/None, 1: Left Blinker, 2: Right Blinker]
        headlight_act: [0: Off, 1: On]
    """
    def __init__(self, track: CityTrack, max_steps: int = 400, dt: float = 0.1):
        self.track = track
        self.max_steps = max_steps
        self.dt = dt
        self.car = Car4Wheel()

        # LiDAR settings
        self.num_lidar_rays = 11
        self.lidar_angles = np.linspace(-math.radians(75), math.radians(75), self.num_lidar_rays)
        self.lidar_max_range = 30.0

        self.step_count = 0
        self.total_distance = 0.0
        self.prev_progress = 0.0

        # Failure telemetry
        self.failure_reason: Optional[str] = None
        self.route_completed: bool = False
        self.history_x: List[float] = []
        self.history_y: List[float] = []
        self.history_speed: List[float] = []
        self.history_lights: List[Dict[str, bool]] = []
        self.history_uncertainty: List[float] = []

    def reset(self, initial_speed: float = 2.0) -> np.ndarray:
        """Reset vehicle to track starting line."""
        p0 = self.track.waypoints[0]
        p1 = self.track.waypoints[1]
        init_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

        self.car = Car4Wheel(
            x=float(p0[0]),
            y=float(p0[1]),
            yaw=float(init_yaw),
            v=float(initial_speed),
            steer=0.0,
            left_indicator=False,
            right_indicator=False,
            headlights=False,
            brake_lights=False,
        )

        self.step_count = 0
        self.total_distance = 0.0
        self.prev_progress = 0.0
        self.failure_reason = None
        self.route_completed = False

        self.history_x = [self.car.x]
        self.history_y = [self.car.y]
        self.history_yaw = [self.car.yaw]
        self.history_speed = [self.car.v]
        self.history_steer = [self.car.steer]
        self.history_lane_dev = [0.0]
        self.history_progress = [0.0]
        self.history_rewards = [0.0]
        self.history_lights = [{
            "left": False, "right": False, "headlights": False, "brake": False
        }]
        self.history_uncertainty = [0.0]

        return self._get_observation()

    def _get_observation(self, uncertainty_score: float = 0.0) -> np.ndarray:
        """Construct normalized observation vector."""
        progress, lane_dev, track_tangent, _ = self.track.get_progress_and_error(self.car.x, self.car.y)
        ambient_light = self.track.get_ambient_illumination(progress)

        # ── 1. LiDAR Raycasting ───────────────────────────────────────────────
        # Effective sensor range decreases in darkness if headlights are OFF!
        if ambient_light < 0.2 and not self.car.headlights:
            effective_max_range = 8.0  # severely reduced vision in dark tunnel
        else:
            effective_max_range = self.lidar_max_range

        lidar_readings = self._cast_lidar(effective_max_range)

        # ── 2. Telemetry ──────────────────────────────────────────────────────
        norm_speed = self.car.v / self.car.max_speed
        half_lane = self.track.lane_width / 2.0
        norm_lane_dev = np.clip(lane_dev / half_lane, -2.0, 2.0)

        # Heading error relative to track tangent (-pi to +pi)
        heading_err = (self.car.yaw - track_tangent + math.pi) % (2 * math.pi) - math.pi
        norm_heading_err = heading_err / math.pi
        norm_speed_limit = self.car.v / self.track.speed_limit

        # ── 3. Rule / Signaling Requirements ──────────────────────────────────
        req_signal = self.track.get_required_signal(progress)
        req_sig_vec = [
            1.0 if req_signal == "NONE" else 0.0,
            1.0 if req_signal == "LEFT" else 0.0,
            1.0 if req_signal == "RIGHT" else 0.0,
        ]

        # ── 4. Car Active Lighting State ──────────────────────────────────────
        lights_vec = [
            1.0 if self.car.left_indicator else 0.0,
            1.0 if self.car.right_indicator else 0.0,
            1.0 if self.car.brake_lights else 0.0,
            1.0 if self.car.headlights else 0.0,
        ]

        # ── 5. Open-World Semantic Context Features ───────────────────────────
        # City A has feature baseline [0.1, 0.1]. City B exhibits novel domain shift [0.85, 0.92]
        if self.track.is_city_b:
            open_world_features = [0.85, 0.92]
        else:
            open_world_features = [0.10, 0.10]

        obs = np.array([
            *lidar_readings,       # 11 floats
            ambient_light,         # 1 float
            norm_speed,            # 1 float
            norm_lane_dev,         # 1 float
            norm_heading_err,      # 1 float
            norm_speed_limit,      # 1 float
            *req_sig_vec,          # 3 floats
            *lights_vec,           # 4 floats
            *open_world_features,  # 2 floats
        ], dtype=np.float32)

        return obs

    def _cast_lidar(self, max_range: float) -> List[float]:
        """Compute raycast intersections against road boundaries and obstacles."""
        readings = []
        car_pos = np.array([self.car.x, self.car.y])
        half_lane = self.track.lane_width / 2.0

        for angle_offset in self.lidar_angles:
            ray_yaw = self.car.yaw + angle_offset
            ray_dir = np.array([math.cos(ray_yaw), math.sin(ray_yaw)])

            min_dist = max_range

            # 1. Road boundaries check along ray
            for test_d in np.linspace(1.0, max_range, 15):
                sample_pt = car_pos + ray_dir * test_d
                _, dev, _, _ = self.track.get_progress_and_error(sample_pt[0], sample_pt[1])
                if abs(dev) >= half_lane:
                    min_dist = min(min_dist, float(test_d))
                    break

            # 2. Obstacles check
            for obs in self.track.obstacles:
                obs_pos = np.array([obs.x, obs.y])
                to_obs = obs_pos - car_pos
                proj = np.dot(to_obs, ray_dir)
                if 0 < proj < min_dist:
                    perp_dist = np.linalg.norm(to_obs - proj * ray_dir)
                    if perp_dist <= obs.radius:
                        hit_dist = float(max(0.1, proj - math.sqrt(max(0.0, obs.radius**2 - perp_dist**2))))
                        min_dist = min(min_dist, hit_dist)

            readings.append(float(min_dist / max_range))

        return readings

    def step(
        self,
        motion_act: int,
        indicator_act: int = 0,
        headlight_act: int = 0,
        uncertainty_score: float = 0.0,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one simulation step.

        Actions:
          motion_act:
            0: Hard Left
            1: Gentle Left
            2: Cruise Straight
            3: Gentle Right
            4: Hard Right
            5: Brake / Slow Down
            6: Accelerate Straight
          indicator_act:
            0: Off
            1: Left Blinker
            2: Right Blinker
          headlight_act:
            0: Off
            1: On
        """
        self.step_count += 1

        # ── Map Actions to Physical Commands ──────────────────────────────────
        throttle = 0.0
        brake = 0.0
        steer_target = 0.0

        if motion_act == 0:     # Hard Left
            steer_target = -self.car.max_steer * 0.85
            throttle = 0.35
        elif motion_act == 1:   # Gentle Left
            steer_target = -self.car.max_steer * 0.40
            throttle = 0.50
        elif motion_act == 2:   # Cruise Straight
            steer_target = 0.0
            throttle = 0.40
        elif motion_act == 3:   # Gentle Right
            steer_target = self.car.max_steer * 0.40
            throttle = 0.50
        elif motion_act == 4:   # Hard Right
            steer_target = self.car.max_steer * 0.85
            throttle = 0.35
        elif motion_act == 5:   # Brake
            brake = 0.80
            throttle = 0.0
        elif motion_act == 6:   # Accelerate
            throttle = 0.80
            steer_target = 0.0

        # Update Signaling state
        self.car.left_indicator = (indicator_act == 1)
        self.car.right_indicator = (indicator_act == 2)
        self.car.headlights = (headlight_act == 1)

        # Update 4-wheel car physics
        self.car.update_physics(throttle, brake, steer_target, dt=self.dt)

        # ── Progress and Road Tracking ────────────────────────────────────────
        progress, lane_dev, track_tangent, _ = self.track.get_progress_and_error(self.car.x, self.car.y)
        progress_gain = max(0.0, progress - self.prev_progress)
        self.total_distance += progress_gain
        self.prev_progress = progress

        half_lane = self.track.lane_width / 2.0
        ambient_light = self.track.get_ambient_illumination(progress)
        req_signal = self.track.get_required_signal(progress)

        # ── Reward Formulation ────────────────────────────────────────────────
        heading_err = (self.car.yaw - track_tangent + math.pi) % (2 * math.pi) - math.pi
        reward = 0.0
        reward += progress_gain * 3.0                                  # Forward progress
        reward += 0.5 * math.cos(heading_err)                          # Heading alignment with track
        reward += 0.5 * max(0.0, 1.0 - abs(lane_dev) / half_lane)      # Lane centering
        reward -= 1.0 * (abs(lane_dev) / half_lane) ** 2               # Quadratic lane deviation penalty
        reward += 0.2 * (self.car.v / self.track.speed_limit)           # Speed efficiency

        # Rule 1: Turn indicator compliance
        if req_signal == "LEFT":
            if self.car.left_indicator:
                reward += 0.5
            else:
                reward -= 1.0
        elif req_signal == "RIGHT":
            if self.car.right_indicator:
                reward += 0.5
            else:
                reward -= 1.0

        # Rule 2: Headlights compliance in dark tunnel
        if ambient_light < 0.2:
            if self.car.headlights:
                reward += 0.8
            else:
                reward -= 2.0  # Violation for driving in dark without headlights

        # ── Collision and Failure Checks ──────────────────────────────────────
        terminated = False
        truncated = (self.step_count >= self.max_steps)

        # 1. Lane Boundary Check (Exact 4 corners of car)
        corners = self.car.get_corners()
        for cx, cy in corners:
            _, c_dev, _, _ = self.track.get_progress_and_error(cx, cy)
            if abs(c_dev) >= half_lane + 0.3:
                terminated = True
                self.failure_reason = "CRASH: Lane Departure / Ran Off Road"
                reward -= 50.0
                break

        # 2. Obstacle Collision Check
        if not terminated:
            car_pos = np.array([self.car.x, self.car.y])
            for obs in self.track.obstacles:
                obs_dist = np.linalg.norm(np.array([obs.x, obs.y]) - car_pos)
                collision_dist = obs.radius + (self.car.width / 2.0)
                if obs_dist <= collision_dist:
                    terminated = True
                    if ambient_light < 0.2 and not self.car.headlights:
                        self.failure_reason = (
                            f"CRASH: Dark Tunnel Blind Collision with {obs.obstacle_type} "
                            f"(Headlights were OFF)"
                        )
                    else:
                        self.failure_reason = (
                            f"CRASH: Collided with {obs.obstacle_type} "
                            f"(Novel={obs.is_open_world_novel})"
                        )
                    reward -= 60.0
                    break

        # 3. Route Completion Check
        if progress >= self.track.total_length - 5.0 and not terminated:
            terminated = True
            self.route_completed = True
            reward += 100.0

        # Record telemetry history
        self.history_x.append(float(self.car.x))
        self.history_y.append(float(self.car.y))
        self.history_yaw.append(float(self.car.yaw))
        self.history_speed.append(float(self.car.v))
        self.history_steer.append(float(self.car.steer))
        self.history_lane_dev.append(float(lane_dev))
        self.history_progress.append(float(progress))
        self.history_rewards.append(float(reward))
        self.history_lights.append({
            "left": bool(self.car.left_indicator),
            "right": bool(self.car.right_indicator),
            "headlights": bool(self.car.headlights),
            "brake": bool(self.car.brake_lights),
        })
        self.history_uncertainty.append(float(uncertainty_score))

        completion_pct = min(100.0, (progress / self.track.total_length) * 100.0)
        info = {
            "step": self.step_count,
            "progress_m": progress,
            "total_track_m": self.track.total_length,
            "completion_pct": completion_pct,
            "speed_mps": self.car.v,
            "lane_dev_m": lane_dev,
            "failure_reason": self.failure_reason,
            "route_completed": self.route_completed,
            "ambient_light": ambient_light,
            "active_lights": {
                "left": self.car.left_indicator,
                "right": self.car.right_indicator,
                "headlights": self.car.headlights,
                "brake": self.car.brake_lights,
            },
        }

        obs = self._get_observation(uncertainty_score=uncertainty_score)
        return obs, reward, terminated, truncated, info

    def get_episode_trajectory(self) -> Dict:
        """
        Package all kinematic, sensory, and environment data for 3D WebGL playback.
        """
        obs_list = []
        for obs in self.track.obstacles:
            obs_list.append({
                "x": round(float(obs.x), 2),
                "y": round(float(obs.y), 2),
                "radius": round(float(obs.radius), 2),
                "type": obs.obstacle_type,
                "is_novel": obs.is_open_world_novel,
            })

        crash_pt = None
        if self.failure_reason is not None and len(self.history_x) > 0:
            crash_pt = [round(self.history_x[-1], 2), round(self.history_y[-1], 2)]

        return {
            "track_name": self.track.name,
            "track_length": round(float(self.track.total_length), 2),
            "lane_width": round(float(self.track.lane_width), 2),
            "waypoints": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in self.track.waypoints],
            "obstacles": obs_list,
            "steps": len(self.history_x),
            "trajectory": [
                {
                    "x": round(self.history_x[i], 2),
                    "y": round(self.history_y[i], 2),
                    "yaw": round(self.history_yaw[i], 4),
                    "v": round(self.history_speed[i], 2),
                    "steer": round(self.history_steer[i], 4),
                    "progress": round(self.history_progress[i], 2),
                    "lane_dev": round(self.history_lane_dev[i], 2),
                    "reward": round(self.history_rewards[i], 2),
                    "lights": self.history_lights[i],
                }
                for i in range(len(self.history_x))
            ],
            "route_completed": bool(self.route_completed),
            "failure_reason": self.failure_reason,
            "crash_point": crash_pt,
            "final_progress": round(float(self.prev_progress), 2),
            "completion_pct": round(min(100.0, (self.prev_progress / max(1e-5, self.track.total_length)) * 100.0), 1),
        }
