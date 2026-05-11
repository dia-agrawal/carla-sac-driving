import carla
from carla import Transform, Location, Rotation
import random
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv

sys.path.append("./CARLA_0.9.16")

from PythonAPI.carla.agents.navigation.global_route_planner import GlobalRoutePlanner
from kalman_filter import KalmanFilter2D


class CarlaDrivingEnv:
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.settings = self.world.get_settings()
        self.blueprint_library = self.world.get_blueprint_library()
        self.debug = self.world.debug

        self.settings.fixed_delta_seconds = 0.05
        self.settings.max_substep_delta_time = 0.01
        self.settings.max_substeps = 10
        self.settings.no_rendering_mode = False
        self.settings.synchronous_mode = True
        self.dt = self.settings.fixed_delta_seconds
        self.world.apply_settings(self.settings)

        self.terminated = False
        self.reward = 0.0
        self.max_steps = 3000
        self.current_step = 0

        # ---- training setup ----
        self.randomize_spawn = True
        self.resume_on_reset = False
        self.reset_interval = 10
        self.episode_counter = 0
        self.use_true_pose_for_training = False
        self.use_true_yaw_for_training = False

        self.draw_route_debug = False
        self.draw_spawn_debug = False
        self.route_debug_life = 0.5

        self.log_pose = False

        # ---- reward params ----
        self.k_progress = 0.7
        self.k_align = 0.05
        self.k_speed = 0.01
        self.k_steer_change = 0.01

        self.time_penalty = -0.01
        self.standstill_speed = 0.2
        self.standstill_penalty = -0.02

        self.waypoint_bonus = 1
        self.goal_bonus = 20.0
        self.collision_penalty = -20.0

        # optional extras
        self.k_wrong_way = 0.05
        self.k_route_align = 0.01

        # ---- new correction behavior params ----
        self.k_route_recover = 0.5
        self.k_heading_fix = 0.05
        self.stuck_bad_heading_penalty = -0.05
        self.bad_heading_threshold = 45.0

        # ---- sensors ----
        self.imu_data = None
        self.gps_data = None
        self.collided = False
        self.impulse = None
        self.collision_intensity = 0.0

        # ---- estimated state ----
        self.vx = 0.0
        self.vy = 0.0
        self.calc_x = 0.0
        self.calc_y = 0.0
        self.calc_yaw = 0.0

        # ---- training state ----
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.waypoint_index = 0
        self.waypoint_x = 0.0
        self.waypoint_y = 0.0
        self.prev_distance_to_waypoint = None
        self.prev_route_error = 0.0

        self.steer_cmd = 0.0
        self.throttle_cmd = 0.0
        self.prev_steer_cmd = 0.0

        self.destination = None
        self.origin = None
        self.route = []

        self.pose_log_file = None
        self.pose_logger = None

        self.lat0 = None
        self.lon0 = None
        self.gps_x_offset = 0.0
        self.gps_y_offset = 0.0
        self.last_gps_frame = None

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_heading_error = 1.0

        self.ax_bias = 0.057
        self.ay_bias = 0.0

        self.episode_reward_totals = None

        # Ego-centric observation
        # [speed, local_wp_x, local_wp_y, heading_err, wp_dist,
        #  vx, vy, sin_yaw, cos_yaw, prev_steer, throttle, standstill_flag]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.actor = None
        self.imu = None
        self.gnss = None
        self.collision_sensor = None

    def _destroy_actor(self, actor):
        if actor is not None:
            try:
                actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass

    def _wrap_angle_deg(self, angle):
        return (angle + 180.0) % 360.0 - 180.0

    def _add_reward(self, breakdown, key, value):
        self.reward += value
        breakdown[key] += value

    def _get_lane_waypoint(self, location):
        return self.map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    def _get_lane_forward_at_location(self, location):
        wp = self._get_lane_waypoint(location)
        if wp is None:
            return None

        forward = wp.transform.get_forward_vector()
        lane_forward = np.array([forward.x, forward.y], dtype=np.float32)

        norm = np.linalg.norm(lane_forward)
        if norm < 1e-6:
            return None

        return lane_forward / norm

    def _get_route_direction(self):
        if self.destination is None:
            return None

        if len(self.route) == 0:
            direction = np.array(
                [self.destination.x - self.x, self.destination.y - self.y],
                dtype=np.float32,
            )
        elif self.waypoint_index < len(self.route) - 1:
            curr_wp = self.route[self.waypoint_index][0].transform.location
            next_wp = self.route[self.waypoint_index + 1][0].transform.location
            direction = np.array(
                [next_wp.x - curr_wp.x, next_wp.y - curr_wp.y],
                dtype=np.float32,
            )
        else:
            curr_wp = self.route[self.waypoint_index][0].transform.location
            direction = np.array(
                [self.destination.x - curr_wp.x, self.destination.y - curr_wp.y],
                dtype=np.float32,
            )

        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None

        return direction / norm

    def _distance_to_route_segment(self):
        if len(self.route) < 2:
            return 0.0

        i = int(np.clip(self.waypoint_index, 0, len(self.route) - 2))
        p1 = self.route[i][0].transform.location
        p2 = self.route[i + 1][0].transform.location

        a = np.array([p1.x, p1.y], dtype=np.float32)
        b = np.array([p2.x, p2.y], dtype=np.float32)
        p = np.array([self.x, self.y], dtype=np.float32)

        ab = b - a
        ab_len2 = float(np.dot(ab, ab))

        if ab_len2 < 1e-6:
            return float(np.linalg.norm(p - a))

        t = float(np.dot(p - a, ab) / ab_len2)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab

        return float(np.linalg.norm(p - proj))

    def _draw_spawn_direction(self, location, yaw_deg, life_time=8.0):
        if not self.draw_spawn_debug:
            return

        start = carla.Location(x=location.x, y=location.y, z=1.0)
        yaw_rad = np.radians(yaw_deg)
        end = carla.Location(
            x=location.x + 3.0 * np.cos(yaw_rad),
            y=location.y + 3.0 * np.sin(yaw_rad),
            z=1.0,
        )

        self.world.debug.draw_point(
            start,
            size=0.15,
            color=carla.Color(0, 255, 255),
            life_time=life_time,
        )
        self.world.debug.draw_line(
            start,
            end,
            thickness=0.08,
            color=carla.Color(0, 255, 255),
            life_time=life_time,
        )

    def _draw_route(self, life_time=8.0):
        if not self.draw_route_debug or len(self.route) < 2:
            return

        for i in range(len(self.route) - 1):
            wp1 = self.route[i][0].transform.location
            wp2 = self.route[i + 1][0].transform.location

            p1 = carla.Location(x=wp1.x, y=wp1.y, z=0.5)
            p2 = carla.Location(x=wp2.x, y=wp2.y, z=0.5)

            self.world.debug.draw_point(
                p1,
                size=0.12,
                color=carla.Color(255, 0, 0),
                life_time=life_time,
            )
            self.world.debug.draw_line(
                p1,
                p2,
                thickness=0.05,
                color=carla.Color(255, 255, 0),
                life_time=life_time,
            )

    def _deduplicate_route(self, route, min_dist=1e-3):
        if route is None or len(route) == 0:
            return []

        filtered = [route[0]]
        for item in route[1:]:
            prev_wp = filtered[-1][0]
            curr_wp = item[0]

            prev_loc = prev_wp.transform.location
            curr_loc = curr_wp.transform.location

            dx = curr_loc.x - prev_loc.x
            dy = curr_loc.y - prev_loc.y

            if np.hypot(dx, dy) > min_dist:
                filtered.append(item)

        return filtered

    def _choose_spawn_transform(self):
        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) == 0:
            raise RuntimeError("No spawn points found on map")

        if not self.randomize_spawn:
            candidate_transform = spawn_points[0]
        else:
            candidate_transform = random.choice(spawn_points)

        nearest_wp = self._get_lane_waypoint(candidate_transform.location)
        if nearest_wp is None:
            raise RuntimeError("Failed to find a road waypoint near spawn")

        lane_transform = nearest_wp.transform

        snapped_transform = Transform(
            Location(
                x=candidate_transform.location.x,
                y=candidate_transform.location.y,
                z=candidate_transform.location.z,
            ),
            Rotation(
                pitch=candidate_transform.rotation.pitch,
                yaw=lane_transform.rotation.yaw,
                roll=candidate_transform.rotation.roll,
            ),
        )
        return snapped_transform

    def _spawn_vehicle(self, max_spawn_tries=20):
        for _ in range(max_spawn_tries):
            self.transform = self._choose_spawn_transform()
            actor = self.world.try_spawn_actor(self.car_design, self.transform)
            if actor is not None:
                self.actor = actor
                return
        raise RuntimeError("Failed to spawn actor")

    def _build_fixed_route(self):
        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) < 2:
            raise RuntimeError("Need at least 2 spawn points on the map")

        if not self.randomize_spawn:
            origin = self.actor.get_transform().location
            dest_idx = min(10, len(spawn_points) - 1)
            destination = spawn_points[dest_idx].location
        else:
            origin = self.actor.get_transform().location
            destination = random.choice(spawn_points).location

        grp = GlobalRoutePlanner(self.map, sampling_resolution=2.0)
        route = grp.trace_route(origin, destination)
        route = self._deduplicate_route(route)

        self.origin = origin
        self.destination = destination
        self.route = route

    def _update_current_waypoint(self):
        if len(self.route) == 0:
            self.waypoint_x = self.destination.x
            self.waypoint_y = self.destination.y
            return

        self.waypoint_index = int(np.clip(self.waypoint_index, 0, len(self.route) - 1))
        wp, _ = self.route[self.waypoint_index]
        self.waypoint_x = wp.transform.location.x
        self.waypoint_y = wp.transform.location.y

    def _build_observation(self):
        speed = np.sqrt(self.vx ** 2 + self.vy ** 2)

        dx = self.waypoint_x - self.x
        dy = self.waypoint_y - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        yaw_rad = np.radians(self.yaw)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)

        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy

        target_heading = np.degrees(np.arctan2(dy, dx))
        heading_error = self._wrap_angle_deg(target_heading - self.yaw)

        obs = np.array([
            speed / 10.0,
            local_x / 20.0,
            local_y / 20.0,
            heading_error / 180.0,
            distance / 20.0,
            self.vx / 10.0,
            self.vy / 10.0,
            np.sin(np.radians(self.yaw)),
            np.cos(np.radians(self.yaw)),
            self.prev_steer_cmd,
            self.throttle_cmd,
            1.0 if speed < self.standstill_speed else 0.0,
        ], dtype=np.float32)

        return obs, speed, dx, dy, heading_error, distance

    def reset(self, debug=False):
        self.episode_reward_totals = {
            "time_penalty": 0.0,
            "progress": 0.0,
            "speed_bonus": 0.0,
            "align_reward": 0.0,
            "route_recover": 0.0,
            "heading_fix": 0.0,
            "stuck_bad_heading": 0.0,
            "standstill_penalty": 0.0,
            "steer_penalty": 0.0,
            "waypoint_bonus": 0.0,
            "goal_bonus": 0.0,
            "collision_penalty": 0.0,
            "wrong_way": 0.0,
            "route_align": 0.0,
        }

        prev_collided = self.collided
        prev_terminated = self.terminated
        prev_distance_to_destination = getattr(self, "distance_to_destination", float("inf"))

        self.terminated = False
        self.collided = False
        self.reward = 0.0
        self.current_step = 0

        self.lat0 = None
        self.lon0 = None
        self.gps_x_offset = 0.0
        self.gps_y_offset = 0.0
        self.last_gps_frame = None
        self.prev_distance_to_waypoint = None
        self.prev_route_error = 0.0

        if debug and self.log_pose:
            if self.pose_log_file is not None:
                self.pose_log_file.close()

            self.pose_log_file = open("pose_log.csv", "w", newline="")
            self.pose_logger = csv.writer(self.pose_log_file)
            self.pose_logger.writerow([
                "step",
                "true_x", "true_y", "true_yaw",
                "gps_x", "gps_y",
                "kf_x", "kf_y", "kf_vx", "kf_vy",
                "err_x", "err_y",
            ])
        else:
            self.pose_logger = None

        self.episode_counter += 1

        resume_ok = (
            self.resume_on_reset
            and getattr(self, "actor", None) is not None
            and not prev_collided
            and not prev_terminated
            and prev_distance_to_destination >= 3.0
            and (self.episode_counter % self.reset_interval) != 0
        )

        if resume_ok:
            reinit_sensors = False
        else:
            self._destroy_actor(getattr(self, "imu", None))
            self.imu = None

            self._destroy_actor(getattr(self, "gnss", None))
            self.gnss = None

            self._destroy_actor(getattr(self, "collision_sensor", None))
            self.collision_sensor = None

            self._destroy_actor(getattr(self, "actor", None))
            self.actor = None

            self.imu_data = None
            self.gps_data = None
            self.impulse = None
            self.collision_intensity = 0.0
            reinit_sensors = True

        self.car_design = self.blueprint_library.find("vehicle.micro.microlino")
        self.IMU_bp = self.blueprint_library.find("sensor.other.imu")
        self.GNSS_bp = self.blueprint_library.find("sensor.other.gnss")

        self.GNSS_bp.set_attribute("sensor_tick", "0.2")
        self.GNSS_bp.set_attribute("noise_lat_bias", "0.0")
        self.GNSS_bp.set_attribute("noise_lat_stddev", "0.00003")
        self.GNSS_bp.set_attribute("noise_lon_bias", "0.0")
        self.GNSS_bp.set_attribute("noise_lon_stddev", "0.00003")
        self.GNSS_bp.set_attribute("noise_alt_bias", "0.0")
        self.GNSS_bp.set_attribute("noise_alt_stddev", "1.0")
        self.GNSS_bp.set_attribute("noise_seed", "42")

        self.IMU_transform = Transform(Location(x=0.0, y=0.0, z=1.5))
        self.GNSS_transform = Transform(Location(x=0.0, y=0.0, z=1.5))

        if reinit_sensors:
            self._spawn_vehicle()

            if self.draw_spawn_debug:
                pass

            self.imu = self.world.spawn_actor(self.IMU_bp, self.IMU_transform, attach_to=self.actor)
            self.gnss = self.world.spawn_actor(self.GNSS_bp, self.GNSS_transform, attach_to=self.actor)
            self.collision_sensor = self.world.spawn_actor(
                self.blueprint_library.find("sensor.other.collision"),
                Transform(Location(x=0.0, y=0.0, z=1.5)),
                attach_to=self.actor,
            )

            self.imu.listen(self.imu_callback)
            self.gnss.listen(self.gps_callback)
            self.collision_sensor.listen(self.collision_callback)

            self.world.tick()
            while self.imu_data is None or self.gps_data is None:
                self.world.tick()
        else:
            self.world.tick()

        true_transform = self.actor.get_transform()
        self.origin = true_transform.location

        self.lat0 = self.gps_data.latitude
        self.lon0 = self.gps_data.longitude
        self.gps_x_offset = self.origin.x
        self.gps_y_offset = self.origin.y
        self.last_gps_frame = self.gps_data.frame

        self.vx = 0.0
        self.vy = 0.0
        self.calc_x = self.origin.x
        self.calc_y = self.origin.y
        self.calc_yaw = true_transform.rotation.yaw

        self.kf = KalmanFilter2D(self.dt)
        self.kf.set_state(
            x0=self.origin.x,
            y0=self.origin.y,
            vx0=0.0,
            vy0=0.0,
        )

        self._build_fixed_route()

        self.waypoint_index = 0
        self._update_current_waypoint()

        self.x = true_transform.location.x
        self.y = true_transform.location.y
        self.yaw = true_transform.rotation.yaw

        self.prev_distance_to_waypoint = np.sqrt(
            (self.waypoint_x - self.x) ** 2 + (self.waypoint_y - self.y) ** 2
        )
        self.prev_route_error = self._distance_to_route_segment()

        if self.draw_route_debug:
            self._draw_route(life_time=self.route_debug_life)

        obs, speed, dx, dy, heading_error, distance = self._build_observation()

        self.prev_steer_cmd = 0.0
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_heading_error = heading_error
        self.throttle_cmd = 0.0

        return obs, {}

    def step(self, action):
        self.reward = 0.0
        truncated = False

        reward_breakdown = {
            "time_penalty": 0.0,
            "progress": 0.0,
            "speed_bonus": 0.0,
            "align_reward": 0.0,
            "route_recover": 0.0,
            "heading_fix": 0.0,
            "stuck_bad_heading": 0.0,
            "standstill_penalty": 0.0,
            "steer_penalty": 0.0,
            "waypoint_bonus": 0.0,
            "goal_bonus": 0.0,
            "collision_penalty": 0.0,
            "wrong_way": 0.0,
            "route_align": 0.0,
        }

        prev_heading_error = self.prev_heading_error

        self.steer_cmd = float(action[0])
        self.throttle_cmd = float(action[1])

        if self.throttle_cmd >= 0:
            throttle = self.throttle_cmd
            brake = 0.0
        else:
            throttle = 0.0
            brake = -self.throttle_cmd

        steer = float(np.clip(self.steer_cmd, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.actor.apply_control(control)

        self.world.tick()
        self.current_step += 1

        if self.current_step >= self.max_steps:
            truncated = True

        if self.imu_data is None or self.gps_data is None:
            obs = np.zeros((12,), dtype=np.float32)
            return obs, 0.0, self.terminated, truncated, {}

        current_imu = self.imu_data
        current_gps = self.gps_data

        a = current_imu.accelerometer
        w = current_imu.gyroscope

        ax = np.clip(a.x - self.ax_bias, -20.0, 20.0)
        ay = np.clip(a.y - self.ay_bias, -20.0, 20.0)

        self.kf.predict(ax, ay)

        self.calc_yaw += np.degrees(w.z * self.dt)
        self.calc_yaw = self._wrap_angle_deg(self.calc_yaw)

        if current_gps.frame != self.last_gps_frame:
            gps_lat = current_gps.latitude
            gps_lon = current_gps.longitude

            gps_x_local, gps_y_local = self.latlon_to_xy(gps_lat, gps_lon)
            gps_x = gps_x_local + self.gps_x_offset
            gps_y = gps_y_local + self.gps_y_offset

            self.kf.update(gps_x, gps_y)
            self.last_gps_frame = current_gps.frame

        self.calc_x, self.calc_y, self.vx, self.vy = self.kf.get_state()

        true_transform = self.actor.get_transform()
        true_x = true_transform.location.x
        true_y = true_transform.location.y
        true_yaw = true_transform.rotation.yaw

        if self.use_true_pose_for_training:
            self.x = true_x
            self.y = true_y
        else:
            self.x = self.calc_x
            self.y = self.calc_y

        if self.use_true_yaw_for_training:
            self.yaw = true_yaw
        else:
            self.yaw = self.calc_yaw

        self.distance_to_destination = np.sqrt(
            (self.destination.x - self.x) ** 2 + (self.destination.y - self.y) ** 2
        )

        if len(self.route) > 0:
            self._update_current_waypoint()
            self.distance_to_waypoint = np.sqrt(
                (self.waypoint_x - self.x) ** 2 + (self.waypoint_y - self.y) ** 2
            )

            while self.waypoint_index < len(self.route) - 1 and self.distance_to_waypoint < 2.5:
                self.waypoint_index += 1
                self._update_current_waypoint()
                self._add_reward(reward_breakdown, "waypoint_bonus", self.waypoint_bonus)

                self.distance_to_waypoint = np.sqrt(
                    (self.waypoint_x - self.x) ** 2 + (self.waypoint_y - self.y) ** 2
                )
        else:
            self.waypoint_x = self.destination.x
            self.waypoint_y = self.destination.y
            self.distance_to_waypoint = self.distance_to_destination

        obs, speed, dx, dy, heading_error, distance = self._build_observation()

        # time penalty
        self._add_reward(reward_breakdown, "time_penalty", self.time_penalty)

        # progress reward
        progress = 0.0
        if self.prev_distance_to_waypoint is not None:
            progress = float(self.prev_distance_to_waypoint - self.distance_to_waypoint)
            self._add_reward(reward_breakdown, "progress", self.k_progress * progress)

        # heading alignment reward
        align_reward = self.k_align * (abs(prev_heading_error) - abs(heading_error))
        self._add_reward(reward_breakdown, "align_reward", align_reward)

        # 1) route recovery reward
        route_error = self._distance_to_route_segment()
        route_error_improve = self.prev_route_error - route_error
        self._add_reward(
            reward_breakdown,
            "route_recover",
            self.k_route_recover * route_error_improve,
        )

        # 2) extra heading-fix reward when badly misaligned
        if abs(prev_heading_error) > self.bad_heading_threshold:
            heading_fix = self.k_heading_fix * (
                abs(prev_heading_error) - abs(heading_error)
            )
            self._add_reward(reward_breakdown, "heading_fix", heading_fix)

        # route alignment reward
        route_dir = self._get_route_direction()
        if route_dir is not None:
            yaw_rad = np.radians(self.yaw)
            heading_vec = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=np.float32)
            route_alignment = float(np.dot(heading_vec, route_dir))
            route_align_reward = self.k_route_align * route_alignment
            self._add_reward(reward_breakdown, "route_align", route_align_reward)

        # wrong-way penalty
        if route_dir is not None and speed > self.standstill_speed:
            vel_vec = np.array([self.vx, self.vy], dtype=np.float32)
            vel_norm = np.linalg.norm(vel_vec)

            if vel_norm > 1e-6:
                vel_dir = vel_vec / vel_norm
                motion_alignment = float(np.dot(vel_dir, route_dir))

                if motion_alignment < 0.0:
                    wrong_way_penalty = -self.k_wrong_way * (-motion_alignment)
                    self._add_reward(reward_breakdown, "wrong_way", wrong_way_penalty)

        # speed bonus
        if progress > 0.0 and abs(heading_error) < 15.0:
            speed_reward = self.k_speed * min(speed, 3.0)
            self._add_reward(reward_breakdown, "speed_bonus", speed_reward)

        # standstill penalty
        if speed < self.standstill_speed:
            self._add_reward(reward_breakdown, "standstill_penalty", self.standstill_penalty)

        # 3) stronger penalty for freezing while badly misaligned
        if speed < self.standstill_speed and abs(heading_error) > self.bad_heading_threshold:
            self._add_reward(
                reward_breakdown,
                "stuck_bad_heading",
                self.stuck_bad_heading_penalty,
            )

        # mild steering smoothness penalty
        steer_change = abs(self.steer_cmd - self.prev_steer_cmd)
        steer_penalty = -self.k_steer_change * steer_change
        self._add_reward(reward_breakdown, "steer_penalty", steer_penalty)

        # success / failure
        if self.distance_to_destination < 3.0:
            self.terminated = True
            self._add_reward(reward_breakdown, "goal_bonus", self.goal_bonus)

        if self.collided:
            self.terminated = True
            self._add_reward(reward_breakdown, "collision_penalty", self.collision_penalty)

        if self.pose_logger is not None:
            gps_x = None
            gps_y = None

            if self.gps_data is not None:
                gps_lat = self.gps_data.latitude
                gps_lon = self.gps_data.longitude
                gps_x_local, gps_y_local = self.latlon_to_xy(gps_lat, gps_lon)
                gps_x = gps_x_local + self.gps_x_offset
                gps_y = gps_y_local + self.gps_y_offset

            kf_x, kf_y, kf_vx, kf_vy = self.kf.get_state()
            err_x = kf_x - true_x
            err_y = kf_y - true_y

            self.pose_logger.writerow([
                self.current_step,
                true_x, true_y, true_yaw,
                gps_x, gps_y,
                kf_x, kf_y, kf_vx, kf_vy,
                err_x, err_y,
            ])

        for key, value in reward_breakdown.items():
            self.episode_reward_totals[key] += value

        reward_sum_check = sum(reward_breakdown.values())

        self.prev_distance_to_waypoint = self.distance_to_waypoint
        self.prev_route_error = route_error
        self.prev_steer_cmd = self.steer_cmd
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_heading_error = heading_error

        info = {
            "reward_breakdown": reward_breakdown,
            "episode_reward_totals": dict(self.episode_reward_totals),
            "reward_sum_check": reward_sum_check,
            "reward_match_error": self.reward - reward_sum_check,
            "speed": speed,
            "heading_error": heading_error,
            "distance_to_waypoint": distance,
            "distance_to_destination": self.distance_to_destination,
            "waypoint_index": self.waypoint_index,
            "route_length": len(self.route),
            "goal_reached": bool(self.distance_to_destination < 3.0),
            "route_error": route_error,
        }

        return obs, self.reward, self.terminated, truncated, info

    def render(self):
        if self.actor is None:
            return

        spectator = self.world.get_spectator()
        vehicle_transform = self.actor.get_transform()

        spectator_transform = carla.Transform(
            vehicle_transform.location + carla.Location(z=10.0),
            carla.Rotation(pitch=-90.0, yaw=0.0),
        )
        spectator.set_transform(spectator_transform)

        text_loc = vehicle_transform.location + carla.Location(z=3.0)
        text = (
            f"TRUE: ({vehicle_transform.location.x:.2f}, {vehicle_transform.location.y:.2f}, {vehicle_transform.rotation.yaw:.1f})\n"
            f"TRAIN: ({self.x:.2f}, {self.y:.2f}, {self.yaw:.1f})\n"
            f"WP IDX: {self.waypoint_index}/{max(len(self.route)-1, 0)}"
        )

        # self.world.debug.draw_string(
        #     text_loc,
        #     text,
        #     draw_shadow=False,
        #     color=carla.Color(255, 255, 255),
        #     life_time=0.1,
        # )

        wp_loc = carla.Location(x=self.waypoint_x, y=self.waypoint_y, z=1.0)

        self.world.debug.draw_point(
            wp_loc,
            size=0.2,
            color=carla.Color(0, 255, 0),
            life_time=0.1,
        )

        vehicle_loc = self.actor.get_transform().location
        self.world.debug.draw_line(
            vehicle_loc,
            wp_loc,
            thickness=0.01,
            color=carla.Color(0, 255, 0),
            life_time=0.1,
        )

    def close(self):
        if self.pose_log_file is not None:
            self.pose_log_file.close()
            self.pose_log_file = None
            self.pose_logger = None

        self._destroy_actor(getattr(self, "imu", None))
        self.imu = None

        self._destroy_actor(getattr(self, "gnss", None))
        self.gnss = None

        self._destroy_actor(getattr(self, "collision_sensor", None))
        self.collision_sensor = None

        self._destroy_actor(getattr(self, "actor", None))
        self.actor = None

    def imu_callback(self, data):
        self.imu_data = data

    def gps_callback(self, data):
        self.gps_data = data

    def collision_callback(self, event):
        self.impulse = event.normal_impulse
        intensity = np.sqrt(self.impulse.x**2 + self.impulse.y**2 + self.impulse.z**2)
        self.collision_intensity = intensity
        self.collided = True

    def latlon_to_xy(self, lat, lon):
        R = 6371000.0
        x = np.radians(lon - self.lon0) * R * np.cos(np.radians(self.lat0))
        y = R * np.radians(lat - self.lat0) - 5.0
        return x, y