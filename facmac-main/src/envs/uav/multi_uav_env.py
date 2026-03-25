import math
import importlib
import logging
import numpy as np
import os, json, time
# Try to import optional helpers; if they are missing provide minimal safe fallbacks
try:
    mod = importlib.import_module('src.envs.obstacle')
    CylinderObstacle = getattr(mod, 'CylinderObstacle')
except Exception:
    class CylinderObstacle:
        def __init__(self, x, y, radius, z_min=0.0, z_max=None):
            self.x = float(x)
            self.y = float(y)
            self.radius = float(radius)
            self.z_min = float(z_min)
            self.z_max = float("inf") if z_max is None else float(z_max)

        def collides_point(self, p):
            p = np.asarray(p, dtype=float)
            radial = np.linalg.norm(p[:2] - np.array([self.x, self.y], dtype=float))
            return radial <= self.radius and self.z_min <= p[2] <= self.z_max

        def distance_to_point(self, p):
            p = np.asarray(p, dtype=float)
            radial_gap = np.linalg.norm(p[:2] - np.array([self.x, self.y], dtype=float)) - self.radius
            if p[2] < self.z_min:
                dz = self.z_min - p[2]
            elif p[2] > self.z_max:
                dz = p[2] - self.z_max
            else:
                dz = 0.0
            if dz == 0.0:
                return radial_gap
            return math.hypot(max(0.0, radial_gap), dz)

try:
    mod = importlib.import_module("src.utils")
    sample_position = getattr(mod, "sample_position")
    clamp_pos = getattr(mod, "clamp_pos")
except Exception:
    def sample_position(rng, margin, space_size):
        low = margin
        high = max(margin, space_size - margin)
        return np.array(
            [rng.uniform(low, high), rng.uniform(low, high), rng.uniform(low, high)],
            dtype=float,
        )

    def clamp_pos(p, space_size):
        p = np.asarray(p, dtype=float)
        return np.clip(p, 0.0, float(space_size))

try:
    mod = importlib.import_module("src.envs.reward")
    compute_step_reward = getattr(mod, "compute_step_reward")
except Exception:
    from src.envs.reward import compute_step_reward

# try:
#     mod = importlib.import_module('src.envs.seed_utils')
#     set_main_seed = getattr(mod, 'set_main_seed')
# except Exception:
#     def set_main_seed(s):
#         try:
#             s = int(s)
#         except Exception:
#             s = 0
#         np.random.seed(s)

# try:
#     mod = importlib.import_module('src.envs.reward')
#     compute_step_reward = getattr(mod, 'compute_step_reward')
# except Exception:
#     def compute_step_reward(positions, velocities, goal, obstacles, prev_positions=None, goal_radius=1.0, v_max=1.0, **kwargs):
#         # Minimal placeholder: zero reward for every agent and a simple info dict.
#         # Accept **kwargs to remain compatible when callers pass extra flags like
#         # already_reached, so we don't raise unexpected-argument errors.
#         n = int(np.shape(positions)[0]) if hasattr(positions, "__len__") else 0
#         rewards = [0.0 for _ in range(n)]
#         info = {"placeholder_reward": True, 'components': None}
#         return rewards, info

def _segment_intersects_sphere(p0, p1, center, radius):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    center = np.asarray(center, dtype=float)
    v = p1 - p0
    vv = np.dot(v, v)
    if vv <= 1e-12:
        return np.linalg.norm(p0 - center) <= radius
    t = np.dot(center - p0, v) / vv
    t = np.clip(t, 0.0, 1.0)
    closest = p0 + t * v
    return np.linalg.norm(closest - center) <= radius

# New: check whether a 3D segment intersects a vertical cylinder obstacle (center x,y, radius, z_min,z_max)
def _segment_intersects_cylinder(p0, p1, cyl):
    """Return True if segment p0->p1 intersects cylinder `cyl` (has x,y,radius,z_min,z_max)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    c_xy = np.array([cyl.x, cyl.y], dtype=float)
    v_xy = p1[:2] - p0[:2]
    vv = np.dot(v_xy, v_xy)
    if vv <= 1e-12:
        return cyl.collides_point(p0)
    t = np.dot(c_xy - p0[:2], v_xy) / vv
    t = np.clip(t, 0.0, 1.0)
    closest = p0 + t * (p1 - p0)
    return cyl.collides_point(closest)

def _pair_segment_min_distance(p0, p1, q0, q1):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    u = p1 - p0
    v = q1 - q0
    w0 = p0 - q0
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    denom = a * c - b * b

    if denom < 1e-12:
        s = 0.0
        t = np.clip(e / max(c, 1e-12), 0.0, 1.0)
    else:
        s = np.clip((b * e - c * d) / denom, 0.0, 1.0)
        t = np.clip((a * e - b * d) / denom, 0.0, 1.0)

    cp = p0 + s * u
    cq = q0 + t * v
    return np.linalg.norm(cp - cq)


# def _point_inside_goal_sphere(p, goal, goal_radius):
#     return np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(goal, dtype=float)) <= float(goal_radius)

class MultiUAVEnv:
    def __init__(self, cfg, main_seed=0, subseed=0, enable_dynamic=False):
        self.cfg = dict(cfg)
        self.log = logging.getLogger(self.cfg.get("logger_name", "MultiUAVEnv"))

        self.n_agents = int(self.cfg.get("n_agents", 3))
        self.space_size = float(self.cfg.get("space_size", 100.0))
        self.n_static = int(self.cfg.get("n_static_obstacles", 1))
        self.max_steps = int(self.cfg.get("max_steps", self.cfg.get("episode_limit", 300)))
        self.dt = float(self.cfg.get("dt", 0.2))
        self.v_max = float(self.cfg.get("v_max", 1.0))
        self.goal_radius = float(self.cfg.get("goal_radius", 8.0))
        self.margin = float(self.cfg.get("spawn_margin", 5.0))
        self.randomize_every_reset = bool(self.cfg.get("randomize_every_reset", True))
        self.fixed_initial_positions = bool(self.cfg.get("fixed_initial_positions", False))
        self.use_normalized_actions = bool(self.cfg.get("use_normalized_actions", True))
        self.safe_uav_dist = float(self.cfg.get("safe_uav_dist", 6.0))
        self.collision_dist = float(self.cfg.get("collision_dist", 2.0))
        self.near_uav_dist = float(self.cfg.get("near_uav_dist", 6.0))
        self.safe_dist_obs = float(self.cfg.get("safe_dist_obs", 6.0))
        self.lidar_num_rays = int(self.cfg.get("lidar_num_rays", 10))
        self.lidar_vertical_rays = int(self.cfg.get("lidar_vertical_rays", 2))
        self.lidar_range = float(self.cfg.get("lidar_range", self.space_size * 0.4))
        self.enable_dynamic = bool(enable_dynamic)

        self.master_seed = int(main_seed)
        self.subseed = int(subseed)
        self.rng = np.random.RandomState(self.master_seed + self.subseed)

        self.action_dim = 3
        self.num_uav = self.n_agents
        self.dynamic_objs = []

        self._fixed_start_pos = None
        self._fixed_goal = None
        self._fixed_obstacles = None

        self._init_world()
        self._reset_episode_buffers()
        self._refresh_dims()

    def _refresh_dims(self):
        self.obs_dim = len(self._agent_obs(0))
        self.state_dim = len(self._get_state_vector())

    def _reset_episode_buffers(self):
        self.step_count = 0
        self.prev_pos = self.pos.copy()
        self.last_actions = np.zeros((self.n_agents, 3), dtype=float)
        self.arrived_mask = np.zeros(self.n_agents, dtype=bool)
        self.crashed_mask = np.zeros(self.n_agents, dtype=bool)
        self.colliding_mask = np.zeros(self.n_agents, dtype=bool)
        self.moved_steps = np.zeros(self.n_agents, dtype=int)
        self.first_arrival_step = np.full(self.n_agents, -1, dtype=int)
        self.first_collision_step = np.full(self.n_agents, -1, dtype=int)
        self.first_collision_reason = [None for _ in range(self.n_agents)]
        self.min_dist_to_goal = np.linalg.norm(self.pos - self.goal, axis=1)
        self.episode_rewards = []
        self.episode_per_agent_rewards = []
        self.episode_components = []
    
    def _sample_goal(self):
        for _ in range(200):
            candidate = sample_position(self.rng, self.margin, self.space_size)
            if all(obs.distance_to_point(candidate) > self.safe_dist_obs for obs in self.obstacles):
                return candidate
        return sample_position(self.rng, self.margin, self.space_size)

    def _sample_obstacles(self):
        obstacles = []
        for _ in range(self.n_static):
            for _ in range(100):
                x = self.rng.uniform(self.margin, self.space_size - self.margin)
                y = self.rng.uniform(self.margin, self.space_size - self.margin)
                radius = self.rng.uniform(3.0, 6.0)
                z_min = self.rng.uniform(0.0, self.space_size * 0.35)
                z_max = self.rng.uniform(z_min + 4.0, self.space_size)
                cand = CylinderObstacle(x, y, radius, z_min=z_min, z_max=z_max)
                if all(
                    np.hypot(cand.x - old.x, cand.y - old.y) > cand.radius + old.radius + self.safe_dist_obs
                    for old in obstacles
                ):
                    obstacles.append(cand)
                    break
        return obstacles

    def _sample_valid_uav_start(self, existing_positions):
        for _ in range(300):
            candidate = sample_position(self.rng, self.margin, self.space_size)
            if np.linalg.norm(candidate - self.goal) <= self.goal_radius + self.safe_uav_dist:
                continue
            if self._collides_any(candidate):
                continue
            if any(np.linalg.norm(candidate - other) < self.safe_uav_dist for other in existing_positions):
                continue
            return candidate
        return sample_position(self.rng, self.margin, self.space_size)
    
    def _init_world(self):
        if self.randomize_every_reset or self._fixed_obstacles is None:
            self.obstacles = self._sample_obstacles()
            self.goal = self._sample_goal()
            self.pos = np.zeros((self.n_agents, 3), dtype=float)
            self.vel = np.zeros((self.n_agents, 3), dtype=float)
            for i in range(self.n_agents):
                self.pos[i] = self._sample_valid_uav_start(self.pos[:i])
            if self.fixed_initial_positions:
                self._fixed_start_pos = self.pos.copy()
                self._fixed_goal = self.goal.copy()
                self._fixed_obstacles = list(self.obstacles)
        else:
            self.obstacles = list(self._fixed_obstacles)
            self.goal = self._fixed_goal.copy()
            self.pos = self._fixed_start_pos.copy()
            self.vel = np.zeros((self.n_agents, 3), dtype=float)

    # TODO(随机种子)：输出随机主种子和子种子
    # print(main_seed)
    # print(sub_seed)
    # TODO(解决完episode输出不完全问题): 输出障碍物、无人机和目标点信息

    def reset(self, subseed=None):
        if subseed is not None:
            self.subseed = int(subseed)
            self.rng = np.random.RandomState(self.master_seed + self.subseed)

        if self.randomize_every_reset:
            self._init_world()
        else:
            if self._fixed_start_pos is None:
                self._init_world()
            else:
                self.obstacles = list(self._fixed_obstacles)
                self.goal = self._fixed_goal.copy()
                self.pos = self._fixed_start_pos.copy()
                self.vel = np.zeros((self.n_agents, 3), dtype=float)

        self._reset_episode_buffers()
        self._refresh_dims()
        return self._get_obs_state()

    def _in_bounds(self, p):
        return clamp_pos(p, self.space_size)

    def _is_out_of_bounds(self, p):
        p = np.asarray(p, dtype=float)
        return bool(np.any(p < 0.0) or np.any(p > self.space_size))

    def _collides_any(self, p):
        return any(obs.collides_point(p) for obs in self.obstacles)
    
    def _simple_ray_cast(self, pos, direction, max_range):
        direction = np.asarray(direction, dtype=float)
        direction = direction / max(np.linalg.norm(direction), 1e-6)
        steps = 40
        for s in range(1, steps + 1):
            q = pos + direction * (max_range * s / steps)
            if self._is_out_of_bounds(q) or self._collides_any(q):
                return max_range * s / steps
        return max_range

    def _lidar_dirs(self):
        dirs = []
        for k in range(self.lidar_num_rays):
            yaw = 2.0 * math.pi * k / max(self.lidar_num_rays, 1)
            dirs.append(np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float))
        if self.lidar_vertical_rays >= 1:
            dirs.append(np.array([0.0, 0.0, 1.0], dtype=float))
            dirs.append(np.array([0.0, 0.0, -1.0], dtype=float))
        return dirs

    def _nearest_obstacle_features(self, idx):
        pos = self.pos[idx]
        if not self.obstacles:
            return np.zeros(3, dtype=float), 1.0
        dists = [obs.distance_to_point(pos) for obs in self.obstacles]
        j = int(np.argmin(dists))
        obs = self.obstacles[j]
        obs_center = np.array([obs.x, obs.y, np.clip(pos[2], obs.z_min, obs.z_max)], dtype=float)
        rel = (obs_center - pos) / max(self.space_size, 1e-6)
        dist_norm = np.clip(dists[j] / max(self.space_size, 1e-6), 0.0, 1.0)
        return rel, dist_norm

    def _nearest_teammate_features(self, idx):
        pos = self.pos[idx]
        vel = self.vel[idx]
        others = []
        for j in range(self.n_agents):
            if j == idx:
                continue
            rel = self.pos[j] - pos
            dist = np.linalg.norm(rel)
            others.append((dist, rel, self.vel[j] - vel, j))
        if not others:
            return np.zeros(3, dtype=float), np.zeros(3, dtype=float), 1.0
        dist, rel, rel_vel, _ = min(others, key=lambda x: x[0])
        return (
            rel / max(self.space_size, 1e-6),
            rel_vel / max(self.v_max, 1e-6),
            np.clip(dist / max(self.space_size, 1e-6), 0.0, 1.0),
        )
    
    def _agent_obs(self, idx):
        pos = self.pos[idx] / max(self.space_size, 1e-6)
        vel = self.vel[idx] / max(self.v_max, 1e-6)
        goal_rel = (self.goal - self.pos[idx]) / max(self.space_size, 1e-6)
        goal_dist = np.array([np.linalg.norm(self.goal - self.pos[idx]) / max(self.space_size, 1e-6)], dtype=float)
        last_action = self.last_actions[idx].copy()
        nearest_obs_rel, nearest_obs_dist = self._nearest_obstacle_features(idx)
        mate_rel, mate_rel_vel, mate_dist = self._nearest_teammate_features(idx)

        lidar = []
        for d in self._lidar_dirs():
            lidar.append(self._simple_ray_cast(self.pos[idx], d, self.lidar_range) / max(self.lidar_range, 1e-6))

        status = np.array(
            [float(self.arrived_mask[idx]), float(self.crashed_mask[idx]), float(self.step_count / max(self.max_steps, 1))],
            dtype=float,
        )

        return np.concatenate(
            [
                pos,
                vel,
                goal_rel,
                goal_dist,
                last_action,
                nearest_obs_rel,
                np.array([nearest_obs_dist], dtype=float),
                mate_rel,
                mate_rel_vel,
                np.array([mate_dist], dtype=float),
                np.array(lidar, dtype=float),
                status,
            ]
        ).astype(np.float32)

    def _get_state_vector(self):
        pos = (self.pos / max(self.space_size, 1e-6)).reshape(-1)
        vel = (self.vel / max(self.v_max, 1e-6)).reshape(-1)
        goal = (self.goal / max(self.space_size, 1e-6)).reshape(-1)
        masks = np.stack([self.arrived_mask.astype(float), self.crashed_mask.astype(float)], axis=1).reshape(-1)

        obs_feats = []
        for obs in self.obstacles:
            obs_feats.extend(
                [
                    obs.x / self.space_size,
                    obs.y / self.space_size,
                    obs.radius / self.space_size,
                    obs.z_min / self.space_size,
                    obs.z_max / self.space_size,
                ]
            )
        if not obs_feats:
            obs_feats = [0.0] * 5

        return np.concatenate([pos, vel, goal, masks, np.array(obs_feats, dtype=float)]).astype(np.float32)

    def _get_obs_state(self):
        obs = [self._agent_obs(i) for i in range(self.n_agents)]
        return obs, self._get_state_vector()

    def _nearest_uav_distances(self):
        out = np.full(self.n_agents, np.inf, dtype=float)
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                out[i] = min(out[i], np.linalg.norm(self.pos[i] - self.pos[j]))
        return out

    def _nearest_obstacle_distances(self):
        if not self.obstacles:
            return np.full(self.n_agents, self.space_size, dtype=float)
        vals = []
        for i in range(self.n_agents):
            vals.append(min(obs.distance_to_point(self.pos[i]) for obs in self.obstacles))
        return np.asarray(vals, dtype=float)


    def step(self, actions):
        self.step_count += 1
        prev_pos = self.pos.copy()
        actions = np.asarray(actions, dtype=float).reshape(self.n_agents, self.action_dim)

        if self.use_normalized_actions:
            clipped_actions = np.clip(actions, -1.0, 1.0)
            desired_vel = clipped_actions * self.v_max
            self.last_actions = clipped_actions.copy()
        else:
            desired_vel = np.clip(actions, -self.v_max, self.v_max)
            self.last_actions = desired_vel / max(self.v_max, 1e-6)
        
        collided_this_step = np.zeros(self.n_agents, dtype=bool)
        collision_reason_this_step = [None for _ in range(self.n_agents)]
        arrived_this_step = np.zeros(self.n_agents, dtype=bool)
        reached_by_segment_step = np.zeros(self.n_agents, dtype=bool)
        reached_by_distance_step = np.zeros(self.n_agents, dtype=bool)
        
        # 更新速度和位置
        for i in range(self.n_agents):
            if self.arrived_mask[i] or self.crashed_mask[i]:
                self.vel[i] = 0.0
                self.pos[i] = prev_pos[i]
                continue

            v = desired_vel[i]
            speed = np.linalg.norm(v)
            if speed > self.v_max:
                v = v / speed * self.v_max

            intended_pos = prev_pos[i] + v * self.dt
            self.vel[i] = v
            self.pos[i] = self._in_bounds(intended_pos)
        
        # MOD-E1: collision is computed on the motion segment, not only on the endpoint.
        for i in range(self.n_agents):
            if self.arrived_mask[i] or self.crashed_mask[i]:
                continue

            intended_pos = prev_pos[i] + self.vel[i] * self.dt
            if self._is_out_of_bounds(intended_pos):
                collided_this_step[i] = True
                collision_reason_this_step[i] = "boundary"

            for obs in self.obstacles:
                if obs.collides_point(self.pos[i]) or _segment_intersects_cylinder(prev_pos[i], self.pos[i], obs):
                    collided_this_step[i] = True
                    if collision_reason_this_step[i] is None:
                        collision_reason_this_step[i] = "obstacle"
                    break

            if _segment_intersects_sphere(prev_pos[i], self.pos[i], self.goal, self.goal_radius):
                reached_by_segment_step[i] = True
                arrived_this_step[i] = True
            elif np.linalg.norm(self.pos[i] - self.goal) <= self.goal_radius:
                reached_by_distance_step[i] = True
                arrived_this_step[i] = True
        
        # MOD-E4: pairwise UAV collision checks.
        for i in range(self.n_agents):
            if self.arrived_mask[i] or self.crashed_mask[i]:
                continue
            for j in range(i + 1, self.n_agents):
                if self.arrived_mask[j] or self.crashed_mask[j]:
                    continue
                seg_dist = _pair_segment_min_distance(prev_pos[i], self.pos[i], prev_pos[j], self.pos[j])
                if seg_dist <= self.collision_dist:
                    collided_this_step[i] = True
                    collided_this_step[j] = True
                    if collision_reason_this_step[i] is None:
                        collision_reason_this_step[i] = "uav_pair"
                    if collision_reason_this_step[j] is None:
                        collision_reason_this_step[j] = "uav_pair"

        # Arrival-priority policy: if an agent reaches the goal this step,
        # same-step collisions are ignored for that agent.
        collided_this_step_raw = collided_this_step.copy()
        collided_this_step = collided_this_step_raw & (~arrived_this_step)
        collision_reason_effective = [
            (collision_reason_this_step[i] if collided_this_step[i] else None)
            for i in range(self.n_agents)
        ]

        self.crashed_mask |= collided_this_step
        new_arrivals = arrived_this_step & (~self.arrived_mask)
        self.arrived_mask |= new_arrivals
        self.colliding_mask = collided_this_step.copy()
        moved_this_step = (np.linalg.norm(self.pos - prev_pos, axis=1) > 1e-9)
        self.moved_steps += moved_this_step.astype(int)

        for i in range(self.n_agents):
            if new_arrivals[i] and self.first_arrival_step[i] < 0:
                self.first_arrival_step[i] = self.step_count
            if collided_this_step[i] and self.first_collision_step[i] < 0:
                self.first_collision_step[i] = self.step_count
                self.first_collision_reason[i] = collision_reason_effective[i]
        
        self.min_dist_to_goal = np.minimum(self.min_dist_to_goal, np.linalg.norm(self.pos - self.goal, axis=1))

        nearest_obstacle_dists = self._nearest_obstacle_distances()
        nearest_uav_dists = self._nearest_uav_distances()

        reward_cfg = dict(self.cfg.get("reward", {}))
        reward_cfg.setdefault("safe_dist_obs", self.safe_dist_obs)
        reward_cfg.setdefault("safe_dist_uav", self.near_uav_dist)
        reward_cfg.setdefault("space_size", self.space_size)

        rewards, reward_info = compute_step_reward(
            positions=self.pos,
            velocities=self.vel,
            goal=self.goal,
            obstacles=self.obstacles,
            prev_positions=prev_pos,
            goal_radius=self.goal_radius,
            v_max=self.v_max,
            already_reached=self.arrived_mask & (~new_arrivals),
            arrived_this_step=new_arrivals,
            collided_this_step=collided_this_step,
            nearest_obstacle_dists=nearest_obstacle_dists,
            nearest_uav_dists=nearest_uav_dists,
            reward_cfg=reward_cfg,
        )

        components = reward_info.get("components", None) if isinstance(reward_info, dict) else None
        self.episode_rewards.append(float(np.sum(rewards)))
        self.episode_per_agent_rewards.append(list(rewards))
        self.episode_components.append(components if components is not None else [])
        
        obs, state = self._get_obs_state()
        all_resolved = bool(np.all(self.arrived_mask | self.crashed_mask))
        reached_by_segment = self.arrived_mask & reached_by_segment_step
        reached_by_distance = self.arrived_mask & reached_by_distance_step
        done = bool(
            self.step_count >= self.max_steps
            or all_resolved
        )

        termination_reason = None
        if done:
            if self.step_count >= self.max_steps and (not all_resolved):
                termination_reason = "max_steps"
            elif all_resolved:
                termination_reason = "all_agents_resolved"
            else:
                termination_reason = "done"

        per_agent_arrival_audit = []
        for i in range(self.n_agents):
            per_agent_arrival_audit.append(
                {
                    "agent_id": int(i),
                    "arrived": bool(self.arrived_mask[i]),
                    "crashed": bool(self.crashed_mask[i]),
                    "reached_by_segment": bool(reached_by_segment[i]),
                    "reached_by_distance": bool(reached_by_distance[i]),
                    "moved_steps": int(self.moved_steps[i]),
                    "first_arrival_step": int(self.first_arrival_step[i]),
                    "first_collision_step": int(self.first_collision_step[i]),
                    "first_collision_reason": self.first_collision_reason[i],
                    "min_dist_to_goal": float(self.min_dist_to_goal[i]),
                }
            )

        info = {
            "arrived_mask": self.arrived_mask.astype(int).tolist(),
            "crashed_mask": self.crashed_mask.astype(int).tolist(),
            "collided_this_step": collided_this_step.astype(int).tolist(),
            "collided_this_step_raw": collided_this_step_raw.astype(int).tolist(),
            "collision_suppressed_by_arrival": (collided_this_step_raw & arrived_this_step).astype(int).tolist(),
            "collision_reason_this_step": collision_reason_effective,
            "collision_reason_this_step_raw": collision_reason_this_step,
            "arrived_this_step": new_arrivals.astype(int).tolist(),
            "nearest_obstacle_dists": nearest_obstacle_dists.tolist(),
            "nearest_uav_dists": nearest_uav_dists.tolist(),
            "episode_step": self.step_count,
            "components": components,
            "episode_components": self.episode_components,
            "reward_info": reward_info,
            "success_all": bool(np.all(self.arrived_mask)),
            "collision_any": bool(np.any(self.crashed_mask)),
            "arrival_mode": "segment_or_distance_arrival_priority",
            "reached_by_distance": reached_by_distance.astype(int).tolist(),
            "reached_by_segment": reached_by_segment.astype(int).tolist(),
            "moved_steps": self.moved_steps.astype(int).tolist(),
            "moved_this_step": moved_this_step.astype(int).tolist(),
            "first_collision_reason": self.first_collision_reason,
            "min_dist_to_goal": self.min_dist_to_goal.astype(float).tolist(),
            "per_agent_arrival_audit": per_agent_arrival_audit,
            "termination_reason": termination_reason,
            "max_steps": int(self.max_steps),
            "episode_limit": bool(self.step_count >= self.max_steps),
            "episode_limit_steps": int(self.max_steps),
        }

        self.prev_pos = prev_pos
        return obs, state, rewards, done, info
    
    # Compatibility helpers
    def get_obs(self):
        return [self._agent_obs(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return self._agent_obs(agent_id)

    def get_state(self):
        return self._get_state_vector()

    def get_stats(self):
        return {
            "success_all": float(np.all(self.arrived_mask)),
            "collision_any": float(np.any(self.crashed_mask)),
            "mean_min_goal_dist": float(np.mean(self.min_dist_to_goal)),
        }

    def render(self, save_path=None, traj=None, info=None, elev=24, azim=45):
        """Render a 3D snapshot/trajectory and optionally save as PNG.

        Args:
            save_path: PNG output path. If None, returns the figure object.
            traj: Optional trajectory sequence with shape [T, n_agents, 3].
            elev: Matplotlib 3D view elevation.
            azim: Matplotlib 3D view azimuth.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            self.log.warning("Matplotlib unavailable, skip rendering: %s", e)
            return None

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Draw environment bounds.
        ax.set_xlim(0.0, self.space_size)
        ax.set_ylim(0.0, self.space_size)
        ax.set_zlim(0.0, self.space_size)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("MultiUAV 3D Trajectory")

        # Draw goal center and radius sphere so "inside goal range" is directly visible.
        ax.scatter(
            [float(self.goal[0])],
            [float(self.goal[1])],
            [float(self.goal[2])],
            s=220,
            c="tab:green",
            marker="*",
            edgecolors="k",
            linewidths=0.6,
            label="goal center",
        )
        phi = np.linspace(0.0, np.pi, 18)
        theta_s = np.linspace(0.0, 2.0 * np.pi, 36)
        gx = self.goal[0] + self.goal_radius * np.outer(np.sin(phi), np.cos(theta_s))
        gy = self.goal[1] + self.goal_radius * np.outer(np.sin(phi), np.sin(theta_s))
        gz = self.goal[2] + self.goal_radius * np.outer(np.cos(phi), np.ones_like(theta_s))
        ax.plot_wireframe(gx, gy, gz, rstride=2, cstride=3, color="tab:green", alpha=0.16, linewidth=0.6)

        # Draw static cylinder obstacles using top/bottom rings and side vertical lines.
        theta = np.linspace(0.0, 2.0 * np.pi, 40)
        for obs in self.obstacles:
            x_ring = obs.x + obs.radius * np.cos(theta)
            y_ring = obs.y + obs.radius * np.sin(theta)
            z0 = np.full_like(theta, obs.z_min, dtype=float)
            z1 = np.full_like(theta, obs.z_max, dtype=float)

            ax.plot(x_ring, y_ring, z0, color="tab:red", alpha=0.55, linewidth=1.0)
            ax.plot(x_ring, y_ring, z1, color="tab:red", alpha=0.55, linewidth=1.0)

            for idx in range(0, len(theta), 6):
                ax.plot(
                    [x_ring[idx], x_ring[idx]],
                    [y_ring[idx], y_ring[idx]],
                    [obs.z_min, obs.z_max],
                    color="tab:red",
                    alpha=0.32,
                    linewidth=0.8,
                )

        traj_arr = None
        if traj is not None:
            try:
                traj_arr = np.asarray(traj, dtype=float)
                if traj_arr.ndim != 3 or traj_arr.shape[-1] < 3:
                    traj_arr = None
            except Exception:
                traj_arr = None

        if traj_arr is not None and traj_arr.shape[1] > 0:
            n_agents = int(traj_arr.shape[1])
            for aid in range(n_agents):
                xs = traj_arr[:, aid, 0]
                ys = traj_arr[:, aid, 1]
                zs = traj_arr[:, aid, 2]
                line_color = "tab:blue"
                ax.plot(xs, ys, zs, linewidth=1.8, color=line_color, label="uav_{} trajectory".format(aid + 1))

                # Fixed semantics:
                # - start: orange triangle
                # - end: blue circle
                ax.scatter([xs[0]], [ys[0]], [zs[0]], s=70, marker="^", c="tab:orange", edgecolors="k", linewidths=0.4,
                           label="start" if aid == 0 else None)
                ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=55, marker="o", c="tab:blue", edgecolors="k", linewidths=0.35,
                           label="end" if aid == 0 else None)

                # Mark the closest point to the 3D goal center to avoid 2D projection confusion.
                dists_to_goal = np.sqrt((xs - self.goal[0]) ** 2 + (ys - self.goal[1]) ** 2 + (zs - self.goal[2]) ** 2)
                min_idx = int(np.argmin(dists_to_goal))
                min_dist = float(dists_to_goal[min_idx])
                ax.scatter([xs[min_idx]], [ys[min_idx]], [zs[min_idx]], s=75, marker="D", c="magenta",
                           edgecolors="k", linewidths=0.35, label="closest-to-goal" if aid == 0 else None)
                ax.text(xs[min_idx], ys[min_idx], zs[min_idx], "dmin={:.2f}".format(min_dist), color="magenta", fontsize=8)

                # Boundary-hit marker (red X): inferred from first collision reason or trajectory touching bounds.
                collision_step = None
                collision_reason = None
                if isinstance(info, dict):
                    audit = info.get("per_agent_arrival_audit")
                    if isinstance(audit, (list, tuple)) and aid < len(audit) and isinstance(audit[aid], dict):
                        collision_step = int(audit[aid].get("first_collision_step", -1))
                        collision_reason = audit[aid].get("first_collision_reason")

                hit_idx = None
                if collision_step is not None and collision_step > 0:
                    hit_idx = max(0, min(len(xs) - 1, collision_step - 1))
                else:
                    eps = 1e-6
                    boundary_hits = np.where(
                        (xs <= eps) | (xs >= self.space_size - eps)
                        | (ys <= eps) | (ys >= self.space_size - eps)
                        | (zs <= eps) | (zs >= self.space_size - eps)
                    )[0]
                    if boundary_hits.size > 0:
                        hit_idx = int(boundary_hits[0])

                if hit_idx is not None and collision_reason in [None, "boundary", "uav_pair", "obstacle"]:
                    marker_label = "collision" if aid == 0 else None
                    ax.scatter([xs[hit_idx]], [ys[hit_idx]], [zs[hit_idx]], s=110, marker="x", c="red", linewidths=2.0,
                               label=marker_label)

                    # Annotate collision reason when available.
                    reason_txt = "collision"
                    if collision_reason == "boundary":
                        reason_txt = "boundary hit"
                    elif collision_reason == "obstacle":
                        reason_txt = "obstacle hit"
                    elif collision_reason == "uav_pair":
                        reason_txt = "uav collision"

                    ax.text(xs[hit_idx], ys[hit_idx], zs[hit_idx], reason_txt, color="red", fontsize=8)
        else:
            # Fallback: draw only current positions.
            for aid in range(self.n_agents):
                p = self.pos[aid]
                ax.scatter([p[0]], [p[1]], [p[2]], s=45, marker="o", label="uav_{}".format(aid + 1))

        try:
            ax.legend(loc="upper right", fontsize=8)
        except Exception:
            pass

        if save_path:
            try:
                out_dir = os.path.dirname(save_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                fig.savefig(save_path, dpi=160, bbox_inches="tight")
            finally:
                plt.close(fig)
            return save_path

        return fig

    def close(self):
        return None