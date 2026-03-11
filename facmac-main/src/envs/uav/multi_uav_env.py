import numpy as np, math
import logging
import importlib
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
            # If z_max is None, make it ceiling of a very large value (handled by env bounds)
            self.z_max = float(z_max) if z_max is not None else float('inf')

        def collides_point(self, p):
            # p is iterable with at least x,y,z
            dx = float(p[0]) - self.x
            dy = float(p[1]) - self.y
            dz = float(p[2])
            in_xy = (dx * dx + dy * dy) <= (self.radius * self.radius)
            in_z = (dz >= self.z_min) and (dz <= self.z_max)
            return in_xy and in_z

        def distance_to_point(self, p):
            p = np.asarray(p, dtype=float)
            dx = p[0] - self.x
            dy = p[1] - self.y
            radial = math.hypot(dx, dy) - self.radius
            if p[2] < self.z_min:
                dz = self.z_min - p[2]
            elif p[2] > self.z_max:
                dz = p[2] - self.z_max
            else:
                dz = 0.0
            if dz == 0.0:
                return radial
            radial_clamped = max(0.0, radial)
            return math.hypot(radial_clamped, dz)

try:
    mod = importlib.import_module('src.utils')
    sample_position = getattr(mod, 'sample_position')
    clamp_pos = getattr(mod, 'clamp_pos')
except Exception:
    def sample_position(rng, margin, space_size):
        # Sample (x,y,z) uniformly inside [margin, space_size-margin]
        low = margin
        high = max(margin, space_size - margin)
        return np.array([rng.uniform(low, high), rng.uniform(low, high), rng.uniform(low, high)], dtype=float)

    def clamp_pos(p, space_size):
        p = np.array(p, dtype=float)
        p = np.minimum(np.maximum(p, 0.0), float(space_size))
        return p

try:
    mod = importlib.import_module('src.envs.seed_utils')
    set_main_seed = getattr(mod, 'set_main_seed')
except Exception:
    def set_main_seed(s):
        try:
            s = int(s)
        except Exception:
            s = 0
        np.random.seed(s)

try:
    mod = importlib.import_module('src.envs.reward')
    compute_step_reward = getattr(mod, 'compute_step_reward')
except Exception:
    def compute_step_reward(positions, velocities, goal, obstacles, prev_positions=None, goal_radius=1.0, v_max=1.0, **kwargs):
        # Minimal placeholder: zero reward for every agent and a simple info dict.
        # Accept **kwargs to remain compatible when callers pass extra flags like
        # already_reached, so we don't raise unexpected-argument errors.
        n = int(np.shape(positions)[0]) if hasattr(positions, "__len__") else 0
        rewards = [0.0 for _ in range(n)]
        info = {"placeholder_reward": True, 'components': None}
        return rewards, info

def _segment_intersects_sphere(p0, p1, center, radius):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    c = np.asarray(center, dtype=float)
    v = p1 - p0
    w = c - p0
    v_dot_v = np.dot(v, v)
    if v_dot_v <= 1e-12:
        return np.linalg.norm(p0 - c) <= radius
    t = np.dot(w, v) / v_dot_v
    t_clamped = max(0.0, min(1.0, t))
    closest = p0 + t_clamped * v
    return np.linalg.norm(closest - c) <= radius

# New: check whether a 3D segment intersects a vertical cylinder obstacle (center x,y, radius, z_min,z_max)
def _segment_intersects_cylinder(p0, p1, cyl):
    """Return True if segment p0->p1 intersects cylinder `cyl` (has x,y,radius,z_min,z_max)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    # 2D closest point on segment to cylinder center in XY plane
    c_xy = np.array([float(cyl.x), float(cyl.y)])
    v_xy = p1[:2] - p0[:2]
    w = c_xy - p0[:2]
    v_dot_v = np.dot(v_xy, v_xy)
    if v_dot_v <= 1e-12:
        # degenerate segment: check point and z-range
        d_xy = np.linalg.norm(p0[:2] - c_xy)
        if d_xy <= float(cyl.radius):
            z = p0[2]
            return (z >= float(cyl.z_min)) and (z <= float(cyl.z_max))
        return False
    t = np.dot(w, v_xy) / v_dot_v
    t_clamped = max(0.0, min(1.0, t))
    closest_xy = p0[:2] + t_clamped * v_xy
    closest_z = p0[2] + t_clamped * (p1[2] - p0[2])
    dist_xy = np.linalg.norm(closest_xy - c_xy)
    if dist_xy <= float(cyl.radius) and (closest_z >= float(cyl.z_min)) and (closest_z <= float(cyl.z_max)):
        return True
    return False

class MultiUAVEnv:
    def __init__(self, cfg, main_seed=0, subseed=0, enable_dynamic=False):
        self.cfg = cfg
        self.n_agents = cfg['n_agents']
        self.space_size = cfg['space_size']
        self.n_static = cfg['n_static_obstacles']
        self.max_steps = cfg['max_steps']
        self.dt = cfg.get('dt',0.2)     #时间步长
        self.v_max = cfg.get('v_max',2.0)
        self.goal_radius = cfg.get('goal_radius',1.0)   #目标点半径
        set_main_seed(main_seed)    # 主随机种子
        self.master_seed = int(main_seed)
        self.subseed = int(subseed) # 子随机种子
        self.rng = np.random.RandomState(self.master_seed + self.subseed)   # 环境随机数生成器
        self.enable_dynamic = enable_dynamic    # 是否启用动态障碍物

        # Configurable behavior: fix world across episodes or randomize each reset
        # Default: randomize each reset (backward compatible)
        self.randomize_every_reset = bool(cfg.get('randomize_every_reset', True))
        # If True, initial positions sampled once and kept across resets when randomize_every_reset is False
        self.fixed_initial_positions = bool(cfg.get('fixed_initial_positions', False))

        # Directory to save per-episode summaries (can be configured in cfg['results_dir'])
        self.log = logging.getLogger(cfg.get('logger_name', 'MultiUAVEnv'))
        self.log_env_reset = bool(cfg.get('log_env_reset', False))
        self.log_goal_event = bool(cfg.get('log_goal_event', False))
        self.log_episode_save = bool(cfg.get('log_episode_save', False))

        self.results_dir = os.path.abspath(cfg.get('results_dir', 'results'))
        self.save_episode_json = bool(cfg.get('save_episode_json', True))
        self.max_episode_summaries = cfg.get('results_keep_last', None)
        self.summary_csv_path = cfg.get('summary_csv_path', None)
        if self.summary_csv_path:
            self.summary_csv_path = os.path.abspath(self.summary_csv_path)
            os.makedirs(os.path.dirname(self.summary_csv_path), exist_ok=True)
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except Exception:
            pass

        # Initialize the world (samples obstacles, goal and initial positions)
        self._init_world()

        # Per-agent flags
        self.arrived_mask = np.zeros(self.n_agents, dtype=bool)  # 记录每个agent是否已到达目标
        self.colliding_mask = np.zeros(self.n_agents, dtype=bool)  # 逐agent碰撞占用状态（用于事件计数）
        self.crashed_mask = np.zeros(self.n_agents, dtype=bool)    # 新增：永久坠毁状态

        # Event tracking: record first step index when each agent arrives or collides (-1 means never)
        self.first_arrival_step = np.full(self.n_agents, -1, dtype=int)
        self.first_collision_step = np.full(self.n_agents, -1, dtype=int)

        # Compatibility attributes used by the wrapper UAVEnv
        self.num_uav = self.n_agents
        # observation dimension matches _agent_obs output length
        self.obs_dim = 15
        # state dimension is flattened positions + goal (n_agents*3 + 3) + obstacles (n_static*5)
        self.state_dim = self.n_agents * 3 + 3 + self.n_static * 5
        # action dimension (3D continuous velocity)
        self.action_dim = 3

        # Per-episode reward recording (per-step total reward and per-agent rewards)
        self.episode_rewards = []  # list of per-step total rewards
        self.episode_per_agent_rewards = []  # list of per-step lists (per-agent rewards)
        # Per-step reward decomposition (list per step of per-agent component dicts)
        self.episode_components = []

    def _init_world(self):
        # 设置障碍物采样时距离边界的最小间距
        margin = 5.0
        self.obstacles = []
        for i in range(self.n_static):
            x = self.rng.uniform(margin, self.space_size - margin)
            y = self.rng.uniform(margin, self.space_size - margin)
            radius = self.rng.uniform(3.0, 8.0)
            z_min = self.rng.uniform(0.0, self.space_size * 0.5)
            # 确保 z_max 的下界不会超过空间大小，避免越界
            z_lower = z_min + 2.0
            if z_lower >= self.space_size:
                z_max = self.space_size
            else:
                z_max = self.rng.uniform(z_lower, self.space_size)
            self.obstacles.append(CylinderObstacle(x, y, radius, z_min=z_min, z_max=z_max))

        # 采样目标点，确保不与障碍物碰撞且距离所有障碍物足够远
        safe_dist = self.cfg.get('reward', {}).get('safe_dist_obs', 2.0)
        max_attempts = 100
        candidate = None
        for attempt in range(max_attempts):
            candidate = sample_position(self.rng, margin, self.space_size)
            too_close = False
            for obs in self.obstacles:
                dist_xy = np.linalg.norm(candidate[:2] - np.array([obs.x, obs.y]))
                if dist_xy < obs.radius + safe_dist + margin:
                    too_close = True
                    break
            if not too_close:
                self.goal = candidate
                break
        else:
            # 如果多次采样都失败，直接用最后一次（极端情况）
            self.goal = candidate

        # 初始化无人机位置和速度
        self.pos = np.zeros((self.n_agents,3), dtype=float)
        self.vel = np.zeros((self.n_agents,3), dtype=float)
        for i in range(self.n_agents):
            self.pos[i] = sample_position(self.rng, margin, self.space_size)
            self.vel[i] = np.zeros(3, dtype=float)

        # If user requested fixed initial positions, store a copy to reuse in reset
        self._fixed_start_pos = np.copy(self.pos) if self.fixed_initial_positions else None

        self.dynamic_objs = []
        self.step_count = 0
        self.prev_pos = None
        self.arrived_mask = np.zeros(self.n_agents, dtype=bool)  # 初始化到达标志
        self.colliding_mask = np.zeros(self.n_agents, dtype=bool)  # 初始化碰撞占用标志
        self.crashed_mask = np.zeros(self.n_agents, dtype=bool)  # 重置坠毁状态

    # TODO(随机种子)：输出随机主种子和子种子
    # print(main_seed)
    # print(sub_seed)
    # TODO(解决完episode输出不完全问题): 输出障碍物、无人机和目标点信息

    def reset(self, subseed=None):
        # 障碍物和目标点只由master_seed决定，环境构造时已初始化，不再变
        # If randomize_every_reset is False and we have stored fixed start positions, reuse them
        if not self.randomize_every_reset and self._fixed_start_pos is not None:
            self.pos = np.copy(self._fixed_start_pos)
            self.vel = np.zeros_like(self.pos)
        else:
            # Otherwise sample new initial positions (optionally using subseed RNG)
            if subseed is not None:
                rng = np.random.RandomState(subseed)
            else:
                rng = self.rng
            margin = 5.0  # 与_init_world保持一致
            self.pos = [sample_position(rng, margin, self.space_size) for _ in range(self.n_agents)]
            self.vel = [np.zeros(3) for _ in range(self.n_agents)]
            self.pos = np.array(self.pos)
            self.vel = np.array(self.vel)
            # If fixed_initial_positions is enabled but user still allows randomize_every_reset,
            # keep the first-sampled positions as fixed reference
            if self.fixed_initial_positions and self._fixed_start_pos is None:
                self._fixed_start_pos = np.copy(self.pos)

        self.prev_pos = None
        self.step_count = 0
        self.arrived_mask = np.zeros(self.n_agents, dtype=bool)  # 重置到达标志
        self.colliding_mask = np.zeros(self.n_agents, dtype=bool)  # 重置碰撞占用标志
        self.crashed_mask = np.zeros(self.n_agents, dtype=bool)  # 重置坠毁状态，避免跨episode泄漏
        # Reset event tracking
        self.first_arrival_step[:] = -1
        self.first_collision_step[:] = -1
        # Reset per-episode reward buffers
        self.episode_rewards = []
        self.episode_per_agent_rewards = []
        # 输出环境初始化信息
        if self.log_env_reset:
            try:
                self.log.info(f"reset positions: {self.pos}")
                self.log.info(f"reset goal: {self.goal}")
            except Exception:
                pass
        return self._get_obs_state()

    def _in_bounds(self,p):
        return clamp_pos(p, self.space_size)

    def _is_out_of_bounds(self, p):
        return np.any(p < 0.0) or np.any(p > float(self.space_size))

    # 用于碰撞检测
    def _collides_any(self,p):
        for c in self.obstacles:
            if c.collides_point(p):
                return True
        for d in self.dynamic_objs:
            if np.linalg.norm(d['pos'] - p) <= d['radius']:
                return True
        return False

    def _simple_ray_cast(self, pos, direction, max_range=None):
        if max_range is None:
            max_range = float(self.space_size)

        p = np.array(pos, dtype=float)
        d = np.array(direction, dtype=float)
        norm_d = np.linalg.norm(d)
        if norm_d > 1e-6:
            d = d / norm_d
        else:
            return max_range

        t = 0.0
        # Use iterative sphere tracing
        for _ in range(30):
            current_p = p + t * d
            min_dist = float('inf')

            # Check static obstacles
            for obs in self.obstacles:
                # distance_to_point returns signed distance if inside (negative)
                dist = obs.distance_to_point(current_p)
                if dist < min_dist:
                    min_dist = dist

            # Check dynamic objects
            for dobj in self.dynamic_objs:
                center_dist = np.linalg.norm(current_p - dobj['pos'])
                dist = center_dist - dobj['radius']
                if dist < min_dist:
                    min_dist = dist

            # If inside or very close, assume hit
            if min_dist <= 0.01:
                return t

            t += min_dist
            if t >= max_range:
                return max_range
        return max_range

    def _agent_obs(self,idx):
        pos = self.pos[idx]
        pos_norm = pos / self.space_size
        v = self.vel[idx] / self.v_max

        # Modification 1: Normalized relative goal vector
        vec_to_goal = (self.goal - pos) / self.space_size

        # Modification 2: 6-direction Lidar sensors
        directions = [
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ]
        # Use space_size as max sensing range for normalization
        sensor_range = float(self.space_size)
        sensor_readings = []
        for d in directions:
            dist = self._simple_ray_cast(pos, d, max_range=sensor_range)
            sensor_readings.append(dist / sensor_range)

        return np.concatenate([
            pos_norm, v, vec_to_goal, np.array(sensor_readings, dtype=float)
        ])

    def _get_obs_state(self):
        obs = [self._agent_obs(i) for i in range(self.n_agents)]

        # 1. 无人机位置归一化
        state_uav = self.pos.flatten() / self.space_size
        # 2. 目标点归一化
        state_goal = self.goal.flatten() / self.space_size

        # 3. 提取并归一化障碍物信息
        state_obs_list = []
        for obs_obj in self.obstacles:
            state_obs_list.extend([
                obs_obj.x / self.space_size,
                obs_obj.y / self.space_size,
                obs_obj.radius / self.space_size,
                obs_obj.z_min / self.space_size,
                obs_obj.z_max / self.space_size
            ])
        state_obs = np.array(state_obs_list, dtype=float)

        # 拼接全新的全局状态
        state = np.concatenate([state_uav, state_goal, state_obs])
        return obs, state

    def step(self, actions):
        self.step_count += 1
        prev_pos = np.copy(self.pos)
        # Use v_max as the action scaling so policy outputs in [-1,1] map to velocity range [-v_max, v_max]
        action_scale = float(self.v_max)
        # 更新速度和位置
        for i in range(self.n_agents):
            if self.arrived_mask[i] or self.crashed_mask[i]:
                self.vel[i] = np.zeros(3, dtype=float)
                self.pos[i] = prev_pos[i]
                continue
            a = np.array(actions[i], dtype=float) * action_scale
            mag = np.linalg.norm(a)
            if mag > self.v_max:
                a = a / mag * self.v_max
            self.vel[i] = a
            self.pos[i] = self._in_bounds(self.pos[i] + self.vel[i] * self.dt)
        # 更新arrived_mask：首次到达标志
        dists = np.linalg.norm(self.pos - self.goal, axis=1)
        # newly_arrived = (dists <= self.goal_radius) & (~self.arrived_mask)
        newly_arrived = (dists <= self.goal_radius) & (~self.arrived_mask) & (~self.crashed_mask)
        # 先记录老的 mask，再更新，以方便计算 reached_new
        old_arrived = self.arrived_mask.copy()
        self.arrived_mask[newly_arrived] = True

        # 检查穿越目标球体的情况
        for i in range(self.n_agents):
            if not self.arrived_mask[i] and not self.crashed_mask[i] and _segment_intersects_sphere(prev_pos[i],self.pos[i],self.goal,self.goal_radius):
            # if not self.arrived_mask[i] and _segment_intersects_sphere(prev_pos[i], self.pos[i], self.goal, self.goal_radius):
                self.arrived_mask[i] = True
                newly_arrived[i] = True
                if self.log_goal_event:
                    try:
                        self.log.info(f"step {self.step_count}: agent {i} reached goal (dist={dists[i]:.3f})")
                    except Exception:
                        pass

        # Record first arrival steps
        for i in range(self.n_agents):
            if newly_arrived[i] and self.first_arrival_step[i] == -1:
                self.first_arrival_step[i] = int(self.step_count)

        # 计算碰撞：既检测当前位置是否在障碍内，也检测移动段是否穿过障碍（高速穿越）
        coll_now = np.zeros(self.n_agents, dtype=bool)
        safe_uav_dist = self.cfg.get('safe_uav_dist', 0.5)

        for i in range(self.n_agents):
            # 如果已经坠毁或到达，不产生新的碰撞事件
            if self.arrived_mask[i] or self.crashed_mask[i]:
                continue

            coll = False

            # 1. 越界检测
            intended_pos = prev_pos[i] + self.vel[i] * self.dt
            if self._is_out_of_bounds(intended_pos):
                coll = True

            # 2. 静态/动态环境障碍物检测
            if not coll:
                coll = self._collides_any(self.pos[i])

            # 3. 高速穿越线段检测
            if (not coll) and (prev_pos is not None):
                for obs in self.obstacles:
                    try:
                        if _segment_intersects_cylinder(prev_pos[i], self.pos[i], obs):
                            coll = True
                            break
                    except Exception:
                        continue
                if (not coll) and len(self.dynamic_objs) > 0:
                    for d in self.dynamic_objs:
                        try:
                            if _segment_intersects_sphere(prev_pos[i], self.pos[i], d['pos'], d['radius']):
                                coll = True
                                break
                        except Exception:
                            continue

            # 4. 无人机互撞检测 (Inter-UAV Collision)
            if not coll:
                for j in range(self.n_agents):
                    if i == j:
                        continue
                    if self.arrived_mask[j]:
                        continue
                    dist_ij = np.linalg.norm(self.pos[i] - self.pos[j])
                    if dist_ij < safe_uav_dist:
                        coll = True
                        break

            coll_now[i] = coll

        collisions_step = coll_now.astype(int).tolist()
        # 确保只有没坠毁的无人机才会触发“新碰撞”
        collisions_new_mask = (~self.colliding_mask) & coll_now & (~self.crashed_mask)
        collisions_new = collisions_new_mask.astype(int).tolist()

        self.colliding_mask = coll_now  # 更新占用状态
        self.crashed_mask |= collisions_new_mask  # <--- 核心：真正把发生碰撞的无人机标记为永久坠毁
        # -----------------------------------

        # Record first collision steps
        for i in range(self.n_agents):
            if collisions_new_mask[i] and self.first_collision_step[i] == -1:
                self.first_collision_step[i] = int(self.step_count)

        for d in self.dynamic_objs:
            d['pos'] = self._in_bounds(d['pos'] + d['vel'] * self.dt)
        rewards, info = compute_step_reward(self.pos, self.vel, self.goal, self.obstacles,
                                            prev_positions=prev_pos,
                                            goal_radius=self.goal_radius, v_max=self.v_max,
                                            already_reached=self.arrived_mask.astype(int).tolist())
        # record per-step reward decomposition if provided by reward function
        # collision_penalty = -5.0  # 建议先设为 -5.0，不要直接用 -30.0，防止梯度爆炸

        for i in range(self.n_agents):
            # 使用 collisions_new_mask 确保只在撞击的那一帧扣分
            # if collisions_new_mask[i]:
            #     rewards[i] += collision_penalty
            # if self.crashed_mask[i] and not collisions_new_mask[i]:
            #     rewards[i] = 0.0  # 变成残骸后，不再产生任何奖励和惩罚

                # 同步更新 info 里的组件信息，方便日志查看
                if 'components' in info and info['components'] is not None:
                    # Guard against malformed/short component lists from the reward function.
                    comps = info['components']
                    # Only attempt to index if comps is a sequence and contains an entry for this agent
                    if isinstance(comps, (list, tuple)) and i < len(comps):
                        try:
                            if isinstance(comps[i], dict):
                                comps[i]['coll_reward'] = self.cfg.get('reward', {}).get('collision_penalty', -5.0)
                        except Exception:
                            pass


        # try:
        #     comps = info.get('components', None)
        #     # 仅在“新发生的碰撞事件”（由 collisions_new_mask 指示）时，记录一次碰撞惩罚，避免持续扣分
        #     if isinstance(comps, list) and len(comps) == self.n_agents:
        #         try:
        #             for i in range(self.n_agents):
        #                 if collisions_new_mask[i]:  # 这一时刻刚进入碰撞
        #                     if isinstance(comps[i], dict):
        #                         comps[i]['coll_reward'] = -30.0
        #                 else:
        #                     # 非新碰撞事件，不覆盖奖励函数自身的 coll_reward（通常为 0）
        #                     pass
        #         except Exception:
        #             pass
        #     # make JSON-friendly copy: convert numpy types if any to python floats/ints
        #     if comps is None:
        #         self.episode_components.append(None)
        #     else:
        #         cleaned = []
        #         for comp in comps:
        #             # comp is expected to be a dict of numeric components
        #             if not isinstance(comp, dict):
        #                 cleaned.append(comp)
        #                 continue
        #             cd = {}
        #             for kk, vv in comp.items():
        #                 try:
        #                     # numpy types and python numbers -> float or int
        #                     if isinstance(vv, (np.floating, float)):
        #                         cd[kk] = float(vv)
        #                     elif isinstance(vv, (np.integer, int)):
        #                         cd[kk] = int(vv)
        #                     else:
        #                         # fallback: try numeric conversion, otherwise string
        #                         try:
        #                             cd[kk] = float(vv)
        #                         except Exception:
        #                             try:
        #                                 cd[kk] = int(vv)
        #                             except Exception:
        #                                 cd[kk] = vv
        #                 except Exception:
        #                     cd[kk] = vv
        #             cleaned.append(cd)
        #         self.episode_components.append(cleaned)
        # except Exception:
        #     # If anything goes wrong, still keep episode_components aligned with steps
        #     try:
        #         self.episode_components.append(None)
        #     except Exception:
        #         pass

        # 只在首次到达时统计reached（与环境到达标记一致）
        info['reached'] = self.arrived_mask.astype(int).tolist()
        # 附加：到达新发生的掩码与距离信息，便于调试/训练侧精确统计
        info['reached_new'] = newly_arrived.astype(int).tolist()
        info['arrived_mask'] = self.arrived_mask.astype(int).tolist()
        info['dist_to_goal'] = dists.astype(float).tolist()
        # 附加：碰撞占用（逐步）与碰撞事件（进入时）
        info['collisions_step'] = collisions_step
        info['collisions_new'] = collisions_new
        info['crashed'] = self.crashed_mask.astype(int).tolist()
        done = False
        if np.all(self.arrived_mask | self.crashed_mask):
            done = True
            if all(self.arrived_mask):
                info['success'] = True
        if self.step_count >= self.max_steps:
            done = True

        # If the episode finished, finalize and save a per-episode summary
        if done:
            try:
                self._finalize_episode(info)
            except Exception:
                pass

        # Record per-step total and per-agent rewards
        self.episode_rewards.append(np.sum(rewards))
        self.episode_per_agent_rewards.append(rewards)

        obs, state = self._get_obs_state()
        self.prev_pos = prev_pos
        return obs, state, rewards, done, info

    def add_dynamic_sphere(self, pos, vel, radius=1.0):
        self.dynamic_objs.append({'pos': np.array(pos, dtype=float), 'vel': np.array(vel, dtype=float), 'radius': float(radius)})

    def render(self, save_path=None, traj=None, pause_time=0.01):
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa
        except Exception as e:
            self.log.warning('matplotlib required for render: %s', e)
            return

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        # 统计轨迹步数 T（若提供 traj）
        T_steps = None

        for c in self.obstacles:
            theta = np.linspace(0, 2 * np.pi, 50)
            z = np.linspace(0, self.space_size, 20)
            theta_grid, z_grid = np.meshgrid(theta, z)
            xs = c.x + c.radius * np.cos(theta_grid)
            ys = c.y + c.radius * np.sin(theta_grid)
            zs = z_grid
            ax.plot_surface(xs, ys, zs, color='red', alpha=0.3, linewidth=0)

        xs = self.pos[:,0]
        ys = self.pos[:,1]
        zs = self.pos[:,2]
        ax.scatter(xs, ys, zs, c='blue', s=30, label='UAVs')
        ax.scatter([self.goal[0]], [self.goal[1]], [self.goal[2]], c='green', s=80, marker='*', label='Goal')

        # 绘制目标点球体（goal_radius）
        try:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = self.goal[0] + self.goal_radius * np.cos(u) * np.sin(v)
            y = self.goal[1] + self.goal_radius * np.sin(u) * np.sin(v)
            z = self.goal[2] + self.goal_radius * np.cos(v)
            ax.plot_surface(x, y, z, color='green', alpha=0.15, linewidth=0)
        except Exception as e:
            self.log.warning('Failed to draw goal sphere: %s', e)

        # 绘制轨迹（使用公共 normalize helper）
        if traj is not None:
            from matplotlib.lines import Line2D
            # Safe import for normalize_traj_to_agent_trajs (may not exist in some packaging setups)
            try:
                mod = importlib.import_module('src.utils.trajectory')
                normalize_traj_to_agent_trajs = getattr(mod, 'normalize_traj_to_agent_trajs')
            except Exception:
                def normalize_traj_to_agent_trajs(traj, n_agents=None):
                    try:
                        arr = np.array(traj)
                        if arr.ndim == 3 and arr.shape[2] == 3:
                            T, N, _ = arr.shape
                            if n_agents is None:
                                n_agents = N
                            out = []
                            for i in range(n_agents):
                                out.append(arr[:, i, :])
                            return out
                    except Exception:
                        pass
                    if isinstance(traj, (list, tuple)) and len(traj) > 0:
                        try:
                            return [np.array(a, dtype=float) for a in traj]
                        except Exception:
                            return None
                    return None

            # 尝试从 traj 推断步数 T
            try:
                arr_try = np.array(traj)
                if isinstance(arr_try, np.ndarray) and arr_try.ndim == 3 and arr_try.shape[2] == 3:
                    T_steps = arr_try.shape[0]
                elif isinstance(traj, (list, tuple)) and len(traj) > 0:
                    first = np.array(traj[0])
                    if first.ndim == 2 and first.shape[1] == 3:
                        T_steps = len(traj)
            except Exception:
                T_steps = None

            n_agents = getattr(self, 'n_agents', None)
            agent_trajs = normalize_traj_to_agent_trajs(traj, n_agents=n_agents)
            legend_handles = []
            if agent_trajs is not None:
                N = len(agent_trajs)
                cmap = plt.get_cmap('tab10')
                colors = cmap(np.linspace(0, 1, max(1, N)))
                for i, a_traj in enumerate(agent_trajs):
                    try:
                        a = np.array(a_traj, dtype=float)
                    except Exception:
                        continue
                    if a.ndim != 2 or a.shape[1] != 3:
                        continue
                    col = colors[i]
                    ax.plot(a[:,0], a[:,1], a[:,2], color=col, lw=2.5, alpha=0.95)
                    ax.scatter(float(a[0,0]), float(a[0,1]), float(a[0,2]), color=col, s=40, marker='o', edgecolors='k')
                    ax.scatter(float(a[-1,0]), float(a[-1,1]), float(a[-1,2]), color=col, s=70, marker='*', edgecolors='k')
                    try:
                        ax.text(float(a[-1,0]), float(a[-1,1]), float(a[-1,2]) + 0.6, f'{i}', fontsize=9, color=col)
                    except Exception:
                        pass
                    legend_handles.append(Line2D([0],[0], color=col, lw=3, label=f'UAV {i}'))
            else:
                # fallback: try original behavior (each element is agent traj)
                try:
                    for i, agent_traj in enumerate(traj):
                        agent_traj = np.array(agent_traj, dtype=float)
                        if agent_traj.ndim == 2 and agent_traj.shape[1] == 3:
                            ax.plot(agent_traj[:,0], agent_traj[:,1], agent_traj[:,2], alpha=0.85, linewidth=2.2, label=f'UAV {i}')
                            legend_handles.append(Line2D([0],[0], color='C{}'.format(i%10), lw=3, label=f'UAV {i}'))
                    # 若走到这里，T_steps 无法从 numpy 判定，按每个 agent 的长度推断
                    if T_steps is None and len(traj) > 0:
                        T_steps = len(traj[0]) if hasattr(traj[0], '__len__') else None
                except Exception:
                    pass

            if legend_handles:
                try:
                    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
                except Exception:
                    ax.legend()

        ax.set_xlim(0,self.space_size)
        ax.set_ylim(0,self.space_size)
        ax.set_zlim(0,self.space_size)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 将步数作为标题，帮助诊断“只有几步”的问题
        if T_steps is not None:
            ax.set_title(f'Trajectory steps: T={T_steps}')

        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
        return fig, ax

    def _finalize_episode(self, info):
        # Build a per-episode summary and save to self.results_dir
        try:
            n = int(self.n_agents)
            reached = np.array(info.get('reached', self.arrived_mask.astype(int).tolist()), dtype=int)
            per_collision_flag = (np.array(self.first_collision_step) >= 0).astype(int)

            summary = {
                'per_agent_reached': reached.tolist(),
                'per_agent_first_arrival_step': [int(x) for x in self.first_arrival_step.tolist()],
                'per_agent_collision_flag': per_collision_flag.tolist(),
                'per_agent_first_collision_step': [int(x) for x in self.first_collision_step.tolist()],
                'episode_steps': int(self.step_count),
                'timestamp': int(time.time()),
                'master_seed': int(getattr(self, 'master_seed', -1)),
                'subseed': int(getattr(self, 'subseed', -1))
            }

            # 指标计算
            success_rate = float(np.sum(reached)) / max(1, n)
            collision_rate = float(np.sum(per_collision_flag)) / max(1, n)
            steps = np.array(self.first_arrival_step, dtype=float)
            # 未到达用 max_steps 代替，方便计算平均延迟
            steps[steps < 0] = float(self.max_steps)
            avg_latency_rate = float(np.mean(steps)) / float(self.max_steps)

            summary.update({
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'avg_latency_rate': avg_latency_rate
            })

            # Add explicit episode-level flags for clarity
            try:
                episode_success_all = bool(np.all(reached == 1))
                episode_collision_any = bool(np.any(per_collision_flag == 1))
                # success with no collision for all agents
                episode_success_no_collision_all = bool(episode_success_all and (not episode_collision_any))
                summary['episode_success_all'] = episode_success_all
                summary['episode_success_no_collision_all'] = episode_success_no_collision_all
                summary['episode_collision_any'] = episode_collision_any
            except Exception:
                pass

            # --- New: mutually exclusive outcome breakdown ---
            # Note: success, collision and avg latency are not mutually exclusive by design.
            # An agent may both collide and later reach the goal, so success_rate + collision_rate
            # does not (and should not) necessarily equal 1. For clearer diagnostics we add
            # mutually exclusive categories below.
            try:
                # Make sure arrays are at least 1-D and of length n
                reached = np.atleast_1d(reached)
                per_collision_flag = np.atleast_1d(per_collision_flag)
                arrived_mask_bool = np.asarray(reached == 1, dtype=bool)
                collision_mask_bool = np.asarray(per_collision_flag == 1, dtype=bool)
                if arrived_mask_bool.size != n:
                    arrived_mask_bool = np.resize(arrived_mask_bool, n)
                if collision_mask_bool.size != n:
                    collision_mask_bool = np.resize(collision_mask_bool, n)

                # Mutually exclusive outcomes per agent (use logical ops for clarity)
                success_no_collision = np.logical_and(arrived_mask_bool, np.logical_not(collision_mask_bool))
                success_with_collision = np.logical_and(arrived_mask_bool, collision_mask_bool)
                collision_only = np.logical_and(np.logical_not(arrived_mask_bool), collision_mask_bool)
                timeout_no_collision = np.logical_and(np.logical_not(arrived_mask_bool), np.logical_not(collision_mask_bool))

                per_agent_outcome = []
                for i in range(n):
                    if bool(success_no_collision[i]):
                        per_agent_outcome.append('success_no_collision')
                    elif bool(success_with_collision[i]):
                        per_agent_outcome.append('success_with_collision')
                    elif bool(collision_only[i]):
                        per_agent_outcome.append('collision_only')
                    else:
                        per_agent_outcome.append('timeout_no_collision')

                summary['per_agent_outcome'] = per_agent_outcome

                # Rates for these mutually exclusive categories
                summary['success_no_collision_rate'] = float(np.sum(success_no_collision)) / max(1, n)
                summary['success_with_collision_rate'] = float(np.sum(success_with_collision)) / max(1, n)
                summary['collision_only_rate'] = float(np.sum(collision_only)) / max(1, n)
                summary['timeout_no_collision_rate'] = float(np.sum(timeout_no_collision)) / max(1, n)
            except Exception:
                # Keep the summary if the additional diagnostics fail
                pass

            # Include per-step rewards if collected
            try:
                if hasattr(self, 'episode_rewards') and len(self.episode_rewards) > 0:
                    summary['episode_rewards'] = [float(x) for x in self.episode_rewards]
                if hasattr(self, 'episode_per_agent_rewards') and len(self.episode_per_agent_rewards) > 0:
                    # ensure nested lists of floats
                    summary['episode_per_agent_rewards'] = [[float(v) for v in step] for step in self.episode_per_agent_rewards]
                    # also include per-agent total reward across episode
                    per_agent_totals = np.sum(np.array(self.episode_per_agent_rewards, dtype=float), axis=0).tolist()
                    summary['per_agent_total_reward'] = [float(x) for x in per_agent_totals]
                # include components breakdown if collected
                if hasattr(self, 'episode_components') and len(self.episode_components) > 0:
                    # keep None entries as null in JSON; components already cleaned per-step
                    summary['episode_components'] = self.episode_components
            except Exception:
                pass

            # 将 summary 加入 info 返回给上层
            for k, v in summary.items():
                info[k] = v

            # 保存为唯一文件
            if self.save_episode_json:
                try:
                    safe_ts = time.time_ns()
                except Exception:
                    safe_ts = int(time.time() * 1e6)
                pid = os.getpid()
                if not hasattr(self, '_episode_save_counter'):
                    self._episode_save_counter = 0
                self._episode_save_counter += 1
                fname = os.path.join(
                    self.results_dir,
                    f'episode_{safe_ts}_{pid}_{summary["master_seed"]}_{summary["subseed"]}_{self._episode_save_counter}.json'
                )
                try:
                    with open(fname, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    if self.log_episode_save:
                        try:
                            self.log.info(f"episode summary saved to {fname}")
                        except Exception:
                            pass
                    self._prune_episode_summaries()
                except Exception as e:
                    self.log.error('failed to write episode summary: %s', e)
        except Exception:
            # 对应最外层的 _finalize_episode 的 try
            pass

    def _prune_episode_summaries(self):
        """Keep at most self.max_episode_summaries most recent files in results_dir.
        This is a lightweight helper used by _finalize_episode; if max is None do nothing.
        """
        try:
            m = self.max_episode_summaries
            if m is None:
                return
            # list files that look like our episode_*.json summaries
            files = [os.path.join(self.results_dir, fn) for fn in os.listdir(self.results_dir) if fn.startswith('episode_') and fn.endswith('.json')]
            if len(files) <= m:
                return
            files.sort(key=lambda p: os.path.getmtime(p))
            to_remove = files[0:len(files)-m]
            for p in to_remove:
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception:
            pass

