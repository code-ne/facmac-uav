# from typing import List, Tuple, Dict, Optional
# import numpy as np
#
#
# def compute_step_reward(
#     positions: np.ndarray,
#     velocities: np.ndarray,
#     goal: np.ndarray,
#     obstacles: List[object],
#     prev_positions: np.ndarray = None,
#     goal_radius: float = 1.0,
#     v_max: float = 1.0,
#     already_reached: Optional[List[int]] = None
# ) -> Tuple[List[float], Dict]:
#     """
#     FACMAC-compatible reward function.
#     Returns:
#         rewards: List[float] (per-agent)
#         info: dict with keys ['reached', 'collisions', 'components']
#     """
#
#     positions = np.asarray(positions)
#     velocities = np.asarray(velocities)
#     N = positions.shape[0]
#
#     rewards = np.zeros(N, dtype=float)
#     reached = np.zeros(N, dtype=int)
#     collisions = np.zeros(N, dtype=int)
#
#     # --- already reached mask ---
#     if already_reached is None:
#         already_reached_mask = np.zeros(N, dtype=bool)
#     else:
#         already_reached_mask = np.asarray(already_reached, dtype=bool)
#         if already_reached_mask.shape[0] != N:
#             already_reached_mask = np.zeros(N, dtype=bool)
#
#     # =============================
#     # 🔧 Reward 超参数（稳定版）
#     # =============================
#     PROGRESS_SCALE = 5.0          # 🚩 主线奖励：向目标推进
#     COLLISION_PENALTY = -2.0      # 碰撞惩罚（不要太大）
#     STAGNATION_PENALTY = -0.1     # 完全不动惩罚
#     ARRIVAL_BONUS = 8.0           # 到达奖励（episode 级信号）
#     REWARD_SCALE = 10.0           # 最终归一化（非常重要）
#
#     goal = np.asarray(goal)
#     obstacles = obstacles or []
#
#     components = []
#
#     for i in range(N):
#         comp = {
#             "progress_reward": 0.0,
#             "collision_penalty": 0.0,
#             "stagnation_penalty": 0.0,
#             "arrival_reward": 0.0,
#             "apf_reward": 0.0,     # 🚩 APF 预留
#             "total": 0.0
#         }
#
#         # 已到达的不再参与
#         if already_reached_mask[i]:
#             reached[i] = 1
#             components.append(comp)
#             continue
#
#         pos = positions[i]
#         vel = velocities[i]
#         dist = np.linalg.norm(pos - goal)
#
#         # ==================================================
#         # 1️⃣ Progress Reward（最核心）
#         # ==================================================
#         if prev_positions is not None:
#             prev_pos = prev_positions[i]
#             prev_dist = np.linalg.norm(prev_pos - goal)
#             progress = prev_dist - dist    # 变近才是正
#             comp["progress_reward"] = PROGRESS_SCALE * progress
#
#         # ==================================================
#         # 2️⃣ 碰撞惩罚
#         # ==================================================
#         for obs in obstacles:
#             fn = getattr(obs, "collides_point", None)
#             if callable(fn) and fn(pos):
#                 collisions[i] = 1
#                 comp["collision_penalty"] = COLLISION_PENALTY
#                 break
#
#         # ==================================================
#         # 3️⃣ 停滞惩罚（几乎不动）
#         # ==================================================
#         speed = np.linalg.norm(vel)
#         if speed < 1e-3:
#             comp["stagnation_penalty"] = STAGNATION_PENALTY
#
#         # ==================================================
#         # 4️⃣ 到达奖励
#         # ==================================================
#         if dist <= goal_radius:
#             reached[i] = 1
#             comp["arrival_reward"] = ARRIVAL_BONUS
#
#         # ==================================================
#         # 5️⃣ 🚧 APF 预留（默认关闭）
#         # ==================================================
#         # comp["apf_reward"] = 0.0
#         # 以后你要加 APF，只在这里写，不要动上面的主线
#
#         # ==================================================
#         # 6️⃣ 合成总 reward
#         # ==================================================
#         total = (
#             comp["progress_reward"]
#             + comp["collision_penalty"]
#             + comp["stagnation_penalty"]
#             + comp["arrival_reward"]
#             + comp["apf_reward"]
#         )
#
#         total = total / REWARD_SCALE
#
#         rewards[i] = total
#         comp["total"] = total
#         components.append(comp)
#
#     info = {
#         "reached": reached.tolist(),
#         "collisions": collisions.tolist(),
#         "components": components
#     }
#
#     return rewards.tolist(), info
from typing import List, Tuple, Dict, Optional
import numpy as np


def compute_step_reward(
        positions: np.ndarray,
        velocities: np.ndarray,
        goal: np.ndarray,
        obstacles: List[object],
        prev_positions: np.ndarray = None,
        goal_radius: float = 1.0,
        v_max: float = 1.0,
        already_reached: Optional[List[int]] = None
) -> Tuple[List[float], Dict]:
    """
    FACMAC-compatible reward function. (激进引导破局版)
    Returns:
        rewards: List[float] (per-agent)
        info: dict with keys ['reached', 'collisions', 'components']
    """

    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    N = positions.shape[0]

    rewards = np.zeros(N, dtype=float)
    reached = np.zeros(N, dtype=int)
    collisions = np.zeros(N, dtype=int)

    # --- already reached mask ---
    if already_reached is None:
        already_reached_mask = np.zeros(N, dtype=bool)
    else:
        already_reached_mask = np.asarray(already_reached, dtype=bool)
        if already_reached_mask.shape[0] != N:
            already_reached_mask = np.zeros(N, dtype=bool)

    # =============================
    # 🔧 Reward 超参数（激进引导版）
    # =============================
    PROGRESS_SCALE = 3.0  # 🚩 稍微调低：防止过度贪婪走直线撞墙
    COLLISION_PENALTY = -5.0  # 🚩 撞墙或互撞惩罚：保持痛感，但不至于让它完全不敢动
    STEP_PENALTY = -0.05  # 🚩 新增核心：无论动不动，每活一步就扣分，逼迫尽快到达
    ARRIVAL_BONUS = 50.0  # 🚩 史诗级加强：将到达终点变成绝对的终极诱惑
    UAV_SAFE_DIST = 0.6  # 🚩 新增：无人机之间的安全距离（防止多机互撞死锁）
    REWARD_SCALE = 10.0  # 保持您的缩放，防止 Q 值爆炸

    goal = np.asarray(goal)
    obstacles = obstacles or []
    components = []

    for i in range(N):
        comp = {
            "progress_reward": 0.0,
            "collision_penalty": 0.0,
            "step_penalty": 0.0,  # 修改了 key 名称
            "arrival_reward": 0.0,
            "apf_reward": 0.0,
            "total": 0.0
        }

        # 已到达的不再参与，且不给额外惩罚
        if already_reached_mask[i]:
            reached[i] = 1
            components.append(comp)
            continue

        pos = positions[i]
        vel = velocities[i]
        dist = np.linalg.norm(pos - goal)

        # ==================================================
        # 1️⃣ Progress Reward (向目标推进)
        # ==================================================
        if prev_positions is not None:
            prev_pos = prev_positions[i]
            prev_dist = np.linalg.norm(prev_pos - goal)
            progress = prev_dist - dist  # 变近才是正
            comp["progress_reward"] = PROGRESS_SCALE * progress

        # ==================================================
        # 2️⃣ 碰撞惩罚 (加入多智能体互撞检测)
        # ==================================================
        is_collided = False

        # A. 撞障碍物
        for obs in obstacles:
            fn = getattr(obs, "collides_point", None)
            if callable(fn) and fn(pos):
                is_collided = True
                break

        # B. 无人机互撞 (防止挤在一起死锁)
        if not is_collided:
            for j in range(N):
                if i != j and not already_reached_mask[j]:
                    if np.linalg.norm(pos - positions[j]) < UAV_SAFE_DIST:
                        is_collided = True
                        break

        if is_collided:
            collisions[i] = 1
            comp["collision_penalty"] = COLLISION_PENALTY

        # ==================================================
        # 3️⃣ 存在时间惩罚 (代替原来的停滞惩罚)
        # ==================================================
        # 只要还没到终点，活着就要扣分。这能彻底解决无人机绕圈圈逃避惩罚的 BUG。
        comp["step_penalty"] = STEP_PENALTY

        # 如果连动都不动，再额外加倍扣分
        speed = np.linalg.norm(vel)
        if speed < 1e-3:
            comp["step_penalty"] += STEP_PENALTY * 2

        # ==================================================
        # 4️⃣ 到达奖励
        # ==================================================
        if dist <= goal_radius:
            reached[i] = 1
            comp["arrival_reward"] = ARRIVAL_BONUS
            comp["step_penalty"] = 0.0  # 到达后当前步免除时间惩罚

        # ==================================================
        # 5️⃣ 🚧 APF 预留
        # ==================================================
        # comp["apf_reward"] = 0.0

        # ==================================================
        # 6️⃣ 合成总 reward
        # ==================================================
        total = (
                comp["progress_reward"]
                + comp["collision_penalty"]
                + comp["step_penalty"]
                + comp["arrival_reward"]
                + comp["apf_reward"]
        )

        total = total / REWARD_SCALE

        rewards[i] = total
        comp["total"] = total
        components.append(comp)

    info = {
        "reached": reached.tolist(),
        "collisions": collisions.tolist(),
        "components": components
    }

    return rewards.tolist(), info