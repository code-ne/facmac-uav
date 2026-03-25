import numpy as np

def _safe_min(arr, default):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float(default)
    return float(np.min(arr))


def compute_step_reward(
    positions,
    velocities,
    goal,
    obstacles,
    prev_positions=None,
    goal_radius=1.0,
    v_max=1.0,
    already_reached=None,
    arrived_this_step=None,
    collided_this_step=None,
    nearest_obstacle_dists=None,
    nearest_uav_dists=None,
    reward_cfg=None,
):
    """
    Rewritten reward for continuous-control FACMAC UAV navigation.

    Key modifications:
    MOD-R1: progress reward is the main dense signal.
    MOD-R2: collision/arrival scale is reduced to avoid sparse-reward domination.
    MOD-R3: add near-obstacle / near-UAV shaping so the policy learns "how" to be safe.
    MOD-R4: add smoothness penalty to stabilize continuous control.
    MOD-R5: add cooperative team bonus/penalty to align with summed FACMAC reward.
    """

    cfg = {
        "progress_scale": 2.0,
        "step_penalty": -0.01,
        "arrival_bonus": 15.0,
        "collision_penalty": -15.0,
        "near_obs_scale": 2.0,
        "near_uav_scale": 2.0,
        "near_boundary_scale": 1.0,
        "safe_dist_obs": 6.0,
        "safe_dist_uav": 6.0,
        "safe_dist_boundary": 6.0,
        "space_size": 40.0,
        "velocity_norm_penalty": 0.00,
        "team_all_arrived_bonus": 20.0,
        "team_any_collision_penalty": -5.0,
        "reward_scale": 1.0,
    }
    if reward_cfg:
        cfg.update(reward_cfg)

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    goal = np.asarray(goal, dtype=float)

    n_agents = positions.shape[0]
    rewards = np.zeros(n_agents, dtype=float)
    reached = np.zeros(n_agents, dtype=int)
    collisions = np.zeros(n_agents, dtype=int)

    if prev_positions is None:
        prev_positions = positions.copy()
    else:
        prev_positions = np.asarray(prev_positions, dtype=float)

    if already_reached is None:
        already_reached = np.zeros(n_agents, dtype=bool)
    else:
        already_reached = np.asarray(already_reached, dtype=bool)

    if arrived_this_step is None:
        arrived_this_step = np.zeros(n_agents, dtype=bool)
    else:
        arrived_this_step = np.asarray(arrived_this_step, dtype=bool)

    if collided_this_step is None:
        collided_this_step = np.zeros(n_agents, dtype=bool)
    else:
        collided_this_step = np.asarray(collided_this_step, dtype=bool)

    if nearest_obstacle_dists is None:
        nearest_obstacle_dists = np.full(n_agents, np.inf, dtype=float)
    else:
        nearest_obstacle_dists = np.asarray(nearest_obstacle_dists, dtype=float)

    if nearest_uav_dists is None:
        nearest_uav_dists = np.full(n_agents, np.inf, dtype=float)
    else:
        nearest_uav_dists = np.asarray(nearest_uav_dists, dtype=float)

    components = []

    for i in range(n_agents):
        comp = {
            "progress_reward": 0.0,
            "step_penalty": 0.0,
            "arrival_reward": 0.0,
            "collision_penalty": 0.0,
            "near_obstacle_penalty": 0.0,
            "near_uav_penalty": 0.0,
            "near_boundary_penalty": 0.0,
            "velocity_penalty": 0.0,
            "team_bonus_share": 0.0,
            "total": 0.0,
        }

        if already_reached[i]:
            reached[i] = 1
            components.append(comp)
            continue

        dist_now = np.linalg.norm(positions[i] - goal)
        dist_prev = np.linalg.norm(prev_positions[i] - goal)
        progress = dist_prev - dist_now
        comp["progress_reward"] = cfg["progress_scale"] * progress

        comp["step_penalty"] = cfg["step_penalty"]

        if arrived_this_step[i] or dist_now <= goal_radius:
            reached[i] = 1
            comp["arrival_reward"] = cfg["arrival_bonus"]

        if collided_this_step[i]:
            collisions[i] = 1
            comp["collision_penalty"] = cfg["collision_penalty"]

        obs_margin = max(0.0, cfg["safe_dist_obs"] - float(nearest_obstacle_dists[i]))
        comp["near_obstacle_penalty"] = -cfg["near_obs_scale"] * obs_margin

        uav_margin = max(0.0, cfg["safe_dist_uav"] - float(nearest_uav_dists[i]))
        comp["near_uav_penalty"] = -cfg["near_uav_scale"] * uav_margin
        
        space_size = float(cfg["space_size"])
        nearest_boundary_dist = min(
            float(positions[i][0]),
            float(positions[i][1]),
            float(positions[i][2]),
            space_size - float(positions[i][0]),
            space_size - float(positions[i][1]),
            space_size - float(positions[i][2]),
        )
        boundary_margin = max(0.0, cfg["safe_dist_boundary"] - nearest_boundary_dist)
        comp["near_boundary_penalty"] = -cfg["near_boundary_scale"] * (
            boundary_margin / max(float(cfg["safe_dist_boundary"]), 1e-6)
        )
        

        speed_norm = np.linalg.norm(velocities[i]) / max(float(v_max), 1e-6)
        comp["velocity_penalty"] = -cfg["velocity_norm_penalty"] * speed_norm

        total = sum(v for k, v in comp.items() if k != "total")
        comp["total"] = total
        rewards[i] = total
        components.append(comp)

    team_bonus = 0.0
    if np.all(already_reached | arrived_this_step | reached.astype(bool)):
        team_bonus += cfg["team_all_arrived_bonus"]
    if np.any(collided_this_step):
        team_bonus += cfg["team_any_collision_penalty"]

    if abs(team_bonus) > 1e-12:
        team_share = team_bonus / max(n_agents, 1)
        rewards += team_share
        for comp in components:
            comp["team_bonus_share"] = team_share
            comp["total"] += team_share

    rewards = rewards / max(float(cfg["reward_scale"]), 1e-6)

    info = {
        "reached": reached.astype(int).tolist(),
        "collisions": collisions.astype(int).tolist(),
        "team_bonus": team_bonus,
        "components": components,
        "stats": {
            "min_nearest_obstacle_dist": _safe_min(nearest_obstacle_dists, np.inf),
            "min_nearest_uav_dist": _safe_min(nearest_uav_dists, np.inf),
            "min_nearest_boundary_dist": float(np.min(np.min(np.stack([
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                float(cfg["space_size"]) - positions[:, 0],
                float(cfg["space_size"]) - positions[:, 1],
                float(cfg["space_size"]) - positions[:, 2],
            ], axis=1), axis=1))),
        },
    }
    return rewards.tolist(), info