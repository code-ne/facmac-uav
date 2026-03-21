import numpy as np
import torch as th
from gym.spaces import Box
from types import SimpleNamespace

from ..multiagentenv import MultiAgentEnv
from .multi_uav_env import MultiUAVEnv

# Optional seeding helper
# todo：有点冗余，reward里没有考虑设计跟种子有关的代码
try:
    from .. import reward as _reward_mod
    _set_main_seed = getattr(_reward_mod, 'set_main_seed', None)
except Exception:
    _set_main_seed = None


class UAVEnv(MultiAgentEnv):
    """
    FACMAC/PyMARL compatible wrapper for the user's MultiUAVEnv.
    """

    def __init__(self, env_args=None, args=None, **kwargs):
        """
        Accept either:
        - env_args: dict of env config (from config file)
        - args: SimpleNamespace of global args (runner args), may contain seed and episode_limit
        EpisodeRunner calls: UAVEnv(env_args=..., args=args)
        """
        # raw config dict for MultiUAVEnv (merge env_args dict and top-level args as fallback)
        # 作为字典存储环境需要的各种键值对（键值对被合并自env_args和args）
        raw_env = {}
        if isinstance(env_args, dict) and len(env_args) != 0:
            raw_env.update(env_args)
        # if env_args missing keys, try to get them from args (global config)
        if isinstance(args, SimpleNamespace):
            args_dict = vars(args)
        elif isinstance(args, dict):
            args_dict = args
        else:
            args_dict = getattr(args, 'env_args', {}) or {}

        # keys that MultiUAVEnv expects
        for k in [
            "n_agents",
            "space_size",
            "n_static_obstacles",
            "max_steps",
            "episode_limit",
            "dt",
            "v_max",
            "goal_radius",
            "randomize_every_reset",
            "fixed_initial_positions",
            "reward",
            "safe_uav_dist",
            "collision_dist",
            "near_uav_dist",
            "safe_dist_obs",
            "use_normalized_actions",
        ]:
            if k not in raw_env and k in args_dict:
                raw_env[k] = args_dict[k]

        cfg = raw_env

        # Pass expected kwargs to MultiAgentEnv so it can set self.args
        super().__init__(env_args=cfg)

        if isinstance(args, SimpleNamespace):
            self.args = args
        else:
            self.args = SimpleNamespace(**cfg)

        seed = getattr(self.args, "seed", 0)
        self.env = MultiUAVEnv(cfg=cfg, main_seed=seed)

        # 初始化你原本的环境
        self.env = MultiUAVEnv(cfg=cfg, main_seed=seed)

        # FACMAC 必须识别的属性
        self.n_agents = self.env.num_uav
        self.obs_size = self.env.obs_dim
        self.state_size = self.env.state_dim
        self.action_size = self.env.action_dim
        self.episode_limit = int(cfg.get("episode_limit", cfg.get("max_steps", 300)))

    # ------------------------------
    # FACMAC API
    # ------------------------------

    def reset(self):
        obs, _state = self.env.reset()
        self.obs_size = self.env.obs_dim
        self.state_size = self.env.state_dim
        return obs

    def step(self, actions):
        # actions 通常由神经网路输出，需要确保是 numpy 格式
        if isinstance(actions, th.Tensor):
            actions = actions.cpu().numpy()
        _obs, _state, rewards, done, info = self.env.step(actions)
        total_reward = float(np.sum(rewards))
        return total_reward, bool(done), info

    # ------------------------------
    # Observation interfaces
    # ------------------------------

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        return self.env.obs_dim

    # ------------------------------
    # State interface
    # ------------------------------

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        return self.env.state_dim

    # ------------------------------
    # Action interface

    # ------------------------------

    def get_total_actions(self):
        """Continuous action dim."""
        return self.action_size

    def get_avail_actions(self):
        """Continuous → all actions always available."""
        return [np.ones(self.action_size, dtype=np.float32) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.action_size, dtype=np.float32)

    def get_stats(self):
        return self.env.get_stats()

    def render(self):
        return None

    def close(self):
        return self.env.close()

    def seed(self, s=None):
        return None

    def save_replay(self):
        return None

    # ------------------------------
    # Required environment info
    # ------------------------------

    def get_env_info(self):
        action_spaces = [
            Box(low=-1.0, high=1.0, shape=(self.action_size,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        return {
            "n_agents": self.n_agents,
            "obs_shape": self.obs_size,
            "state_shape": self.state_size,
            "n_actions": self.action_size,
            "episode_limit": self.episode_limit,
            "action_spaces": action_spaces,
            "actions_dtype": np.float32,
            "normalise_actions": True,
        }