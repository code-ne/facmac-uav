import numpy as np
import torch as th

from ..multiagentenv import MultiAgentEnv
from types import SimpleNamespace

# 导入你原来的文件
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
        for k in ['n_agents', 'space_size', 'n_static_obstacles', 'max_steps', 'dt', 'v_max', 'goal_radius', 'randomize_every_reset', 'fixed_initial_positions', 'reward']:
            if k not in raw_env and k in args_dict:
                raw_env[k] = args_dict[k]

        cfg = raw_env

        # Pass expected kwargs to MultiAgentEnv so it can set self.args
        super().__init__(env_args=cfg)

        # create a simple args namespace to keep backward-compatible attribute access
        if isinstance(args, SimpleNamespace):
            ns = args
        else:
            ns = SimpleNamespace(**(cfg if isinstance(cfg, dict) else {}))
            if args is not None and hasattr(args, 'seed'):
                ns.seed = args.seed
            if args is not None and hasattr(args, 'episode_limit'):
                ns.episode_limit = args.episode_limit

        self.args = ns

        # 初始化你原本的环境
        self.env = MultiUAVEnv(cfg=cfg, main_seed=getattr(ns, 'seed', 0))

        # FACMAC 必须识别的属性
        self.n_agents = self.env.num_uav
        self.obs_size = self.env.obs_dim
        self.state_size = self.env.state_dim
        self.action_size = self.env.action_dim

        # Use dict get for episode_limit to avoid None when cfg is dict
        self.episode_limit = getattr(ns, 'episode_limit', None)
        if self.episode_limit is None and isinstance(cfg, dict):
            self.episode_limit = cfg.get('episode_limit', None)
        # Provide a sensible fallback if still None
        if self.episode_limit is None:
            # try max_steps or default 200
            self.episode_limit = cfg.get('max_steps', 200) if isinstance(cfg, dict) else 200

    # ------------------------------
    # FACMAC API
    # ------------------------------

    def reset(self):
        """
        FACMAC expects reset() to return observations only.
        """
        res = self.env.reset()
        # MultiUAVEnv.reset returns (obs, state). Return obs only for FACMAC.
        if isinstance(res, tuple) and len(res) == 2:
            obs, _ = res
            return obs
        return res

    def step(self, actions):
        # actions 通常由神经网路输出，需要确保是 numpy 格式
        if isinstance(actions, th.Tensor):
            actions = actions.cpu().numpy()

        next_obs, _state, rewards, done, info = self.env.step(actions)

        # 1. 奖励必须求和 (FACMAC 的架构决定了它通常处理团队总奖励)
        total_reward = np.sum(rewards)

        # 将求和改为求平均，以适应多智能体环境中的奖励分配
        # total_reward = np.mean(rewards)

        # 2. 这里的 done 如果是 bool，需要确保 Runner 能识别
        return total_reward, done, info

    # ------------------------------
    # Observation interfaces
    # ------------------------------

    def get_obs(self):
        """Returns list of obs for each agent."""
        res = self.env._get_obs_state()
        if isinstance(res, tuple) and len(res) == 2:
            obs, _ = res
            return obs
        return res

    def get_obs_agent(self, agent_id):
        """Return obs of a single agent."""
        obs_all = self.get_obs()
        return obs_all[agent_id]

    def get_obs_size(self):
        return self.obs_size

    # ------------------------------
    # State interface
    # ------------------------------

    def get_state(self):
        """Return global state."""
        res = self.env._get_obs_state()
        if isinstance(res, tuple) and len(res) == 2:
            _obs, state = res
            return state
        return res

    def get_state_size(self):
        return self.state_size

    # ------------------------------
    # Action interface
    # ------------------------------

    def get_total_actions(self):
        """Continuous action dim."""
        return self.action_size

    def get_avail_actions(self):
        """Continuous → all actions always available."""
        return [np.ones(self.action_size) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.action_size)

    # ------------------------------
    # Required environment info
    # ------------------------------

    def get_env_info(self):
        # Provide continuous action metadata so runner and controllers can handle continuous actions.
        try:
            from gym.spaces import Box
        except Exception:
            Box = None
        action_spaces = None
        if Box is not None:
            low = -getattr(self.env, 'v_max', 1.0)
            high = getattr(self.env, 'v_max', 1.0)
            action_spaces = [Box(low=low, high=high, shape=(self.action_size,)) for _ in range(self.n_agents)]

        return {
            "n_agents": self.n_agents,
            "obs_shape": self.obs_size,
            "state_shape": self.state_size,
            "n_actions": self.action_size,
            "episode_limit": self.episode_limit,
            "action_spaces": action_spaces,
            "actions_dtype": np.float32,
            "normalise_actions": False,
        }

    def close(self):
        """Attempt to close the underlying env; no-op if not implemented."""
        try:
            if hasattr(self.env, 'close'):
                self.env.close()
        except Exception:
            pass

    def seed(self, s=None):
        try:
            if s is None:
                s = getattr(self.args, 'seed', None)
            if s is not None:
                if _set_main_seed is not None and callable(_set_main_seed):
                    _set_main_seed(s)
                else:
                    try:
                        np.random.seed(int(s))
                    except Exception:
                        pass
                if hasattr(self.env, 'rng'):
                    try:
                        self.env.rng.seed(int(s))
                    except Exception:
                        pass
        except Exception:
            pass
