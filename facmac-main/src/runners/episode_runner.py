from ..envs import REGISTRY as env_REGISTRY
from functools import partial
from ..components.episode_buffer import EpisodeBatch
from numbers import Number
import torch as th
import numpy as np
import copy
import os
import json


def safe_add(a, b):
    """Robustly add/merge two values that may be numbers, lists, numpy arrays or dicts.
    - If both are dicts: recursively merge keys using safe_add
    - If one is dict: prefer the dict (don't coerce dicts to numbers)
    - If values are numeric or list-like numeric: produce numeric sum with broadcasting
    - Otherwise: prefer the existing (truthy) value
    Returns a Python-native type (float, list, dict) where appropriate.
    """
    # If either is a dict, handle dict merging or prefer dict
    if isinstance(a, dict) or isinstance(b, dict):
        if isinstance(a, dict) and isinstance(b, dict):
            out = {}
            for k in set(a) | set(b):
                out[k] = safe_add(a.get(k, 0), b.get(k, 0))
            return out
        # If only one side is dict, keep that dict (prefer b over a)
        return b if isinstance(b, dict) else a

    def to_array(x):
        # Return a numpy array of floats when x is numeric/list-like, else return None
        if isinstance(x, np.ndarray):
            try:
                return x.astype(float)
            except Exception:
                return None
        if isinstance(x, (list, tuple)):
            out = []
            for y in x:
                if isinstance(y, bool):
                    out.append(int(y))
                    continue
                if isinstance(y, Number):
                    out.append(float(y))
                    continue
                # try to coerce to float (e.g., numpy scalars)
                try:
                    out.append(float(y))
                except Exception:
                    return None
            try:
                return np.array(out, dtype=float)
            except Exception:
                return None
        if isinstance(x, bool):
            try:
                return np.array(float(int(x)), dtype=float)
            except Exception:
                return None
        if isinstance(x, Number):
            try:
                return np.array(float(x), dtype=float)
            except Exception:
                return None
        # everything else (strings, objects) -> None
        try:
            return np.array(float(x), dtype=float)
        except Exception:
            return None

    arr_a = to_array(a)
    arr_b = to_array(b)

    # If neither side is numeric/array-like, prefer to keep the existing value (a if truthy else b)
    if arr_a is None and arr_b is None:
        return a if a else b
    # If only one side is numeric, return the non-numeric side (we don't want to coerce dicts to numbers)
    if arr_a is None:
        return b
    if arr_b is None:
        return a

    # Both sides are numeric arrays -> perform addition with sensible broadcasting
    try:
        res = arr_a + arr_b
    except ValueError:
        try:
            if arr_a.size == 1:
                arr_a = np.broadcast_to(arr_a, arr_b.shape)
            if arr_b.size == 1:
                arr_b = np.broadcast_to(arr_b, arr_a.shape)
            res = arr_a + arr_b
        except Exception:
            # fallback: return a
            return a

    if res.ndim == 0:
        return float(res)
    return res.tolist()


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        if 'sc2' in self.args.env:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        else:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        # NEW: placeholders for per-episode logging payload
        self.last_episode_env_scalars = None
        self.last_episode_details = None
        self._global_episode_id = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, **kwargs):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        # NEW: episode-wise collections
        actions_seq = []  # list[ per-step actions for all agents ]
        step_rewards = []
        step_components_seq = []
        # Per-episode component accumulators (must reset at each run/episode).
        episode_component_sums_all_agents = {}
        episode_component_sums_per_agent = []
        traj_seq = []  # optional trajectory if env exposes positions
        last_env_info = {}
        base_env = getattr(self.env, "env", self.env)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)
            if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode))
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                actions = th.argmax(actions, dim=-1).long()

            # record actions to python list
            try:
                actions_seq.append(actions[0].detach().cpu().numpy().tolist())
            except Exception:
                try:
                    actions_seq.append(actions[0].tolist())
                except Exception:
                    actions_seq.append(None)

            if self.args.env in ["particle"]:
                cpu_actions = copy.deepcopy(actions).to("cpu").numpy()
                reward, terminated, env_info = self.env.step(cpu_actions[0])
                if isinstance(reward, (list, tuple)):
                    assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                    reward = reward[0]
                episode_return += reward
            else:
                reward, terminated, env_info = self.env.step(actions[0].cpu())
                if isinstance(reward, (list, tuple)):
                    # Expect cooperative reward (same for all agents). If so, take first element.
                    try:
                        assert all([r == reward[0] for r in reward]), "reward has to be cooperative!"
                    except Exception:
                        raise ValueError(
                            f"Non-cooperative reward detected: {reward}. "
                            f"Please handle team reward in env."
                        )
                        # If not cooperative, sum as a team reward fallback
                        # reward = float(sum(reward))
                    else:
                        reward = reward[0]
                episode_return += reward

            # record step reward and optional trajectory
            step_rewards.append(float(reward))
            try:
                if hasattr(base_env, "pos"):
                    # ensure serializable
                    traj_seq.append(np.array(base_env.pos).tolist())
            except Exception:
                pass

            # Collect per-step reward components as a robust fallback when
            # final env_info does not include episode_components.
            try:
                comps = env_info.get("components", None) if isinstance(env_info, dict) else None
                if comps is None:
                    step_components_seq.append(None)
                else:
                    cleaned = []
                    for comp in comps:
                        if not isinstance(comp, dict):
                            cleaned.append(comp)
                            continue
                        cd = {}
                        for kk, vv in comp.items():
                            if isinstance(vv, (np.floating, float)):
                                cd[kk] = float(vv)
                            elif isinstance(vv, (np.integer, int)):
                                cd[kk] = int(vv)
                            else:
                                cd[kk] = vv
                        cleaned.append(cd)
                    step_components_seq.append(cleaned)

                    # Update per-episode sums online so stats cannot leak across episodes.
                    while len(episode_component_sums_per_agent) < len(cleaned):
                        episode_component_sums_per_agent.append({})

                    for aid, comp in enumerate(cleaned):
                        if not isinstance(comp, dict):
                            continue
                        for k, v in comp.items():
                            if isinstance(v, (bool, int, float, np.integer, np.floating)):
                                fv = float(v)
                                episode_component_sums_all_agents[k] = float(
                                    episode_component_sums_all_agents.get(k, 0.0) + fv
                                )
                                episode_component_sums_per_agent[aid][k] = float(
                                    episode_component_sums_per_agent[aid].get(k, 0.0) + fv
                                )
            except Exception:
                step_components_seq.append(None)
            last_env_info = env_info

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode,
                                              explore=(not test_mode))
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            actions = th.argmax(actions, dim=-1).long()

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        # Safely combine numeric and list-valued env_info entries into cur_stats
        # def _safe_add(a, b):
        #     # If both are lists/tuples, do elementwise add when possible
        #     if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        #         if len(a) == len(b):
        #             return [ (x if not isinstance(x, bool) else int(x)) + (y if not isinstance(y, bool) else int(y)) for x, y in zip(a, b) ]
        #         else:
        #             # fallback: sum numeric elements
        #             try:
        #                 return float(sum(a)) + float(sum(b))
        #             except Exception:
        #                 return a
        #     # If one is list and other scalar, add scalar to each element
        #     if isinstance(a, (list, tuple)) and isinstance(b, (int, float)):
        #         return [ (float(x) if not isinstance(x, bool) else int(x)) + float(b) for x in a ]
        #     if isinstance(b, (list, tuple)) and isinstance(a, (int, float)):
        #         return [ float(a) + (float(x) if not isinstance(x, bool) else int(x)) for x in b ]
        #     # default numeric add
        #     try:
        #         return a + b
        #     except Exception:
        #         return a

        for k in set(cur_stats) | set(last_env_info):
            cur_stats[k] = safe_add(cur_stats.get(k, 0), last_env_info.get(k, 0))

        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if self.args.action_selector is not None and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # NEW: package per-episode payload for external logger
        # scalars from env/episode
        success_rate = last_env_info.get("success_rate") if isinstance(last_env_info, dict) else None
        collision_rate = last_env_info.get("collision_rate") if isinstance(last_env_info, dict) else None
        latency_rate = last_env_info.get("avg_latency_rate") if isinstance(last_env_info, dict) else None
        # Prefer explicit episode flags when available
        episode_success_all = last_env_info.get("episode_success_all") if isinstance(last_env_info, dict) else None
        episode_collision_any = last_env_info.get("episode_collision_any") if isinstance(last_env_info, dict) else None
        self.last_episode_env_scalars = {
            "total_reward": float(episode_return),
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "avg_latency_rate": latency_rate,
            "episode_steps": int(self.t),
        }
        # detailed JSONL payload
        self.last_episode_details = {
            "episode_id": kwargs.get("episode", 0),
            "success": bool(episode_success_all) if isinstance(episode_success_all, bool) else (bool(success_rate > 0) if isinstance(success_rate, (int, float)) else None),
            "collision": bool(episode_collision_any) if isinstance(episode_collision_any, bool) else (bool(collision_rate > 0) if isinstance(collision_rate, (int, float)) else None),
            "steps": int(self.t),
            "step_rewards": step_rewards,
            "actions": actions_seq,
            "trajectory": traj_seq if len(traj_seq) > 0 else None,
            "env_info": last_env_info,
        }

        # Episode-level component aggregation for easy reward diagnosis.
        source_episode_id = int(kwargs.get("episode", 0))
        episode_id = int(self._global_episode_id)
        self._global_episode_id += 1
        run_token = str(getattr(self.args, "unique_token", "run"))
        save_interval = max(1, int(getattr(self.args, "artifact_save_interval", 10)))
        should_save_artifacts = (episode_id % save_interval) == 0
        render_dir = os.path.join(self.args.local_results_path, "renders", run_token)
        comp_json_path = None
        render_png_path = None
        ep_components = None
        if isinstance(last_env_info, dict):
            ep_components = last_env_info.get("episode_components", None)
        if not (isinstance(ep_components, (list, tuple)) and len(ep_components) > 0):
            if len(step_components_seq) > 0:
                ep_components = step_components_seq
        comp_sums_all = None
        comp_sums_per_agent = None
        comp_sums_per_agent_labeled = None
        if isinstance(ep_components, (list, tuple)) and len(ep_components) > 0:
            comp_sums_all, comp_sums_per_agent = self._summarize_episode_components(ep_components)
        elif len(episode_component_sums_all_agents) > 0:
            comp_sums_all = episode_component_sums_all_agents
            comp_sums_per_agent = episode_component_sums_per_agent

        if comp_sums_all is not None:
            if isinstance(comp_sums_per_agent, list):
                comp_sums_per_agent_labeled = {}
                for i, item in enumerate(comp_sums_per_agent):
                    comp_sums_per_agent_labeled["uav_{}".format(i + 1)] = item if isinstance(item, dict) else {}
            self.last_episode_details["episode_component_sums"] = comp_sums_all
            self.last_episode_details["episode_component_sums_per_agent"] = comp_sums_per_agent
            self.last_episode_details["episode_component_sums_per_agent_labeled"] = comp_sums_per_agent_labeled

            if getattr(self.args, "print_components", False):
                try:
                    self.logger.console_logger.info(
                        "[components][episode-sum] t_env={} episode={} sums={}".format(
                            self.t_env, episode_id, comp_sums_all
                        )
                    )
                except Exception:
                    pass

        # Save component summary JSON every N episodes to control artifact volume.
        if should_save_artifacts:
            try:
                os.makedirs(render_dir, exist_ok=True)
                comp_json_path = os.path.join(render_dir, "episode_{:06d}_components.json".format(episode_id))
                arrival_audit = last_env_info.get("per_agent_arrival_audit") if isinstance(last_env_info, dict) else None
                arrival_mode = last_env_info.get("arrival_mode") if isinstance(last_env_info, dict) else None
                reached_by_distance = last_env_info.get("reached_by_distance") if isinstance(last_env_info, dict) else None
                reached_by_segment = last_env_info.get("reached_by_segment") if isinstance(last_env_info, dict) else None
                moved_steps = last_env_info.get("moved_steps") if isinstance(last_env_info, dict) else None
                min_dist_to_goal = last_env_info.get("min_dist_to_goal") if isinstance(last_env_info, dict) else None
                termination_reason = last_env_info.get("termination_reason") if isinstance(last_env_info, dict) else None
                env_max_steps = last_env_info.get("max_steps") if isinstance(last_env_info, dict) else None
                env_episode_limit = last_env_info.get("episode_limit") if isinstance(last_env_info, dict) else None
                with open(comp_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "episode_id": episode_id,
                            "source_episode_id": source_episode_id,
                            "t_env": int(self.t_env),
                            "episode_steps": int(self.t),
                            "save_interval": save_interval,
                            "has_episode_components": bool(isinstance(ep_components, (list, tuple)) and len(ep_components) > 0),
                            "component_sums_all_agents": comp_sums_all,
                            "component_sums_per_agent": comp_sums_per_agent,
                            "component_sums_per_agent_labeled": comp_sums_per_agent_labeled,
                            "arrival_audit": arrival_audit,
                            "arrival_mode": arrival_mode,
                            "reached_by_distance": reached_by_distance,
                            "reached_by_segment": reached_by_segment,
                            "moved_steps": moved_steps,
                            "min_dist_to_goal": min_dist_to_goal,
                            "termination_reason": termination_reason,
                            "max_steps": env_max_steps,
                            "episode_limit": env_episode_limit,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                self.last_episode_details["component_summary_json"] = comp_json_path
                self.logger.console_logger.info("Saved component summary to {}".format(comp_json_path))
            except Exception as e:
                self.logger.console_logger.warning("Failed to save component summary JSON: {}".format(e))

        # Episode-level components print (once per episode) to avoid noisy per-step logs.
        if getattr(self.args, "print_components", False):
            try:
                final_components = None
                ep_components = None
                if isinstance(last_env_info, dict):
                    final_components = last_env_info.get("components", None)
                    ep_components = last_env_info.get("episode_components", None)
                if final_components is not None or ep_components is not None:
                    n_steps = len(ep_components) if isinstance(ep_components, (list, tuple)) else 0
                    self.logger.console_logger.info(
                        "[components][episode] t_env={} episode={} steps={} final_step_components={}".format(
                            self.t_env,
                            episode_id,
                            n_steps,
                            final_components,
                        )
                    )
            except Exception:
                pass

        # Optional: save a 3D trajectory image every N episodes via env.render(...).
        if should_save_artifacts and getattr(self.args, "save_3d_trajectory", False) and hasattr(base_env, "render"):
            try:
                os.makedirs(render_dir, exist_ok=True)
                render_png_path = os.path.join(render_dir, "episode_{:06d}_3d.png".format(episode_id))
                base_env.render(save_path=render_png_path, traj=traj_seq)
                self.last_episode_details["render_3d_path"] = render_png_path
                self.logger.console_logger.info("Saved 3D trajectory render to {}".format(render_png_path))
            except Exception as e:
                self.logger.console_logger.warning("Failed to save 3D trajectory render: {}".format(e))

        # Append one-line artifact index (JSONL) for easy lookup and pairing.
        if should_save_artifacts:
            try:
                os.makedirs(render_dir, exist_ok=True)
                index_path = os.path.join(render_dir, "artifacts_index.jsonl")
                with open(index_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "episode_id": episode_id,
                        "source_episode_id": source_episode_id,
                        "t_env": int(self.t_env),
                        "episode_steps": int(self.t),
                        "save_interval": save_interval,
                        "component_summary_json": comp_json_path,
                        "trajectory_png": render_png_path,
                    }, ensure_ascii=False) + "\n")
                self.last_episode_details["artifact_index_jsonl"] = index_path
            except Exception as e:
                self.logger.console_logger.warning("Failed to append artifact index: {}".format(e))

        return self.batch

    def _summarize_episode_components(self, episode_components):
        """Aggregate numeric component keys across all steps.
        Returns:
            - all_agents: dict[key -> total]
            - per_agent: list[dict[key -> total]]
        """
        all_agents = {}
        per_agent = []

        for step_comps in episode_components:
            if not isinstance(step_comps, (list, tuple)):
                continue
            while len(per_agent) < len(step_comps):
                per_agent.append({})

            for aid, comp in enumerate(step_comps):
                if not isinstance(comp, dict):
                    continue
                for k, v in comp.items():
                    if isinstance(v, (bool, int, float, np.integer, np.floating)):
                        fv = float(v)
                        all_agents[k] = float(all_agents.get(k, 0.0) + fv)
                        per_agent[aid][k] = float(per_agent[aid].get(k, 0.0) + fv)

        return all_agents, per_agent

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        n_eps = stats.get("n_episodes", 1)
        for k, v in list(stats.items()):
            if k == "n_episodes":
                continue
            # compute a scalar mean for logging
            try:
                if isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.array(v, dtype=float)
                    scalar = float(np.mean(arr)) / float(n_eps)
                else:
                    scalar = float(v) / float(n_eps)
            except Exception:
                # skip non-numeric or incompatible entries
                continue
            self.logger.log_stat(prefix + k + "_mean", scalar, self.t_env)
        stats.clear()