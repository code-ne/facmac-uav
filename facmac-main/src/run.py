import datetime
import numpy as np
import os
import pprint
import time
import threading
import torch as th
from gym import spaces
from types import SimpleNamespace as SN
from .utils.logging import Logger
from .utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from .learners import REGISTRY as le_REGISTRY
from .runners import REGISTRY as r_REGISTRY
from .controllers import REGISTRY as mac_REGISTRY
from .components.episode_buffer import ReplayBuffer
from .components.transforms import OneHot
from .utils.episode_logger import EpisodeDualLogger  # NEW


def run(_run, _config, _log, pymongo_client=None):

    # 把config中的参数转换为适合程序使用的格式
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    checkpoint_path = getattr(args, "checkpoint_path", "")
    resume_token = ""
    if checkpoint_path:
        # If resuming, reuse the original run token so logs continue under the same run directory.
        cp_norm = os.path.normpath(checkpoint_path)
        cp_base = os.path.basename(cp_norm)
        if cp_base.isdigit():
            resume_token = os.path.basename(os.path.dirname(cp_norm))
        else:
            resume_token = cp_base

    if resume_token:
        unique_token = resume_token
    else:
        unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # NEW: dual episode logger (TensorBoard scalars via SummaryWriter + JSONL details)
    results_root = os.path.join(dirname(dirname(abspath(__file__))), "results")
    dual_logger = EpisodeDualLogger(
        base_results_dir=results_root,
        run_token=unique_token,
        enable_tb=True,
        enable_jsonl=True,
    )

    # Run and train
    run_sequential(args=args, logger=logger, dual_logger=dual_logger)

    # Clean up after finishing
    # Clean up after finishing
    try:
        _log.info("Exiting Main")
    except Exception:
        pass

    if pymongo_client is not None:
        try:
            _log.info("Attempting to close mongodb client")
            pymongo_client.close()
            _log.info("Mongodb client closed")
        except Exception:
            pass

    try:
        _log.info("Stopping all threads")
    except Exception:
        pass
    for t in threading.enumerate():
        if t.name != "MainThread":
            try:
                _log.info("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            except Exception:
                pass
            t.join(timeout=1)
            try:
                _log.info("Thread joined")
            except Exception:
                pass

    # NEW: close dual logger
    try:
        dual_logger.close()
    except Exception:
        pass

    try:
        _log.info("Exiting script")
    except Exception:
        pass

    # Making sure framework really exits
    try:
        os._exit(os.EX_OK)
    except Exception:
        pass


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger, dual_logger=None):

    # Init runner so we can get env info
    # runner职责：用MAC控制agent，与环境交互，收集数据，提供给learner学习
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    # Treat particle/cts_matrix_game/mujoco_multi/uav as "continuous-type" envs handled by the else branch
    if 'particle' not in args.env and "cts_matrix_game" not in args.env and "mujoco_multi" not in args.env and "uav" not in args.env:
        # 离散动作环境
        # 获取环境信息
        env_info = runner.get_env_info()
        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]

        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
        }
    else:
        # 连续动作环境
        env_info = runner.get_env_info()
        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.action_spaces = env_info["action_spaces"]
        args.actions_dtype = env_info["actions_dtype"]
        args.normalise_actions = env_info.get("normalise_actions", False) # if true, action vectors need to sum to one

        ttype = th.FloatTensor if not args.use_cuda else th.cuda.FloatTensor
        mult_coef_tensor = ttype(args.n_agents, args.n_actions)
        action_min_tensor = ttype(args.n_agents, args.n_actions)
        if all([isinstance(act_space, spaces.Box) for act_space in args.action_spaces]):
            for _aid in range(args.n_agents):
                for _actid in range(args.action_spaces[_aid].shape[0]):
                    _action_min = args.action_spaces[_aid].low[_actid]
                    _action_max = args.action_spaces[_aid].high[_actid]
                    mult_coef_tensor[_aid, _actid] = float(_action_max - _action_min)
                    action_min_tensor[_aid, _actid] = float(_action_min)
        elif all([isinstance(act_space, spaces.Tuple) for act_space in args.action_spaces]):
            for _aid in range(args.n_agents):
                for _actid in range(args.action_spaces[_aid].spaces[0].shape[0]):
                    _action_min = args.action_spaces[_aid].spaces[0].low[_actid]
                    _action_max = args.action_spaces[_aid].spaces[0].high[_actid]
                    mult_coef_tensor[_aid, _actid] = float(_action_max - _action_min)
                    action_min_tensor[_aid, _actid] = float(_action_min)
                for _actid in range(args.action_spaces[_aid].spaces[1].shape[0]):
                    _action_min = args.action_spaces[_aid].spaces[1].low[_actid]
                    _action_max = args.action_spaces[_aid].spaces[1].high[_actid]
                    tmp_idx = _actid + args.action_spaces[_aid].spaces[0].shape[0]
                    mult_coef_tensor[_aid, tmp_idx] = float(_action_max - _action_min)
                    action_min_tensor[_aid, tmp_idx] = float(_action_min)

        args.actions2unit_coef = mult_coef_tensor
        args.actions2unit_coef_cpu = mult_coef_tensor.cpu()
        args.actions2unit_coef_numpy = mult_coef_tensor.cpu().numpy()
        args.actions_min = action_min_tensor
        args.actions_min_cpu = action_min_tensor.cpu()
        args.actions_min_numpy = action_min_tensor.cpu().numpy()

        def actions_to_unit_box(actions):
            if isinstance(actions, np.ndarray):
                return args.actions2unit_coef_numpy * actions + args.actions_min_numpy
            elif actions.is_cuda:
                return args.actions2unit_coef * actions + args.actions_min
            else:
                return args.actions2unit_coef_cpu  * actions + args.actions_min_cpu

        def actions_from_unit_box(actions):
            if isinstance(actions, np.ndarray):
                return (actions - args.actions_min_numpy) / args.actions2unit_coef_numpy
            elif actions.is_cuda:
                return th.div((actions - args.actions_min), args.actions2unit_coef)
            else:
                return th.div((actions - args.actions_min_cpu), args.actions2unit_coef_cpu)

        # make conversion functions globally available
        args.actions2unit = actions_to_unit_box
        args.unit2actions = actions_from_unit_box

        action_dtype = th.long if not args.actions_dtype == np.float32 else th.float
        # Ensure initialization to avoid possible unbound local warning
        actions_vshape = 1
        if all([isinstance(act_space, spaces.Box) for act_space in args.action_spaces]):
            actions_vshape = 1 if not args.actions_dtype == np.float32 else max([i.shape[0] for i in args.action_spaces])
        elif all([isinstance(act_space, spaces.Tuple) for act_space in args.action_spaces]):
            actions_vshape = 1 if not args.actions_dtype == np.float32 else \
                                           max([i.spaces[0].shape[0] + i.spaces[1].shape[0] for i in args.action_spaces])
        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (actions_vshape,), "group": "agents", "dtype": action_dtype},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.n_agents
        }

        if not args.actions_dtype == np.float32:
            preprocess = {
                "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
            }
        else:
            preprocess = {}

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1 if args.runner_scope == "episodic" else 2,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    resumed_timestep = 0

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load
        resumed_timestep = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = - args.test_interval - 1
    last_log_T = 0
    model_save_time = resumed_timestep

    start_time = time.time()
    last_time = start_time

    # If user provided t_max_episodes, prefer stopping after that many episodes;
    # otherwise use legacy behavior (t_max interpreted as total env timesteps).
    if getattr(args, 't_max_episodes', None) is not None:
        logger.console_logger.info("Beginning training for {} episodes".format(args.t_max_episodes))
        stop_by_episodes = True
        target_episodes = int(args.t_max_episodes)
    else:
        logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
        stop_by_episodes = False

    # Main training loop: either by episode count or by total timesteps
    if stop_by_episodes:
        condition_fn = lambda runner, episode: episode < target_episodes
    else:
        condition_fn = lambda runner, episode: runner.t_env <= args.t_max

    while condition_fn(runner, episode):

        # Run for a whole episode at a time
        # 采样（执行阶段）
        if getattr(args, "runner_scope", "episodic") == "episodic":
            episode_batch = runner.run(test_mode=False, learner=learner, dual_logger=dual_logger, episode=episode)
            # 存入回放缓冲区
            buffer.insert_episode_batch(episode_batch)

            # 训练阶段
            if buffer.can_sample(args.batch_size) and (buffer.episodes_in_buffer > getattr(args, "buffer_warmup", 0)):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                # ====== 奖励归一化处理（min-max归一化到[-1,1]） ======
                rewards = episode_sample["reward"]  # shape: [batch, T, 1]
                min_r = rewards.min()
                max_r = rewards.max()
                if (max_r > min_r):
                    rewards = 2 * (rewards - min_r) / (max_r - min_r) - 1
                else:
                    # 当所有奖励都相等时，跳过归一化，避免将其强制置零
                    pass
                episode_sample.update({"reward": rewards})
                # ====== END 奖励归一化 ======

                learner.train(episode_sample, runner.t_env, episode)

            # NEW: write per-episode logs (TensorBoard scalars + JSONL details)
            if dual_logger is not None and getattr(runner, "last_episode_env_scalars", None) is not None:
                scalars = dict(runner.last_episode_env_scalars)
                # 增加最新的loss与Q/TD统计到JSON/TensorBoard
                if hasattr(learner, "last_pg_loss") and learner.last_pg_loss is not None:
                    scalars["actor_loss"] = learner.last_pg_loss
                if hasattr(learner, "last_critic_loss") and learner.last_critic_loss is not None:
                    scalars["critic_loss"] = learner.last_critic_loss
                # if hasattr(learner, "last_q_mean") and learner.last_q_mean is not None:
                #     scalars["q_mean"] = learner.last_q_mean
                # if hasattr(learner, "last_q_std") and learner.last_q_std is not None:
                #     scalars["q_std"] = learner.last_q_std
                # if hasattr(learner, "last_td_error_mean") and learner.last_td_error_mean is not None:
                #     scalars["td_error_mean"] = learner.last_td_error_mean
                # if hasattr(learner, "last_td_error_std") and learner.last_td_error_std is not None:
                #     scalars["td_error_std"] = learner.last_td_error_std
                # 去掉None值避免TB异常
                # scalars = {k: v for k, v in scalars.items() if v is not None}
                scalars = {k: scalars[k] for k in ["total_reward", "critic_loss", "actor_loss",
                                                   "success_rate", "collision_rate"]
                           if k in scalars}


                # dual_logger.log_details(details)


                details = {}
                if getattr(runner, "last_episode_details", None) is not None:
                    details.update(runner.last_episode_details)
                details.update({
                    "t_env": int(runner.t_env),
                    "unique_token": args.unique_token,
                })

                # 自动生成轨迹图（如有）
                images = None
                try:
                    traj = details.get('trajectory')
                    if traj:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        import numpy as _np
                        fig, ax = plt.subplots()
                        traj_arr = _np.array(traj)
                        if traj_arr.ndim == 3 and traj_arr.shape[2] >= 2:
                            T, n_agents, dim = traj_arr.shape
                            for aid in range(n_agents):
                                xs = traj_arr[:, aid, 0]
                                ys = traj_arr[:, aid, 1]
                                ax.plot(xs, ys, marker='o', label=f'agent_{aid}')
                            ax.set_title(f'Episode {episode} trajectory')
                            ax.set_xlabel('x')
                            ax.set_ylabel('y')
                            ax.legend()
                            images = { 'trajectory': fig }
                        else:
                            plt.close(fig)
                            images = None
                except Exception:
                    images = None

                SAVE_INTERVAL = 100
                if runner.t_env % SAVE_INTERVAL == 0:
                    dual_logger.log_episode(episode=episode, scalars=scalars, details=details, images=images)

                # try:
                #     dual_logger.log_episode(episode=episode, scalars=scalars, details=details, images=images)
                # except Exception:
                #     pass

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        # 触发条件：当自上次测试环境时间步达到或者超过test_interval时出发测试块
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            # 打印当前进度
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            # 打印时间估计
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            if getattr(args, "testing_on", True):
                for _ in range(n_test_runs):
                    if getattr(args, "runner_scope", "episodic") == "episodic":
                        runner.run(test_mode=True, learner=learner)
                    elif getattr(args, "runner_scope", "episode") == "transition":
                        runner.run(test_mode=True,
                                   buffer = buffer,
                                   learner = learner,
                                   episode = episode)
                    else:
                        raise Exception("Undefined runner scope!")

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            # learner.save_models(save_path, args.unique_token, model_save_time)

            learner.save_models(save_path)

        episode += args.batch_size_run

        # 只在每100轮输出一次终端统计信息
        if episode > 0 and episode % 10 == 0:
            # 确保 scalars 已定义，否则用空字典
            if 'scalars' not in locals():
                scalars = {}
            logger.log_stat("total_reward", scalars.get("total_reward", 0), runner.t_env)
            logger.log_stat("critic_loss", scalars.get("critic_loss", 0), runner.t_env)
            logger.log_stat("actor_loss", scalars.get("actor_loss", 0), runner.t_env)
            logger.log_stat("success_rate", scalars.get("success_rate", 0), runner.t_env)
            logger.log_stat("collision_rate", scalars.get("collision_rate", 0), runner.t_env)
            logger.log_stat("episode", episode, runner.t_env)

            # 打印终端，只打印关键指标
            print(
                f"[E{episode}] "
                f"R: {scalars.get('total_reward', 0):.3f} | "
                f"CriticL: {scalars.get('critic_loss', 0):.2f} | "
                f"ActorL: {scalars.get('actor_loss', 0):.2f} | "
                f"Succ: {scalars.get('success_rate', 0):.2f} | "
                f"Coll: {scalars.get('collision_rate', 0):.2f}"
            )

            last_log_T = runner.t_env
        # 原有：if (runner.t_env - last_log_T) >= args.log_interval:
        #     logger.log_stat("episode", episode, runner.t_env)
        #     logger.print_recent_stats()
        #     last_log_T = runner.t_env

    # 训练结束：尝试根据JSONL生成loss曲线
    try:
        if dual_logger is not None:
            p = dual_logger.generate_loss_plot(output_filename="loss_curve.png", keys=["critic_loss", "actor_loss"])
            if p:
                logger.console_logger.info(f"Saved loss curve to {p}")
    except Exception:
        pass

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
