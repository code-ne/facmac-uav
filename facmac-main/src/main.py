import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from collections.abc import Mapping
import sys
import torch as th
try:
    from .utils.logging import get_logger
    from .run import run
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from utils.logging import get_logger
    from run import run
import yaml

SETTINGS['CAPTURE_MODE'] = "no" # disable Sacred capture so stdout/stderr appear in console
SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
logger = get_logger()

ex = Experiment("pymarl", save_git_info=False)
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r", encoding="utf-8") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


import collections.abc as _collections_abc

# def recursive_dict_update(d, u):
#     for k, v in u.items():
#         if isinstance(v, _collections_abc.Mapping):
#             d[k] = recursive_dict_update(d.get(k, {}), v)
#         else:
#             d[k] = v
#     return d

def recursive_dict_update(d, u):
    """
    把 u 合并到 d：
    - 若 u 不是 Mapping（字典类），直接返回 u（覆盖 d 的位置）。
    - 若 d 不是 Mapping，则先置为空 dict（便于合并）。
    - 对于同名键，若双方都是 Mapping 则递归合并，否则直接以 u 的值覆盖。
    返回合并后的结果（dict 或标量）。
    """
    if not isinstance(u, Mapping):
        # u 为标量/列表/None/字符串时，直接用 u 覆盖
        return u
    if not isinstance(d, Mapping):
        d = {}
    for k, v in u.items():
        if k in d and isinstance(d[k], Mapping) and isinstance(v, Mapping):
            d[k] = recursive_dict_update(d[k], v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r", encoding="utf-8") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
