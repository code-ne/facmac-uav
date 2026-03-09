import sys, os
import importlib.util

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src = os.path.join(root)
if src not in sys.path:
    sys.path.insert(0, src)

# Load MultiUAVEnv directly from file to avoid importing the package's __init__ (which imports other envs)
module_path = os.path.join(src, 'envs', 'uav', 'multi_uav_env.py')
spec = importlib.util.spec_from_file_location('multi_uav_env_mod', module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MultiUAVEnv = mod.MultiUAVEnv

# Build minimal cfg dict expected by MultiUAVEnv
cfg = {
    'n_agents': 3,
    'space_size': 50.0,
    'n_static_obstacles': 2,
    'max_steps': 100,
    # optional keys
    'dt': 0.2,
    'v_max': 2.0,
    'goal_radius': 1.0,
}

env = MultiUAVEnv(cfg=cfg, main_seed=123)
print('MultiUAVEnv created successfully')
obs, state = env.reset()
print('reset returned obs length', len(obs))
actions = [[0.0,0.0,0.0] for _ in range(cfg['n_agents'])]
obs, state, rewards, done, info = env.step(actions)
print('step returned rewards len', len(rewards), 'done', done)
print('info keys:', list(info.keys()))
