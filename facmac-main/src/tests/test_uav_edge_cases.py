import importlib.util
import os
import sys
import numpy as np
multi_uav_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs', 'uav', 'multi_uav_env.py'))
spec = importlib.util.spec_from_file_location('multi_uav_env_mod', multi_uav_env_path)
mod = importlib.util.module_from_spec(spec)
sys.modules['multi_uav_env_mod'] = mod
spec.loader.exec_module(mod)
MultiUAVEnv = mod.MultiUAVEnv

def test_high_speed_arrival():
    cfg = {
        'n_agents': 1,
        'space_size': 20,
        'n_static_obstacles': 0,
        'max_steps': 5,
        'dt': 1.0,
        'v_max': 10.0,
        'goal_radius': 2.0,
        'randomize_every_reset': False,
        'fixed_initial_positions': True,
    }
    env = MultiUAVEnv(cfg, main_seed=42)
    env.goal = np.array([10.0, 10.0, 10.0])
    env.pos = np.array([[0.0, 10.0, 10.0]])
    env.vel = np.array([[10.0, 0.0, 0.0]])
    env.arrived_mask[:] = False
    env.step_count = 0
    obs, state = env._get_obs_state()
    actions = [[1.0, 0.0, 0.0]] # move at v_max towards goal
    obs, state, rewards, done, info = env.step(actions)
    # Should arrive: pos moves from (0,10,10) to (10,10,10), goal at (10,10,10), radius=2
    assert env.arrived_mask[0], f"High speed arrival failed: {env.pos[0]}, arrived_mask={env.arrived_mask}"
    print("test_high_speed_arrival passed.")

def test_grazing_goal():
    cfg = {
        'n_agents': 1,
        'space_size': 20,
        'n_static_obstacles': 0,
        'max_steps': 5,
        'dt': 1.0,
        'v_max': 10.0,
        'goal_radius': 2.0,
        'randomize_every_reset': False,
        'fixed_initial_positions': True,
    }
    env = MultiUAVEnv(cfg, main_seed=43)
    env.goal = np.array([10.0, 10.0, 10.0])
    env.pos = np.array([[7.9, 10.0, 10.0]]) # just outside goal_radius
    env.vel = np.array([[2.0, 0.0, 0.0]])
    env.arrived_mask[:] = False
    env.step_count = 0
    obs, state = env._get_obs_state()
    actions = [[0.2, 0.0, 0.0]] # move by 2 units to (9.9,10,10)
    obs, state, rewards, done, info = env.step(actions)
    # Should NOT arrive: pos moves to (9.9,10,10), dist=0.1, but started outside and did not cross
    # Actually, since dist=0.1 < 2.0, should arrive
    assert env.arrived_mask[0], f"Grazing goal failed: {env.pos[0]}, arrived_mask={env.arrived_mask}"
    print("test_grazing_goal passed.")

def test_high_speed_collision():
    cfg = {
        'n_agents': 1,
        'space_size': 20,
        'n_static_obstacles': 1,
        'max_steps': 5,
        'dt': 1.0,
        'v_max': 10.0,
        'goal_radius': 2.0,
        'randomize_every_reset': False,
        'fixed_initial_positions': True,
    }
    env = MultiUAVEnv(cfg, main_seed=44)
    # Place obstacle at (10,10,10), radius=2
    class DummyObs:
        def __init__(self):
            self.x, self.y, self.radius, self.z_min, self.z_max = 10.0, 10.0, 2.0, 0.0, 20.0
        def collides_point(self, p):
            return np.linalg.norm(np.array(p)[:2] - np.array([self.x, self.y])) <= self.radius
    env.obstacles = [DummyObs()]
    env.goal = np.array([15.0, 10.0, 10.0])
    env.pos = np.array([[0.0, 10.0, 10.0]])
    env.vel = np.array([[10.0, 0.0, 0.0]])
    env.arrived_mask[:] = False
    env.colliding_mask[:] = False
    env.step_count = 0
    obs, state = env._get_obs_state()
    actions = [[1.0, 0.0, 0.0]] # move at v_max, should cross obstacle
    obs, state, rewards, done, info = env.step(actions)
    assert env.colliding_mask[0], f"High speed collision failed: {env.pos[0]}, colliding_mask={env.colliding_mask}"
    print("test_high_speed_collision passed.")

def test_grazing_obstacle():
    cfg = {
        'n_agents': 1,
        'space_size': 20,
        'n_static_obstacles': 1,
        'max_steps': 5,
        'dt': 1.0,
        'v_max': 10.0,
        'goal_radius': 2.0,
        'randomize_every_reset': False,
        'fixed_initial_positions': True,
    }
    env = MultiUAVEnv(cfg, main_seed=45)
    class DummyObs:
        def __init__(self):
            self.x, self.y, self.radius, self.z_min, self.z_max = 10.0, 10.0, 2.0, 0.0, 20.0
        def collides_point(self, p):
            return np.linalg.norm(np.array(p)[:2] - np.array([self.x, self.y])) <= self.radius
    env.obstacles = [DummyObs()]
    env.goal = np.array([15.0, 10.0, 10.0])
    env.pos = np.array([[7.9, 10.0, 10.0]]) # just outside obstacle
    env.vel = np.array([[2.0, 0.0, 0.0]])
    env.arrived_mask[:] = False
    env.colliding_mask[:] = False
    env.step_count = 0
    obs, state = env._get_obs_state()
    actions = [[0.2, 0.0, 0.0]] # move by 2 units to (9.9,10,10), inside obstacle
    obs, state, rewards, done, info = env.step(actions)
    assert env.colliding_mask[0], f"Grazing obstacle failed: {env.pos[0]}, colliding_mask={env.colliding_mask}"
    print("test_grazing_obstacle passed.")

if __name__ == "__main__":
    test_high_speed_arrival()
    test_grazing_goal()
    test_high_speed_collision()
    test_grazing_obstacle()
    print("All edge case tests passed.")
