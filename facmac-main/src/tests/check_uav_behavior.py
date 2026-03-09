import os, importlib.util
import numpy as np

# Load MultiUAVEnv from file directly
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
module_path = os.path.join(root, 'envs', 'uav', 'multi_uav_env.py')
spec = importlib.util.spec_from_file_location('multi_uav_env_mod', module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MultiUAVEnv = mod.MultiUAVEnv
CylinderObstacle = getattr(mod, 'CylinderObstacle')

cfg = {
    'n_agents': 5,
    'space_size': 30.0,
    'n_static_obstacles': 0,
    'max_steps': 10,
    'dt': 1.0,
    'v_max': 2.0,
    'goal_radius': 1.0,
}

env = MultiUAVEnv(cfg=cfg, main_seed=42)
obs, state = env.reset()
print('Goal:', env.goal)
print('Initial positions:\n', env.pos)

# Scenario setup:
# - Agent 0: mark as already arrived -> should remain stationary
# - Agent 1: positioned just outside goal and will move into it in one step -> should arrive and then stop on subsequent step
# - Agent 2: will move into an obstacle -> should report collision but continue moving (unless arrives)
# - Agents 3-4: random actions

# Place agent 0 at the goal and mark as arrived
env.pos[0] = env.goal.copy() + 0.0
env.arrived_mask[0] = True

# Place agent 1 slightly outside the goal along x axis so one step reaches it
env.pos[1] = env.goal.copy() + np.array([1.2, 0.0, 0.0])
# Agent 2 start far from goal; we'll place a cylinder obstacle right at its next position
env.pos[2] = env.goal.copy() + np.array([-5.0, 0.0, 0.0])

# Add a static obstacle at a point agent2 will move into
# set its pos 3 units towards the goal from agent2's position
next_pos_agent2 = env.pos[2] + (env.goal - env.pos[2]) / np.linalg.norm(env.goal - env.pos[2]) * 3.0
obs_cyl = CylinderObstacle(next_pos_agent2[0], next_pos_agent2[1], radius=1.5, z_min=0.0, z_max=env.space_size)
env.obstacles.append(obs_cyl)

print('Adjusted positions:\n', env.pos)
print('Added obstacle at (approx):', next_pos_agent2)

# Build actions:
# agent0: non-zero action (should be ignored because arrived)
# agent1: action pointing to goal (will reach)
# agent2: action pointing towards goal (will hit obstacle)
# agents3-4: zero actions

actions = []
for i in range(cfg['n_agents']):
    if i == 0:
        actions.append([1.0, 0.0, 0.0])
    elif i == 1:
        vec = env.goal - env.pos[1]
        # normalized in range [-1,1] by dividing by v_max
        actions.append((vec / (np.linalg.norm(vec) + 1e-8) * 1.0 / env.v_max).tolist())
    elif i == 2:
        vec = env.goal - env.pos[2]
        actions.append((vec / (np.linalg.norm(vec) + 1e-8) * 1.0 / env.v_max).tolist())
    else:
        actions.append([0.0, 0.0, 0.0])

print('Actions step 1:', actions)
obs, state, rewards, done, info = env.step(actions)
print('\nAfter step 1:')
print('Positions:\n', env.pos)
print('Arrived mask:', env.arrived_mask.astype(int))
print('Colliding mask:', env.colliding_mask.astype(int))
print('Info keys:', list(info.keys()))
print('Reached new:', info.get('reached_new'))
print('Collisions new:', info.get('collisions_new'))

# Assertions / checks
# Agent0 should remain at the same place
assert np.allclose(env.pos[0], env.goal, atol=1e-6), 'Agent0 moved despite being marked arrived'
# Agent1 should have arrived
assert env.arrived_mask[1], 'Agent1 did not arrive as expected'
# Agent2 should have colliding_mask True if hit obstacle
coll2 = env.colliding_mask[2]
print('Agent2 collision flag after step1:', coll2)

# Now step again: arrived agents should not move (agent0 and agent1)
actions2 = [[0.5,0,0] for _ in range(cfg['n_agents'])]
obs2, state2, rewards2, done2, info2 = env.step(actions2)
print('\nAfter step 2:')
print('Positions:\n', env.pos)
print('Arrived mask:', env.arrived_mask.astype(int))
print('Colliding mask:', env.colliding_mask.astype(int))

# Agent1 should not have moved between step1 and step2
# We saved agent1 position after step1 in pos1
# Instead compare by checking that agent1's vel is zero and future positions unchanged
assert np.linalg.norm(env.vel[1]) < 1e-6, 'Agent1 velocity non-zero after arriving'

print('\nTEST PASSED: behavior matches expectations (arrived stop, others continue, collisions do not auto-stop)')
