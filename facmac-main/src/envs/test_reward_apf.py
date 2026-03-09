"""Quick smoke test for APF reward component in src.envs.reward.compute_step_reward

Run this file directly (python src/envs/test_reward_apf.py) to see printed outputs.
"""
from reward import compute_step_reward
import numpy as np


class CircleObstacle:
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)

    def collides_point(self, p):
        return float(np.linalg.norm(np.array(p) - self.center)) <= self.radius

    def distance_to_point(self, p):
        return float(np.linalg.norm(np.array(p) - self.center) - self.radius)


def run_smoke():
    # Two agents: one moves closer to the goal, one moves away / stays
    prev_positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    positions = np.array([[0.5, 0.0, 0.0], [4.5, 0.0, 0.0]])
    velocities = np.zeros_like(positions)
    goal = np.array([10.0, 0.0, 0.0])

    # Place a circular obstacle near agent 2
    obs = CircleObstacle(center=[5.0, 0.0, 0.0], radius=0.5)
    obstacles = [obs]

    rewards, info = compute_step_reward(positions, velocities, goal, obstacles,
                                        prev_positions=prev_positions, goal_radius=1.0, v_max=1.0)

    print('Rewards:', rewards)
    print('Info reached:', info['reached'])
    print('Info collisions:', info['collisions'])
    print('Components:')
    for i, c in enumerate(info['components']):
        print(f' Agent {i}:', c)


if __name__ == '__main__':
    run_smoke()
