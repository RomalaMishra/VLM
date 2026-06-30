"""
Evaluate a trained PPO model on OutdoorNavEnv.

Usage:
    python -m rl.evaluate --model outputs/ppo_bench_final
    python -m rl.evaluate --model outputs/ppo_bench_final --goal bench --episodes 30 --render
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from env.nav_env import OutdoorNavEnv, GOAL_LABELS, REACH_DIST

ACTION_NAMES = ["turn_left", "turn_right", "forward"]


def evaluate(model_path: str, goal: str = "bench", episodes: int = 20, render: bool = False):
    render_mode = "human" if render else None
    env = OutdoorNavEnv(goal_label=goal, render_mode=render_mode)
    model = PPO.load(model_path, env=env)

    print(f"\n[v2 Evaluate] model={model_path} | goal={goal} | episodes={episodes}\n")

    successes, step_counts = [], []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            steps += 1

        dist = env._dist_to_goal()
        success = dist < REACH_DIST
        successes.append(success)
        step_counts.append(steps)
        print(f"  ep {ep:3d} | {'SUCCESS' if success else 'FAIL   '} | steps={steps:4d} | dist={dist:.2f}m")

    env.close()

    print(f"\n{'─'*45}")
    print(f"  Success rate : {np.mean(successes)*100:.1f}%  ({sum(successes)}/{episodes})")
    print(f"  Avg steps    : {np.mean(step_counts):.0f}")
    print(f"{'─'*45}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True)
    p.add_argument("--goal",     default="bench", choices=GOAL_LABELS)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render",   action="store_true")
    args = p.parse_args()
    evaluate(args.model, args.goal, args.episodes, args.render)
