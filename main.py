"""
v2 Navigation Assistant — interactive inference.

Loads a trained PPO model and runs the robot in OutdoorNavEnv.
Prints step-by-step telemetry; renders the simulator window if --render is set.

Usage:
    python main.py --model outputs/ppo_bench_final --goal bench
    python main.py --model outputs/ppo_bench_final --goal tree --render

Train first:
    python -m rl.train --goal bench
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from env.nav_env import OutdoorNavEnv, GOAL_LABELS, REACH_DIST

ACTION_NAMES = ["turn_left", "turn_right", "forward"]


def run(model_path: str, goal: str, render: bool = False, episodes: int = 5):
    render_mode = "human" if render else None
    env = OutdoorNavEnv(goal_label=goal, render_mode=render_mode)
    model = PPO.load(model_path, env=env)

    print(f"\n{'='*50}")
    print(f"  v2 Navigation Assistant")
    print(f"  Goal   : {goal.upper()}")
    print(f"  Model  : {model_path}")
    print(f"{'='*50}\n")
    print("Press Ctrl+C to quit.\n")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        ep_start = time.time()
        print(f"--- Episode {ep} ---")

        try:
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                step += 1
                dist = env._dist_to_goal()
                print(
                    f"  step={step:4d} | action={ACTION_NAMES[int(action)]:12s} "
                    f"| dist={dist:.2f}m | reward={reward:+.4f}",
                    end="\r",
                )

            elapsed = time.time() - ep_start
            dist = env._dist_to_goal()
            result = "REACHED GOAL" if dist < REACH_DIST else "TIMEOUT"
            print(f"\n  [{result}] steps={step} | dist={dist:.2f}m | time={elapsed:.1f}s\n")
            time.sleep(0.8)

        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            break

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="v2 Navigation Assistant — inference")
    p.add_argument("--model",    required=True)
    p.add_argument("--goal",     default="bench", choices=GOAL_LABELS)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render",   action="store_true")
    args = p.parse_args()
    run(args.model, args.goal, args.render, args.episodes)
