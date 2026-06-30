"""
PPO training for OutdoorNavEnv.

Usage:
    python -m rl.train                         # defaults
    python -m rl.train --goal tree
    python -m rl.train --goal bench --timesteps 1000000 --envs 4

Outputs:
    outputs/tensorboard/          — TensorBoard logs
    outputs/checkpoints/          — periodic model saves
    outputs/ppo_<goal>_final.zip  — final trained model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.nav_env import OutdoorNavEnv, GOAL_LABELS

OUTPUTS = Path(__file__).parent.parent / "outputs"


def train(goal: str = "bench", timesteps: int = 500_000, n_envs: int = 2):
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "checkpoints").mkdir(exist_ok=True)

    print(f"\n[v2 Train] goal={goal} | steps={timesteps:,} | parallel_envs={n_envs}")
    print(f"  Outputs → {OUTPUTS}\n")

    vec_env = make_vec_env(
        lambda: OutdoorNavEnv(goal_label=goal),
        n_envs=n_envs,
    )
    eval_env = Monitor(OutdoorNavEnv(goal_label=goal))

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(OUTPUTS / "tensorboard"),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=str(OUTPUTS / "checkpoints"),
            name_prefix=f"ppo_{goal}",
            verbose=0,
        ),
        EvalCallback(
            eval_env,
            eval_freq=max(25_000 // n_envs, 1),
            n_eval_episodes=5,
            best_model_save_path=str(OUTPUTS / "best"),
            log_path=str(OUTPUTS / "eval_logs"),
            verbose=1,
        ),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    save_path = str(OUTPUTS / f"ppo_{goal}_final")
    model.save(save_path)
    vec_env.close()
    eval_env.close()
    print(f"\n[v2 Train] Done. Model saved → {save_path}.zip")
    print(f"  TensorBoard: tensorboard --logdir {OUTPUTS / 'tensorboard'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train PPO on OutdoorNavEnv")
    p.add_argument("--goal",      default="bench", choices=GOAL_LABELS)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--envs",      type=int, default=2)
    args = p.parse_args()
    train(args.goal, args.timesteps, args.envs)
