"""
Outdoor-style navigation environment built on MiniWorld.

Layout: open flat room with 4 named goal objects (coloured boxes)
        and N grey obstacle boxes. All positions randomised at reset (L1 dynamic).

Observation space (Dict):
    "image" : uint8  (H, W, 3)   — first-person RGB camera
    "goal"  : float32 (N_GOALS,) — one-hot vector for active goal

Action space: Discrete(3)
    0 → turn left
    1 → turn right
    2 → move forward

Reward:
    +5.0   on reaching goal (dist < REACH_DIST)
    shaping: (prev_dist - curr_dist) * 2.0  (rewards getting closer)
    -0.002 per step (encourages efficiency)
"""

import numpy as np
import gymnasium as gym
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box

GOAL_LABELS = ["bench", "tree", "cone", "barrier"]
GOAL_COLORS = {
    "bench":   [1.0, 0.85, 0.20],
    "tree":    [0.10, 0.65, 0.10],
    "cone":    [1.0,  0.35, 0.00],
    "barrier": [0.70, 0.10, 0.80],
}
OBSTACLE_COLOR = [0.35, 0.35, 0.35]
N_GOALS = len(GOAL_LABELS)
REACH_DIST = 1.2


class OutdoorNavEnv(MiniWorldEnv):
    """Open outdoor-like navigation environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        goal_label: str = "bench",
        n_obstacles: int = 4,
        max_episode_steps: int = 500,
        render_mode=None,
        **kwargs,
    ):
        assert goal_label in GOAL_LABELS, f"Unknown goal '{goal_label}'. Choose from {GOAL_LABELS}"
        self.goal_label = goal_label
        self.n_obstacles = n_obstacles
        self._goal_entity = None
        self._prev_dist = 0.0

        super().__init__(
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            **kwargs,
        )

        H, W = self.obs_height, self.obs_width
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
            "goal":  gym.spaces.Box(low=0.0, high=1.0, shape=(N_GOALS,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(3)

    def _gen_world(self):
        self.add_rect_room(
            min_x=-8, max_x=8,
            min_z=-8, max_z=8,
            wall_tex="brick_wall",
            floor_tex="floor_tiles_bw",
        )

        self._goal_entity = None
        for label, color in GOAL_COLORS.items():
            ent = self.place_entity(Box(color=color, size=0.6))
            ent._nav_label = label
            if label == self.goal_label:
                self._goal_entity = ent

        for _ in range(self.n_obstacles):
            self.place_entity(Box(color=OBSTACLE_COLOR, size=0.45))

        self.place_agent()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_dist = self._dist_to_goal()
        return self._wrap(obs), info

    def step(self, action: int):
        action_map = [
            self.actions.turn_left,
            self.actions.turn_right,
            self.actions.move_forward,
        ]
        raw_obs, _, terminated, truncated, info = super().step(int(action_map[action]))

        dist = self._dist_to_goal()
        reached = dist < REACH_DIST
        reward = self._reward(dist, reached)
        self._prev_dist = dist

        return self._wrap(raw_obs), reward, reached or terminated, truncated, info

    def _wrap(self, image: np.ndarray) -> dict:
        goal_vec = np.zeros(N_GOALS, dtype=np.float32)
        goal_vec[GOAL_LABELS.index(self.goal_label)] = 1.0
        return {"image": image, "goal": goal_vec}

    def _dist_to_goal(self) -> float:
        if self._goal_entity is None:
            return 0.0
        ax, az = self.agent.pos[0], self.agent.pos[2]
        gx, gz = self._goal_entity.pos[0], self._goal_entity.pos[2]
        return float(np.sqrt((ax - gx) ** 2 + (az - gz) ** 2))

    def _reward(self, dist: float, reached: bool) -> float:
        if reached:
            return 5.0
        return (self._prev_dist - dist) * 2.0 - 0.002


if __name__ == "__main__":
    env = OutdoorNavEnv(goal_label="bench", render_mode="human")
    obs, _ = env.reset()
    print(f"Image shape  : {obs['image'].shape}")
    print(f"Goal vector  : {obs['goal']}")
    print(f"Action space : {env.action_space}")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"  step={step+1} action={action} reward={reward:.4f} dist={env._dist_to_goal():.2f}m")
        if terminated or truncated:
            break
    env.close()
    print("Smoke test passed.")
