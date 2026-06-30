# v2 — Navigation Baseline

Second version of the AI Accessibility / Navigation project.

Extends v1 (static accessibility assistant) into a full simulation loop:
a robot navigates an outdoor-like environment using only RGB camera input,
trained with PPO (Proximal Policy Optimisation).

```
RGB Camera Frame
      |
      v
 YOLOv8n (detect_frame)        ← adapted from v1
      |
      v
 Structured Observation
 {image: (H,W,3), goal: (4,)}
      |
      v
 PPO Policy (stable-baselines3)
      |
      v
 Action: turn_left / turn_right / move_forward
      |
      v
 MiniWorld Simulator (OutdoorNavEnv)
```

**What is new vs v1:**
- Simulation environment (`env/nav_env.py`) with L1 dynamic obstacle placement
- `detect_frame()` in `vision/detector.py` — feeds simulator frames into YOLO
- Depth Anything V2 Small (`perception/depth.py`) — monocular depth on any RGB frame
- PPO training + evaluation scripts (`rl/train.py`, `rl/evaluate.py`)
- Goal-conditioned policy: robot navigates to a named object (bench / tree / cone / barrier)

**What is coming in v3:**
- Depth + YOLO integrated into the training observation
- Groq vision endpoint (VLM) for natural language goal input
- Real-time QnA: user types to change destination or ask the robot questions
- L2 dynamic obstacles (moving on fixed paths)

---

## Setup

```bash
cd v2
pip install -r requirements.txt
```

No `.env` file is needed for v2.

---

## Usage

### Train
```bash
python -m rl.train --goal bench --timesteps 500000 --envs 2
```

| Flag | Default | Options |
|------|---------|---------|
| `--goal` | `bench` | `bench` / `tree` / `cone` / `barrier` |
| `--timesteps` | `500000` | any int |
| `--envs` | `2` | parallel training envs |

Monitor: `tensorboard --logdir outputs/tensorboard`

### Evaluate
```bash
python -m rl.evaluate --model outputs/ppo_bench_final --episodes 20 --render
```

### Inference
```bash
python main.py --model outputs/ppo_bench_final --goal bench --render
```

---

## Project Layout

```
v2/
├── env/
│   └── nav_env.py       # OutdoorNavEnv — MiniWorld gym environment
├── vision/
│   └── detector.py      # YOLOv8 + detect_frame() for numpy input
├── perception/
│   └── depth.py         # Depth Anything V2 Small (CPU-safe)
├── rl/
│   ├── train.py         # PPO training (stable-baselines3)
│   └── evaluate.py      # Success rate evaluation
├── outputs/             # Models, checkpoints, tensorboard logs
├── main.py              # Interactive inference entry point
├── requirements.txt
├── .env.example         # Placeholder for v3 API keys
└── README.md
```

---

## Environment Details

| Property | Value |
|----------|-------|
| Simulator | MiniWorld (Farama Foundation) |
| Observation | `Dict{"image": uint8 (H,W,3), "goal": float32 (4,)}` |
| Action space | `Discrete(3)` — turn_left, turn_right, move_forward |
| Goal objects | bench (yellow), tree (green), cone (orange), barrier (purple) |
| Obstacles | 4 grey boxes, random positions each episode (L1 dynamic) |
| Reach distance | 1.2 m |
| Reward | `+5.0` at goal · `(prev_dist − dist) × 2.0 − 0.002` per step |
| Max steps | 500 per episode |
