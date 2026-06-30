# v2 — Navigation Baseline

Second version of the AI Accessibility / Navigation project.

Extends v1 (static accessibility assistant) into a full simulation loop:
a robot navigates an outdoor-like environment using only RGB camera input,
trained with PPO (Proximal Policy Optimisation).

```
RGB Camera Frame
      |
      v
 YOLOv8n (detect_frame)     
      |
      v
 Structured Observation
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

## Setup
```bash
cd v2
pip install -r requirements.txt
```

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
