# AI Navigation Research Project

A research-oriented, vision-language robot navigation system built progressively
across four versions. The project begins with a static accessibility assistant and
evolves into a full embodied AI system where a robot navigates a simulated
environment, understands its surroundings through a Vision Language Model, and
explains every decision in natural language.

---

## Final Research Vision (v4)

```
RGB Camera Feed
        |
        v
YOLOv8 + Depth Anything V2
(Structured Perception — object labels, positions, depth bins)
        |
        v
Scene State  [(label, x, depth_bin), ...]
        |
        +─────────────────────────────────────+
        |                                     |
        v  every step (~10 Hz)               v  async (~0.5 Hz)
PPO Navigation Policy               Vision Language Model
→ turn / move / stop                → parse natural language goal
                                    → assess scene risk
                                    → generate explanation
        |                                     |
        +─────────────────────────────────────+
        |
        v
Robot Action in Simulation
```

The VLM enables four capabilities a standalone policy cannot provide:

| Capability | Example |
|------------|---------|
| Language goal input | *"Go to the bench"* |
| Dynamic replanning | *"Change destination to the tree"* |
| Real-time Q&A | *"Why did you stop?"* |
| Decision explanation | *"Obstacle entered path — slowing down"* |

---

## Versions

### v1 — AI Accessibility Assistant
**Branch:** [`v1`](../../tree/v1) · **Author:** RomalaMishra

A static vision-language pipeline designed to assist visually impaired users.
Given a single image and a natural language question, the system detects objects
with YOLOv8, retrieves safety guidance from a FAISS vector store via RAG, and
produces a spoken-style navigation instruction using Groq / Llama 3.1.

```
Image + Question
      |
      v
YOLOv8n  (object detection)
      |
      v
FAISS RAG  (retrieve safety guidance per detected object)
      |
      v
Groq / Llama 3.1 8B  (generate navigation instruction)
      |
      v
Guidance text
```

**Stack**

| Component | Technology |
|-----------|-----------|
| Object detection | YOLOv8n (Ultralytics) |
| Knowledge retrieval | FAISS + LangChain RAG |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Language model | Groq API — Llama 3.1 8B Instant |
| Knowledge base | 15 accessibility object classes |

**Limitations (motivating v2+)**
- Static single-image input — no simulation loop
- No spatial reasoning — bounding box positions not used
- Text output only — no real-time interaction or replanning
- Knowledge base limited to 15 COCO-class objects

---

### v2 — Navigation Baseline
**Branch:** [`v2`](../../tree/v2) · **Author:** Sree-harsh

Extends the v1 perception pipeline into a full simulation loop. A robot navigates
an outdoor-like MiniWorld environment using only RGB camera input, trained with
Proximal Policy Optimisation (PPO). Introduces structured visual observation,
goal-conditioned navigation, and monocular depth estimation — laying the foundation
for VLM integration in v3.

```
Simulator RGB Frame
      |
      v
YOLOv8n  detect_frame()   ←  adapted from v1 (numpy array input)
      |
      v
Depth Anything V2 Small   (monocular depth → near / mid / far bins)
      |
      v
Structured Observation
{ image: (H,W,3),  goal: one-hot (4,) }
      |
      v
PPO Policy  (stable-baselines3 MultiInputPolicy)
      |
      v
Action: turn_left | turn_right | move_forward
      |
      v
MiniWorld Simulator  (OutdoorNavEnv)
```

**Environment design**

| Property | Value |
|----------|-------|
| Simulator | MiniWorld (Farama Foundation) |
| Scene | Open flat room, 16×16 m |
| Goal objects | bench · tree · cone · barrier (coloured boxes) |
| Obstacles | 4 grey boxes, random positions each episode (L1 dynamic) |
| Reach condition | Distance < 1.2 m |
| Reward | +5.0 at goal · (Δdist × 2.0) − 0.002 per step |
| Max steps | 500 per episode |

**RL training**

| Hyperparameter | Value |
|---------------|-------|
| Algorithm | PPO |
| Policy | MultiInputPolicy (CNN + MLP) |
| Learning rate | 3 × 10⁻⁴ |
| Steps per update | 512 per env |
| Batch size | 64 |
| Entropy coefficient | 0.01 |

**What transfers from v1:** `detect_frame()` wraps the v1 YOLO pipeline to accept
numpy arrays directly from the simulator. The RAG knowledge base and Groq chain
are extended in v3.

---

### v3 — VLM Integration + Real-Time Q&A *(planned)*
**Author:** Sree-harsh

Connects the v2 navigation stack to a Vision Language Model running asynchronously
alongside the PPO policy. The VLM operates at ~0.5 Hz in a background thread;
the policy runs at ~10 Hz uninterrupted.

**New in v3**
- Groq vision endpoint replaces text-only chain — VLM receives the actual camera frame
- Natural language goal input: *"Go to the bench"* → VLM extracts goal label → policy navigates
- Real-time Q&A terminal: user types freely while the robot keeps moving
- L2 dynamic obstacles — one or two objects move on fixed paths each episode
- Knowledge base extended with outdoor object classes

**Q&A architecture**

```
Main thread     PPO control loop           ~10 Hz
Input thread    waits for user text  →  query queue
VLM thread      reads queue  →  Groq vision call  →  response queue
Display thread  prints responses
```

Two query types:
- **Command** — *"Go to the tree"* → updates shared goal state immediately
- **Query** — *"Why did you stop?"* → VLM answers from current frame + scene state

---

### v4 — Full Research System *(planned)*
**Authors:** RomalaMishra · Sree-harsh

The complete pipeline. Ablation study comparing PPO-only vs PPO+VLM across
success rate, steps-to-goal, and response to language. Demo video with bounding
boxes, scene state overlay, VLM explanations, and action labels.

**Research claim:** *The VLM layer adds goal generalisation, dynamic replanning,
and interpretability that no standalone RL policy can provide.*

---

## Branch Guide

| Branch | Content | Author |
|--------|---------|--------|
| `main` | Project overview and roadmap | RomalaMishra · Sree-harsh |
| `v1` | AI Accessibility Assistant | RomalaMishra |
| `v2` | Navigation Baseline (PPO + MiniWorld) | Sree-harsh |
| `v3` | VLM Integration + Q&A *(coming)* | Sree-harsh |
| `v4` | Full Research System *(coming)* | RomalaMishra · Sree-harsh |

---

## Setup

Each version is self-contained. Navigate to the branch and follow its own README.

```bash
# v1
git checkout v1
pip install -r requirements.txt

# v2
git checkout v2
pip install -r requirements.txt
python -m rl.train --goal bench
```
