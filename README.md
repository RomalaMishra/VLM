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
