# AI Accessibility Assistant

A navigation assistant for visually impaired users. It takes an image and a natural language question, detects objects using YOLOv8, retrieves safety guidance via RAG, and produces a spoken-style navigation instruction using an LLM.

## How it works

```
Image + Question
      |
      v
 YOLOv8 (object detection)
      |
      v
 FAISS RAG (retrieve safety guidance per detected object)
      |
      v
 Groq / Llama 3.1 (generate navigation instruction)
      |
      v
 guidance
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Build the FAISS index (run once):

```bash
python -m rag.build_index
```

**Example:**

```bash
python main.py photo.jpg "Can I walk forward?"
```

**Output:**

```
Detecting objects...
Detected: person, chair, dining table
Annotated image: outputs/result_photo.jpg

Generating guidance...

--- Guidance ---
There is a person ahead — slow down and move to the side to give them at least 1 meter of space.
A dining table and chairs are present; navigate around the perimeter and watch for chair legs extending into your path.
Proceed slowly toward the left side of the room to avoid these obstacles.
----------------
```

The annotated image with bounding boxes is saved to `outputs/`.


## Limitations (v1)

- Knowledge base is limited to 15 object classes; objects outside this set receive no tailored guidance.
- No spatial reasoning — bounding box positions are not passed to the LLM, so left/right/near/far directionality is not object-specific.
- Single-image input only; no video or real-time stream support.
- Text output only; no text-to-speech integration.
