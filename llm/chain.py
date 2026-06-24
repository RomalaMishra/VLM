"""
Combines YOLO detections + RAG guidance + user question into a
natural language navigation answer via Groq (Llama 3).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from groq import Groq

from rag.retriever import retrieve

load_dotenv()

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=api_key)
    return _client


def _truncate(text: str, max_sentences: int = 2) -> str:
    sentences = text.split(". ")
    return ". ".join(sentences[:max_sentences]).strip(".") + "."


def _build_prompt(detected_objects: list[str], guidance_chunks: list[str], question: str) -> str:
    objects_str = ", ".join(detected_objects) if detected_objects else "none"
    guidance_str = " | ".join(guidance_chunks) if guidance_chunks else "No guidance available."

    return (
        f"Visually impaired user. Scene objects: {objects_str}. "
        f"Safety guidance: {guidance_str} "
        f"Question: {question} "
        f"Give 2-3 sentence navigation instruction. Mention direction. Prioritize danger. Plain text only, no markdown."
    )


def answer(detected_objects: list[str], question: str) -> str:
    raw_chunks = retrieve(detected_objects, top_k=2)
    guidance_chunks = [_truncate(c) for c in raw_chunks]
    prompt = _build_prompt(detected_objects, guidance_chunks, question)

    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Smoke test using objects the bus.jpg sample would detect
    test_objects = ["person", "bus", "truck"]
    test_question = "Is it safe to walk forward?"

    print(f"Objects: {test_objects}")
    print(f"Question: {test_question}")
    print("\nAnswer:")
    print(answer(test_objects, test_question))
