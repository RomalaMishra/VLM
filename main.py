"""
AI Accessibility Assistant
Usage: python main.py <image_path> "<question>"
Example: python main.py test.jpg "Can I walk forward?"
"""

import sys
import argparse
from pathlib import Path

from vision.detector import detect
from llm.chain import answer


def run(image_path: str, question: str):
    image_path = str(Path(image_path).resolve())

    print("Detecting objects...")
    result = detect(image_path)

    objects = result["objects"]
    annotated_path = result["annotated_path"]

    if not objects:
        print("No objects detected in the image.")
        return

    print(f"Detected: {', '.join(objects)}")
    print(f"Annotated image: {annotated_path}")
    print("\nGenerating guidance...")

    guidance = answer(objects, question)

    print("\n--- Guidance ---")
    print(guidance)
    print("----------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Accessibility Assistant")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("question", help='Navigation question e.g. "Can I walk forward?"')
    args = parser.parse_args()

    run(args.image, args.question)
