"""
Runs YOLOv8 on an image, draws bounding boxes, and saves the annotated image.
Returns the list of detected object names for downstream RAG + LLM use.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

_model = None


def _load_model() -> YOLO:
    global _model
    if _model is None:
        # yolov8n = nano, fastest, good enough for v1
        _model = YOLO("yolov8n.pt")
    return _model


def detect(image_path: str, conf_threshold: float = 0.4) -> dict:
    """
    Args:
        image_path: path to input image
        conf_threshold: minimum confidence to count a detection

    Returns:
        {
            "objects": ["chair", "person"],          # unique detected class names
            "detections": [                          # all raw detections
                {"label": "chair", "confidence": 0.87, "box": [x1, y1, x2, y2]},
                ...
            ],
            "annotated_path": "outputs/result_<name>.jpg"
        }
    """
    model = _load_model()
    results = model(image_path, conf=conf_threshold, verbose=False)[0]

    img = cv2.imread(image_path)
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = round(float(box.conf[0]), 2)
        label = model.names[int(box.cls[0])]

        detections.append({"label": label, "confidence": conf, "box": [x1, y1, x2, y2]})
        _draw_box(img, label, conf, x1, y1, x2, y2)

    output_name = f"result_{Path(image_path).stem}.jpg"
    output_path = str(OUTPUTS_DIR / output_name)
    cv2.imwrite(output_path, img)

    unique_objects = list({d["label"] for d in detections})

    return {
        "objects": unique_objects,
        "detections": detections,
        "annotated_path": output_path,
    }


def _draw_box(img, label: str, conf: float, x1: int, y1: int, x2: int, y2: int):
    color = (0, 200, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Use ultralytics built-in sample image if no argument given
        from ultralytics.utils import ASSETS
        test_image = str(ASSETS / "bus.jpg")
        print(f"No image provided. Using built-in sample: {test_image}")
    else:
        test_image = sys.argv[1]

    result = detect(test_image)
    print(f"Detected objects: {result['objects']}")
    print(f"Annotated image saved to: {result['annotated_path']}")
    for d in result["detections"]:
        print(f"  {d['label']} ({d['confidence']}) @ {d['box']}")
