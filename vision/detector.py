"""
YOLO object detector — v2 extension of v1/vision/detector.py.

New in v2: detect_frame(frame) accepts a numpy array directly,
so the same detector works on simulator camera output without
writing to disk first.

Model resolution order for yolov8n.pt:
  1. v2/yolov8n.pt          (local copy inside this folder)
  2. VLM/yolov8n.pt         (shared with v1 during dev — parent dir)
  3. auto-download           (ultralytics fetches it on first use)
"""

from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

_V2_ROOT = Path(__file__).parent.parent
_V1_ROOT = _V2_ROOT.parent
_MODEL_PATH: str = (
    str(_V2_ROOT / "yolov8n.pt") if (_V2_ROOT / "yolov8n.pt").exists()
    else str(_V1_ROOT / "yolov8n.pt") if (_V1_ROOT / "yolov8n.pt").exists()
    else "yolov8n.pt"
)

_model = None


def _load_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(_MODEL_PATH)
    return _model


def detect(image_path: str, conf_threshold: float = 0.4) -> dict:
    """Detect from a file path (unchanged from v1)."""
    model = _load_model()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return _run(model, img, conf_threshold, Path(image_path).stem)


def detect_frame(frame: np.ndarray, conf_threshold: float = 0.4,
                 save_name: str = "frame") -> dict:
    """
    Detect from a numpy array (H, W, 3) uint8 RGB image.
    Used to feed simulator camera output directly into YOLO.

    Returns:
        {
            "objects":        list[str],   # unique detected class names
            "detections":     list[dict],  # label, confidence, box per detection
            "annotated_path": str,         # saved annotated image path
        }
    """
    model = _load_model()
    img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    return _run(model, img, conf_threshold, save_name)


def _run(model: YOLO, img_bgr: np.ndarray, conf_threshold: float, stem: str) -> dict:
    results = model(img_bgr, conf=conf_threshold, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = round(float(box.conf[0]), 2)
        label = model.names[int(box.cls[0])]
        detections.append({"label": label, "confidence": conf, "box": [x1, y1, x2, y2]})
        _draw_box(img_bgr, label, conf, x1, y1, x2, y2)

    out_path = str(OUTPUTS_DIR / f"result_{stem}.jpg")
    cv2.imwrite(out_path, img_bgr)

    return {
        "objects":        list({d["label"] for d in detections}),
        "detections":     detections,
        "annotated_path": out_path,
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
    if len(sys.argv) > 1:
        result = detect(sys.argv[1])
    else:
        dummy = np.random.randint(0, 255, (80, 60, 3), dtype=np.uint8)
        result = detect_frame(dummy, save_name="smoke_test")
        print("detect_frame() smoke test (random frame — no real detections expected)")
    print(f"Detected objects : {result['objects']}")
    print(f"Detections       : {len(result['detections'])}")
    print(f"Annotated image  : {result['annotated_path']}")
