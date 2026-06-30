"""
Monocular depth estimation using Depth Anything V2 Small (HuggingFace).

Runs on CPU (~200 ms/frame on a modern laptop).
Model is lazy-loaded on first call and cached for the session.

Public API:
    estimate(frame)                            -> (H, W) float32 depth map
    depth_bin(depth_map, box)                  -> "near" | "mid" | "far"
    annotate(frame, detections, depth_map)     -> annotated RGB image
"""

import numpy as np
from PIL import Image

_pipe = None


def _load():
    global _pipe
    if _pipe is None:
        from transformers import pipeline as hf_pipeline
        print("[depth] Loading Depth Anything V2 Small — first call only...")
        _pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
        print("[depth] Model loaded.")
    return _pipe


def estimate(frame: np.ndarray) -> np.ndarray:
    """
    Args:
        frame: (H, W, 3) uint8 RGB image

    Returns:
        (H, W) float32 depth map — larger values = further away.
    """
    pipe = _load()
    result = pipe(Image.fromarray(frame))
    return np.array(result["depth"], dtype=np.float32)


def depth_bin(depth_map: np.ndarray, box: list) -> str:
    """
    Classify depth of a bounding box region.

    Args:
        depth_map : (H, W) float32 from estimate()
        box       : [x1, y1, x2, y2]

    Returns:
        "near" | "mid" | "far" | "unknown"
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    d_min, d_max = depth_map.min(), depth_map.max()
    norm = (roi.mean() - d_min) / (d_max - d_min + 1e-6)
    if norm < 0.33:
        return "near"
    if norm < 0.66:
        return "mid"
    return "far"


def annotate(frame: np.ndarray, detections: list, depth_map: np.ndarray) -> np.ndarray:
    """Overlay depth bins on bounding boxes. Returns annotated RGB copy."""
    import cv2
    out = frame.copy()
    for det in detections:
        label = det["label"]
        box = det["box"]
        bin_label = depth_bin(depth_map, box)
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(out, f"{label} [{bin_label}]", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)
    return out


if __name__ == "__main__":
    import sys, cv2
    if len(sys.argv) > 1:
        frame = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    else:
        print("No image provided. Using random 480x640 frame.")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = estimate(frame)
    print(f"Depth map shape : {depth.shape}")
    print(f"Depth range     : {depth.min():.2f} – {depth.max():.2f}")
    print(f"Depth bin [50,50,200,200]: {depth_bin(depth, [50, 50, 200, 200])}")
