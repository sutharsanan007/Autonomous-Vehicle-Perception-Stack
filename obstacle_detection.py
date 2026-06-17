# ─────────────────────────────────────────────────────────────────────────────
#  obstacle_detection.py  |  Modular Hybrid ADAS — Obstacle Perception Module
#
#  Changes from original:
#    • CONF_THRESHOLD = 0.45  — the YOLOv8n default (0.25) generates too many
#      false positives on road signs and foliage.  0.45 keeps only confident hits.
#    • ROAD_CLASSES filter — only annotate objects that are actually hazardous
#      on a road.  Without this, YOLO labels trees, signs, and fences which
#      clutter the display and can overlap the lane lines.
# ─────────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO

# Confidence threshold — only detections above this score are rendered.
CONF_THRESHOLD = 0.45

# COCO class indices that are relevant on public roads.
# Full COCO list: https://docs.ultralytics.com/datasets/detect/coco/
ROAD_CLASSES = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    11,  # stop sign
}


def initialize_model(model_path: str = 'yolov8n.pt') -> YOLO:
    """Load and initialise the YOLOv8 obstacle detection model."""
    print(f"Loading YOLO AI Model from {model_path}...")
    return YOLO(model_path)


def detect_obstacles(frame, model: YOLO):
    """
    Run YOLOv8 inference and draw bounding boxes for road-relevant objects.

    Parameters
    ----------
    frame : BGR image (unmodified — we never draw on the caller's copy)
    model : initialised YOLO instance

    Returns
    -------
    Annotated BGR frame with bounding boxes.
    """
    results = model(
        frame,
        stream=True,
        verbose=False,
        conf=CONF_THRESHOLD,
        classes=list(ROAD_CLASSES),
    )
    for r in results:
        frame = r.plot()    # draws boxes, labels, confidence scores
    return frame