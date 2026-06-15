from ultralytics import YOLO

def initialize_model(model_path='yolov8n.pt'):
    """Loads and initializes the YOLOv8 AI model."""
    print(f"Loading YOLO AI Model from {model_path}...")
    return YOLO(model_path)

def detect_obstacles(frame, model):
    """Feeds the frame to the AI and draws bounding boxes."""
    results = model(frame, stream=True, verbose=False)
    for r in results:
        frame = r.plot() # Automatically draws the boxes
    return frame