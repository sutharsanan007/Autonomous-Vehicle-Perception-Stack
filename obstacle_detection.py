# ─────────────────────────────────────────────────────────────────────────────
#  obstacle_detection.py  |  Modular Hybrid ADAS — Obstacle Perception Module
# ─────────────────────────────────────────────────────────────────────────────

import cv2
from ultralytics import YOLO

# Confidence threshold — only detections above this score are rendered.
CONF_THRESHOLD = 0.45

# COCO class indices that are relevant on public roads.
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

# ── FCW Constants ─────────────────────────────────────────────────────────────
FOCAL_LENGTH = 800.0   # Estimated pixel focal length for a standard dashcam
KNOWN_WIDTH  = 1.8     # Average width of a car in meters
FCW_DISTANCE = 10.0    # Warning triggers if a car is closer than 10 meters


def initialize_model(model_path: str = 'yolov8n.pt') -> YOLO:
    """Load and initialise the YOLOv8 obstacle detection model."""
    print(f"Loading YOLO AI Model from {model_path}...")
    return YOLO(model_path)


def detect_obstacles(frame, model: YOLO):
    """
    Run YOLOv8 inference, calculate distance to vehicles, and trigger FCW.
    Returns: (annotated_frame, fcw_warning_flag)
    """
    results = model(
        frame,
        stream=True,
        verbose=False,
        conf=CONF_THRESHOLD,
        classes=list(ROAD_CLASSES),
    )
    
    h, w = frame.shape[:2]
    # Define our "Ego Lane" (the path directly in front of our car)
    ego_left_bound  = w * 0.35
    ego_right_bound = w * 0.65
    
    fcw_warning: bool = False  
    
    # We will draw our own text over the YOLO plotted frame
    annotated_frame = frame.copy()

    for r in results:
        annotated_frame = r.plot()  # Draw base YOLO boxes and labels
        
        for box in r.boxes:
            cls_id = int(box.cls[0])
            
            # Only calculate distance for vehicles (Car, Bus, Truck)
            if cls_id in [2, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pixel_width = x2 - x1
                
                if pixel_width > 0:
                    # Pinhole Camera Math: Distance = (Real Width * Focal Length) / Pixel Width
                    dist = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
                    
                    cx = (x1 + x2) / 2.0
                    
                    # Check if the vehicle is in our lane AND dangerously close
                    in_path  = ego_left_bound < cx < ego_right_bound
                    is_close = dist < FCW_DISTANCE
                    
                    if in_path and is_close:
                        fcw_warning = True
                        color = (0, 0, 255) # Red text for danger
                    else:
                        color = (255, 255, 0) # Cyan text for safe distance
                    
                    # FIXED PLACEMENT: Shifted UP by 35 pixels
                    # This clears the default YOLO label so they stack perfectly
                    # Added a safeguard (max) so it doesn't clip off the top of the screen
                    text_y = max(20, int(y1) - 35)
                    
                    cv2.putText(annotated_frame, f"Dist: {dist:.1f}m", 
                                (int(x1), text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return annotated_frame, fcw_warning