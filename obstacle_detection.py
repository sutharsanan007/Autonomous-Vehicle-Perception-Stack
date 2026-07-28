# ─────────────────────────────────────────────────────────────────────────────
#  obstacle_detection.py  |  Modular Hybrid ADAS — Obstacle Perception Module
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import time
from ultralytics import YOLO

# Confidence threshold
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

# ── FCW & TTC Constants ───────────────────────────────────────────────────────
FOCAL_LENGTH  = 800.0   
KNOWN_WIDTH   = 1.8     
TTC_THRESHOLD = 2.5     
MIN_SAFE_DIST = 5.0     

# ── Persistent State Tracker ──────────────────────────────────────────────────
_ego_state = {"dist": None, "time": None, "v_rel": 0.0}
_track_history = {}


def initialize_model(model_path: str = 'yolov8n.pt') -> YOLO:
    print(f"Loading YOLO AI Model from {model_path}...")
    return YOLO(model_path)


def detect_obstacles(frame, model: YOLO):
    """
    Returns: (annotated_frame, fcw_warning_flag, traffic_alert_string, vehicle_boxes, radar_targets)
    """
    results = model.track(
        frame,
        persist=True,
        stream=True,
        verbose=False,
        conf=CONF_THRESHOLD,
        classes=list(ROAD_CLASSES),
    )
    
    h, w = frame.shape[:2]
    ego_left_bound  = w * 0.35
    ego_right_bound = w * 0.65
    
    fcw_warning   = False  
    traffic_alert = None
    current_time  = time.time()
    
    annotated_frame = frame.copy()
    closest_ego_dist = None
    
    # ── SENSOR FUSION & RADAR DATA ──
    vehicle_boxes = []
    radar_targets = []  # Stores (distance, center_x, color) for the HUD map

    for r in results:
        if r.boxes.id is None:
            continue
            
        track_ids = r.boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(r.boxes, track_ids):
            cls_id = int(box.cls[0])
            
            if cls_id == 9:
                traffic_alert = "🚥 TRAFFIC LIGHT AHEAD"
                continue
            elif cls_id == 11:
                traffic_alert = "🛑 STOP SIGN AHEAD"
                continue
            
            if cls_id in [2, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pixel_width = x2 - x1
                
                vehicle_boxes.append((x1, y1, x2, y2))
                
                if pixel_width > 0:
                    dist = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
                    cx = (x1 + x2) / 2.0
                    
                    in_path = ego_left_bound < cx < ego_right_bound
                    if in_path:
                        if closest_ego_dist is None or dist < closest_ego_dist:
                            closest_ego_dist = dist

                    if track_id not in _track_history:
                        _track_history[track_id] = {'times': [], 'dists': []}
                        
                    hist = _track_history[track_id]
                    hist['times'].append(current_time)
                    hist['dists'].append(dist)
                    
                    if len(hist['times']) > 15:
                        hist['times'].pop(0)
                        hist['dists'].pop(0)
                        
                    is_stationary_parked = False
                    
                    if len(hist['times']) > 5:
                        dt = hist['times'][-1] - hist['times'][0]
                        if dt > 0:
                            dist_change = hist['dists'][0] - hist['dists'][-1]
                            closing_speed = dist_change / dt 
                            
                            if closing_speed > 6.0 and not in_path:
                                is_stationary_parked = True
                    
                    if not is_stationary_parked:
                        # ── Dynamic RGB Target Grading ──
                        if in_path:
                            d_clamp = max(5.0, min(dist, 25.0))
                            ratio = (d_clamp - 5.0) / 20.0
                            r = 255 if ratio < 0.5 else int(255 * (1.0 - ratio) * 2)
                            g = 255 if ratio > 0.5 else int(255 * ratio * 2)
                            color = (0, g, r) 
                        else:
                            color = (255, 200, 0) # Cyan for adjacent lanes
                        
                        # Add object to Radar Map
                        radar_targets.append((dist, cx, color))
                        
                        L = 20  
                        t = 2   
                        
                        cv2.line(annotated_frame, (int(x1), int(y1)), (int(x1)+L, int(y1)), color, t)
                        cv2.line(annotated_frame, (int(x1), int(y1)), (int(x1), int(y1)+L), color, t)
                        cv2.line(annotated_frame, (int(x2), int(y1)), (int(x2)-L, int(y1)), color, t)
                        cv2.line(annotated_frame, (int(x2), int(y1)), (int(x2), int(y1)+L), color, t)
                        cv2.line(annotated_frame, (int(x1), int(y2)), (int(x1)+L, int(y2)), color, t)
                        cv2.line(annotated_frame, (int(x1), int(y2)), (int(x1), int(y2)-L), color, t)
                        cv2.line(annotated_frame, (int(x2), int(y2)), (int(x2)-L, int(y2)), color, t)
                        cv2.line(annotated_frame, (int(x2), int(y2)), (int(x2), int(y2)-L), color, t)
                        
                        text_y = max(20, int(y1) - 10)
                        cv2.putText(annotated_frame, f"Dist: {dist:.1f}m", 
                                    (int(x1), text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Advanced TTC Math
    if closest_ego_dist is not None:
        if _ego_state["dist"] is not None:
            dt = current_time - _ego_state["time"]
            if 0 < dt < 1.0:
                raw_v_rel = (_ego_state["dist"] - closest_ego_dist) / dt
                _ego_state["v_rel"] = (_ego_state["v_rel"] * 0.75) + (raw_v_rel * 0.25)
                
                if _ego_state["v_rel"] > 0.5:
                    ttc = closest_ego_dist / _ego_state["v_rel"]
                    if ttc < TTC_THRESHOLD:
                        fcw_warning = True
        
        if closest_ego_dist < MIN_SAFE_DIST:
            fcw_warning = True
            
        _ego_state["dist"] = closest_ego_dist
        _ego_state["time"] = current_time
    else:
        _ego_state["dist"] = None
        _ego_state["v_rel"] = 0.0

    return annotated_frame, fcw_warning, traffic_alert, vehicle_boxes, radar_targets