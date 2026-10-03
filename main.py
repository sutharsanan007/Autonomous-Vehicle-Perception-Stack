# ─────────────────────────────────────────────────────────────────────────────
#  main.py  |  Modular Hybrid ADAS — Orchestrator (v6.7 - UI Mastery + Radar)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import cv2
import numpy as np
import time
from obstacle_detection import initialize_model, detect_obstacles
from lane_detection import process_lanes, WARNING_THRESH

def resolve_source(arg: str):
    return int(arg) if arg.isdigit() else arg

def draw_hud(image, stats: dict, fps: float, fcw_warning: bool, traffic_alert: str, radar_targets: list):
    h, w = image.shape[:2]

    # ── Emergency Collision Override ──
    if fcw_warning:
        red_tint = image.copy()
        red_tint[:] = (0, 0, 200) 
        image = cv2.addWeighted(red_tint, 0.35, image, 0.65, 0)
        warn_text = "!!! BRAKE - COLLISION WARNING !!!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(warn_text, font, 1.2, 4)
        tx, ty = (w - tw) // 2, (h // 2) - 50
        cv2.putText(image, warn_text, (tx, ty), font, 1.2, (255, 255, 255), 6)
        cv2.putText(image, warn_text, (tx, ty), font, 1.2, (0, 0, 255), 4)

    # ── Traffic Control Popup ──
    if traffic_alert:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(traffic_alert, font, 0.8, 2)
        box_y = h - 80
        cv2.rectangle(image, (w//2 - tw//2 - 15, box_y - th - 10), (w//2 + tw//2 + 15, box_y + 10), (0, 0, 0), -1)
        cv2.putText(image, traffic_alert, (w//2 - tw//2, box_y), font, 0.8, (0, 255, 255), 2)

    # ── Top-Left Visual Telemetry Dashboard ──
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (460, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    deviation   = stats["deviation"]
    road_status = stats["road_status"]
    
    if abs(deviation) < 3.0: 
        dev_color = (60, 220, 60)   # Green
    elif abs(deviation) < WARNING_THRESH: 
        dev_color = (0, 200, 255)  # Cyan/Amber
    else: 
        dev_color = (60, 60, 255)   # Red Warning

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "<HYBRID ADAS SYSTEM>", (20, 40), font, 0.7, (255, 255, 255), 2)
    
    is_live = "Live" in road_status
    mode_dot_color = (60, 220, 60) if is_live else (0, 150, 255)
    
    cv2.circle(image, (32, 77), 6, mode_dot_color, -1)
    if not is_live:
        cv2.circle(image, (32, 77), 11, mode_dot_color, 1)
        
    cv2.putText(image, f"Mode: {road_status}", (52, 83), font, 0.6, (230, 230, 230), 2)

    dev_dir = "L" if deviation > 0 else "R"
    cv2.putText(image, f"Dev: {abs(deviation):.1f}% {dev_dir}", (20, 130), font, 0.6, dev_color, 2)
    
    track_x, track_y = 230, 122
    track_w, track_h = 200, 8
    
    cv2.rectangle(image, (track_x, track_y), (track_x + track_w, track_y + track_h), (70, 75, 80), -1)
    cv2.line(image, (track_x + track_w//2, track_y - 4), (track_x + track_w//2, track_y + track_h + 4), (255, 255, 255), 2)
    
    max_track_scale = WARNING_THRESH * 2.0  
    scale_ratio = np.clip(-deviation / max_track_scale, -1.0, 1.0)
    
    slider_cx = int((track_x + track_w//2) + (scale_ratio * (track_w // 2)))
    cv2.rectangle(image, (slider_cx - 4, track_y - 3), (slider_cx + 4, track_y + track_h + 3), dev_color, -1)

    # ── Steering Wheel Correction Arc (Bottom Center) ──
    arc_cx, arc_cy = w // 2, h - 140
    arc_radius = 50
    cv2.ellipse(image, (arc_cx, arc_cy), (arc_radius, arc_radius), 0, 180, 360, (70, 75, 80), 2)
    cv2.ellipse(image, (arc_cx, arc_cy), (arc_radius, arc_radius), 0, 268, 272, (255, 255, 255), 2) 
    
    steer_angle = np.clip(deviation * 4.0, -70, 70)
    active_angle = 270 + steer_angle
    
    cv2.ellipse(image, (arc_cx, arc_cy), (arc_radius, arc_radius), 0, active_angle - 25, active_angle + 25, dev_color, 6)
    
    tick_in_x  = int(arc_cx + (arc_radius - 10) * np.cos(np.radians(active_angle)))
    tick_in_y  = int(arc_cy + (arc_radius - 10) * np.sin(np.radians(active_angle)))
    tick_out_x = int(arc_cx + (arc_radius + 10) * np.cos(np.radians(active_angle)))
    tick_out_y = int(arc_cy + (arc_radius + 10) * np.sin(np.radians(active_angle)))
    cv2.line(image, (tick_in_x, tick_in_y), (tick_out_x, tick_out_y), (255, 255, 255), 2)


    # ── Mini Top-Down Radar Box (Bottom Right) ──
    radar_w, radar_h = 160, 220
    radar_x0 = w - radar_w - 20
    radar_y0 = h - radar_h - 20
    
    overlay_radar = image.copy()
    cv2.rectangle(overlay_radar, (radar_x0, radar_y0), (radar_x0 + radar_w, radar_y0 + radar_h), (20, 25, 30), -1)
    cv2.addWeighted(overlay_radar, 0.85, image, 0.15, 0, image)
    cv2.rectangle(image, (radar_x0, radar_y0), (radar_x0 + radar_w, radar_y0 + radar_h), (80, 80, 90), 1)
    
    max_radar_dist = 40.0
    ego_cx = radar_x0 + radar_w // 2
    ego_cy = radar_y0 + radar_h - 25
    
    # Draw Grid Lines (10m, 20m, 30m)
    for d in [10, 20, 30]:
        grid_y = ego_cy - int((d / max_radar_dist) * (radar_h - 40))
        cv2.line(image, (radar_x0, grid_y), (radar_x0 + radar_w, grid_y), (60, 65, 70), 1)
        cv2.putText(image, f"{d}m", (radar_x0 + 3, grid_y - 3), font, 0.35, (150, 150, 150), 1)
        
    # Draw Ego Lane projection bounds
    lane_w = int(radar_w * 0.35)
    cv2.line(image, (ego_cx - lane_w//2, radar_y0), (ego_cx - lane_w//2, ego_cy), (60, 90, 60), 1)
    cv2.line(image, (ego_cx + lane_w//2, radar_y0), (ego_cx + lane_w//2, ego_cy), (60, 90, 60), 1)
    
    # Plot Dynamic Radar Targets
    for (dist, obj_cx, color) in radar_targets:
        if dist > max_radar_dist: dist = max_radar_dist
        # Map distance to Y axis
        plot_y = ego_cy - int((dist / max_radar_dist) * (radar_h - 40))
        # Map camera X bounds to Radar X bounds
        plot_x = radar_x0 + int((obj_cx / w) * radar_w)
        
        cv2.circle(image, (plot_x, plot_y), 5, color, -1)
        cv2.circle(image, (plot_x, plot_y), 5, (255, 255, 255), 1) # Clean white border
        
    # Draw Ego Car (White Triangle)
    pts = np.array([[ego_cx, ego_cy - 8], [ego_cx - 6, ego_cy + 8], [ego_cx + 6, ego_cy + 8]], np.int32)
    cv2.fillPoly(image, [pts], (255, 255, 255))
    
    cv2.putText(image, "[BIRD'S EYE RADAR]", (radar_x0 + 10, radar_y0 + 20), font, 0.45, (230, 230, 230), 1)


    # ── Top-Right Status Panel ──
    panel_w = 480
    panel_h = 135
    panel_x0 = w - panel_w - 20
    panel_y0 = 15
    
    overlay2 = image.copy()
    cv2.rectangle(overlay2, (panel_x0, panel_y0), (w - 20, panel_y0 + panel_h), (25, 30, 35), -1)
    cv2.rectangle(overlay2, (panel_x0, panel_y0), (w - 20, panel_y0 + panel_h), (120, 80, 50), 1)  
    cv2.addWeighted(overlay2, 0.85, image, 0.15, 0, image)

    cv2.putText(image, "[Lane Keeping Status]", (panel_x0 + 20, panel_y0 + 30), font, 0.6, (230, 230, 230), 2)
    
    if stats["is_warning"]:
        status_txt = "WARNING! OFF LANE"
        status_clr = (60, 60, 255) 
    else:
        status_txt = "Good Lane Keeping"
        status_clr = (60, 220, 60) 
        
    (txt_w, _), _ = cv2.getTextSize(status_txt, font, 0.9, 2)
    center_x = panel_x0 + (panel_w // 2)
    
    icon_x = center_x - (txt_w // 2) - 25
    icon_y = panel_y0 + 72
    if stats["is_warning"]:
        cv2.line(image, (icon_x-8, icon_y-8), (icon_x+8, icon_y+8), status_clr, 3)
        cv2.line(image, (icon_x+8, icon_y-8), (icon_x-8, icon_y+8), status_clr, 3)
    else:
        cv2.line(image, (icon_x-10, icon_y), (icon_x-4, icon_y+8), status_clr, 3)
        cv2.line(image, (icon_x-4, icon_y+8), (icon_x+10, icon_y-10), status_clr, 3)

    cv2.putText(image, status_txt, (center_x - (txt_w // 2) + 10, panel_y0 + 80), font, 0.9, status_clr, 2)

    road = stats['upcoming_road']
    bottom_str = f"[Upcoming Road] : {road}"
    cv2.putText(image, bottom_str, (panel_x0 + 20, panel_y0 + 120), font, 0.6, (230, 230, 230), 2)
    
    (bot_w, _), _ = cv2.getTextSize(bottom_str, font, 0.6, 2)
    arr_cx = panel_x0 + 35 + bot_w
    arr_cy = panel_y0 + 115
    arr_clr = (255, 255, 255) if road == "Stay Straight" else (0, 220, 220)
    
    if road == "Stay Straight":
        pts = np.array([[arr_cx, arr_cy+8], [arr_cx, arr_cy-8], [arr_cx-5, arr_cy], [arr_cx, arr_cy-8], [arr_cx+5, arr_cy]], np.int32)
        cv2.polylines(image, [pts[:2]], False, arr_clr, 2)
        cv2.polylines(image, [pts[2:]], False, arr_clr, 2)
    elif road == "Left Curve Ahead":
        pts = np.array([[arr_cx+8, arr_cy+8], [arr_cx-2, arr_cy+8], [arr_cx-2, arr_cy-4], [arr_cx+3, arr_cy], [arr_cx-2, arr_cy-4], [arr_cx-7, arr_cy]], np.int32)
        cv2.polylines(image, [pts[:3]], False, arr_clr, 2)
        cv2.polylines(image, [pts[3:]], False, arr_clr, 2)
    elif road == "Right Curve Ahead":
        pts = np.array([[arr_cx-8, arr_cy+8], [arr_cx+2, arr_cy+8], [arr_cx+2, arr_cy-4], [arr_cx-3, arr_cy], [arr_cx+2, arr_cy-4], [arr_cx+7, arr_cy]], np.int32)
        cv2.polylines(image, [pts[:3]], False, arr_clr, 2)
        cv2.polylines(image, [pts[3:]], False, arr_clr, 2)

    return image


def main():
    model = initialize_model()
    arg        = sys.argv[1] if len(sys.argv) > 1 else "data/test_video.mp4"
    source     = resolve_source(arg)
    cap        = cv2.VideoCapture(source)
    prev_time  = 0

    if not cap.isOpened():
        print(f"ERROR: could not open video source: {source!r}")
        return

    print(f"System Online. Source: {source!r}. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps          = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
        prev_time    = current_time

        # ── PIPELINE WITH RADAR EXTENSION ──
        frame_with_objects, fcw_warning, traffic_alert, vehicle_boxes, radar_targets = detect_obstacles(frame, model)
        lane_overlay, lane_stats = process_lanes(frame, vehicle_boxes)
        blended_frame = cv2.addWeighted(frame_with_objects, 1.0, lane_overlay, 0.55, 0)
        
        # Pass radar targets into the draw_hud function
        final_output = draw_hud(blended_frame, lane_stats, fps, fcw_warning, traffic_alert, radar_targets)

        cv2.imshow("Modular Hybrid ADAS Prototype", final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()