# ─────────────────────────────────────────────────────────────────────────────
#  main.py  |  Modular Hybrid ADAS — Orchestrator
#
#  CHANGE IN THIS VERSION
#    The video source was hardcoded to "data/test_video.mp4", which is the
#    opposite of "100% autonomous, works on any video or webcam."
#    It now accepts a command-line argument:
#      python main.py                  → defaults to data/test_video.mp4
#      python main.py path/to/clip.mp4 → any video file
#      python main.py 0                → webcam index 0 (built-in camera)
#      python main.py 1                → webcam index 1 (external camera)
#    No other logic changed — lane_detection.py's autonomy (no hardcoded
#    pixel coordinates, fully dynamic horizons) means it adapts to whatever
#    resolution/source main.py hands it.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import cv2
import time
from obstacle_detection import initialize_model, detect_obstacles
from lane_detection import process_lanes, WARNING_THRESH


def resolve_source(arg: str):
    """
    Turn a CLI argument into something cv2.VideoCapture understands.
    A pure integer string ("0", "1", ...) is treated as a webcam index;
    anything else is treated as a file path.
    """
    return int(arg) if arg.isdigit() else arg


def draw_hud(image, stats: dict, fps: float):
    """
    Draws the transparent ADAS dashboard on the final frame.

    Two panels, matching the reference-image layout:
      • Top-left  — system telemetry (FPS, deviation, visibility, mode).
      • Top-right — lane-keeping status + "Upcoming Road" curve classifier
        (NEW — sourced from lane_detection's _classify_curvature output).
    """
    h, w = image.shape[:2]

    # ── Top-left telemetry panel ──────────────────────────────────────────────
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (460, 235), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    deviation  = stats["deviation"]
    brightness = stats["brightness"]
    is_warning = stats["is_warning"]

    vis_color  = (0, 255, 0) if brightness > 80 else (0, 255, 255)

    if abs(deviation) < 10.0:
        dev_color = (0, 255, 0)
    elif abs(deviation) < WARNING_THRESH:
        dev_color = (0, 200, 255)
    else:
        dev_color = (0, 0, 255)

    dev_dir   = "L" if deviation > 0 else "R"
    dev_label = f"Deviation: {abs(deviation):.1f}%  {dev_dir}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "<HYBRID ADAS SYSTEM>", (20,  40), font, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"System FPS: {fps:.1f}",   (20,  75), font, 0.6, (0, 255, 255),   2)
    cv2.putText(image, dev_label,                  (20, 110), font, 0.6, dev_color,        2)
    cv2.putText(image, f"Visibility: {brightness}",(20, 145), font, 0.6, vis_color,        2)
    cv2.putText(image, f"Mode: {stats['road_status']}", (20, 180), font, 0.6, (255,255,255), 2)

    # ── Top-right status panel (reference-image style) ───────────────────────
    panel_w   = 440
    panel_x0  = w - panel_w - 10
    overlay2  = image.copy()
    cv2.rectangle(overlay2, (panel_x0, 10), (w - 10, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, image, 0.4, 0, image)

    warn_color = (0, 0, 255) if is_warning else (0, 255, 0)
    warn_text  = "WARNING! LANE DEPARTURE" if is_warning else "Good Lane Keeping"

    cv2.putText(image, "[Lane Keeping Status]", (panel_x0 + 15, 35),
                font, 0.6, (255, 255, 255), 2)
    cv2.putText(image, warn_text, (panel_x0 + 15, 75),
                font, 0.8, warn_color, 2)
    cv2.putText(image, f"[Upcoming Road]: {stats['upcoming_road']}",
                (panel_x0 + 15, 110), font, 0.55, (255, 255, 255), 2)

    return image


def main():
    model = initialize_model()

    # Default to the bundled test clip if no source is given on the CLI
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

        # ── Pipeline ──────────────────────────────────────────────────────────
        lane_overlay, lane_stats = process_lanes(frame)
        frame_with_objects       = detect_obstacles(frame, model)
        blended_frame            = cv2.addWeighted(frame_with_objects, 1.0, lane_overlay, 0.55, 0)
        final_output             = draw_hud(blended_frame, lane_stats, fps)

        cv2.imshow("Modular Hybrid ADAS Prototype", final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()