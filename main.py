# ─────────────────────────────────────────────────────────────────────────────
#  main.py  |  Modular Hybrid ADAS — Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import time
from obstacle_detection import initialize_model, detect_obstacles
from lane_detection import process_lanes


def draw_hud(image: "np.ndarray", stats: dict, fps: float) -> "np.ndarray":
    """
    Draws the transparent ADAS dashboard on the final frame.

    Changes from original:
      • "Route:" renamed to "Mode:" — more accurate label for the tracker state.
      • Deviation now shows direction (L = car drifting left, R = drifting right).
      • Deviation value itself is colour-coded: green < 10%, yellow 10–20%, red > 20%.
    """
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (460, 235), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    deviation   = stats["deviation"]
    brightness  = stats["brightness"]
    is_warning  = stats["is_warning"]

    # Colour logic
    vis_color  = (0, 255, 0) if brightness > 80 else (0, 255, 255)
    warn_color = (0, 0, 255) if is_warning else (0, 255, 0)
    warn_text  = "WARNING! LANE DEPARTURE" if is_warning else "Good Lane Keeping"

    if abs(deviation) < 10.0:
        dev_color = (0, 255, 0)       # green — well centred
    elif abs(deviation) < WARNING_THRESH:
        dev_color = (0, 200, 255)     # amber — drifting
    else:
        dev_color = (0, 0, 255)       # red   — departure

    # Direction label: positive deviation → lane centre is RIGHT of car centre
    # → car is drifting LEFT
    dev_dir   = "L" if deviation > 0 else "R"
    dev_label = f"Deviation: {abs(deviation):.1f}%  {dev_dir}"

    font  = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "<HYBRID ADAS SYSTEM>", (20,  40), font, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"System FPS: {fps:.1f}",  (20,  75), font, 0.6, (0, 255, 255),   2)
    cv2.putText(image, dev_label,                 (20, 110), font, 0.6, dev_color,        2)
    cv2.putText(image, f"Visibility: {brightness}",(20, 145), font, 0.6, vis_color,       2)
    cv2.putText(image, f"Mode: {stats['road_status']}", (20, 180), font, 0.6, (255,255,255), 2)
    cv2.putText(image, warn_text,                 (20, 215), font, 0.7, warn_color,       2)

    return image


# Import the warning threshold so the HUD colour logic is always in sync
try:
    from lane_detection import WARNING_THRESH
except ImportError:
    WARNING_THRESH = 20.0


def main():
    model = initialize_model()

    video_path = "data/test_video.mp4"
    cap        = cv2.VideoCapture(video_path)
    prev_time  = 0

    print("System Online. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps          = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
        prev_time    = current_time

        # ── Pipeline ──────────────────────────────────────────────────────────

        # Step A: Lane overlay (pure black bg with green fill + yellow borders)
        lane_overlay, lane_stats = process_lanes(frame)

        # Step B: Object detection bounding boxes drawn on the clean frame
        frame_with_objects = detect_obstacles(frame, model)

        # Step C: Blend lane overlay onto the object-annotated frame
        blended_frame = cv2.addWeighted(frame_with_objects, 1.0, lane_overlay, 0.4, 0)

        # Step D: Draw ADAS dashboard
        final_output = draw_hud(blended_frame, lane_stats, fps)

        cv2.imshow("Modular Hybrid ADAS Prototype", final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()