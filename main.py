import cv2
import time
from obstacle_detection import initialize_model, detect_obstacles
from lane_detection import process_lanes

def draw_hud(image, stats, fps):
    """Draws the transparent dashboard on the final frame using dictionary stats."""
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (450, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    vis_color = (0, 255, 0) if stats["brightness"] > 80 else (0, 255, 255)
    status_color = (0, 0, 255) if stats["is_warning"] else (0, 255, 0)
    status_text = "WARNING! OFF LANE" if stats["is_warning"] else "Good Lane Keeping"

    cv2.putText(image, "<HYBRID ADAS SYSTEM>", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"System FPS: {fps:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, f"Deviation: {abs(stats['deviation']):.1f}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, f"Visibility: {stats['brightness']}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color, 2)
    cv2.putText(image, f"Route: {stats['road_status']}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, status_text, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    return image

def main():
    # 1. Boot up AI
    model = initialize_model()
    
    # 2. Open Video Stream
    video_path = "data/test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    prev_time = 0
    
    print("System Online. Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Track Time for FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # --- THE PIPELINE ---
        # Step A: Find cars using the AI module
        frame_with_objects = detect_obstacles(frame, model)
        
        # Step B: Find curves using the Classical module
        lane_overlay, lane_stats = process_lanes(frame_with_objects)
        
        # Step C: Blend the AI frame with the green lane mask
        blended_frame = cv2.addWeighted(frame_with_objects, 1, lane_overlay, 0.4, 0)
        
        # Step D: Draw the ADAS Dashboard
        final_output = draw_hud(blended_frame, lane_stats, fps)
        
        # Show result
        cv2.imshow("Modular Hybrid ADAS Prototype", final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()