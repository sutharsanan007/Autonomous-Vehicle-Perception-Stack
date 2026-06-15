import cv2
import numpy as np
import time

# --- Global Variables ---
prev_left_line = None
prev_right_line = None
dynamic_horizon_y = None

def get_horizon_y(image):
    global dynamic_horizon_y
    if dynamic_horizon_y is None: dynamic_horizon_y = int(image.shape[0] * 0.65)
    return dynamic_horizon_y

def select_region_of_interest(image):
    height, width = image.shape[:2]
    horizon = get_horizon_y(image)
    polygons = np.array([
        [(int(width * 0.10), height), (int(width * 0.90), height), 
         (int(width * 0.55), horizon), (int(width * 0.45), horizon)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    height = image.shape[0]
    horizon = get_horizon_y(image)
    y1 = height
    y2 = horizon
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    global prev_left_line, prev_right_line, dynamic_horizon_y
    left_fit, right_fit = [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters[0], parameters[1]
            if -2.0 < slope < -0.4: left_fit.append((slope, intercept))
            elif 0.4 < slope < 2.0: right_fit.append((slope, intercept))
                
    left_avg = np.average(left_fit, axis=0) if left_fit else None
    right_avg = np.average(right_fit, axis=0) if right_fit else None
    
    if left_avg is not None and right_avg is not None:
        m1, c1 = left_avg
        m2, c2 = right_avg
        if (m1 - m2) != 0:
            intersect_x = (c2 - c1) / (m1 - m2)
            intersect_y = m1 * intersect_x + c1
            min_safe_y, max_safe_y = int(image.shape[0] * 0.50), int(image.shape[0] * 0.80)
            intersect_y = max(min_safe_y, min(intersect_y, max_safe_y))
            dynamic_horizon_y = int(get_horizon_y(image) * 0.9 + intersect_y * 0.1)

    if left_avg is not None:
        left_line = make_coordinates(image, left_avg)
        if prev_left_line is not None: left_line = (prev_left_line * 0.85 + left_line * 0.15).astype(int)
        prev_left_line = left_line
    else: left_line = prev_left_line
        
    if right_avg is not None:
        right_line = make_coordinates(image, right_avg)
        if prev_right_line is not None: right_line = (prev_right_line * 0.85 + right_line * 0.15).astype(int)
        prev_right_line = right_line
    else: right_line = prev_right_line
    
    return [left_line, right_line]

def draw_hud(image, deviation, fps, is_warning):
    """Draws the transparent dashboard UI matching the capstone video style"""
    height, width = image.shape[:2]
    
    # Draw a semi-transparent black background for the HUD panel
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Set colors based on warning status
    status_color = (0, 0, 255) if is_warning else (0, 255, 0) # Red or Green
    status_text = "WARNING! OFF LANE" if is_warning else "Good Lane Keeping"

    # Write text to the screen
    cv2.putText(image, "<LANE STATUS>", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Deviation: {abs(deviation):.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, f"FPS: {fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, status_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

def process_frame_with_hud(frame, prev_time):
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    cropped_edges = select_region_of_interest(edges)
    
    lines = cv2.HoughLinesP(cropped_edges, rho=2, theta=np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
    averaged_lines = average_slope_intercept(frame, lines)
    
    lane_image = np.zeros_like(frame)
    deviation = 0
    is_warning = False

    if averaged_lines is not None:
        left_line, right_line = averaged_lines
        if left_line is not None and right_line is not None:
            
            # --- LANE DEPARTURE MATH ---
            car_center = frame.shape[1] // 2
            lane_center = (left_line[0] + right_line[0]) // 2
            
            # Calculate how far off center we are as a percentage
            max_drift = frame.shape[1] // 2
            deviation = ((lane_center - car_center) / max_drift) * 100
            
            # If deviation is more than 10%, trigger warning
            is_warning = abs(deviation) > 10.0
            carpet_color = (0, 0, 255) if is_warning else (0, 255, 0) # Red if warning, Green if safe
            # ---------------------------
            
            y_top = get_horizon_y(frame) + 25
            def adjust_x_to_y(line, target_y):
                if (line[3] - line[1]) == 0: return line[2]
                slope = (line[3] - line[1]) / (line[2] - line[0])
                intercept = line[1] - slope * line[0]
                return int((target_y - intercept) / slope)

            left_x2_adjusted = adjust_x_to_y(left_line, y_top)
            right_x2_adjusted = adjust_x_to_y(right_line, y_top)

            pts = np.array([
                [left_line[0], left_line[1]],
                [left_x2_adjusted, y_top],
                [right_x2_adjusted, y_top],
                [right_line[0], right_line[1]]
            ], np.int32)
            
            cv2.fillPoly(lane_image, [pts], carpet_color)
            cv2.line(lane_image, (left_line[0], left_line[1]), (left_x2_adjusted, y_top), (0, 255, 255), 6)
            cv2.line(lane_image, (right_line[0], right_line[1]), (right_x2_adjusted, y_top), (0, 255, 255), 6)

    # Blend the lane over the video
    final_output = cv2.addWeighted(frame, 1, lane_image, 0.4, 0)
    
    # Draw the dashboard on top of everything
    draw_hud(final_output, deviation, fps, is_warning)
    
    return final_output, current_time

def main():
    video_path = "data/test_video.mp4" 
    cap = cv2.VideoCapture(video_path)
    prev_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        processed_frame, prev_time = process_frame_with_hud(frame, prev_time)
        cv2.imshow("ADAS Dashboard - Lane Departure System", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()